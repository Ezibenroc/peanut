import os
import time
from .peanut import Job, logger, ExpFile, RunError


class Simdjson(Job):
    simdjson_dir = 'simdjson'
    build_dir = os.path.join(simdjson_dir, 'singleheader')
    installfile_types = {'version': str, 'nb_calls': int, 'nb_runs': int, 'warmup_time': int, 'monitoring': int}

    def setup(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        self.apt_install(
            'build-essential',
            'zip',
            'make',
            'git',
            'time',
            'hwloc',
            'pciutils',
            'cmake',
            'cpufrequtils',
            'linux-cpupower',
        )
        simdjson_version = install_options['version']
        self.git_clone('https://github.com/simdjson/simdjson.git', self.simdjson_dir, checkout=simdjson_version)
        self.nodes.write_files(self.benchmark_src, os.path.join(self.build_dir, 'benchmark.cpp'))
        self.nodes.run('c++ -O3 -std=c++17 -pthread  -o benchmark benchmark.cpp simdjson.cpp', directory=self.build_dir)
        if self.nodes.frequency_information.active_driver == 'intel_pstate':
            self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
            self.nodes.disable_hyperthreading()
            self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        else:
            self.nodes.disable_hyperthreading()
            self.nodes.set_frequency_performance()
        self.nodes.disable_idle_state()
        self.nodes.enable_turboboost()


    def run_exp(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        self.nodes.write_files(expfile.raw_content, os.path.join(self.build_dir, 'expfile.json'))
        nb_cores = len(self.nodes.cores)
        numactl_str = 'numactl --physcpubind=%d --localalloc '
        monocore_files = []
        nb_runs = install_options['nb_runs']
        nb_calls = install_options['nb_calls']
        for i in range(nb_cores):
            numactl = numactl_str % i
            filename = 'result_%d.csv' % i
            monocore_files.append(filename)
            command = 'tmux new-session -d -s tmux_simdjson_%d' % i
            command += ' "%s ./benchmark expfile.json %d %d %d %s"' % (numactl, nb_runs, nb_calls, i, filename)
            self.nodes.run(command, directory=self.build_dir)
        # Waiting for all the commands to be finished
        waiting_nodes = list(self.nodes)
        while len(waiting_nodes) > 0:
            node = waiting_nodes[0]
            try:
                node.run('tmux ls | grep tmux_simdjson')
            except RunError:  # this node has finished, let's remove it
                waiting_nodes = waiting_nodes[1:]
            else:  # this node has not finished yet
                time.sleep(6)
        # Merging the per-core files into a single file for each host
        header = self.nodes.run_unique('head -n 1 %s' % monocore_files[0], directory=self.build_dir).stdout.strip()
        self.nodes.run('rm -f ./result.csv && touch ./result.csv', directory=self.build_dir)
        for filename in monocore_files:
            self.nodes.run('tail -n +2 %s >> ./result.csv' % filename, directory=self.build_dir)
        # Adding a hostname column to each file
        result_files = []
        for node in self.nodes:
            name = node.hostnames[0]
            resfile = 'result_%s.csv' % name
            result_files.append(resfile)
            node.run('awk \'{print $0",%s"}\' result.csv > %s' % (name, resfile), directory=self.build_dir)
            self.director.run("rsync -a '%s:%s' ." % (name, os.path.join(self.nodes.working_dir, self.build_dir, resfile)),
                directory=self.build_dir)
        self.director.run('cat %s > ./result.csv' % (' '.join(result_files)), directory=self.build_dir)
        # Adding the header back
        self.director.run("sed -i '1s/^/%s,hostname\\n/' ./result.csv" % header, directory=self.build_dir)
        # Retrieving the file
        self.add_local_to_archive(os.path.join(self.build_dir, './result.csv'))


    benchmark_src = r'''
#include <sys/time.h>
#include <sys/stat.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <chrono>
#include <ctime>

#include "simdjson.h"

using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::to_string;


void exit_error(string message) {
    cerr << message << endl;
    exit(EXIT_FAILURE);
}


std::string get_timestamp(void) {
    // Taken from https://codereview.stackexchange.com/a/11922
    timeval curTime;
    gettimeofday(&curTime, NULL);
    int milli = curTime.tv_usec / 1000;

    char buffer [80];
    strftime(buffer, 80, "%Y-%m-%d %H:%M:%S", localtime(&curTime.tv_sec));

    char currentTime[84] = "";
    sprintf(currentTime, "%s:%03d", buffer, milli);
    return std::string(currentTime);
}


long get_file_size(std::string filename) {
    // Taken from https://stackoverflow.com/a/6039648/4110059
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}


void measure_calls(simdjson::padded_string &json, long filesize, int proc_id, int run_id, int nb_iter, std::ofstream &outfile) {
    std::vector<double> durations;
    durations.resize(nb_iter);
    auto start_iterations = get_timestamp();
    for(int iter=0; iter<nb_iter; iter++) {
        auto start = std::chrono::steady_clock::now();
        auto result = build_parsed_json(json);
        if( ! result.is_valid() ) {
            // something went wrong
            std::cout << result.get_error_message() << std::endl;
            exit(1);
        }
        auto end = std::chrono::steady_clock::now();
        durations[iter] = ((std::chrono::duration<double>)(end-start)).count();
    }
    for(int iter=0; iter<nb_iter; iter++) {
        outfile << start_iterations << "," << proc_id  << "," << run_id << "," << iter << "," << durations[iter];
        outfile << "," << (filesize / durations[iter] * 1e-9) << endl;
    }
}


int main(int argc, char *argv[]) {
    if(argc != 6) {
        cerr << "Syntax: " << argv[0] << "<input_file> <nb_runs> <nb_calls> <proc_id> <output_file>" << endl;
        exit(1);
    }
    const char *input_name = argv[1];
    int nb_runs = atoi(argv[2]);
    int nb_calls = atoi(argv[3]);
    int proc_id = atoi(argv[4]);
    const char *output_name = argv[5];
    long filesize = get_file_size(input_name);
    simdjson::padded_string json = simdjson::get_corpus(input_name);
    std::ofstream outfile;
    outfile.open(output_name);
    outfile << "timestamp,proc_id,run_id,call_id,duration,speed" << endl;
    for(int i = 0; i < nb_runs; i++) {
        measure_calls(json, filesize, proc_id, i, nb_calls, outfile);
    }
    outfile.close();
}
'''
