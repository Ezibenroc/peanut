import os
from .peanut import Job, logger, ExpFile


class Simdjson(Job):
    simdjson_dir = 'simdjson'
    build_dir = os.path.join(simdjson_dir, 'build')
    simple_dir = os.path.join(simdjson_dir, 'singleheader')
    installfile_types = {'version': str, 'nb_calls': int, 'nb_runs': int, 'warmup_time': int, 'monitoring': int,
            'core': int, 'mode': str}

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
        if install_options['mode'] == 'complete':
            assert not (simdjson_version.startswith('v0.2') or simdjson_version.startswith('v0.1'))
            self.nodes.run('mkdir %s' % self.build_dir)
            self.nodes.run('cmake ..', directory=self.build_dir)
            self.nodes.run('cmake --build . --config Release', directory=self.build_dir)
        elif install_options['mode'] == 'simple':
            self.nodes.write_files(self.benchmark_src, os.path.join(self.simple_dir, 'benchmark.cpp'))
            self.nodes.run('c++ -O3 -std=c++17 -pthread  -o benchmark benchmark.cpp simdjson.cpp', directory=self.simple_dir)
        else:
            assert False

    def make_runs_complete(self, base_command, nb_calls):
        command = base_command + './benchmark/parse -n %d -t -i 1 expfile.json' % nb_calls
        all_outputs = self.nodes.run(command, directory=self.build_dir)
        results = []
        for node, output in all_outputs.items():
            stdout = output.stdout.strip().split('\t')
            new_res = {
                'hostname': node.host,
                'cycle_per_byte_allocation'     : float(stdout[1]),
                'cycle_per_byte_stage1'         : float(stdout[2]),
                'cycle_per_byte_stage2'         : float(stdout[3]),
                'cycle_per_byte_allstages'      : float(stdout[4]),
                'gigabyte_per_second_allstages' : float(stdout[5]),
                'gigabyte_per_second_stage1'    : float(stdout[6]),
                'gigabyte_per_second_stage2'    : float(stdout[7]),
            }
            results.append(new_res)
        return results

    def make_runs_simple(self, base_command, nb_calls, nb_bytes):
        command = base_command + './benchmark expfile.json %d' % nb_calls
        all_outputs = self.nodes.run(command, directory=self.simple_dir)
        results = []
        for node, output in all_outputs.items():
            all_exp = output.stdout.strip().split('\n')
            for exp in all_exp:
                new_res = {
                    'hostname': node.host,
                    'duration': float(exp),
                    'gigabytes_per_second': nb_bytes/1e9/float(exp),
                }
                results.append(new_res)
        return results

    def run_exp(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        self.nodes.write_files(expfile.raw_content, os.path.join(self.build_dir, 'expfile.json'),
                                                    os.path.join(self.simple_dir, 'expfile.json'))
        nb_cores = len(self.nodes.cores)
        core = install_options['core']
        assert core in range(0, nb_cores)
        command = 'numactl --physcpubind=%d --localalloc ' % core
        results = []
        nb_bytes = len(expfile.raw_content.encode('utf-8'))
        for i in range(install_options['nb_runs']):
            start = self.add_timestamp('sub_exp_start', i)
            if install_options['mode'] == 'complete':
                new_results = self.make_runs_complete(command, install_options['nb_calls'])
            else:
                new_results = self.make_runs_simple(command, install_options['nb_calls'], nb_bytes)
            stop = self.add_timestamp('sub_exp_stop', i)
            for d in new_results:
                d['start'] = start
                d['stop'] = stop
                results.append(d)
        results = ExpFile(content=results, filename='results.csv')
        self.add_content_to_archive(results.raw_content, 'results.csv')


    benchmark_src = r'''
#include <vector>
#include <iostream>
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


void measure_calls(simdjson::padded_string &json, int nb_iter) {
    std::vector<double> durations;
    durations.resize(nb_iter);
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
        cout << durations[iter] << endl;
    }
}


int main(int argc, char *argv[]) {
    if(argc != 3) {
        cerr << "Syntax: " << argv[0] << "<input_file> <nb_iterations>" << endl;
        exit(1);
    }
    const char *filename = argv[1];
    int nb_iter = atoi(argv[2]);
    simdjson::padded_string json = simdjson::get_corpus(filename);
    measure_calls(json, nb_iter);
}
'''
