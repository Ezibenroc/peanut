import os
from .peanut import Job, logger, ExpFile


class Simdjson(Job):
    simdjson_dir = 'simdjson'
    build_dir = os.path.join(simdjson_dir, 'build')
    installfile_types = {'version': str, 'nb_calls': int, 'nb_runs': int, 'warmup_time': int, 'monitoring': int,
                         'core': int}

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
        self.nodes.run('mkdir %s' % self.build_dir)
        self.nodes.run('cmake ..', directory=self.build_dir)
        self.nodes.run('cmake --build . --config Release', directory=self.build_dir)

    def run_exp(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        self.nodes.write_files(expfile.raw_content, os.path.join(self.build_dir, 'expfile.json'))
        nb_cores = len(self.nodes.cores)
        core = install_options['core']
        assert core in range(0, nb_cores)
        command = 'numactl --physcpubind=%d --localalloc ' % core
        command += './benchmark/parse -n %d -t -i 1 expfile.json' % (install_options['nb_calls'])
        results = []
        for i in range(install_options['nb_runs']):
            start = self.add_timestamp('sub_exp_start', i)
            all_outputs = self.nodes.run(command, directory=self.build_dir)
            stop = self.add_timestamp('sub_exp_stop', i)
            for node, output in all_outputs.items():
                stdout = output.stdout.strip().split('\t')
                new_res = {
                    'start': start,
                    'stop': stop,
                    'hostname': node.host,
                    'cycle_per_byte_allocation'     : stdout[1],
                    'cycle_per_byte_stage1'         : stdout[2],
                    'cycle_per_byte_stage2'         : stdout[3],
                    'cycle_per_byte_allstages'      : stdout[4],
                    'gigabyte_per_second_allstages' : stdout[5],
                    'gigabyte_per_second_stage1'    : stdout[6],
                    'gigabyte_per_second_stage2'    : stdout[7],
                }
                results.append(new_res)
        results = ExpFile(content=results, filename='results.csv')
        self.add_content_to_archive(results.raw_content, 'results.csv')
