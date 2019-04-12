import os
from .peanut import Job, logger, RunError


class StressTest(Job):
    expfile_types = {'size': int, 'nb_calls': int, 'nb_runs': int, 'nb_sleeps': int, 'sleep_time': float}
    expfile_header_in_file = True

    @classmethod
    def check_exp(cls, exp):
        for k, v in exp.items():
            if v < 0:
                raise ValueError('Error with experiment %s, negative %s' % (exp, k))

    def setup(self):
        super().setup()
        self.apt_install(
            'build-essential',
            'python3',
            'python3-dev',
            'python3-numpy',
            'libopenblas-base',
            'libopenblas-dev',
            'zip',
            'make',
            'git',
            'time',
            'hwloc',
            'pciutils',
            'net-tools',
            'cpufrequtils',
            'linux-cpupower',
            'numactl',
        )
        self.git_clone('https://github.com/Ezibenroc/Stress-Test', 'stress-test')
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_performance()
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        for i, exp in enumerate(expfile):
            perf_file = os.path.join(self.nodes.working_dir, 'stress_perf_%d.csv' % i)
            temp_file = os.path.join(self.nodes.working_dir, 'stress_temp_%d.csv' % i)
            cmd = 'python3 stress-test/stress_test.py --size %d --nb_calls %d' % (exp['size'], exp['nb_calls'])
            cmd += ' --nb_runs %d --nb_sleeps %d --sleep_time %d' % (exp['nb_runs'], exp['nb_sleeps'], exp['sleep_time'])
            cmd += ' --temp_output %s --perf_output %s' % (temp_file, perf_file)
            self.nodes.run(cmd)
            self.add_local_to_archive(perf_file)
            self.add_local_to_archive(temp_file)

    @classmethod
    def gen_exp(cls):
        return [{
            'size': 16384,
            'nb_calls': 10,
            'nb_runs': 5,
            'nb_sleeps': 60,
            'sleep_time': 1.0
            }]
