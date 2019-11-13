import os
from .peanut import Job, logger


class StressTest(Job):
    expfile_types = {'mode': str, 'size': int, 'nb_calls': int, 'nb_runs': int, 'nb_sleeps': int, 'sleep_time': float,
                     'cores': str}
    expfile_header_in_file = True

    @classmethod
    def check_exp(cls, exp):
        for k, v in exp.items():
            if k == 'mode':
                if v not in ('blas', 'stress', 'loop'):
                    raise ValueError('Error with experiment %s, unknown mode "%s"' % (exp, v))
            elif k == 'cores':
                try:
                    [int(c) for c in v.split()]
                except ValueError:
                    raise ValueError('Error with experiment %s, unparsable core list %s' % (exp, k))
            elif v < 0:
                raise ValueError('Error with experiment %s, negative %s' % (exp, k))

    def setup(self):
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
            'stress',
        )
        self.git_clone('https://github.com/Ezibenroc/Stress-Test', 'stress-test')
        self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        self.nodes.disable_idle_state()
        # self.nodes.disable_turboboost()
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        for i, exp in enumerate(expfile):
            cores = [int(c) for c in exp['cores'].split()]
            possible_cores = sum(self.nodes.cores, [])  # aggregating the list of lists
            diff = set(cores) - set(possible_cores)
            if len(diff) > 0:
                logger.error('The following cores are not available on the nodes: %s' % diff)
            temp_file = os.path.join(self.nodes.working_dir, 'stress_temp_%d.csv' % i)
            perf_file = None
            freq_file = None
            cmd = 'python3 stress_test.py'
            cmd += ' --nb_runs %d --nb_calls %d --nb_sleeps %d --sleep_time %d' % (exp['nb_runs'], exp['nb_calls'],
                                                                                   exp['nb_sleeps'], exp['sleep_time'])
            cmd += ' --temp_output %s' % temp_file
            if exp['mode'] == 'blas':
                perf_file = os.path.join(self.nodes.working_dir, 'stress_perf_%d.csv' % i)
                cmd += ' blas --size %d' % exp['size']
                cmd += ' --perf_output %s' % perf_file
            elif exp['mode'] == 'stress':
                freq_file = os.path.join(self.nodes.working_dir, 'stress_freq_%d.csv' % i)
                cmd += ' command "stress -c %d -t %ds"' % (len(self.nodes.cores), exp['size'])
                cmd += ' --freq_output %s' % freq_file
            elif exp['mode'] == 'loop':
                perf_file = os.path.join(self.nodes.working_dir, 'stress_perf_%d.csv' % i)
                freq_file = os.path.join(self.nodes.working_dir, 'stress_freq_%d.csv' % i)
                cmd += ' loop --size %d' % exp['size']
                cmd += ' --perf_output %s' % perf_file
                cmd += ' --freq_output %s' % freq_file
                cmd += ' --cores %s' % (' '.join([str(n) for n in cores]))
            self.nodes.run(cmd, directory=os.path.join(self.nodes.working_dir, 'stress-test'))
            self.add_local_to_archive(temp_file)
            if perf_file:
                self.add_local_to_archive(perf_file)
            if freq_file:
                self.add_local_to_archive(freq_file)

    @classmethod
    def gen_exp(cls):
        return [{
            'mode': 'blas',
            'size': 16384,
            'nb_calls': 10,
            'nb_runs': 5,
            'nb_sleeps': 60,
            'sleep_time': 1.0
            }]
