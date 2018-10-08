import itertools
import random
from .peanut import Job, logger


class BLASCalibration(Job):
    expfile_types = {'operation': str, 'm': int, 'n': int, 'k': int}
    all_op = ['dgemm', 'dtrsm']
    expfile_header = False

    @classmethod
    def check_exp(cls, exp):
        if exp['m'] < 0 or exp['n'] < 0 or (exp['operation'] != 'dtrsm' and exp['k'] < 0):
            raise ValueError('Error with experiment %s, negative size.' % exp)
        if exp['operation'] not in cls.all_op:
            raise ValueError('Error with experiment %s, unknown operation.' % exp)

    def setup(self):
        super().setup()
        self.apt_install(
            'build-essential',
            'python3',
            'python3-dev',
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
            'tmux',
        )
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout='v0.3.1')
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.nodes.run('ln -s libopenblas.so libblas.so', directory='lib')
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration')
        self.nodes.run('BLAS_INSTALLATION=%s make calibrate_blas' % self.nodes.working_dir,
                       directory='platform-calibration/src/calibration')
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_performance()
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        ldlib = 'LD_LIBRARY_PATH=%s/lib' % self.nodes.working_dir
        cmd = './calibrate_blas -s ./zoo_sizes'
        nb_cores = len(self.nodes.cores)
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(expfile.raw_content, path + '/zoo_sizes')
        self.nodes.run('OMP_NUM_THREADS=%d %s %s -o ./result_multicore.csv' % (nb_cores, ldlib, cmd),
                       directory=path)
        numactl_str = 'numactl --physcpubind=%d --localalloc'
        numactl = numactl_str % (nb_cores-1)
        self.nodes.run('OMP_NUM_THREADS=1 %s %s %s -o ./result_monocore.csv' % (ldlib, numactl, cmd),
                       directory=path)
        for i in range(nb_cores-1):
            numactl = numactl_str % i
            self.nodes.run('tmux new-session -d -s tmux_%d "OMP_NUM_THREADS=1 %s %s %s -l"' % (i, ldlib, numactl, cmd),
                           directory=path)
        numactl = numactl_str % (nb_cores-1)
        self.nodes.run('OMP_NUM_THREADS=1 %s %s %s -o ./result_monocore_contention.csv' % (ldlib, numactl, cmd),
                       directory=path)
        self.add_local_to_archive(path + '/result_multicore.csv')
        self.add_local_to_archive(path + '/result_monocore.csv')
        self.add_local_to_archive(path + '/result_monocore_contention.csv')
        self.nodes.run('tmux kill-server')

    @classmethod
    def gen_exp(cls):
        max_size = 4000
        exp = []
        for _ in range(100):
            m = random.randint(1, max_size)
            n = random.randint(1, max_size)
            k = random.randint(1, max_size)
            for op in cls.all_op:
                if op == 'dtrsm':
                    exp.append({'operation': op, 'm': m, 'n': n, 'k': -1})
                else:
                    exp.append({'operation': op, 'm': m, 'n': n, 'k': k})
        exp *= 5
        random.shuffle(exp)
        return exp
