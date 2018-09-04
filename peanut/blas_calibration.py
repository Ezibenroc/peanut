import itertools
import random
from .peanut import Job, logger


class BLASCalibration(Job):
    expfile_types = {'size': int, 'operation': str}
    all_op = ['dgemm', 'dtrsm']

    @classmethod
    def check_exp(cls, exp):
        if exp['size'] < 0:
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
        )
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout='v0.3.1')
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration')
        self.nodes.run('BLAS_INSTALLATION=%s make calibrate_blas' % self.nodes.working_dir,
                       directory='platform-calibration/src/calibration')
        self.nodes.enable_hyperthreading()
        self.nodes.set_frequency_performance()
        return self

    def run_exp(self):
        nb_cores = len(self.nodes.cores)
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(self.expfile.raw_content, path + '/zoo_sizes')
        self.nodes.run('OMP_NUM_THREADS=%d ./calibrate_blas -s ./zoo_sizes -o ./result_multicore.csv' % nb_cores,
                       directory=path)
        self.nodes.run('OMP_NUM_THREADS=1 ./calibrate_blas -s ./zoo_sizes -o ./result_monocore.csv',
                       directory=path)
        for i in range(nb_cores-1):
            self.nodes.run('tmux new-session -d -s tmux_%d "OMP_NUM_THREADS=1 ./calibrate_blas -s ./zoo_sizes -l"' % i,
                           directory=path)
        self.nodes.run('OMP_NUM_THREADS=1 ./calibrate_blas -s ./zoo_sizes -o ./result_monocore_contention.csv',
                       directory=path)
        self.add_local_to_archive(path + '/result_multicore.csv')
        self.add_local_to_archive(path + '/result_monocore.csv')
        self.add_local_to_archive(path + '/result_monocore_contention.csv')
        self.nodes.run('tmux kill-server')

    @classmethod
    def gen_exp(cls):
        sizes = {int(10**random.uniform(0, 3.6)) for _ in range(1000)}
        exp = list(itertools.product(cls.all_op, sizes))
        exp *= 50
        random.shuffle(exp)
        return [{'operation': op, 'size': size} for op, size in exp]
