import itertools
import random
import time
from .peanut import Job, logger, RunError


class BLASCalibration(Job):
    expfile_types = {'operation': str, 'm': int, 'n': int, 'k': int}
    all_op = ['dgemm', 'dtrsm']
    expfile_header_in_file = False
    expfile_header = ['operation', 'm', 'n', 'k']
    installfile_types = {'warmup_time': int, 'multicore': bool, 'openblas': str}

    @classmethod
    def check_exp(cls, exp):
        if exp['m'] < 0 or exp['n'] < 0 or (exp['operation'] != 'dtrsm' and exp['k'] < 0):
            raise ValueError('Error with experiment %s, negative size.' % exp)
        if exp['operation'] not in cls.all_op:
            raise ValueError('Error with experiment %s, unknown operation.' % exp)

    def setup(self):
        super().setup()
        assert self.installfile is not None
        install_options = self.installfile.content
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
            'stress',
        )
        openblas_version = install_options['openblas']
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout=openblas_version)
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
        assert self.installfile is not None
        install_options = self.installfile.content
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        warmup = install_options['warmup_time']
        if warmup > 0:
            cmd = 'stress -c %d -t %ds' % (4*len(self.nodes.cores), warmup)
            self.nodes.run(cmd)
        ldlib = 'LD_LIBRARY_PATH=%s/lib' % self.nodes.working_dir
        cmd = './calibrate_blas -s ./zoo_sizes'
        nb_cores = len(self.nodes.cores)
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(expfile.raw_content, path + '/zoo_sizes')
        if install_options['multicore']:
            self.nodes.run('OMP_NUM_THREADS=%d %s %s -o ./result.csv' % (nb_cores, ldlib, cmd),
                           directory=path)
            # Adding a column "all" at the end of the CSV, representing the cores used
            self.nodes.run('awk \'{print $0, "all"}\' %s > tmp && mv tmp %s' % (filename, filename), directory=path)
        else:
            numactl_str = 'numactl --physcpubind=%d --localalloc'
            monocore_files = []
            for i in range(nb_cores):
                numactl = numactl_str % i
                filename = 'result_monocore_%d.csv' % i
                monocore_files.append(filename)
                command = 'tmux new-session -d -s tmux_%d "OMP_NUM_THREADS=1' % i
                command += ' %s %s %s -l 2 -o %s"' % (ldlib, numactl, cmd, filename)
                self.nodes.run(command, directory=path)
            # Waiting for all the commands to be finished
            while True:
                try:
                    time.sleep(60)
                    self.nodes.run('tmux ls')
                except RunError:
                    break
            # Adding a core ID column to each file, then merge all the files into a single one
            for core, filename in enumerate(monocore_files):
                self.nodes.run('awk \'{print $0",%d"}\' %s > tmp && mv tmp %s' % (core, filename, filename), directory=path)
            self.nodes.run('cat %s > ./result.csv' % (' '.join(monocore_files)), directory=path)
        # Adding a header to the file
        self.nodes.run("sed -i '1s/^/function,m,n,k,timestamp,duration,core\\n/' ./result.csv", directory=path)
        self.add_local_to_archive(path + '/result.csv')

    @classmethod
    def gen_exp(cls):
        max_size = 4000
        exp = []
        for _ in range(60):
            m = random.randint(1, max_size)
            n = random.randint(1, max_size)
            k = random.randint(1, max_size)
            for op in cls.all_op:
                exp.append({'operation': op, 'm': m, 'n': n, 'k': k})
        for _ in range(30):
            big_sizes = [random.randint(1, max_size*4) for _ in range(2)]
            small_size = random.randint(1, max_size//16)
            exp.append({'operation': 'dgemm', 'm': big_sizes[0], 'n': big_sizes[1], 'k': small_size})
            exp.append({'operation': 'dgemm', 'm': big_sizes[0], 'n': small_size, 'k': big_sizes[1]})
            exp.append({'operation': 'dgemm', 'm': small_size, 'n': big_sizes[0], 'k': big_sizes[1]})
            exp.append({'operation': 'dtrsm', 'm': big_sizes[0], 'n': small_size, 'k': -1})
            exp.append({'operation': 'dtrsm', 'm': small_size, 'n': big_sizes[0], 'k': -1})
        for e in exp:
            if e['operation'] == 'dtrsm':
                e['k'] = -1
        exp *= 3
        random.shuffle(exp)
        return exp
