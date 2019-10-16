import itertools
import random
import time
import itertools
from .peanut import Job, logger, RunError


class BLASCalibration(Job):
    expfile_types = {'operation': str, 'm': int, 'n': int, 'k': int, 'lda': int, 'ldb': int, 'ldc': int}
    all_op = ['dgemm', 'dtrsm']
    expfile_header_in_file = False
    expfile_header = ['operation', 'm', 'n', 'k', 'lda', 'ldb', 'ldc']
    installfile_types = {'warmup_time': int, 'multicore': bool, 'openblas': str,
            'remote_url': str, 'path_in_repo': str, 'token_path': str, 'monitoring': int}

    @classmethod
    def check_exp(cls, exp):
        if exp['m'] < 0 or exp['n'] < 0 or (exp['operation'] != 'dtrsm' and exp['k'] < 0):
            raise ValueError('Error with experiment %s, negative size.' % exp)
        if exp['lda'] < exp['m'] or exp['ldb'] < exp['k'] or exp['ldc'] < exp['m']:
            raise ValueError('Error with experiment %s, leading dimension is too small.' % exp)
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
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration', checkout='653f49d247eb583b9d414e2b95e79653b438f87f')
        self.nodes.run('BLAS_INSTALLATION=%s make calibrate_blas' % self.nodes.working_dir,
                       directory='platform-calibration/src/calibration')
        self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        self.nodes.disable_idle_state()
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
        if install_options['monitoring'] > 0:
            self.start_monitoring(period=install_options['monitoring'])
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
                command = 'tmux new-session -d -s tmux_blas_%d "OMP_NUM_THREADS=1' % i
                command += ' %s %s %s -l 1 -o %s"' % (ldlib, numactl, cmd, filename)
                self.nodes.run(command, directory=path)
            # Waiting for all the commands to be finished
            while True:
                try:
                    time.sleep(60)
                    self.nodes.run('tmux ls | grep tmux_blas')
                except RunError:
                    break
            # Adding a core ID column to each file, then merge all the files into a single one
            for core, filename in enumerate(monocore_files):
                self.nodes.run('awk \'{print $0",%d"}\' %s > tmp && mv tmp %s' % (core, filename, filename), directory=path)
            self.nodes.run('cat %s > ./result.csv' % (' '.join(monocore_files)), directory=path)
        if install_options['monitoring'] > 0:
            self.stop_monitoring()
        # Adding a header to the file
        self.nodes.run("sed -i '1s/^/function,m,n,k,lda,ldb,ldc,timestamp,duration,core\\n/' ./result.csv", directory=path)
        self.add_local_to_archive(path + '/result.csv')

    @classmethod
    def gen_exp(cls, max_prod=int(1e10), max_size=15500):

        def get_sizes(N, target_product):
            '''
            Return a list of N random sizes such that their product is close to the target
            (would be exactly the target without the rounding).
            '''
            if N == 1:
                return [target_product]
            s = round(random.uniform(1, target_product**(1/N)))
            return [s] + get_sizes(N-1, round(target_product/s))

        def get_sizes_limit(N, target_product, max_size):
            '''
            Return a list of N random sizes such that their product is close to the target
            and no size is larger than the limit.
            '''
            while True:
                sizes = get_sizes(N, target_product)
                if all(i <= max_size for i in sizes):
                    return sizes

        def get_batch(nb_batch, N, target_product, max_size):
            '''
            Return a list of nb_batch * N! lists.
            '''
            result = []
            for i in range(nb_batch):
                result.extend(itertools.permutations(get_sizes_limit(N, target_product, max_size)))
            return result

        products = random.sample(range(1, int(max_prod)), 30)
        products = list(range(max_prod, 10, -max_prod//30))
        for i in range(len(products)):
            products[i] += random.randint(-max_prod//1000, max_prod//1000)
        sizes = []
        for prod in products:
            sizes.extend(get_batch(3, 3, prod, max_size))
        # Adding special sizes
        sizes.append((2048, 2048, 2048))
        for i in range(1, 5):
            sizes.append((i, i, i))
        random.shuffle(sizes)
        return [{'operation': 'dgemm', 'm': m, 'n': n, 'k': k} for (m, n, k) in sizes]
