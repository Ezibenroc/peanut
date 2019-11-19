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
    installfile_types = {'warmup_time': int, 'multicore': bool, 'openblas': str, 'matrix_initialization': (str,int,float),
            'matrix_initialization_mask_size': int,
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
        assert self.installfile is not None
        install_options = self.installfile.content
        matrix_init = install_options['matrix_initialization']
        if matrix_init not in ('random', 'sequential'):
            try:
                float(matrix_init)
            except ValueError:
                logger.error('Wrong value "%s" to initialize the matrix: neither "random" nor a float' % matrix_init)
                matrix_init = 'random'
        if matrix_init == 'sequential':
            matrix_init = 'i/(double)(size*size-1)'
        matrix_mask = install_options['matrix_initialization_mask_size']
        if matrix_mask not in range(0, 64):
            logger.error('Wrong value "%s" to use as a bitmask size, should be in [0, 63]' % matrix_mask)
            matrix_mask = 0
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
        openblas_version = install_options['openblas']
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout=openblas_version)
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.nodes.run('ln -s libopenblas.so libblas.so', directory='lib')
        patch = None if matrix_init == 'random' else self.initialization_patch(matrix_init)
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration',
                       checkout='153a4a131f3d9eb28bd31a5b15eba5cb7df86999', patch=patch)
        make_var = 'CFLAGS="-DMASK_SIZE=%d"' % matrix_mask if matrix_mask else ''
        self.nodes.run('BLAS_INSTALLATION=%s make calibrate_blas %s' % (self.nodes.working_dir, make_var),
                       directory='platform-calibration/src/calibration')
        if self.nodes.frequency_information.active_driver == 'intel_pstate':
            self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
            self.nodes.disable_hyperthreading()
            self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        else:
            self.nodes.disable_hyperthreading()
            self.nodes.set_frequency_performance()
        self.nodes.disable_idle_state()
        self.nodes.enable_turboboost()
        return self

    def run_exp(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        warmup = install_options['warmup_time']
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
            waiting_nodes = list(self.nodes)
            while len(waiting_nodes) > 0:
                node = waiting_nodes[0]
                try:
                    node.run('tmux ls | grep tmux_blas')
                except RunError:  # this node has finished, let's remove it
                    waiting_nodes = waiting_nodes[1:]
                else:  # this node has not finished yet
                    time.sleep(60)
            # Adding a core ID column to each file, then merge all the files into a single one
            for core, filename in enumerate(monocore_files):
                self.nodes.run('awk \'{print $0",%d"}\' %s > tmp && mv tmp %s' % (core, filename, filename), directory=path)
            self.nodes.run('cat %s > ./result.csv' % (' '.join(monocore_files)), directory=path)
            # Adding a hostname column to each file
            result_files = []
            for node in self.nodes:
                name = node.hostnames[0]
                resfile = 'result_%s.csv' % name
                result_files.append(resfile)
                node.run('awk \'{print $0",%s"}\' result.csv > %s' % (name, resfile), directory=path)
                self.director.run("rsync -a '%s:%s' ." % (name, path + '/' + resfile), directory=path)
            self.director.run('cat %s > ./result.csv' % (' '.join(result_files)), directory=path)
        # Adding a header to the file
        self.nodes.run("sed -i '1s/^/function,m,n,k,lda,ldb,ldc,timestamp,duration,core,hostname\\n/' ./result.csv", directory=path)
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
        return [{
                'operation': 'dgemm',
                'm': m,
                'n': n,
                'k': k,
                'lda': max(m, n, k),
                'ldb': max(m, n, k),
                'ldc': max(m, n, k)
            } for (m, n, k) in sizes]

    @classmethod
    def initialization_patch(cls, value):
        return r'''
diff --git a/src/calibration/calibrate_blas.c b/src/calibration/calibrate_blas.c
index 43df27b..1338b97 100644
--- a/src/calibration/calibrate_blas.c
+++ b/src/calibration/calibrate_blas.c
@@ -142,7 +142,7 @@ double *allocate_matrix(int size) {
     assert(x == apply_mask(x, get_mask(0, 0)));
     assert(x != apply_mask(x, get_mask(0, 1)));
     for(int i = 0; i < size*size; i++) {
-        result[i] = apply_mask((double)rand()/(double)(RAND_MAX), mask);
+        result[i] = (double)%s;
     }
     return result;
 }
''' % value
