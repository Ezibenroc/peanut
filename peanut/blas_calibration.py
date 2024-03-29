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
        if exp['lda'] < exp['m'] or exp['ldb'] < exp['n'] or exp['ldc'] < exp['m']:
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
        openblas_version = install_options['openblas']
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout=openblas_version)
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.nodes.run('ln -s libopenblas.so libblas.so', directory='lib')
        patch = None if matrix_init == 'random' else self.initialization_patch(matrix_init)
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration',
                       checkout='b04b719df563185a8420329d41a1c03b4eedf40a', patch=patch)
        make_var = 'CFLAGS="-DMASK_SIZE=%d"' % matrix_mask if matrix_mask else ''
        self.nodes.run('BLAS_INSTALLATION=%s make calibrate_blas %s' % (self.nodes.working_dir, make_var),
                       directory='platform-calibration/src/calibration')
        return self

    @staticmethod
    def get_alloc_size(experiments):
        max_m = max(exp['m'] for exp in experiments)
        max_n = max(exp['n'] for exp in experiments)
        max_k = max(exp['k'] for exp in experiments)
        return (max_m*max_k + max_k*max_n + max_m*max_n)*8

    def run_exp(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        warmup = install_options['warmup_time']
        ldlib = 'LD_LIBRARY_PATH=%s/lib' % self.nodes.working_dir
        cmd = './calibrate_blas -s ./zoo_sizes'
        nb_cores = len(self.nodes.cores)
        alloc_size = self.get_alloc_size(expfile)
        if not install_options['multicore']:
            alloc_size *= nb_cores
        mem_available = self.nodes.run('grep MemAvailable /proc/meminfo')
        for node, result in mem_available.items():
            available = result.stdout.strip().split(' ')
            assert available[-1] == 'kB'
            available = int(available[-2]) * 1024
            if available < alloc_size:
                msg = 'The experiment will allocate about %d GB of memory, but only %d GB are available on node %s'
                logger.warning(msg % (alloc_size*1e-9, available*1e-9, node.host))
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(expfile.raw_content, path + '/zoo_sizes')
        if install_options['multicore']:
            filename = './result.csv'
            self.nodes.run('OMP_NUM_THREADS=%d %s %s -o %s' % (nb_cores, ldlib, cmd, filename),
                           directory=path)
            # Adding a column "all" at the end of the CSV, representing the cores used
            self.nodes.run('awk \'{print $0",all"}\' %s > tmp && mv tmp %s' % (filename, filename), directory=path)
        else:
            numactl_str = 'numactl --physcpubind=%d --localalloc'
            monocore_files = {}
            cores = [c[0] for c in self.nodes.cores]
            for i in cores:
                numactl = numactl_str % i
                filename = 'result_monocore_%d.csv' % i
                monocore_files[i] = filename
                command = 'tmux new-session -d -s tmux_blas_%d "OMP_NUM_THREADS=1' % i
                command += ' %s %s %s -l 1 -o %s"' % (ldlib, numactl, cmd, filename)
                self.nodes.run(command, directory=path)
            nb = int(self.nodes.run_unique('tmux ls | grep tmux_blas | wc -l').stdout.strip())
            if nb != nb_cores:
                logger.warning('Started %d processes, but only %d are alive' % (nb_cores, nb))
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
            for core, filename in monocore_files.items():
                self.nodes.run('awk \'{print $0",%d"}\' %s > tmp && mv tmp %s' % (core, filename, filename), directory=path)
            self.nodes.run('cat %s > ./result.csv' % (' '.join(monocore_files.values())), directory=path)
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
    def gen_exp(cls, max_prod=int(1e10), max_size=15500, fixed_sizes={}):
#    def gen_exp(cls, max_prod=int(1e12), max_size=80000, fixed_sizes={'k': 512}): # multithread case, with K fixed
        '''
        Generate a random sequence of experiments, list of tuples (m,n,k,lda,ldb,ldc) such that:
        - The product m*n*k is regularly and uniformly distributed in [1, max_prod] (with some randomness).
        - All the elements of the tuple are bounded by max_size.
        - A subset of (m, n, k) can be fixed with the argument fixed_sizes (e.g. fixed_sizes={'k': 128}).
        '''
        assert set(fixed_sizes.keys()) <= {'m', 'n', 'k'}
        assert all([isinstance(val, int) for val in fixed_sizes.values()])
        # Replacing the parameter name by its index (m->0, n->1, k->2)
        parameters = {k:v for k, v in zip('mnk', range(3))}
        fixed_sizes = {parameters[k]:v for k, v in fixed_sizes.items()}
        base_sizes = [-1]*3
        for idx,size in fixed_sizes.items():
            base_sizes[idx] = size
        non_fixed_indices = list(sorted(set(range(3)) - set(fixed_sizes.keys())))

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

        nb_fixed = len(fixed_sizes)
        prod_fixed = 1
        for size in fixed_sizes.values():
            prod_fixed *= size

        sizes = []
        for prod in products:
            sizes.extend(get_batch(3, 3-nb_fixed, round(prod/prod_fixed), max_size))
        for i, curr_size in enumerate(sizes):
            tmp = list(base_sizes)
            for idx, s in zip(non_fixed_indices, curr_size):
                tmp[idx] = s
            sizes[i] = tmp

        # Adding special sizes
        sizes.append((2048, 2048, 2048))
        for i in range(1, 5):
            sizes.append((i, i, i))
        random.shuffle(sizes)
        # Now, let's estimate the duration and the performance for this sample.
        # We will use regression coefficients computed for CPU 0 of dahu-1 between 2019-10-18 and 2020-05-05
        total_flops = 0
        total_duration = 0
        mnk       = 7.335735e-11
        mn        = 5.547941e-11
        nk        = 3.463183e-09
        mk        = 2.369342e-09
        intercept = 2.457025e-06
        for m, n, k in sizes:
            total_flops += 2*m*n*k
            total_duration += intercept + mnk*m*n*k + mn*m*n + mk*m*k + nk*n*k
        logger.info(f'Generated a new experiment file for the dgemm calibration')
        logger.info(f'Estimated duration    : {total_duration*3:.2f} s')
        logger.info(f'Estimated performance : {total_flops/total_duration*1e-9:.2f} Gflop/s')
        result = [{
                'operation': 'dgemm',
                'm': m,
                'n': n,
                'k': k,
                'lda': max(m, n, k),
                'ldb': max(m, n, k),
                'ldc': max(m, n, k)
            } for (m, n, k) in sizes]
        alloc_size = cls.get_alloc_size(result)
        logger.info(f'Estimated allocation  : {alloc_size*1e-9:.2f} GB per core')
        return result

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
