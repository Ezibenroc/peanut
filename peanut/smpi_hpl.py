import os
from .peanut import logger, ExpFile, RunError
from .abstract_hpl import AbstractHPL
import re
import random
from lxml import etree

float_string = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
sim_time_str = 'The simulation took (?P<simulation>%s) seconds \(after parsing and platform setup\)' % float_string
app_time_str = '(?P<application>%s) seconds were actual computation of the application' % float_string
full_time_str = 'Simulated time: (?P<full_time>%s) seconds.' % float_string
smpi_reg = re.compile('[\S\s]*%s[\S\s]*%s\n%s' % (full_time_str, sim_time_str, app_time_str))


class TopoFile:
    def __init__(self, expfile):
        self.expfile = expfile
        self.xml = etree.fromstring(expfile.raw_content)
        self.core = None
        self.hostnames = self.parse_hosts()

    def parse_hosts(self):
        AS = self.xml.findall('AS')[0]
        cluster = AS.findall('cluster')
        host_list = []
        if len(cluster) > 0:
            assert len(cluster) == 1
            cluster = cluster[0]
            self.core = int(cluster.get('core', default=1))
            prefix = cluster.get('prefix')
            suffix = cluster.get('suffix')
            radical = cluster.get('radical').split('-')
            for i in range(int(radical[0]), int(radical[1])+1):
                host_list.extend(['%s%d%s' % (prefix, i, suffix)]*self.core)
        else:
            for host in AS.findall('host'):
                hostname = host.get('id')
                nb_cores = int(host.get('core', default=1))
                if self.core is None:
                    self.core = nb_cores
                else:
                    if self.core != nb_cores:
                        logger.warning('Heterogeneous number of cores (found %d and %d).' % (self.core, nb_cores))
                host_list.extend([hostname]*nb_cores)
        return host_list


def parse_smpi(output):
    match = smpi_reg.match(output)
    try:
        simulation_time = float(match.group('simulation'))
        application_time = float(match.group('application'))
    except AttributeError:
        logger.warning('Could not parse SMPI metrics')
        return {}
    last_line = output.strip().split('\n')[-1]
    values = last_line.split()
    if values[0] != '/usr/bin/time:output' or len(values) != 6:
        logger.warning('Could not parse SMPI metrics')
        return {}
    return dict(
        simulation_time=simulation_time,
        application_time=application_time,
        usr_time=float(values[1]),
        sys_time=float(values[2]),
        major_page_fault=int(values[3]),
        minor_page_fault=int(values[4]),
        cpu_utilization=float(values[5][:-1])/100  # given in percentage, with '%'
    )


def model_to_c_code(model):
    cols = ['intercept', 'mnk', 'mn', 'mk', 'nk', 'm', 'n', 'k']
    cols = cols + ['%s_residual' % c for c in cols]
    def get_reg(reg, residual):
        if residual:
            reg = {k[:-9]: v for k, v in reg.items() if k.endswith('_residual')}
        else:
            reg = {k: v for k, v in reg.items() if not k.endswith('_residual')}
        return reg

    def model_str(reg, residual=False):
        reg = get_reg(reg, residual)
        parameters = []
        for name, param in reg.items():
            if param == 0:
                continue
            if name != 'intercept':
                parameters.append('%e*%s' % (param, name))
            else:
                parameters.append('%e' % reg[name])
        if len(parameters) == 0:
            parameters = ['0']
        return ' + '.join(parameters)

    def __return_stmt(reg):
        mu    = model_str(reg, False)
        sigma = model_str(reg, True)
        result  = '        mu    = %s;\n' % mu
        result += '        sigma = %s;\n' % sigma
        result += '        return mu + random_halfnormal_shifted(0, sigma);'
        return result

    def __reg_to_c(reg, granularity):
        tmp = {}
        for c in cols:
            try:
                tmp[c] = reg[c]
            except KeyError:
                pass
        key = 'cpu_id' if granularity == 'cpu' else 'node'
        return '''    case %d: // node %d\n%s''' % (reg[key], reg['node'], __return_stmt(tmp))

    def reg_to_c(all_reg, granularity):
        result = [__reg_to_c(reg, granularity) for reg in all_reg]
        return '\n'.join(result)

    def compute_mean_reg(reg):
        mean_reg = {}
        for var in cols:
            try:
                mean_reg[var] = sum([tmp[var] for tmp in reg]) / len(reg)
            except KeyError:
                pass
        return mean_reg

    def dump_reg(all_reg, granularity):
        reg_code = reg_to_c(all_reg, granularity)
        reg_code_default = __return_stmt(compute_mean_reg(all_reg))
        switch = 'get_cpuid' if granularity == 'cpu' else 'get_nodeid'
        reg_code = 'double mu, sigma;\nswitch(%s()) {\n%s\n    default:\n%s\n}' % (switch, reg_code, reg_code_default)
        return reg_code

    try:
        granularity = model['metadata']['granularity']
    except KeyError:  # all format, the granularity was not here
        granularity = 'cpu'
    assert granularity in ('cpu', 'node')

    return dump_reg(model['model'], granularity)


class SMPIHPL(AbstractHPL):
    installfile_types = {'stochastic_network': bool, 'stochastic_cpu': bool, 'disable_hpl_kernels': bool,
            'disable_nondgemm_randomness': bool, 'loopback_model': bool, 'random_seed': int,
            'simgrid_version': str,
                         **AbstractHPL.installfile_types}

    def setup(self):
        super().setup()
        assert self.installfile is not None
        assert len(self.expfile) == 3
        assert {f.extension for f in self.expfile} == {'xml', 'csv', 'yaml'}
        install_options = self.installfile.content
        files = {f.extension: f for f in self.expfile}
        expfile = files['csv']
        dgemm_model = files['yaml'].content
        dgemm_c = model_to_c_code(dgemm_model)
        self.add_content_to_archive(dgemm_c, 'dgemm_model.c')
        self.apt_install('python3', 'libboost-dev', 'pajeng')
        self.git_clone('https://github.com/Ezibenroc/memwatch.git', 'memwatch')
        if install_options['stochastic_network']:
            assert not install_options['loopback_model']
            simgrid_patch = self.simgrid_stochastic_patch
        elif install_options['loopback_model']:
            simgrid_patch = self.simgrid_loopback_patch + self.simgrid_debug_patch
        else:
            simgrid_patch = None
        self.git_clone('https://framagit.org/simgrid/simgrid.git', 'simgrid',
                       checkout=install_options['simgrid_version'], patch=simgrid_patch)
        self.nodes.run('mkdir build && cd build && cmake -Denable_documentation=OFF ..', directory='simgrid')
        self.nodes.run('make -j 64 && make install', directory='simgrid/build')
        hpl_branch = 'master'
        patches = [self.makefile_patch, AbstractHPL.first_bcast_trace_patch]
        if not install_options['stochastic_cpu']:
            patches.append(self.no_noise_patch)
        if install_options['terminate_early']:
            patches.append(self.hpl_early_termination_patch)
        if install_options['insert_bcast']:
            patches.append(self.hpl_bcast_patch)
        if install_options['disable_hpl_kernels']:
            patches.append(self.no_hpl_kernels_patch)
        if install_options['disable_nondgemm_randomness']:
            patches.append(self.no_random_kernels_patch)
        patch = '\n'.join(patches) if patches else None
        self.git_clone('https://github.com/Ezibenroc/hpl.git', self.hpl_dir, patch=patch, checkout=hpl_branch)
        self.nodes.run('make startup arch=SMPI', directory=self.hpl_dir)
        options = '-DSMPI_OPTIMIZATION -DSMPI_SEED=%d' % install_options['random_seed']
        if install_options['trace_execution']:
            options += ' -DSMPI_MEASURE'
        self.nodes.write_files(dgemm_c, os.path.join(self.hpl_dir, 'src/blas/dgemm_model.c'))
        while True:
            try:
                self.nodes.run('make SMPI_OPTS="%s" arch=SMPI' % options, directory=self.hpl_dir)
            except RunError as e:  # for some reason, this command fails sometime...
                msg = str(e).split('\n')[0]
                logger.error('Previous command failed with message %s' % msg)
            else:
                break
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_performance()
        self.nodes.run('sysctl -w vm.overcommit_memory=1')
        self.nodes.run('sysctl -w vm.max_map_count=2000000000')
        self.nodes.run('mkdir -p /root/huge')
        self.nodes.run('mount none /root/huge -t hugetlbfs -o rw,mode=0777')
        self.nodes.write_files('1', '/proc/sys/vm/nr_hugepages')

    def run_exp(self):
        install_options = self.installfile.content
        results = []
        files = {f.extension: f for f in self.expfile}
        platform = TopoFile(files['xml'])
        expfile = files['csv']
        nb_cores = platform.core
        self.nodes.write_files(platform.expfile.raw_content, os.path.join(self.hpl_dir, 'bin/SMPI/platform.xml'))
        self.nodes.write_files('\n'.join(platform.hostnames), os.path.join(self.hpl_dir, 'bin/SMPI/hosts.txt'))
        for i, exp in enumerate(expfile):
            proc_p = exp['proc_p']
            proc_q = exp['proc_q']
            nb_hpl_proc = proc_p * proc_q
            process_per_node = exp['process_per_node']
            thread_per_process = exp['thread_per_process']
            if nb_cores % (process_per_node*thread_per_process) != 0:
                msg = 'Requested %d process per node and %d thread per process, but %d cores are available'
                logger.warning(msg % (process_per_node, thread_per_process, nb_cores))
            hpl_file = self.generate_hpl_file(**exp)
            self.nodes.write_files(hpl_file, os.path.join(self.hpl_dir, 'bin/SMPI/HPL.dat'))

            memwatch_file = os.path.join(self.nodes.working_dir, 'memory_%d.csv' % i)
            memwatch_script = os.path.join(self.nodes.working_dir, 'memwatch/memwatch.py')

            cmd = 'TIME="/usr/bin/time:output %U %S %F %R %P" '
            cmd += 'LD_LIBRARY_PATH=/tmp/lib '
            subcmd = 'smpirun -wrapper /usr/bin/time --cfg=smpi/privatize-global-variables:dlopen -np %d ' % nb_hpl_proc
            subcmd += '--cfg=smpi/simulate-computation:no '
            if install_options['trace_execution']:
                paje_file = os.path.join(self.director.working_dir, 'trace_%d.paje' % i)
                subcmd += '--cfg=tracing:yes --cfg=tracing/filename:%s --cfg=tracing/smpi:1 ' % paje_file
                subcmd += '--cfg=tracing/smpi/display-sizes:yes '
                subcmd += '--cfg=tracing/smpi/computing:yes '
            subcmd += '--cfg=smpi/shared-malloc-hugepage:/root/huge '
            subcmd += '--cfg=smpi/shared-malloc-blocksize:%d ' % (1 << 21)
            subcmd += '--cfg=smpi/display-timing:yes -platform platform.xml -hostfile hosts.txt ./xhpl'
            cmd += 'python3 %s -t 1 -o %s "%s"' % (memwatch_script, memwatch_file, subcmd)
            output = self.director.run_unique(cmd, directory=self.hpl_dir+'/bin/SMPI')
            self.add_local_to_archive(memwatch_file)
            if install_options['trace_execution']:
                mpi_trace = 'trace_mpi_%d.csv' % i
                blas_trace = os.path.join(self.director.working_dir, 'trace_blas_%d.csv' % i)
                self.director.run('pj_dump -u %s | grep -v MPI_Iprobe > %s' % (paje_file, mpi_trace))
                self.director.run('cat blas*trace > %s' % blas_trace, directory=self.hpl_dir+'/bin/SMPI')
                self.nodes.run('rm -f blas*trace', directory=self.hpl_dir+'/bin/SMPI')
                self.add_local_to_archive(mpi_trace)
                self.add_local_to_archive(blas_trace)
            total_time, gflops, residual = self.parse_hpl(output.stdout)
            new_res = dict(exp)
            new_res['time'] = total_time
            new_res['gflops'] = gflops
            smpi_info = parse_smpi(output.stderr)
            for key, val in smpi_info.items():
                new_res[key] = val
            results.append(new_res)
        results = ExpFile(content=results, filename='results.csv')
        self.add_content_to_archive(results.raw_content, 'results.csv')

    @classmethod
    def gen_exp(cls):
        factors = dict(cls.expfile_sets)
        factors['matrix_size'] = [int(x) for x in [5e5, 1e6, 2e6, 4e6]]
        factors['block_size'] = [2**7]
        factors['proc_p'] = [16, 32, 64, 128]
        factors['proc_q'] = [32]
        factors['rfact'] = [2]
        factors['pfact'] = [1]
        factors['mem_align'] = [8]
        factors['bcast'] = [2]
        factors['swap'] = [0]
        factors['depth'] = [1]
        factors['process_per_node'] = [32]
        factors['thread_per_process'] = [1]
        exp = cls.fact_design(factors)
        random.shuffle(exp)
        return exp

    no_noise_patch = r'''
diff --git a/src/blas/HPL_dgemm.c b/src/blas/HPL_dgemm.c
index 0119820..dfe826c 100644
--- a/src/blas/HPL_dgemm.c
+++ b/src/blas/HPL_dgemm.c
@@ -144,6 +144,7 @@ double random_halfnormal(void) {
 double random_halfnormal_shifted(double exp, double std) {
     // Here, exp and std are the desired expectation and standard deviation.
     // We compute the corresponding mu and sigma parameters for the normal distribution.
+    return exp;
     double mu, sigma;
     sigma = std/sqrt(1-2/M_PI);
     mu = exp - sigma*sqrt(2/M_PI);
    '''

    linear_dgemm_patch = r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 35dea84..3803d8c 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -214,7 +214,7 @@ static double dtrsm_intercept = -1;
     double mn =  (double)(M) * (double)(N);\
     double mk =  (double)(M) * (double)(K);\
     double nk =  (double)(N) * (double)(K);\
-    double raw_duration = 2.844700e-07 + 6.317136e-11*mnk + 1.489053e-10*mn + 2.107985e-09*mk + 3.332944e-09*nk;\
+    double raw_duration = 6.484604e-11*mnk + 1e-6;\
     double sigma = 1.087202e-07 + 2.976703e-12*mnk + 8.365868e-12*mn + 1.528598e-10*mk + 9.931248e-11*nk;\
     double noise = random_halfnormal_shifted(0, sigma);\
     double injected_duration = raw_duration + noise;\
    '''

    no_random_kernels_patch = r'''
diff --git a/src/blas/HPL_dgemm.c b/src/blas/HPL_dgemm.c
index dd235b7..ab40276 100644
--- a/src/blas/HPL_dgemm.c
+++ b/src/blas/HPL_dgemm.c
@@ -204,14 +204,14 @@ double random_halfnormal_shifted(double exp, double std) {
 }

 void smpi_execute_normal(double mu, double sigma) {
-    double coefficient = random_halfnormal_shifted(mu, sigma);
+    double coefficient = mu;
     if(coefficient > 0) {
         usleep((useconds_t)(coefficient*1e6));
     }
 }

 void smpi_execute_normal_size(double mu, double sigma, double size) {
-    double coefficient = random_halfnormal_shifted(mu, sigma);
+    double coefficient = mu;
     if(coefficient > 0 && size > 0) {
         usleep((useconds_t)(size*coefficient*1e6));
     }
    '''

    no_hpl_kernels_patch = r'''
diff --git a/src/auxil/HPL_dlacpy.c b/src/auxil/HPL_dlacpy.c
index 70ccbce..0fd833b 100644
--- a/src/auxil/HPL_dlacpy.c
+++ b/src/auxil/HPL_dlacpy.c
@@ -342,8 +342,6 @@ void HPL_dlacpy
 /*
  * End of HPL_dlacpy
  */
-#else
-   smpi_execute_normal_size(3.871806e-09, 1.328595e-09, ((double)M)*((double)N));
 #endif // SMPI_OPTIMIZATION_LEVEL
     timestamp_t duration = get_timestamp() - start;
     record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
diff --git a/src/auxil/HPL_dlatcpy.c b/src/auxil/HPL_dlatcpy.c
index 50d71eb..8bdca86 100644
--- a/src/auxil/HPL_dlatcpy.c
+++ b/src/auxil/HPL_dlatcpy.c
@@ -397,8 +397,6 @@ void HPL_dlatcpy
 /*
  * End of HPL_dlatcpy
  */
-#else
-    smpi_execute_normal_size(4.893900e-09, 4.691039e-10, ((double)M)*N);
 #endif // SMPI_OPTIMIZATION_LEVEL
     timestamp_t duration = get_timestamp() - start;
     record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
diff --git a/src/pauxil/HPL_dlaswp01T.c b/src/pauxil/HPL_dlaswp01T.c
index dc0f8b3..e0d8879 100644
--- a/src/pauxil/HPL_dlaswp01T.c
+++ b/src/pauxil/HPL_dlaswp01T.c
@@ -251,8 +251,6 @@ void HPL_dlaswp01T
 /*
  * End of HPL_dlaswp01T
  */
-#else
-    smpi_execute_normal_size(7.547639e-09, 1.371708e-09, ((double)M)*((double)N));
 #endif // SMPI_OPTIMIZATION_LEVEL
     timestamp_t duration = get_timestamp() - start;
     record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
diff --git a/src/pauxil/HPL_dlaswp02N.c b/src/pauxil/HPL_dlaswp02N.c
index ba461fc..d7345d9 100644
--- a/src/pauxil/HPL_dlaswp02N.c
+++ b/src/pauxil/HPL_dlaswp02N.c
@@ -204,8 +204,6 @@ void HPL_dlaswp02N
 /*
  * End of HPL_dlaswp02N
  */
-#else
-    smpi_execute_normal_size(2.822241e-08, 5.497050e-09, ((double)M)*N);
 #endif // SMPI_OPTIMIZATION_LEVEL
     timestamp_t duration = get_timestamp() - start;
     record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
diff --git a/src/pauxil/HPL_dlaswp03T.c b/src/pauxil/HPL_dlaswp03T.c
index 8e54bfe..92c4665 100644
--- a/src/pauxil/HPL_dlaswp03T.c
+++ b/src/pauxil/HPL_dlaswp03T.c
@@ -185,8 +185,6 @@ void HPL_dlaswp03T
 /*
  * End of HPL_dlaswp03T
  */
-#else
-    smpi_execute_normal_size(3.775240e-09, 2.968320e-10, ((double)M)*N);
 #endif // SMPI_OPTIMIZATION_LEVEL
     timestamp_t duration = get_timestamp() - start;
     record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
diff --git a/src/pauxil/HPL_dlaswp04T.c b/src/pauxil/HPL_dlaswp04T.c
index 61dd638..8e05cf3 100644
--- a/src/pauxil/HPL_dlaswp04T.c
+++ b/src/pauxil/HPL_dlaswp04T.c
@@ -269,8 +269,6 @@ void HPL_dlaswp04T
 /*
  * End of HPL_dlaswp04T
  */
-#else
-   smpi_execute_normal_size(8.498957e-09, 2.218070e-09, ((double)M1)*((double)N));
 #endif // SMPI_OPTIMIZATION_LEVEL
     timestamp_t duration = get_timestamp() - start;
     record_measure("", 0, __func__, start, duration, 3, (int []){M0, M1, N});
    '''

    makefile_patch = '''
diff --git a/Make.SMPI b/Make.SMPI
index c34be62..a610089 100644
--- a/Make.SMPI
+++ b/Make.SMPI
@@ -68,7 +68,7 @@ ARCH         = $(arch)
 # - HPL Directory Structure / HPL library ------------------------------
 # ----------------------------------------------------------------------
 #
-TOPdir       = /home/tom/Documents/Fac/2017_Stage_LIG/hpl-2.2
+TOPdir=/tmp/hpl-2.2
 INCdir       = $(TOPdir)/include
 BINdir       = $(TOPdir)/bin/$(ARCH)
 LIBdir       = $(TOPdir)/lib/$(ARCH)
@@ -93,9 +93,9 @@ MPlib        =
 # header files,  LAlib  is defined  to be the name of  the library to be
 # used. The variable LAdir is only used for defining LAinc and LAlib.
 #
-LAdir        = /usr/lib
+LAdir        = /tmp/lib
 LAinc        =
-LAlib        = -lblas
+LAlib        = /tmp/lib/libopenblas.so
 #
 # ----------------------------------------------------------------------
 # - F77 / C interface --------------------------------------------------
 '''

    blas_randomization_patch = r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 023ec77..741b225 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -188,6 +188,7 @@ STDC_ARGS(

 FILE *get_measure_file();
 double get_timestamp(struct timeval timestamp);
+double get_random_factor();

 #ifdef SMPI_MEASURE
 #pragma message "[SMPI] Tracing the calls to BLAS functions."
@@ -232,6 +233,7 @@ static double dgemm_intercept = -1;
     }\
     double expected_time;\
     expected_time = dgemm_coefficient*((double)(M))*((double)(N))*((double)(K)) + dgemm_intercept;\
+    expected_time *= get_random_factor();\
     struct timeval before = {};\
     START_MEASURE(before);\
     if(expected_time > 0)\
@@ -264,6 +266,7 @@ static double dtrsm_intercept = -1;
     } else {\
         expected_time = dtrsm_coefficient*((double)(M))*((double)(N))*((double)(N)) + dtrsm_intercept;\
     }\
+    expected_time *= get_random_factor();\
     struct timeval before = {};\
     START_MEASURE(before);\
     if(expected_time > 0)\
diff --git a/src/blas/HPL_dgemm.c b/src/blas/HPL_dgemm.c
index 7c017f3..8ee8fb4 100644
--- a/src/blas/HPL_dgemm.c
+++ b/src/blas/HPL_dgemm.c
@@ -76,6 +76,20 @@ double get_timestamp(struct timeval timestamp) {
     return t;
 }

+double get_random_factor() {
+    double min_f = 0.95;
+    double max_f = 1.05;
+    static double factor = -1;
+    if(factor < 0) {
+        int my_rank;
+        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
+        srand(my_rank + 12);  // we cannot do srand(my_rank) as srand(0) and srand(1) are equivalent
+        double x = (double)rand()/(double)(RAND_MAX);  // x is in [0, 1]
+        factor = min_f + x*(max_f-min_f);
+    }
+    return factor;
+}
+

 #ifndef HPL_dgemm

'''

    blas_reg_patch = r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 741b225..3ced2e4 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -228,8 +228,43 @@ static double dgemm_intercept = -1;
 #pragma message "[SMPI] Using smpi_execute for HPL_dgemm."
 #define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
     if(dgemm_coefficient < 0 || dgemm_intercept < 0) {\
-        dgemm_coefficient = get_param("SMPI_DGEMM_COEFFICIENT");\
-        dgemm_intercept = get_param("SMPI_DGEMM_INTERCEPT");\
+        int rank;\
+        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
+        switch(rank) {\
+            case 0:\
+                dgemm_coefficient=2.323719e-12;\
+                dgemm_intercept=3.401798e-04;\
+                break;\
+            case 1:\
+                dgemm_coefficient=2.394360e-12;\
+                dgemm_intercept=9.812780e-05;\
+                break;\
+            case 2:\
+                dgemm_coefficient=2.494063e-12;\
+                dgemm_intercept=8.229821e-04;\
+                break;\
+            case 3:\
+                dgemm_coefficient=2.345799e-12;\
+                dgemm_intercept=4.526956e-04;\
+                break;\
+            case 4:\
+                dgemm_coefficient=2.462303e-12;\
+                dgemm_intercept=3.162167e-04;\
+                break;\
+            case 5:\
+                dgemm_coefficient=2.999396e-12;\
+                dgemm_intercept=3.729884e-05;\
+                break;\
+            case 6:\
+                dgemm_coefficient=2.344958e-12;\
+                dgemm_intercept=8.719184e-04;\
+                break;\
+            case 7:\
+                dgemm_coefficient=2.338696e-12;\
+                dgemm_intercept=6.135174e-04;\
+                break;\
+            default: exit(1);\
+        }\
     }\
     double expected_time;\
     expected_time = dgemm_coefficient*((double)(M))*((double)(N))*((double)(K)) + dgemm_intercept;\
@@ -257,8 +292,43 @@ static double dtrsm_intercept = -1;
 #pragma message "[SMPI] Using smpi_execute for HPL_dtrsm."
 #define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
     if(dtrsm_coefficient < 0 || dtrsm_intercept < 0) {\
-        dtrsm_coefficient = get_param("SMPI_DTRSM_COEFFICIENT");\
-        dtrsm_intercept = get_param("SMPI_DTRSM_INTERCEPT");\
+        int rank;\
+        MPI_Comm_rank(MPI_COMM_WORLD, &rank);\
+        switch(rank) {\
+            case 0:\
+                dtrsm_coefficient=4.330126e-12;\
+                dtrsm_intercept=1.597083e-05;\
+                break;\
+            case 1:\
+                dtrsm_coefficient=4.319634e-12;\
+                dtrsm_intercept=7.507639e-06;\
+                break;\
+            case 2:\
+                dtrsm_coefficient=5.607205e-12;\
+                dtrsm_intercept=4.197323e-05;\
+                break;\
+            case 3:\
+                dtrsm_coefficient=4.338905e-12;\
+                dtrsm_intercept=2.482316e-05;\
+                break;\
+            case 4:\
+                dtrsm_coefficient=3.765359e-12;\
+                dtrsm_intercept=1.683950e-05;\
+                break;\
+            case 5:\
+                dtrsm_coefficient=5.830476e-12;\
+                dtrsm_intercept=6.822817e-06;\
+                break;\
+            case 6:\
+                dtrsm_coefficient=2.984086e-12;\
+                dtrsm_intercept=4.615099e-05;\
+                break;\
+            case 7:\
+                dtrsm_coefficient=4.266092e-12;\
+                dtrsm_intercept=3.448889e-05;\
+                break;\
+            default: exit(1);\
+        }\
     }\
     double expected_time;\
     if((Side) == HplLeft) {\
'''

    normal_durations_patch = r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 19a665a..5de474a 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -163,14 +163,15 @@ STDC_ARGS(

 #if SMPI_OPTIMIZATION_LEVEL >= 2
 #define    HPL_dswap(...)      {}
-#define    HPL_dgemv(...)      {}
-#define    HPL_daxpy(...)      {}
-#define    HPL_dscal(...)      {}
-#define    HPL_idamax(N, X, incX) (rand()%N)
-#define    HPL_dtrsv(...)      {}
+#define    HPL_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY) ({smpi_execute_normal(1.389908e+09, 6.011379e+07, ((double)M)*N);})
+#define    HPL_daxpy(...)      ({smpi_execute_normal(6.907377e+08, 2.553646e+08, 4.920070e+02);})
+#define    HPL_dscal(...)      ({smpi_execute_normal(2.779018e+09, 4.668436e+08, 2.311699e+04);})
+#define    HPL_idamax(N, X, incX) ({smpi_execute_normal(1.796604e+09, 5.786329e+08, 2.311625e+04); rand()%N;})
+#define    HPL_dtrsv(...)      ({smpi_execute_normal(6.561277e+08, 7.395658e+07, 1.637382e+04);})
 #define    HPL_dger(...)       {}
 #pragma message "[SMPI] Using no-op for the cheapest BLAS functions."
 #else
+#define    HPL_dcopy           cblas_dcopy
 #define    HPL_dswap           cblas_dswap
 #define    HPL_dgemv           cblas_dgemv
 #define    HPL_daxpy           cblas_daxpy
@@ -188,6 +189,7 @@ STDC_ARGS(

 FILE *get_measure_file();
 double get_timestamp(struct timeval timestamp);
+void smpi_execute_normal(double mu, double sigma, double size);

 #ifdef SMPI_MEASURE
 #pragma message "[SMPI] Tracing the calls to BLAS functions."
@@ -234,8 +236,7 @@ static double dgemm_intercept = -1;
     expected_time = dgemm_coefficient*((double)(M))*((double)(N))*((double)(K)) + dgemm_intercept;\
     struct timeval before = {};\
     START_MEASURE(before);\
-    if(expected_time > 0)\
-        smpi_execute_benched(expected_time);\
+    smpi_execute_normal(1.500643e+10, 6.046520e+08, ((double)M)*((double)N)*K);\
     STOP_MEASURE(before, "dgemm", M, N, K, lda, ldb, ldc);\
 })
 #else
@@ -266,8 +267,7 @@ static double dtrsm_intercept = -1;
     }\
     struct timeval before = {};\
     START_MEASURE(before);\
-    if(expected_time > 0)\
-        smpi_execute_benched(expected_time);\
+    smpi_execute_normal(1.831074e+10, 8.947492e+08, ((double)M)*N);\
     STOP_MEASURE(before, "dtrsm", M, N, -1, lda, ldb, -1);\
 })
 #else
diff --git a/src/auxil/HPL_dlacpy.c b/src/auxil/HPL_dlacpy.c
index 8c1396a..a2cda2c 100644
--- a/src/auxil/HPL_dlacpy.c
+++ b/src/auxil/HPL_dlacpy.c
@@ -128,6 +128,7 @@ void HPL_dlacpy
  * .. Local Variables ..
  */
 #if SMPI_OPTIMIZATION_LEVEL >= 2
+    smpi_execute_normal(3.739279e+08, 2.089563e+08, 2.929817e+06);
     return;
 #endif
 #ifdef HPL_LACPY_USE_COPY
diff --git a/src/auxil/HPL_dlatcpy.c b/src/auxil/HPL_dlatcpy.c
index 417a1d5..5d82cd1 100644
--- a/src/auxil/HPL_dlatcpy.c
+++ b/src/auxil/HPL_dlatcpy.c
@@ -128,6 +128,7 @@ void HPL_dlatcpy
  * .. Local Variables ..
  */
 #if SMPI_OPTIMIZATION_LEVEL >= 2
+    smpi_execute_normal(2.025287e+08, 2.064743e+07, M*N);
     return;
 #endif
 #ifdef HPL_LATCPY_USE_COPY
diff --git a/src/blas/HPL_dgemm.c b/src/blas/HPL_dgemm.c
index 7c017f3..374504b 100644
--- a/src/blas/HPL_dgemm.c
+++ b/src/blas/HPL_dgemm.c
@@ -48,6 +48,7 @@
  * Include files
  */
 #include <sys/time.h>
+#include <math.h>
 #include "hpl.h"

 FILE *get_measure_file() {
@@ -76,6 +77,24 @@ double get_timestamp(struct timeval timestamp) {
     return t;
 }

+double random_normal(double mu, double sigma) {
+    // From https://rosettacode.org/wiki/Statistics/Normal_distribution#C
+    double x, y, rsq, f;
+    do {
+        x = 2.0 * rand() / (double)RAND_MAX - 1.0;
+        y = 2.0 * rand() / (double)RAND_MAX - 1.0;
+        rsq = x * x + y * y;
+    }while( rsq >= 1. || rsq == 0. );
+    f = sqrt( -2.0 * log(rsq) / rsq );
+    return (x * f)*sigma + mu; // y*f would also be good
+}
+
+void smpi_execute_normal(double mu, double sigma, double size) {
+    double coefficient = random_normal(mu, sigma);
+    if(coefficient > 0) {
+        smpi_execute_benched(size / coefficient);
+    }
+}

 #ifndef HPL_dgemm

diff --git a/src/pauxil/HPL_dlaswp01T.c b/src/pauxil/HPL_dlaswp01T.c
index afb24b6..dce614e 100644
--- a/src/pauxil/HPL_dlaswp01T.c
+++ b/src/pauxil/HPL_dlaswp01T.c
@@ -142,6 +142,7 @@ void HPL_dlaswp01T
  * .. Local Variables ..
  */
 #if SMPI_OPTIMIZATION_LEVEL >= 2
+    smpi_execute_normal(1.326539e+08, 2.558332e+07, 7.942974e+05);
     return;
 #endif
    double                     * a0, * a1;
diff --git a/src/pauxil/HPL_dlaswp02N.c b/src/pauxil/HPL_dlaswp02N.c
index a49d28b..3a7305a 100644
--- a/src/pauxil/HPL_dlaswp02N.c
+++ b/src/pauxil/HPL_dlaswp02N.c
@@ -138,6 +138,7 @@ void HPL_dlaswp02N
  * .. Local Variables ..
  */
 #if SMPI_OPTIMIZATION_LEVEL >= 2
+    smpi_execute_normal(4.411560e+07, 8.231826e+06, ((double)M)*N);
     return;
 #endif
    const double               * A0 = A, * a0;
diff --git a/src/pauxil/HPL_dlaswp03T.c b/src/pauxil/HPL_dlaswp03T.c
index efffab5..0d425b3 100644
--- a/src/pauxil/HPL_dlaswp03T.c
+++ b/src/pauxil/HPL_dlaswp03T.c
@@ -128,6 +128,7 @@ void HPL_dlaswp03T
  * .. Local Variables ..
  */
 #if SMPI_OPTIMIZATION_LEVEL >= 2
+    smpi_execute_normal(2.740416e+08, 3.444054e+07, ((double)M)*N);
     return;
 #endif
    const double               * w = W, * w0;
diff --git a/src/pauxil/HPL_dlaswp04T.c b/src/pauxil/HPL_dlaswp04T.c
index f279771..f935e63 100644
--- a/src/pauxil/HPL_dlaswp04T.c
+++ b/src/pauxil/HPL_dlaswp04T.c
@@ -160,6 +160,7 @@ void HPL_dlaswp04T
  * .. Local Variables ..
  */
 #if SMPI_OPTIMIZATION_LEVEL >= 2
+    smpi_execute_normal(7.956960e+07, 2.261901e+07, 2.679820e+05);
     return;
 #endif
    const double               * w = W, * w0;
'''

    simgrid_stochastic_patch = r'''
diff --git a/src/smpi/internals/smpi_host.cpp b/src/smpi/internals/smpi_host.cpp
index 95c7284f6..93859b31c 100644
--- a/src/smpi/internals/smpi_host.cpp
+++ b/src/smpi/internals/smpi_host.cpp
@@ -11,9 +11,68 @@
 #include <string>
 #include <vector>
 #include <xbt/log.h>
+#include <math.h>

 XBT_LOG_NEW_DEFAULT_SUBCATEGORY(smpi_host, smpi, "Logging specific to SMPI (host)");

+double random_normal(void) {
+    // From https://rosettacode.org/wiki/Statistics/Normal_distribution#C
+    double x, y, rsq, f;
+    do {
+        x = 2.0 * rand() / (double)RAND_MAX - 1.0;
+        y = 2.0 * rand() / (double)RAND_MAX - 1.0;
+        rsq = x * x + y * y;
+    }while( rsq >= 1. || rsq == 0. );
+    f = sqrt( -2.0 * log(rsq) / rsq );
+    return (x * f); // y*f would also be good
+}
+
+double random_halfnormal(void) {
+    double x = random_normal();
+    if(x < 0) {
+        x = -x;
+    }
+    return x;
+}
+
+double random_halfnormal_shifted(double exp, double std) {
+    // Here, exp and std are the desired expectation and standard deviation.
+    // We compute the corresponding mu and sigma parameters for the normal distribution.
+    double mu, sigma;
+    sigma = std/sqrt(1-2/M_PI);
+    mu = exp - sigma*sqrt(2/M_PI);
+    double x = random_halfnormal();
+    return x*sigma + mu;
+}
+
+double random_mixture(int nb_modes, double mixtures[][3]) {
+    // Selecting randomly a mode according to the desired probabilities
+    int i;
+    do {
+        double proba_sum = 0;
+        double x = rand() / (double)RAND_MAX;  // random value in [0, 1]
+        i=-1;
+        while(i < nb_modes && x > proba_sum) {
+            i++;
+            if(i >= nb_modes)
+                break;
+            proba_sum += mixtures[i][2];
+            assert(mixtures[i][2] >= 0);
+            assert(proba_sum <= 1.0001);  // the sum may be slightly higher than 1
+        }
+    } while(i >= nb_modes); // the sum may be slightly lower than 1, in this case we redraw
+    // Drawing a random number on this mode
+    double mu = mixtures[i][0];
+    double sigma = mixtures[i][1];
+    return random_halfnormal_shifted(mu, sigma);
+}
+
+double max(double a, double b) {
+    if(a < b)
+        return b;
+    return a;
+}
+
 namespace simgrid {
 namespace smpi {

@@ -21,6 +80,51 @@ simgrid::xbt::Extension<simgrid::s4u::Host, Host> Host::EXTENSION_ID;

 double Host::orecv(size_t size)
 {
+  double smpi_stochastic_intercept=-1, smpi_stochastic_coefficient=-1;
+  if(size < 8133) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {1.578858e-05, 1.353790e-06, 7.197802e-04},
+          {6.255998e-06, 1.015674e-06, 3.538462e-03},
+          {2.297108e-06, 1.768100e-07, 2.844286e-01},
+          {9.681853e-07, 8.229993e-08, 7.113132e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 8.443925e-11;
+  }
+  if(8133 <= size && size < 15831) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {1.480942e-05, 2.686375e-06, 9.777778e-04},
+          {6.427250e-06, 8.791199e-07, 4.088889e-03},
+          {2.489158e-06, 2.181154e-07, 2.819111e-01},
+          {4.415818e-07, 1.243068e-07, 7.130222e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.044152e-10;
+  }
+  if(15831 <= size && size < 33956) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {1.434377e-05, 2.540300e-06, 1.964286e-03},
+          {7.252870e-06, 5.616525e-07, 4.892857e-03},
+          {3.168695e-06, 4.400812e-07, 3.339286e-01},
+          {2.804051e-08, 2.379107e-07, 6.592143e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.015274e-10;
+  }
+  if(33956 <= size && size < 64000) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {1.550638e-05, 2.360998e-06, 3.000000e-03},
+          {8.684909e-06, 9.668691e-07, 1.303846e-02},
+          {4.243792e-06, 6.296865e-07, 4.060769e-01},
+          {-1.003977e-06, 5.748014e-07, 5.778846e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.074226e-10;
+  }
+  if(size < 64000) {
+    double time = smpi_stochastic_coefficient*size + smpi_stochastic_intercept;
+    return max(time, 0);
+  }
   double current = orecv_parsed_values.empty() ? 0.0 : orecv_parsed_values.front().values[0] +
                                                            orecv_parsed_values.front().values[1] * size;

@@ -44,6 +148,47 @@ double Host::orecv(size_t size)

 double Host::osend(size_t size)
 {
+  double smpi_stochastic_intercept=-1, smpi_stochastic_coefficient=-1;
+  if(size < 8133) {
+      double mixtures_smpi_stochastic_intercept[3][3] = {
+          {1.158459e-05, 4.445288e-06, 3.681319e-04},
+          {7.342084e-07, 1.921238e-07, 1.978187e-01},
+          {1.834565e-07, 6.848359e-08, 8.018132e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(3, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 9.630674e-11;
+  }
+  if(8133 <= size && size < 15831) {
+      double mixtures_smpi_stochastic_intercept[3][3] = {
+          {1.309743e-05, 2.247476e-06, 1.560000e-02},
+          {5.666558e-06, 3.963607e-07, 8.555556e-02},
+          {3.811149e-06, 3.445008e-07, 8.988444e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(3, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.002390e-10;
+  }
+  if(15831 <= size && size < 33956) {
+      double mixtures_smpi_stochastic_intercept[3][3] = {
+          {1.351552e-05, 2.434263e-06, 1.028571e-02},
+          {4.800520e-06, 5.476791e-07, 2.700357e-01},
+          {3.429396e-06, 1.783164e-07, 7.196786e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(3, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.099663e-10;
+  }
+  if(33956 <= size && size < 64000) {
+      double mixtures_smpi_stochastic_intercept[3][3] = {
+          {2.499927e-05, 2.469454e-06, 5.730769e-03},
+          {1.538620e-05, 9.485493e-07, 1.705385e-01},
+          {7.027786e-06, 3.911404e-07, 8.237308e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(3, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.263659e-10;
+  }
+  if(size < 64000) {
+    double time = smpi_stochastic_coefficient*size + smpi_stochastic_intercept;
+    return max(time, 0);
+  }
   double current =
       osend_parsed_values.empty() ? 0.0 : osend_parsed_values[0].values[0] + osend_parsed_values[0].values[1] * size;
   // Iterate over all the sections that were specified and find the right
@@ -67,6 +212,59 @@ double Host::osend(size_t size)

 double Host::oisend(size_t size)
 {
+  double smpi_stochastic_intercept=-1, smpi_stochastic_coefficient=-1;
+  if(size < 8133) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {4.492700e-05, 4.074614e-06, 2.730769e-03},
+          {2.100718e-05, 2.427905e-06, 1.167582e-02},
+          {7.387183e-07, 1.654137e-07, 2.104066e-01},
+          {2.189611e-07, 6.260189e-08, 7.751868e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 7.050642e-11;
+  }
+  if(8133 <= size && size < 15831) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {4.590579e-05, 4.614774e-06, 2.755556e-03},
+          {2.034218e-05, 2.321272e-06, 1.097778e-02},
+          {8.584810e-07, 2.402363e-07, 2.008444e-01},
+          {-1.862171e-07, 1.632566e-07, 7.854222e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 1.229780e-10;
+  }
+  if(15831 <= size && size < 33956) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {5.106743e-05, 5.243046e-06, 2.785714e-03},
+          {2.200488e-05, 2.521526e-06, 1.325000e-02},
+          {6.032292e-06, 4.870055e-07, 1.873571e-01},
+          {1.792845e-06, 1.766367e-07, 7.966071e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 4.059840e-11;
+  }
+  if(33956 <= size && size < 64000) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {5.093095e-05, 3.380393e-06, 2.461538e-03},
+          {2.201740e-05, 2.466355e-06, 1.430769e-02},
+          {7.167819e-06, 5.962693e-07, 1.693077e-01},
+          {1.817755e-06, 1.954229e-07, 8.139231e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 3.293459e-11;
+  }
+  if(64000 <= size) {
+      double mixtures_smpi_stochastic_intercept[4][3] = {
+          {4.549290e-05, 4.842656e-06, 1.382353e-03},
+          {2.117993e-05, 2.147733e-06, 1.298039e-02},
+          {7.524123e-07, 8.528449e-08, 1.205392e-01},
+          {2.759115e-07, 3.453358e-08, 8.650980e-01}
+      };
+      smpi_stochastic_intercept = random_mixture(4, mixtures_smpi_stochastic_intercept);
+      smpi_stochastic_coefficient = 0.000000e+00;
+  }
+  double time = smpi_stochastic_coefficient*size + smpi_stochastic_intercept;
+  return max(time, 0);
   double current =
       oisend_parsed_values.empty() ? 0.0 : oisend_parsed_values[0].values[0] + oisend_parsed_values[0].values[1] * size;

diff --git a/src/surf/network_cm02.cpp b/src/surf/network_cm02.cpp
index a85a3d5ed..ca09e78f5 100644
--- a/src/surf/network_cm02.cpp
+++ b/src/surf/network_cm02.cpp
@@ -12,6 +12,8 @@
 #include "src/surf/surf_interface.hpp"
 #include "surf/surf.hpp"

+double random_mixture(int nb_modes, double mixtures[][3]);
+
 XBT_LOG_EXTERNAL_DEFAULT_CATEGORY(surf_network);

 double sg_latency_factor = 1.0; /* default value; can be set by model or from command line */
@@ -257,6 +259,18 @@ Action* NetworkCm02Model::communicate(s4u::Host* src, s4u::Host* dst, double siz
   int constraints_per_variable = route.size();
   constraints_per_variable += back_route.size();

+  // TODO HERE!
+  if(size < 64000) {
+    // nothing here, the randomness is done in or/os/oi
+  }
+  else {
+    double mixtures[2][3] = {
+      {2.575923e-05, 2.313502e-06, 4.061920e-02},
+      {1.175278e-05, 8.834412e-07, 9.593808e-01}
+    };
+    action->latency_ = random_mixture(2, mixtures);
+  }
+
   if (action->latency_ > 0) {
     action->set_variable(get_maxmin_system()->variable_new(action, 0.0, -1.0, constraints_per_variable));
     if (get_update_algorithm() == Model::UpdateAlgo::LAZY) {
'''

    simgrid_loopback_patch = r'''
diff --git a/src/simgrid/sg_config.cpp b/src/simgrid/sg_config.cpp
index 6e09b7661e..524c4179fe 100644
--- a/src/simgrid/sg_config.cpp
+++ b/src/simgrid/sg_config.cpp
@@ -362,6 +362,18 @@ void sg_config_init(int *argc, char **argv)
                                              "65472:11.6436;15424:3.48845;9376:2.59299;5776:2.18796;3484:1.88101;"
                                              "1426:1.61075;732:1.9503;257:1.95341;0:2.01467");
   simgrid::config::alias("smpi/lat-factor", {"smpi/lat_factor"});
+
+  simgrid::config::declare_flag<std::string>("smpi/loopback-bw-factor",
+                                             "Bandwidth factors for smpi loopback.",
+                                             "65472:0.940694;15424:0.697866;9376:0.58729;5776:1.08739;3484:0.77493;"
+                                             "1426:0.608902;732:0.341987;257:0.338112;0:0.812084");
+  simgrid::config::alias("smpi/loopback-bw-factor", {"smpi/loopback_bw_factor"});
+
+  simgrid::config::declare_flag<std::string>("smpi/loopback-lat-factor", "Latency factors for smpi loopback.",
+                                             "65472:11.6436;15424:3.48845;9376:2.59299;5776:2.18796;3484:1.88101;"
+                                             "1426:1.61075;732:1.9503;257:1.95341;0:2.01467");
+  simgrid::config::alias("smpi/loopback-lat-factor", {"smpi/loopback-lat_factor"});
+
   simgrid::config::declare_flag<std::string>("smpi/IB-penalty-factors",
                                              "Correction factor to communications using Infiniband model with "
                                              "contention (default value based on Stampede cluster profiling)",
diff --git a/src/surf/network_cm02.cpp b/src/surf/network_cm02.cpp
index 25b0ba3211..8624c7ed56 100644
--- a/src/surf/network_cm02.cpp
+++ b/src/surf/network_cm02.cpp
@@ -200,14 +200,15 @@ Action* NetworkCm02Model::communicate(s4u::Host* src, s4u::Host* dst, double siz
         });
   }

-  double bandwidth_bound = route.empty() ? -1.0 : get_bandwidth_factor(size) * route.front()->get_bandwidth();
+  double bandwidth_bound = route.empty() ? -1.0 : get_bandwidth_factor(size, route.front()) * route.front()->get_bandwidth();

   for (auto const& link : route)
-    bandwidth_bound = std::min(bandwidth_bound, get_bandwidth_factor(size) * link->get_bandwidth());
+    bandwidth_bound = std::min(bandwidth_bound, get_bandwidth_factor(size, link) * link->get_bandwidth());

   action->lat_current_ = action->latency_;
-  action->latency_ *= get_latency_factor(size);
-  action->rate_ = get_bandwidth_constraint(action->rate_, bandwidth_bound, size);
+  action->latency_ *= get_latency_factor(size, route.front());
+  action->rate_ = get_bandwidth_constraint(action->rate_, bandwidth_bound, size, route.front());
+

   size_t constraints_per_variable = route.size();
   constraints_per_variable += back_route.size();
diff --git a/src/surf/network_interface.cpp b/src/surf/network_interface.cpp
index ccb7e034fc..172c058d3d 100644
--- a/src/surf/network_interface.cpp
+++ b/src/surf/network_interface.cpp
@@ -39,17 +39,17 @@ simgrid::config::Flag<bool> NetworkModel::cfg_crosstraffic(

 NetworkModel::~NetworkModel() = default;

-double NetworkModel::get_latency_factor(double /*size*/)
+double NetworkModel::get_latency_factor(double /*size*/, LinkImpl* const &link)
 {
   return sg_latency_factor;
 }

-double NetworkModel::get_bandwidth_factor(double /*size*/)
+double NetworkModel::get_bandwidth_factor(double /*size*/, LinkImpl* const &link)
 {
   return sg_bandwidth_factor;
 }

-double NetworkModel::get_bandwidth_constraint(double rate, double /*bound*/, double /*size*/)
+double NetworkModel::get_bandwidth_constraint(double rate, double /*bound*/, double /*size*/, LinkImpl* const &link)
 {
   return rate;
 }
diff --git a/src/surf/network_interface.hpp b/src/surf/network_interface.hpp
index 3ecd3bbd5e..a17e154f28 100644
--- a/src/surf/network_interface.hpp
+++ b/src/surf/network_interface.hpp
@@ -72,7 +72,7 @@ public:
    * @param size The size of the message.
    * @return The latency factor.
    */
-  virtual double get_latency_factor(double size);
+  virtual double get_latency_factor(double size, LinkImpl* const &link);

   /**
    * @brief Get the right multiplicative factor for the bandwidth.
@@ -83,7 +83,7 @@ public:
    * @param size The size of the message.
    * @return The bandwidth factor.
    */
-  virtual double get_bandwidth_factor(double size);
+  virtual double get_bandwidth_factor(double size, LinkImpl* const &link);

   /**
    * @brief Get definitive bandwidth.
@@ -94,7 +94,7 @@ public:
    * @param size The size of the message.
    * @return The new bandwidth.
    */
-  virtual double get_bandwidth_constraint(double rate, double bound, double size);
+  virtual double get_bandwidth_constraint(double rate, double bound, double size, LinkImpl* const &link);
   double next_occurring_event_full(double now) override;

   LinkImpl* loopback_ = nullptr;
diff --git a/src/surf/network_smpi.cpp b/src/surf/network_smpi.cpp
index 2c9ab11646..e07a55fef7 100644
--- a/src/surf/network_smpi.cpp
+++ b/src/surf/network_smpi.cpp
@@ -13,6 +13,8 @@ XBT_LOG_EXTERNAL_DEFAULT_CATEGORY(surf_network);

 std::vector<s_smpi_factor_t> smpi_bw_factor;
 std::vector<s_smpi_factor_t> smpi_lat_factor;
+std::vector<s_smpi_factor_t> smpi_loopback_bw_factor;
+std::vector<s_smpi_factor_t> smpi_loopback_lat_factor;

 /*********
  * Model *
@@ -47,45 +49,69 @@ NetworkSmpiModel::NetworkSmpiModel() : NetworkCm02Model()
   /* Do not add this into all_existing_models: our ancestor already does so */
 }

-double NetworkSmpiModel::get_bandwidth_factor(double size)
+double NetworkSmpiModel::get_bandwidth_factor(double size, LinkImpl* const&link)
 {
   if (smpi_bw_factor.empty())
     smpi_bw_factor = parse_factor(config::get_value<std::string>("smpi/bw-factor"));
+  if (smpi_loopback_bw_factor.empty())
+    smpi_loopback_bw_factor = parse_factor(simgrid::config::get_value<std::string>("smpi/loopback-bw-factor"));
+
+  const std::string link_name = link->get_name();
+  bool is_loopback = (link_name.find("loopback") != std::string::npos);
+
+  std::vector<s_smpi_factor_t> *factors;
+  if(is_loopback)
+    factors = &smpi_loopback_bw_factor;
+  else
+    factors = &smpi_bw_factor;
+

   double current = 1.0;
-  for (auto const& fact : smpi_bw_factor) {
+  for (auto const& fact : *factors) {
     if (size <= fact.factor) {
       XBT_DEBUG("%f <= %zu return %f", size, fact.factor, current);
       return current;
     } else
       current = fact.values.front();
   }
-  XBT_DEBUG("%f > %zu return %f", size, smpi_bw_factor.back().factor, current);
+  XBT_DEBUG("%f > %zu return %f", size, factors->back().factor, current);

   return current;
 }

-double NetworkSmpiModel::get_latency_factor(double size)
+double NetworkSmpiModel::get_latency_factor(double size, LinkImpl* const&link)
 {
   if (smpi_lat_factor.empty())
     smpi_lat_factor = parse_factor(config::get_value<std::string>("smpi/lat-factor"));
+  if (smpi_loopback_lat_factor.empty())
+    smpi_loopback_lat_factor = parse_factor(simgrid::config::get_value<std::string>("smpi/loopback-lat-factor"));
+
+  const std::string link_name = link->get_name();
+  bool is_loopback = (link_name.find("loopback") != std::string::npos);
+
+  std::vector<s_smpi_factor_t> *factors;
+  if(is_loopback)
+    factors = &smpi_loopback_lat_factor;
+  else
+    factors = &smpi_lat_factor;
+

   double current = 1.0;
-  for (auto const& fact : smpi_lat_factor) {
+  for (auto const& fact : *factors) {
     if (size <= fact.factor) {
       XBT_DEBUG("%f <= %zu return %f", size, fact.factor, current);
       return current;
     } else
       current = fact.values.front();
   }
-  XBT_DEBUG("%f > %zu return %f", size, smpi_lat_factor.back().factor, current);
+  XBT_DEBUG("%f > %zu return %f", size, factors->back().factor, current);

   return current;
 }

-double NetworkSmpiModel::get_bandwidth_constraint(double rate, double bound, double size)
+double NetworkSmpiModel::get_bandwidth_constraint(double rate, double bound, double size, LinkImpl* const &link)
 {
-  return rate < 0 ? bound : std::min(bound, rate * get_bandwidth_factor(size));
+  return rate < 0 ? bound : std::min(bound, rate * get_bandwidth_factor(size, link));
 }
 } // namespace resource
 } // namespace kernel
diff --git a/src/surf/network_smpi.hpp b/src/surf/network_smpi.hpp
index 391509789c..031f779864 100644
--- a/src/surf/network_smpi.hpp
+++ b/src/surf/network_smpi.hpp
@@ -17,9 +17,9 @@ public:
   NetworkSmpiModel();
   ~NetworkSmpiModel() = default;

-  double get_latency_factor(double size);
-  double get_bandwidth_factor(double size);
-  double get_bandwidth_constraint(double rate, double bound, double size);
+  double get_latency_factor(double size, LinkImpl* const &link);
+  double get_bandwidth_factor(double size, LinkImpl* const &link);
+  double get_bandwidth_constraint(double rate, double bound, double size, LinkImpl* const &link);
 };
 } // namespace resource
 } // namespace kernel
'''

    simgrid_debug_patch = r'''
diff --git a/include/simgrid/kernel/resource/Model.hpp b/include/simgrid/kernel/resource/Model.hpp
index ac5ccd72ef..e94bb791dd 100644
--- a/include/simgrid/kernel/resource/Model.hpp
+++ b/include/simgrid/kernel/resource/Model.hpp
@@ -20,6 +20,7 @@ namespace resource {
 class XBT_PUBLIC Model {
 public:
   /** @brief Possible update mechanisms */
+  std::string get_name() {return "unknown";};
   enum class UpdateAlgo {
     FULL, /**< Full update mechanism: the remaining time of every action is recomputed at each step */
     LAZY  /**< Lazy update mechanism: only the modified actions get recomputed.
diff --git a/src/kernel/resource/DiskImpl.hpp b/src/kernel/resource/DiskImpl.hpp
index 11e4829e40..7f41311357 100644
--- a/src/kernel/resource/DiskImpl.hpp
+++ b/src/kernel/resource/DiskImpl.hpp
@@ -36,6 +36,7 @@ class DiskAction;
  *********/
 class DiskModel : public Model {
 public:
+  std::string get_name() {return "disk";};
   DiskModel();
   DiskModel(const DiskModel&) = delete;
   DiskModel& operator=(const DiskModel&) = delete;
diff --git a/src/kernel/resource/Model.cpp b/src/kernel/resource/Model.cpp
index 640df1ee84..0d95ce0651 100644
--- a/src/kernel/resource/Model.cpp
+++ b/src/kernel/resource/Model.cpp
@@ -60,8 +60,8 @@ double Model::next_occurring_event_lazy(double now)
     double min   = -1;
     double share = action->get_variable()->get_value();

+    double time_to_completion = -42;
     if (share > 0) {
-      double time_to_completion;
       if (action->get_remains() > 0) {
         time_to_completion = action->get_remains_no_update() / share;
       } else {
@@ -85,8 +85,40 @@ double Model::next_occurring_event_lazy(double now)
     if (min > -1) {
       action_heap_.update(action, min, action_type);
       XBT_DEBUG("Insert at heap action(%p) min %f now %f", action, min, now);
-    } else
+    } else {
+      char *action_type;
+      if(action->get_type() == ActionHeap::Type::latency)
+          action_type = "latency";
+      else if(action->get_type() == ActionHeap::Type::max_duration)
+          action_type = "max_duration";
+      else if(action->get_type() == ActionHeap::Type::normal)
+          action_type = "normal";
+      else if(action->get_type() == ActionHeap::Type::unset)
+          action_type = "unset";
+      else
+          action_type = "error:unknown";
+
+      fprintf(stderr, "## ERROR ##\n\n");
+      fprintf(stderr, "action->get_type()              = %s\n", action_type);
+      fprintf(stderr, "action->get_model()->get_name() = %s\n", action->get_model()->get_name().c_str());
+      fprintf(stderr, "action->get_sharing_penalty()   = %e\n", action->get_sharing_penalty());
+      fprintf(stderr, "action->get_max_duration()      = %e\n", action->get_max_duration());
+      fprintf(stderr, "action->get_remains()           = %e\n", action->get_remains());
+      fprintf(stderr, "action->get_remains_no_update() = %e\n", action->get_remains_no_update());
+      fprintf(stderr, "action->get_start_time()        = %e\n", action->get_start_time());
+      fprintf(stderr, "action->get_finish_time()       = %e\n", action->get_finish_time());
+      fprintf(stderr, "action->get_cost()              = %e\n", action->get_cost());
+      fprintf(stderr, "action->get_last_update()       = %e\n", action->get_last_update());
+      fprintf(stderr, "action->get_last_value()        = %e\n", action->get_last_value());
+      fprintf(stderr, "share                           = %e\n", share);
+      fprintf(stderr, "min                             = %e\n", min);
+      fprintf(stderr, "now                             = %e\n", now);
+      fprintf(stderr, "time_to_completion              = %e\n", time_to_completion);
+      fprintf(stderr, "\n");
+
+
       DIE_IMPOSSIBLE;
+    }
   }

   // hereafter must have already the min value for this resource model
diff --git a/src/surf/cpu_interface.hpp b/src/surf/cpu_interface.hpp
index 0215897243..8db117ded6 100644
--- a/src/surf/cpu_interface.hpp
+++ b/src/surf/cpu_interface.hpp
@@ -27,6 +27,7 @@ namespace resource {
  */
 class XBT_PUBLIC CpuModel : public Model {
 public:
+  std::string get_name() {return "CPU";};
   explicit CpuModel(Model::UpdateAlgo algo) : Model(algo) {}

   /**
diff --git a/src/surf/network_interface.hpp b/src/surf/network_interface.hpp
index c48f4649d4..a628bc1508 100644
--- a/src/surf/network_interface.hpp
+++ b/src/surf/network_interface.hpp
@@ -32,6 +32,7 @@ namespace resource {
  */
 class NetworkModel : public Model {
 public:
+  std::string get_name() {return "network";};
   static config::Flag<double> cfg_tcp_gamma;
   static config::Flag<bool> cfg_crosstraffic;

'''
