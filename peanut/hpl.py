import os
import time
from .peanut import logger, ExpFile, Time, RunError
from .abstract_hpl import AbstractHPL


class HPL(AbstractHPL):
    installfile_types = {'warmup_time': int, 'trace_dgemm': bool,
                         **AbstractHPL.installfile_types}
    def install_akypuera(self):
        self.git_clone('https://github.com/Ezibenroc/akypuera.git', 'akypuera', recursive=True,
                       patch=self.akypuera_patch)
        self.nodes.run('mkdir build && cd build && cmake ..', directory='akypuera')
        self.nodes.run('make -j 32', directory='akypuera/build')

    @property
    def akypuera_dir(self):
        return os.path.join(self.nodes.working_dir, 'akypuera/build')

    def setup(self):
        super().setup()
        assert self.installfile is not None
        install_options = self.installfile.content
        self.apt_install(
            'openmpi-bin',
            'libopenmpi-dev',
            'net-tools',
            'stress',
        )
        if install_options['trace_execution']:
            self.apt_install('pajeng')
            self.install_akypuera()
        self.nodes.run('wget http://www.netlib.org/benchmark/hpl/hpl-2.2.tar.gz')
        self.nodes.run('tar -xvf hpl-2.2.tar.gz')
        if install_options['trace_execution']:
            self.nodes.write_files(self.patch, self.hpl_dir + '/patch.diff')
            self.nodes.run('git apply --whitespace=fix patch.diff', directory=self.hpl_dir)
        if install_options['trace_dgemm'] and not install_options['trace_execution']:
            self.nodes.write_files(self.trace_dgemm_patch, self.hpl_dir + '/patch.diff')
            self.nodes.run('git apply --whitespace=fix patch.diff', directory=self.hpl_dir)
        if install_options['terminate_early']:
            self.nodes.write_files(self.hpl_early_termination_patch, self.hpl_dir + '/patch.diff')
            self.nodes.run('git apply --whitespace=fix patch.diff', directory=self.hpl_dir)
        if install_options['insert_bcast']:
            self.nodes.write_files(self.hpl_bcast_patch, self.hpl_dir + '/patch.diff')
            self.nodes.run('git apply --whitespace=fix patch.diff', directory=self.hpl_dir)
        self.nodes.write_files(self.makefile, os.path.join(self.hpl_dir, 'Make.Debian'))
        self.nodes.run('make startup arch=Debian', directory=self.hpl_dir)
        while True:
            try:
                self.nodes.run('LD_LIBRARY_PATH=/tmp/lib make -j 64 arch=Debian', directory=self.hpl_dir)
            except RunError as e:  # for some reason, this command fails sometime...
                msg = str(e).split('\n')[0]
                logger.error('Previous command failed with message %s' % msg)
            else:
                break
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_performance()

    def run_exp(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        nb_cores = len(self.nodes.cores)
        warmup = install_options['warmup_time']
        if warmup > 0:
            cmd = 'stress -c %d -t %ds' % (4*nb_cores, warmup)
            self.nodes.run(cmd)
        results = []
        start = time.time()
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        for i, exp in enumerate(expfile):
            proc_p = exp['proc_p']
            proc_q = exp['proc_q']
            nb_hpl_proc = proc_p * proc_q
            process_per_node = exp['process_per_node']
            thread_per_process = exp['thread_per_process']
            if nb_cores % (process_per_node*thread_per_process) != 0:
                msg = 'Requested %d process per node and %d thread per process, but %d cores are available'
                logger.warning(msg % (process_per_node, thread_per_process, nb_cores))
            nb_proc = len(self.hostnames)*process_per_node
            if nb_proc != nb_hpl_proc:
                msg = 'Requested %d*%d=%d processes for HPL, but the total number of processes is %d*%d=%d'
                logger.warning(msg % (proc_p, proc_q, nb_hpl_proc, len(self.hostnames), process_per_node, nb_proc))
            nb_ranks = max(nb_hpl_proc, nb_proc)
            mapping = []
            for rank in range(nb_ranks):
                host = self.hostnames[rank // process_per_node]
                core = rank % process_per_node
                mapping.append('rank %d=%s slot=%d' % (rank, host, core))
            mapping = '\n'.join(mapping)
            hosts = '\n'.join('%s slots=%d' % (host, process_per_node) for host in self.hostnames)
            hostfile = os.path.join('/tmp/hosts.txt')
            rankfile = os.path.join('/tmp/ranks.txt')
            self.nodes.write_files(hosts, hostfile)
            self.nodes.write_files(mapping, rankfile)
            hpl_file = self.generate_hpl_file(**exp)
            self.nodes.write_files(hpl_file, os.path.join(self.hpl_dir, 'bin/Debian/HPL.dat'))
            cmd = 'mpirun --allow-run-as-root --report-bindings --timestamp-output -np %d -x OMP_NUM_THREADS=%d'
            cmd += ' -hostfile %s' % hostfile
            cmd += ' --rankfile %s' % rankfile
            cmd += ' -x LD_LIBRARY_PATH=/tmp/lib'
            if install_options['trace_execution']:
                lib = os.path.join(self.akypuera_dir, 'libaky.so')
                cmd += ' -x LD_PRELOAD=%s' % lib
                cmd = 'LD_PRELOAD=%s %s' % (lib, cmd)
            cmd += ' ./xhpl'
            cmd = cmd % (nb_ranks, thread_per_process)
            self.register_temperature()
            start_timestamp = self.get_timestamp()
            output = self.director.run_unique(cmd, directory=self.hpl_dir+'/bin/Debian')
            stop_timestamp = self.get_timestamp()
            self.register_temperature()
            if install_options['trace_execution']:
                self.director.run('ls -l rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                rstdir = os.path.join(self.orchestra.working_dir, self.hpl_dir, 'bin/Debian/rastro-*.rst')
                for node in self.orchestra.hostnames:
                    self.director.run("rsync -a '%s:%s' ." % (node, rstdir), directory=self.hpl_dir+'/bin/Debian')
                converter = os.path.join(self.akypuera_dir, 'aky_converter')
                paje_file = os.path.join(self.director.working_dir, 'trace_%d.paje' % i)
                self.director.run('ls -l rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                aky_output = self.director.run_unique('%s rastro-*rst > %s' % (converter, paje_file),
                                                  directory=self.hpl_dir+'/bin/Debian')
                if 'do you synchronize' in aky_output.stderr:
                    logger.warning('Could not build a communication trace with Akypuera')
                    self.director.run('%s --no-links rastro-*rst > %s' % (converter, paje_file),
                                      directory=self.hpl_dir+'/bin/Debian')
                mpi_trace = 'trace_mpi_%d.csv' % i
                self.director.run('pj_dump -u %s | grep -v MPI_Iprobe > %s' % (paje_file, mpi_trace))
                self.nodes.run('rm -f rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                self.add_local_to_archive(mpi_trace)
            if install_options['trace_execution'] or install_options['trace_dgemm']:
                blasdir = os.path.join(self.orchestra.working_dir, self.hpl_dir, 'bin/Debian/blas*trace')
                for node in self.orchestra.hostnames:
                    self.director.run("rsync -a '%s:%s' ." % (node, blasdir), directory=self.hpl_dir+'/bin/Debian')
                blas_trace = os.path.join(self.director.working_dir, 'trace_blas_%d.csv' % i)
                self.director.run('cat blas*trace > %s' % blas_trace, directory=self.hpl_dir+'/bin/Debian')
                self.nodes.run('rm -f blas*trace', directory=self.hpl_dir+'/bin/Debian')
                self.add_local_to_archive(blas_trace)
            total_time, gflops, residual = self.parse_hpl(output.stdout)
            new_res = dict(exp)
            new_res['time'] = total_time
            new_res['gflops'] = gflops
            new_res['residual'] = residual
            new_res['start_timestamp'] = start_timestamp
            new_res['stop_timestamp'] = stop_timestamp
            results.append(new_res)
            ellapsed = time.time() - start
            exp_i = i+1
            speed = exp_i/ellapsed
            rest = (len(expfile)-exp_i)/speed
            if rest > 0:
                rest = Time.from_seconds(rest)
                time_info = ' | estimated remaining time: %s' % rest
            else:
                time_info = ''
            logger.debug('Done experiment %d / %d%s' % (i+1, len(expfile), time_info))
        results = ExpFile(content=results, filename='results.csv')
        self.add_content_to_archive(results.raw_content, 'results.csv')

    @property
    def akypuera_patch(self):
        return r'''
diff --git a/src/aky.c b/src/aky.c
index 47feb15..5b2e4e5 100644
--- a/src/aky.c
+++ b/src/aky.c
@@ -939,9 +939,7 @@ MPI_Comm comm;
 int *flag;
 MPI_Status *status;
 {
-  rst_event(MPI_IPROBE_IN);
   int returnVal = PMPI_Iprobe(source, tag, comm, flag, status);
-  rst_event(MPI_IPROBE_OUT);
   return returnVal;
 }

@@ -1060,9 +1058,7 @@ int tag;
 MPI_Comm comm;
 MPI_Status *status;
 {
-  rst_event(MPI_PROBE_IN);
   int returnVal = PMPI_Probe(source, tag, comm, status);
-  rst_event(MPI_PROBE_OUT);
   return returnVal;
 }

'''

    trace_functions_patch = r'''
diff --git a/src/blas/HPL_dgemm.c b/src/blas/HPL_dgemm.c
index f27888a..7c6b73f 100644
--- a/src/blas/HPL_dgemm.c
+++ b/src/blas/HPL_dgemm.c
@@ -48,6 +48,74 @@
  * Include files
  */
 #include "hpl.h"
+#include "unistd.h"
+#if _POSIX_TIMERS
+#include <time.h>
+#define HAVE_CLOCKGETTIME 1
+#else
+#include <sys/time.h>
+#define HAVE_GETTIMEOFDAY 1
+#endif
+
+FILE *get_measure_file() {
+    static FILE *measure_file=NULL;
+    if(!measure_file) {
+        int my_rank;
+        char filename[50];
+        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
+        sprintf (filename, "blas_%d.trace", my_rank);
+        measure_file=fopen(filename, "w");
+        if(!measure_file) {
+            fprintf(stderr, "Error opening file %s\n", filename);
+            exit(1);
+        }
+    }
+    return measure_file;
+}
+
+
+#ifdef HAVE_CLOCKGETTIME
+#define PRECISION 1000000000.0
+#elif HAVE_GETTIMEOFDAY
+#define PRECISION 1000000.0
+#else
+#define PRECISION 1
+#endif
+
+timestamp_t get_time(){
+#ifdef HAVE_CLOCKGETTIME
+    struct timespec tp;
+    clock_gettime (CLOCK_REALTIME, &tp);
+    return (tp.tv_sec * 1000000000 + tp.tv_nsec);
+#elif HAVE_GETTIMEOFDAY
+    struct timeval tv;
+    gettimeofday (&tv, NULL);
+    return (tv.tv_sec * 1000000 + tv.tv_usec)*1000;
+#endif
+}
+
+timestamp_t get_timestamp(void) {
+    static timestamp_t start = 0;
+    if(start == 0) {
+        start = get_time();
+        return 0;
+    }
+    return get_time() - start;
+}
+
+void record_measure(const char *file, int line, const char *function, timestamp_t start, timestamp_t duration, int n_args, int *args) {
+    static int my_rank = -1;
+    if(my_rank < 0) {
+        MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
+    }
+    FILE *measure_file = get_measure_file();
+    if(!measure_file) {fprintf(stderr, "error with measure_file\n"); exit(1);}
+    fprintf(measure_file, "%s, %d, %s, %d, %e, %e", file, line, function, my_rank, start/PRECISION, duration/PRECISION);
+    for(int i = 0; i < n_args; i++) {
+        fprintf(measure_file, ", %d", args[i]);
+    }
+    fprintf(measure_file, "\n");
+}

 #ifndef HPL_dgemm

    '''

    trace_dgemm_patch = trace_functions_patch + r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 41e5afd..5350ea5 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -169,11 +169,23 @@ STDC_ARGS(
 #define    HPL_dtrsv           cblas_dtrsv
 #define    HPL_dger            cblas_dger

-#define    HPL_dgemm           cblas_dgemm
 #define    HPL_dtrsm           cblas_dtrsm

+
 #endif

+FILE *get_measure_file();
+typedef unsigned long long timestamp_t;
+timestamp_t get_timestamp(void);
+void record_measure(const char *file, int line, const char *function, timestamp_t start, timestamp_t duration, int n_args, int *args);
+
+#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);\
+    timestamp_t duration = get_timestamp() - start;\
+    if(M > 0 && N > 0 && K > 0) record_measure(__FILE__, __LINE__, "dgemm", start, duration, 3, (int []){M, N, K});\
+})
+
 #ifdef HPL_CALL_FBLAS
 /*
  * ---------------------------------------------------------------------

    '''

    patch = trace_functions_patch + r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 41e5afd..f52e826 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -159,21 +159,78 @@ STDC_ARGS(
  * HPL C BLAS macro definition
  * ---------------------------------------------------------------------
  */
-#define    HPL_dswap           cblas_dswap
-#define    HPL_dcopy           cblas_dcopy
-#define    HPL_daxpy           cblas_daxpy
-#define    HPL_dscal           cblas_dscal
-#define    HPL_idamax          cblas_idamax
-
-#define    HPL_dgemv           cblas_dgemv
-#define    HPL_dtrsv           cblas_dtrsv
-#define    HPL_dger            cblas_dger
-
-#define    HPL_dgemm           cblas_dgemm
-#define    HPL_dtrsm           cblas_dtrsm

 #endif

+FILE *get_measure_file();
+typedef unsigned long long timestamp_t;
+timestamp_t get_timestamp(void);
+void record_measure(const char *file, int line, const char *function, timestamp_t start, timestamp_t duration, int n_args, int *args);
+
+#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dgemm", start, duration, 3, (int []){M, N, K});\
+})
+
+#define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dtrsm", start, duration, 2, (int []){M, N});\
+})
+
+#define HPL_dswap(N, X, incX, Y, incY) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dswap(N, X, incX, Y, incY);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dswap", start, duration, 1, (int []){N});\
+})
+#define HPL_dcopy(N, X, incX, Y, incY) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dcopy(N, X, incX, Y, incY);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dcopy", start, duration, 1, (int []){N});\
+})
+#define HPL_daxpy(N, alpha, X, incX, Y, incY) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_daxpy(N, alpha, X, incX, Y, incY);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "daxpy", start, duration, 1, (int []){N});\
+})
+#define HPL_dscal(N, alpha, X, incX) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dscal(N, alpha, X, incX);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dscal", start, duration, 1, (int []){N});\
+})
+#define HPL_idamax(N, X, incX) ({\
+    timestamp_t start = get_timestamp();\
+    int result = cblas_idamax(N, X, incX);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "idamax", start, duration, 1, (int []){N});\
+    result;\
+})
+#define HPL_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dgemv(layout, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dgemv", start, duration, 2, (int []){M, N});\
+})
+#define HPL_dtrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dtrsv(layout, Uplo, TransA, Diag, N, A, lda, X, incX);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dtrsv", start, duration, 1, (int []){N});\
+})
+#define HPL_dger(layout, M, N, alpha, X, incX, Y, incY, A, ldA) ({\
+    timestamp_t start = get_timestamp();\
+    cblas_dger(layout, M, N, alpha, X, incX, Y, incY, A, ldA);\
+    timestamp_t duration = get_timestamp() - start;\
+    record_measure(__FILE__, __LINE__, "dger", start, duration, 2, (int []){M, N});\
+})
+
 #ifdef HPL_CALL_FBLAS
 /*
  * ---------------------------------------------------------------------
diff --git a/src/auxil/HPL_dlacpy.c b/src/auxil/HPL_dlacpy.c
index 89ae13b..ea15e65 100644
--- a/src/auxil/HPL_dlacpy.c
+++ b/src/auxil/HPL_dlacpy.c
@@ -127,6 +127,7 @@ void HPL_dlacpy
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
 #ifdef HPL_LACPY_USE_COPY
    register int               j;
 #else
@@ -340,4 +341,6 @@ void HPL_dlacpy
 /*
  * End of HPL_dlacpy
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/auxil/HPL_dlatcpy.c b/src/auxil/HPL_dlatcpy.c
index 7643676..eb84bea 100644
--- a/src/auxil/HPL_dlatcpy.c
+++ b/src/auxil/HPL_dlatcpy.c
@@ -127,6 +127,7 @@ void HPL_dlatcpy
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
 #ifdef HPL_LATCPY_USE_COPY
    register int               j;
 #else
@@ -395,4 +396,6 @@ void HPL_dlatcpy
 /*
  * End of HPL_dlatcpy
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp00N.c b/src/pauxil/HPL_dlaswp00N.c
index 60ae8b1..72952e3 100644
--- a/src/pauxil/HPL_dlaswp00N.c
+++ b/src/pauxil/HPL_dlaswp00N.c
@@ -121,6 +121,7 @@ void HPL_dlaswp00N
 /* ..
  * .. Executable Statements ..
  */
+    timestamp_t start = get_timestamp();
    if( ( M <= 0 ) || ( N <= 0 ) ) return;

    nr = N - ( nu = (int)( ( (unsigned int)(N) >> HPL_LASWP00N_LOG2_DEPTH )
@@ -195,4 +196,6 @@ void HPL_dlaswp00N
 /*
  * End of HPL_dlaswp00N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp01N.c b/src/pauxil/HPL_dlaswp01N.c
index f467470..eab7b51 100644
--- a/src/pauxil/HPL_dlaswp01N.c
+++ b/src/pauxil/HPL_dlaswp01N.c
@@ -140,6 +140,7 @@ void HPL_dlaswp01N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    double                     * a0, * a1;
    const int                  incA = (int)( (unsigned int)(LDA) <<
                                             HPL_LASWP01N_LOG2_DEPTH ),
@@ -206,4 +207,6 @@ void HPL_dlaswp01N
 /*
  * End of HPL_dlaswp01N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp01T.c b/src/pauxil/HPL_dlaswp01T.c
index c3c9e4a..3e0b4cb 100644
--- a/src/pauxil/HPL_dlaswp01T.c
+++ b/src/pauxil/HPL_dlaswp01T.c
@@ -150,6 +150,7 @@ void HPL_dlaswp01T
 /* ..
  * .. Executable Statements ..
  */
+    timestamp_t start = get_timestamp();
    if( ( M <= 0 ) || ( N <= 0 ) ) return;

    nr = N - ( nu = (int)( ( (unsigned int)(N) >> HPL_LASWP01T_LOG2_DEPTH ) <<
@@ -249,4 +250,6 @@ void HPL_dlaswp01T
 /*
  * End of HPL_dlaswp01T
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp02N.c b/src/pauxil/HPL_dlaswp02N.c
index 84a887b..fa1fa1b 100644
--- a/src/pauxil/HPL_dlaswp02N.c
+++ b/src/pauxil/HPL_dlaswp02N.c
@@ -137,6 +137,7 @@ void HPL_dlaswp02N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * A0 = A, * a0;
    double                     * w0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -202,4 +203,6 @@ void HPL_dlaswp02N
 /*
  * End of HPL_dlaswp02N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp03N.c b/src/pauxil/HPL_dlaswp03N.c
index 711c211..e7ae6b5 100644
--- a/src/pauxil/HPL_dlaswp03N.c
+++ b/src/pauxil/HPL_dlaswp03N.c
@@ -127,6 +127,7 @@ void HPL_dlaswp03N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * w = W, * w0;
    double                     * u0;
    const int                  incU = (int)( (unsigned int)(LDU) <<
@@ -191,4 +192,6 @@ void HPL_dlaswp03N
 /*
  * End of HPL_dlaswp03N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp03T.c b/src/pauxil/HPL_dlaswp03T.c
index d6629de..b6b9922 100644
--- a/src/pauxil/HPL_dlaswp03T.c
+++ b/src/pauxil/HPL_dlaswp03T.c
@@ -127,6 +127,7 @@ void HPL_dlaswp03T
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * w = W, * w0;
    double                     * u0;
    const int                  incU = ( 1 << HPL_LASWP03T_LOG2_DEPTH );
@@ -183,4 +184,6 @@ void HPL_dlaswp03T
 /*
  * End of HPL_dlaswp03T
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp04N.c b/src/pauxil/HPL_dlaswp04N.c
index 822a5ac..9ba343b 100644
--- a/src/pauxil/HPL_dlaswp04N.c
+++ b/src/pauxil/HPL_dlaswp04N.c
@@ -158,6 +158,7 @@ void HPL_dlaswp04N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * w = W, * w0;
    double                     * a0, * u0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -282,4 +283,6 @@ void HPL_dlaswp04N
 /*
  * End of HPL_dlaswp04N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 3, (int []){M0, M1, N});
 }
diff --git a/src/pauxil/HPL_dlaswp04T.c b/src/pauxil/HPL_dlaswp04T.c
index 4b62689..c907d85 100644
--- a/src/pauxil/HPL_dlaswp04T.c
+++ b/src/pauxil/HPL_dlaswp04T.c
@@ -159,6 +159,7 @@ void HPL_dlaswp04T
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * w = W, * w0;
    double                     * a0, * u0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -267,4 +268,6 @@ void HPL_dlaswp04T
 /*
  * End of HPL_dlaswp04T
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 3, (int []){M0, M1, N});
 }
diff --git a/src/pauxil/HPL_dlaswp05N.c b/src/pauxil/HPL_dlaswp05N.c
index 928e7f7..a172ded 100644
--- a/src/pauxil/HPL_dlaswp05N.c
+++ b/src/pauxil/HPL_dlaswp05N.c
@@ -129,6 +129,7 @@ void HPL_dlaswp05N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * U0 = U, * u0;
    double                     * a0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -192,4 +193,6 @@ void HPL_dlaswp05N
 /*
  * End of HPL_dlaswp05N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp05T.c b/src/pauxil/HPL_dlaswp05T.c
index 50f337a..c83d981 100644
--- a/src/pauxil/HPL_dlaswp05T.c
+++ b/src/pauxil/HPL_dlaswp05T.c
@@ -129,6 +129,7 @@ void HPL_dlaswp05T
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    const double               * U0 = U, * u0;
    double                     * a0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -193,4 +194,6 @@ void HPL_dlaswp05T
 /*
  * End of HPL_dlaswp05T
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp06N.c b/src/pauxil/HPL_dlaswp06N.c
index 8954577..472b86a 100644
--- a/src/pauxil/HPL_dlaswp06N.c
+++ b/src/pauxil/HPL_dlaswp06N.c
@@ -124,6 +124,7 @@ void HPL_dlaswp06N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    double                     r;
    double                     * U0 = U, * a0, * u0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -203,4 +204,6 @@ void HPL_dlaswp06N
 /*
  * End of HPL_dlaswp06N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp06T.c b/src/pauxil/HPL_dlaswp06T.c
index 481b53b..36a4908 100644
--- a/src/pauxil/HPL_dlaswp06T.c
+++ b/src/pauxil/HPL_dlaswp06T.c
@@ -124,6 +124,7 @@ void HPL_dlaswp06T
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    double                     r;
    double                     * U0 = U, * a0, * u0;
    const int                  incA = (int)( (unsigned int)(LDA) <<
@@ -204,4 +205,6 @@ void HPL_dlaswp06T
 /*
  * End of HPL_dlaswp06T
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/src/pauxil/HPL_dlaswp10N.c b/src/pauxil/HPL_dlaswp10N.c
index 8b33de5..a6238f7 100644
--- a/src/pauxil/HPL_dlaswp10N.c
+++ b/src/pauxil/HPL_dlaswp10N.c
@@ -111,6 +111,7 @@ void HPL_dlaswp10N
 /*
  * .. Local Variables ..
  */
+    timestamp_t start = get_timestamp();
    double                     r;
    double                     * a0, * a1;
    const int                  incA = ( 1 << HPL_LASWP10N_LOG2_DEPTH );
@@ -183,4 +184,6 @@ void HPL_dlaswp10N
 /*
  * End of HPL_dlaswp10N
  */
+    timestamp_t duration = get_timestamp() - start;
+    record_measure("", 0, __func__, start, duration, 2, (int []){M, N});
 }
diff --git a/testing/ptest/HPL_pddriver.c b/testing/ptest/HPL_pddriver.c
index dd2b3fd..07f83c3 100644
--- a/testing/ptest/HPL_pddriver.c
+++ b/testing/ptest/HPL_pddriver.c
@@ -107,6 +107,7 @@ int main( ARGC, ARGV )
  * .. Executable Statements ..
  */
    MPI_Init( &ARGC, &ARGV );
+   FILE *measure_file = get_measure_file();
 #ifdef HPL_CALL_VSIPL
    vsip_init((void*)0);
 #endif
@@ -283,6 +284,7 @@ label_end_of_npqs: ;
 #ifdef HPL_CALL_VSIPL
    vsip_finalize((void*)0);
 #endif
+   fclose(measure_file);
    MPI_Finalize();
    exit( 0 );

diff --git a/testing/ptest/HPL_pdtest.c b/testing/ptest/HPL_pdtest.c
index 9039693..395cbe9 100644
--- a/testing/ptest/HPL_pdtest.c
+++ b/testing/ptest/HPL_pdtest.c
@@ -48,6 +48,7 @@
  * Include files
  */
 #include "hpl.h"
+#include <sys/time.h>

 #ifdef STDC_HEADERS
 void HPL_pdtest
@@ -136,6 +137,7 @@ void HPL_pdtest
 /* ..
  * .. Executable Statements ..
  */
+   get_timestamp(); // initialize the timer...
    (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );

    mat.n  = N; mat.nb = NB; mat.info = 0;
'''
