import os
import time
from .peanut import logger, ExpFile, Time, RunError
from .abstract_hpl import AbstractHPL


class HPL(AbstractHPL):
    def install_akypuera(self):
        self.git_clone('https://github.com/schnorr/akypuera.git', 'akypuera', recursive=True)
        self.nodes.run('mkdir build && cd build && cmake ..', directory='akypuera')
        self.nodes.run('make -j 32', directory='akypuera/build')

    @property
    def akypuera_dir(self):
        return os.path.join(self.nodes.working_dir, 'akypuera/build')

    def setup(self):
        super().setup()
        self.apt_install(
            'openmpi-bin',
            'libopenmpi-dev',
            'net-tools',
        )
        if self.trace_execution:
            self.apt_install('pajeng')
            self.install_akypuera()
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout='v0.3.1')
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.nodes.run('wget http://www.netlib.org/benchmark/hpl/hpl-2.2.tar.gz')
        self.nodes.run('tar -xvf hpl-2.2.tar.gz')
        if self.trace_execution:
            self.nodes.write_files(self.patch, self.hpl_dir + '/patch.diff')
            self.nodes.run('git apply --whitespace=fix patch.diff', directory=self.hpl_dir)
        if self.terminate_early:
            self.nodes.write_files(self.hpl_early_termination_patch, self.hpl_dir + '/patch.diff')
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
        nb_cores = len(self.nodes.cores)
        results = []
        start = time.time()
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        script = '''
            head -1 $1
            for filename in $*; do
               tail -n +2  $filename
            done
        '''
        self.director.write_files(script, self.hpl_dir+'/bin/Debian/concatenate.sh')
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
            hostnames = [host for host in self.hostnames for _ in range(process_per_node)]
            hosts = ','.join(hostnames)
            hpl_file = self.generate_hpl_file(**exp)
            self.nodes.write_files(hpl_file, os.path.join(self.hpl_dir, 'bin/Debian/HPL.dat'))
            cmd = 'mpirun --allow-run-as-root --bind-to none --timestamp-output -np %d -x OMP_NUM_THREADS=%d -H %s'
            cmd += ' --mca orte_rsh_agent ssh --mca pml ob1 --mca btl tcp,self'  # https://gitlab.inria.fr/fheinric/paper-simgrid-energy/blob/master/experiments/mpi_hpl_dvfs_1core/taurus_2017-01-16/execution/HPL_1core_launcher.zsh
            cmd += ' -x LD_LIBRARY_PATH=/tmp/lib'
            if self.trace_execution:
                lib = os.path.join(self.akypuera_dir, 'libaky.so')
                cmd += ' -x LD_PRELOAD=%s' % lib
                cmd = 'LD_PRELOAD=%s %s' % (lib, cmd)
            cmd += ' ./xhpl'
            cmd = cmd % (max(nb_hpl_proc, nb_proc), thread_per_process, hosts)
            output = self.director.run_unique(cmd, directory=self.hpl_dir+'/bin/Debian')
            if self.trace_execution:
                self.director.run('ls -l rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                rstdir = os.path.join(self.orchestra.working_dir, self.hpl_dir, 'bin/Debian/rastro-*.rst')
                blasdir = os.path.join(self.orchestra.working_dir, self.hpl_dir, 'bin/Debian/blas*trace')
                for node in self.orchestra.hostnames:
                    self.director.run("rsync -a '%s:%s' ." % (node, rstdir), directory=self.hpl_dir+'/bin/Debian')
                    self.director.run("rsync -a '%s:%s' ." % (node, blasdir), directory=self.hpl_dir+'/bin/Debian')
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
                blas_trace = os.path.join(self.director.working_dir, 'trace_blas_%d.csv' % i)
                self.director.run('pj_dump -u %s | grep -v MPI_Iprobe > %s' % (paje_file, mpi_trace))
                self.nodes.run('rm -f rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                self.director.run('bash concatenate.sh blas*trace > %s' % blas_trace, directory=self.hpl_dir+'/bin/Debian')
                self.nodes.run('rm -f blas*trace', directory=self.hpl_dir+'/bin/Debian')
                self.add_local_to_archive(mpi_trace)
                self.add_local_to_archive(blas_trace)
            total_time, gflops, residual = self.parse_hpl(output.stdout)
            new_res = dict(exp)
            new_res['time'] = total_time
            new_res['gflops'] = gflops
            new_res['residual'] = residual
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
diff --git a/include/aky_rastro.h b/include/aky_rastro.h
index 30a0163..7b955c6 100644
--- a/include/aky_rastro.h
+++ b/include/aky_rastro.h
@@ -17,6 +17,8 @@ void rst_event_iiiii_ptr(rst_buffer_t *ptr, u_int16_t type, u_int32_t i0, u_int3
 #define rst_event_iiiii(type, i0, i1, i2, i3, i4) rst_event_iiiii_ptr(RST_PTR, type, i0, i1, i2, i3, i4)
 void rst_event_l_ptr(rst_buffer_t *ptr, u_int16_t type, u_int64_t l0);
 #define rst_event_l(type, l0) rst_event_l_ptr(RST_PTR, type, l0)
+void rst_event_il_ptr(rst_buffer_t *ptr, u_int16_t type, u_int32_t i0, u_int64_t l0);
+#define rst_event_il(type, i0, l0) rst_event_il_ptr(RST_PTR, type, i0, l0)
 void rst_event_iil_ptr(rst_buffer_t *ptr, u_int16_t type, u_int32_t i0, u_int32_t i1, u_int64_t l0);
 #define rst_event_iil(type, i0, i1, l0) rst_event_iil_ptr(RST_PTR, type, i0, i1, l0)

diff --git a/src/aky.c b/src/aky.c
index dde4789..5b2e4e5 100644
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

@@ -954,7 +952,9 @@ int tag;
 MPI_Comm comm;
 MPI_Request *request;
 {
-  rst_event(MPI_IRECV_IN);
+  int tsize;
+  PMPI_Type_size(datatype, &tsize);
+  rst_event_i(MPI_IRECV_IN, count * tsize);
   int returnVal =
       PMPI_Irecv(buf, count, datatype, source, tag, comm, request);
   rst_event(MPI_IRECV_OUT);
@@ -993,7 +993,7 @@ MPI_Request *request;
 {
   int tsize;
   PMPI_Type_size(datatype, &tsize);
-  rst_event_l(MPI_ISEND_IN, send_mark);
+  rst_event_il(MPI_ISEND_IN, count * tsize, send_mark);
   rst_event_iil(AKY_PTP_SEND, AKY_translate_rank(comm, dest), count * tsize, send_mark);
   int returnVal =
       PMPI_Isend(buf, count, datatype, dest, tag, comm, request);
@@ -1058,9 +1058,7 @@ int tag;
 MPI_Comm comm;
 MPI_Status *status;
 {
-  rst_event(MPI_PROBE_IN);
   int returnVal = PMPI_Probe(source, tag, comm, status);
-  rst_event(MPI_PROBE_OUT);
   return returnVal;
 }

@@ -1073,8 +1071,10 @@ int tag;
 MPI_Comm comm;
 MPI_Status *status;
 {
+  int tsize;
+  PMPI_Type_size(datatype, &tsize);
   MPI_Status stat2;
-  rst_event(MPI_RECV_IN);
+  rst_event_i(MPI_RECV_IN, count * tsize);
   int returnVal =
       PMPI_Recv(buf, count, datatype, source, tag, comm, &stat2);
   rst_event_i(AKY_PTP_RECV, AKY_translate_rank(comm, stat2.MPI_SOURCE));
@@ -1128,7 +1128,7 @@ MPI_Comm comm;
 {
   int tsize;
   PMPI_Type_size(datatype, &tsize);
-  rst_event_l(MPI_SEND_IN, send_mark);
+  rst_event_il(MPI_SEND_IN, count * tsize, send_mark);
   rst_event_iil(AKY_PTP_SEND, AKY_translate_rank(comm, dest), count * tsize, send_mark);
   int returnVal = PMPI_Send(buf, count, datatype, dest, tag, comm);
   rst_event(MPI_SEND_OUT);
diff --git a/src/aky/aky_converter.c b/src/aky/aky_converter.c
index 860f3a6..586fae9 100644
--- a/src/aky/aky_converter.c
+++ b/src/aky/aky_converter.c
@@ -15,9 +15,11 @@
     along with Akypuera. If not, see <http://www.gnu.org/licenses/>.
 */
 #include <inttypes.h>
+#include <assert.h>
 #include "aky2paje.h"

-int specialPushState; //Identifier for PushState with mark
+int specialPushStateSend; //Identifier for PushState for MPI_Send and MPI_Isend
+int specialPushStateRecv; //Identifier for PushState for MPI_Recv
 int specialStartLinkSizeMark; //Identifier for StartLink with size and mark

 /* DRY */
@@ -188,7 +190,8 @@ int main(int argc, char **argv)
     /* output build version, date and conversion for aky in the trace */
     aky_dump_version (PROGRAM, argv, argc);
     poti_header();
-    specialPushState = poti_header_DeclareEvent (PAJE_PushState, 1, "SendMark string");
+    specialPushStateSend = poti_header_DeclareEvent (PAJE_PushState, 2, "MsgSize string", "SendMark string");
+    specialPushStateRecv = poti_header_DeclareEvent (PAJE_PushState, 1, "MsgSize string");

     specialStartLinkSizeMark = poti_header_DeclareEvent (PAJE_StartLink, 2, "Size string", "Mark string");
     aky_paje_hierarchy();
@@ -293,10 +296,6 @@ int main(int argc, char **argv)
     case MPI_SCATTER_IN:
     case MPI_SCATTERV_IN:
     case MPI_WAIT_IN:
-    case MPI_IRECV_IN:
-    case MPI_ISEND_IN:
-    case MPI_RECV_IN:
-    case MPI_SEND_IN:
     case MPI_BCAST_IN:
     case MPI_BARRIER_IN:
     case MPI_GATHER_IN:
@@ -419,12 +418,38 @@ int main(int argc, char **argv)
           /* uint64 upper range is a 20 digits integer in base 10 */
           char mark[21];
           snprintf(mark, 21, "%"PRIu64, event.v_uint64[0]);
-	  poti_user_PushState (specialPushState, timestamp, mpi_process, "STATE", value, 1, mark);
+	  poti_user_PushState (specialPushStateSend, timestamp, mpi_process, "STATE", value, 1, mark);
         }else{
           poti_PushState(timestamp, mpi_process, "STATE", value);
         }
       }
       break;
+    case MPI_ISEND_IN:
+    case MPI_SEND_IN:
+      if (!arguments.no_states){
+        char value[AKY_DEFAULT_STR_SIZE];
+        snprintf(value, AKY_DEFAULT_STR_SIZE, "%s", name_get(event.type));
+        assert(event.ct.n_uint32 >= 1); // for obscure reasons, there is a second uint32...
+        assert(event.ct.n_uint64 == 1);
+          /* uint64 upper range is a 20 digits integer in base 10 */
+          char size[21], mark[21];
+          snprintf(mark, 21, "%"PRIu64, event.v_uint64[0]);
+          snprintf(size, 21, "%"PRIu64, event.v_uint32[0]);
+          poti_user_PushState (specialPushStateSend, timestamp, mpi_process, "STATE", value, 2, size, mark);
+      }
+      break;
+    case MPI_IRECV_IN:
+    case MPI_RECV_IN:
+      if (!arguments.no_states){
+        char value[AKY_DEFAULT_STR_SIZE];
+        snprintf(value, AKY_DEFAULT_STR_SIZE, "%s", name_get(event.type));
+        assert(event.ct.n_uint32 == 1);
+          /* uint64 upper range is a 20 digits integer in base 10 */
+          char size[21];
+          snprintf(size, 21, "%"PRIu64, event.v_uint32[0]);
+          poti_user_PushState (specialPushStateRecv, timestamp, mpi_process, "STATE", value, 1, size);
+      }
+      break;
     case MPI_COMM_SPAWN_OUT:
     case MPI_COMM_GET_NAME_OUT:
     case MPI_COMM_SET_NAME_OUT:
diff --git a/src/aky_rastro.c b/src/aky_rastro.c
index 7adac2f..26615b8 100644
--- a/src/aky_rastro.c
+++ b/src/aky_rastro.c
@@ -60,6 +60,14 @@ void rst_event_l_ptr(rst_buffer_t *ptr, u_int16_t type, u_int64_t l0)
   rst_endevent(ptr);
 }

+void rst_event_il_ptr(rst_buffer_t *ptr, u_int16_t type, u_int32_t i0, u_int64_t l0)
+{
+  rst_startevent(ptr, type<<18|0x24770);
+  RST_PUT(ptr, u_int64_t, l0);
+  RST_PUT(ptr, u_int32_t, i0);
+  rst_endevent(ptr);
+}
+
 void rst_event_iil_ptr(rst_buffer_t *ptr, u_int16_t type, u_int32_t i0, u_int32_t i1, u_int64_t l0)
 {
   rst_startevent(ptr, type<<18|0x24770);
'''

    @property
    def patch(self):
        return r'''
diff --git a/include/hpl_blas.h b/include/hpl_blas.h
index 41e5afd..1b8d40e 100644
--- a/include/hpl_blas.h
+++ b/include/hpl_blas.h
@@ -169,11 +169,40 @@ STDC_ARGS(
 #define    HPL_dtrsv           cblas_dtrsv
 #define    HPL_dger            cblas_dger

-#define    HPL_dgemm           cblas_dgemm
-#define    HPL_dtrsm           cblas_dtrsm
-
 #endif

+FILE *get_measure_file();
+double get_timestamp(struct timeval timestamp);
+
+#define START_MEASURE(before) ({\
+    gettimeofday(&before, NULL);\
+})
+#define STOP_MEASURE(before, function, M, N, K, lda, ldb, ldc)  ({\
+    struct timeval after = {};\
+    gettimeofday(&after, NULL);\
+    double duration = (after.tv_sec-before.tv_sec) + 1e-6*(after.tv_usec-before.tv_usec);\
+    int my_rank, buff=0;\
+    double timestamp = get_timestamp(before);\
+    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);\
+    FILE *measure_file = get_measure_file();\
+    if(!measure_file) {fprintf(stderr, "error with measure_file\n"); exit(1);}\
+    fprintf(measure_file, "%s, %s, %d, %d, %d, %d, %d, %d, %d, %d, %g, %g\n", function, __FILE__, __LINE__, my_rank, M, N, K, lda, ldb, ldc, duration, timestamp);\
+})
+
+#define  HPL_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc)  ({\
+    struct timeval before = {};\
+    START_MEASURE(before);\
+    cblas_dgemm(layout, TransA, TransB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);\
+    STOP_MEASURE(before, "dgemm", M, N, K, lda, ldb, ldc);\
+})
+
+#define HPL_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb) ({\
+    struct timeval before = {};\
+    START_MEASURE(before);\
+    cblas_dtrsm(layout, Side, Uplo, TransA, Diag, M, N, alpha, A, lda, B, ldb);\
+    STOP_MEASURE(before, "dtrsm", M, N, -1, lda, ldb, -1);\
+})
+
 #ifdef HPL_CALL_FBLAS
 /*
  * ---------------------------------------------------------------------
diff --git a/src/blas/HPL_dgemm.c b/src/blas/HPL_dgemm.c
index f27888a..4e3f7e9 100644
--- a/src/blas/HPL_dgemm.c
+++ b/src/blas/HPL_dgemm.c
@@ -47,8 +47,35 @@
 /*
  * Include files
  */
+#include <sys/time.h>
 #include "hpl.h"

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
+        fprintf(measure_file, "function, file, line, rank, m, n, k, lead_A, lead_B, lead_C, duration, timestamp\n");
+    }
+    return measure_file;
+}
+
+double get_timestamp(struct timeval timestamp) {
+    static struct timeval start = {.tv_sec=-1, .tv_usec=-1};
+    if(start.tv_sec < 0) {
+        gettimeofday(&start, NULL);
+    }
+    double t = (timestamp.tv_sec-start.tv_sec) + 1e-6*(timestamp.tv_usec-start.tv_usec);
+    return t;
+}
+
 #ifndef HPL_dgemm

 #ifdef HPL_CALL_VSIPL
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
index 9039693..33b11ac 100644
--- a/testing/ptest/HPL_pdtest.c
+++ b/testing/ptest/HPL_pdtest.c
@@ -48,6 +48,7 @@
  * Include files
  */
 #include "hpl.h"
+#include <sys/time.h>

 #ifdef STDC_HEADERS
 void HPL_pdtest
@@ -136,6 +137,8 @@ void HPL_pdtest
 /* ..
  * .. Executable Statements ..
  */
+   struct timeval tmp_time = {};
+   get_timestamp(tmp_time); // initialize the timer...
    (void) HPL_grid_info( GRID, &nprow, &npcol, &myrow, &mycol );

    mat.n  = N; mat.nb = NB; mat.info = 0;
'''
