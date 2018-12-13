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


class SMPIHPL(AbstractHPL):
    expfile_types = dict(dgemm_coefficient=float, dgemm_intercept=float, dtrsm_coefficient=float, dtrsm_intercept=float,
                         **AbstractHPL.expfile_types)

    def setup(self):
        super().setup()
        self.apt_install('python3', 'libboost-dev', 'libatlas-base-dev', 'pajeng')
        self.git_clone('https://framagit.org/simgrid/simgrid.git', 'simgrid',
                       checkout='a6f883f0e28e60a805227007ec71cac80bced118')
        self.nodes.run('mkdir build && cd build && cmake -Denable_documentation=OFF ..', directory='simgrid')
        self.nodes.run('make -j 64 && make install', directory='simgrid/build')
        patches = []
        if self.terminate_early:
            patches.append(self.hpl_early_termination_patch)
        if self.insert_bcast:
            patches.append(self.hpl_bcast_patch)
        patches.append(self.blas_reg_patch)
        patch = '\n'.join(patches) if patches else None
        self.git_clone('https://github.com/Ezibenroc/hpl.git', self.hpl_dir, patch=patch,
                       checkout='2a2823f19b5a981f2470dc7403c369ac48f60a6d')
        self.nodes.run('sed -ri "s|TOPdir\s*=.+|TOPdir="`pwd`"|g" Make.SMPI', directory=self.hpl_dir)
        self.nodes.run('make startup arch=SMPI', directory=self.hpl_dir)
        options = '-DSMPI_OPTIMIZATION'
        if self.trace_execution:
            options += ' -DSMPI_MEASURE'
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
        script = '''
            head -1 $1
            for filename in $*; do
               tail -n +2  $filename
            done
        '''
        self.director.write_files(script, self.hpl_dir+'/bin/SMPI/concatenate.sh')
        results = []
        assert len(self.expfile) == 2
        platform = [f for f in self.expfile if f.extension == 'xml']
        assert len(platform) == 1
        platform = TopoFile(platform[0])
        expfile = [f for f in self.expfile if f.extension != 'xml']
        assert len(expfile) == 1
        expfile = expfile[0]
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

            dgemm_coeff = exp['dgemm_coefficient']
            dgemm_inter = exp['dgemm_intercept']
            dtrsm_coeff = exp['dtrsm_coefficient']
            dtrsm_inter = exp['dtrsm_intercept']

            cmd = 'SMPI_DGEMM_COEFFICIENT=%e SMPI_DGEMM_INTERCEPT=%e ' % (dgemm_coeff, dgemm_inter)
            cmd += 'SMPI_DTRSM_COEFFICIENT=%e SMPI_DTRSM_INTERCEPT=%e ' % (dtrsm_coeff, dtrsm_inter)
            cmd += 'TIME="/usr/bin/time:output %U %S %F %R %P" '
            cmd += 'smpirun -wrapper /usr/bin/time --cfg=smpi/privatize-global-variables:dlopen -np %d ' % nb_hpl_proc
            if self.trace_execution:
                paje_file = os.path.join(self.director.working_dir, 'trace_%d.paje' % i)
                cmd += '--cfg=tracing:yes --cfg=tracing/filename:%s --cfg=tracing/smpi:1 ' % paje_file
                cmd += '--cfg=tracing/smpi/display-sizes:yes '
                cmd += '--cfg=tracing/smpi/computing:yes '
            cmd += '--cfg=smpi/shared-malloc-hugepage:/root/huge '
            cmd += '--cfg=smpi/shared-malloc-blocksize:%d ' % (1 << 21)
            cmd += '--cfg=smpi/display-timing:yes -platform platform.xml -hostfile hosts.txt ./xhpl'
            output = self.director.run_unique(cmd, directory=self.hpl_dir+'/bin/SMPI')
            if self.trace_execution:
                mpi_trace = 'trace_mpi_%d.csv' % i
                blas_trace = os.path.join(self.director.working_dir, 'trace_blas_%d.csv' % i)
                self.director.run('pj_dump -u %s | grep -v MPI_Iprobe > %s' % (paje_file, mpi_trace))
                self.director.run('bash concatenate.sh blas*trace > %s' % blas_trace, directory=self.hpl_dir+'/bin/SMPI')
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
        factors['matrix_size'] = [2**i for i in range(12, 18)]
#        factors['matrix_size'] += [s + s//2 for s in factors['matrix_size'][:-1]]
        factors['matrix_size'] = list(range(300000, 500001, 50000))  # list(range(50000, 250001, 25000))
        factors['block_size'] = [2**7]
        factors['dgemm_coefficient'] = [6.484604e-11]
        factors['dgemm_intercept'] = [2.401076e-04]
        factors['dtrsm_coefficient'] = [8.021068e-11]
        factors['dtrsm_intercept'] = [6.929164e-07]
        factors['proc_p'] = [32]
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

    simgrid_loopback_patch = '''
diff --git a/src/surf/sg_platf.cpp b/src/surf/sg_platf.cpp
index f521fd925..fcb273088 100644
--- a/src/surf/sg_platf.cpp
+++ b/src/surf/sg_platf.cpp
@@ -220,7 +220,7 @@ void sg_platf_new_cluster(simgrid::kernel::routing::ClusterCreationArgs* cluster
       link.id        = tmp_link;
       link.bandwidth = cluster->loopback_bw;
       link.latency   = cluster->loopback_lat;
-      link.policy    = simgrid::s4u::Link::SharingPolicy::FATPIPE;
+      link.policy    = simgrid::s4u::Link::SharingPolicy::SHARED;
       sg_platf_new_link(&link);
       linkUp   = simgrid::s4u::Link::by_name_or_null(tmp_link);
       linkDown = simgrid::s4u::Link::by_name_or_null(tmp_link);
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
