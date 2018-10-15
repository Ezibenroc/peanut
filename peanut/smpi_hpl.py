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
        self.apt_install('python3', 'libboost-dev', 'libatlas-base-dev')  # we don't care about which BLAS is installed
        self.git_clone('https://github.com/simgrid/simgrid.git', 'simgrid', checkout='v3.20', patch=self.simgrid_patch)
        self.nodes.run('mkdir build && cd build && cmake -Denable_documentation=OFF ..', directory='simgrid')
        self.nodes.run('make -j 64 && make install', directory='simgrid/build')
        self.git_clone('https://github.com/Ezibenroc/hpl.git', self.hpl_dir)
        self.nodes.run('sed -ri "s|TOPdir\s*=.+|TOPdir="`pwd`"|g" Make.SMPI', directory=self.hpl_dir)
        self.nodes.run('make startup arch=SMPI', directory=self.hpl_dir)
        while True:
            try:
                self.nodes.run('make SMPI_OPTS="-DSMPI_OPTIMIZATION" arch=SMPI', directory=self.hpl_dir)
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
                cmd += '--cfg=tracing/smpi/computing:yes '
            cmd += '--cfg=smpi/shared-malloc-hugepage:/root/huge '
            cmd += '--cfg=smpi/shared-malloc-blocksize:%d ' % (1 << 21)
            cmd += '--cfg=smpi/display-timing:yes -platform platform.xml -hostfile hosts.txt ./xhpl'
            output = self.director.run_unique(cmd, directory=self.hpl_dir+'/bin/SMPI')
            if self.trace_execution:
                self.add_local_to_archive(paje_file)
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

    simgrid_patch = '''
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
