import os
import time
from .peanut import logger, ExpFile, Time, RunError
from .abstract_hpl import AbstractHPL


class HPL(AbstractHPL):
    def setup(self):
        super().setup()
        self.apt_install(
            'openmpi-bin',
            'libopenmpi-dev',
            'net-tools',
        )
        if self.trace_execution:
            self.install_akypuera(smpi=False)
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout='v0.3.1')
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.nodes.run('wget http://www.netlib.org/benchmark/hpl/hpl-2.2.tar.gz')
        self.nodes.run('tar -xvf hpl-2.2.tar.gz')
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
            cmd += ' -x LD_LIBRARY_PATH=/tmp/lib'
            if self.trace_execution:
                lib = os.path.join(self.akypuera_dir, 'libaky.so')
                cmd += ' -x LD_PRELOAD=%s' % lib
            cmd += ' ./xhpl'
            cmd = cmd % (max(nb_hpl_proc, nb_proc), thread_per_process, hosts)
            output = self.director.run_unique(cmd, directory=self.hpl_dir+'/bin/Debian')
            if self.trace_execution:
                rstdir = os.path.join(self.orchestra.working_dir, self.hpl_dir, 'bin/Debian/rastro-*.rst')
                for node in self.orchestra.hostnames:
                    self.director.run("rsync -a '%s:%s' ." % (node, rstdir), directory=self.hpl_dir+'/bin/Debian')
                converter = os.path.join(self.akypuera_dir, 'aky_converter')
                paje_file = os.path.join(self.director.working_dir, 'trace_%d.paje' % i)
                self.director.run('ls -l rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                self.director.run('%s rastro-*rst > %s' % (converter, paje_file), directory=self.hpl_dir+'/bin/Debian')
                self.nodes.run('rm -f rastro-*rst', directory=self.hpl_dir+'/bin/Debian')
                self.add_local_to_archive(paje_file)
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
