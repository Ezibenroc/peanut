import os
import time
from .peanut import logger, ExpFile, Time, RunError
from .abstract_hpl import AbstractHPL


class SMPIHPL(AbstractHPL):
    def setup(self):
        super().setup()
        self.apt_install('python3', 'libboost-dev', 'libatlas-base-dev')  # we don't care about which BLAS is installed
        self.git_clone('https://github.com/simgrid/simgrid.git', 'simgrid', checkout='v3.20')
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
