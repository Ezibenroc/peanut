import os
import time
from .peanut import Job, logger, RunError


class SW4lite(Job):

    def setup(self):
        super().setup()
        self.apt_install(
            'build-essential',
            'linux-tools',
            'gfortran',
            'python3',
            'python3-dev',
            'zip',
            'make',
            'cmake',
            'git',
            'libboost-dev',
            'libopenblas-base',
            'libopenblas-dev',
            'time',
            'libopenmpi-dev',
            'openmpi-bin',
            'libxml2',
            'libxml2-dev',
            'hwloc',
            'pciutils',
            'net-tools',
        )

        self.git_clone('https://framagit.org/simgrid/simgrid.git', 'simgrid',
                       checkout='a6f883f0e28e60a805227007ec71cac80bced118')
        self.nodes.run('mkdir build && cd build && cmake -Denable_documentation=OFF ..', directory='simgrid')
        self.nodes.run('make -j 64 && make install', directory='simgrid/build')
        self.git_clone('https://github.com/geodynamics/sw4lite.git', 'sw4lite_mpi')
        self.nodes.write_files(self.pointsource, 'sw4lite_mpi/tests/pointsource/pointsource.in')
        self.nodes.run('mkdir -p sw4lite_mpi/optimize/pointsource-h0p04')
        self.nodes.run('cp -r sw4lite_mpi sw4lite_smpi')
        self.nodes.run('sed -i -e "s/\mpicxx/\smpicxx/g" Makefile', directory='sw4lite_smpi')
        self.nodes.run('sed -i -e "s/\mpic++/\smpicxx/g" Makefile', directory='sw4lite_smpi')
        self.nodes.run('sed -i -e "s/\mpif90/\smpiff/g" Makefile', directory='sw4lite_smpi')
        while True:
            try:
                self.nodes.run('make -j 64 openmp=no', directory='sw4lite_mpi')
                self.nodes.run('make -j 64 openmp=no', directory='sw4lite_smpi')
            except RunError:
                logger.warning('Previous command failed, retrying')
                time.sleep(1)
            else:
                break
        self.git_clone('https://github.com/brendangregg/FlameGraph.git', 'FlameGraph')
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        platform = [f for f in self.expfile if f.extension == 'xml']
        assert len(platform) == 1
        platform = platform[0]
        nb = len(self.hostnames)
        hosts = ','.join(self.hostnames)
        cmd = 'mpirun --allow-run-as-root -np %d -host %s ./sw4lite ' % (nb, hosts)
        cmd += '../tests/pointsource/pointsource.in'
        reality = self.director.run_unique(cmd, directory='sw4lite_mpi/optimize')
        self.nodes.write_files('\n'.join(self.hostnames), 'sw4lite_smpi/optimize/hosts.txt')
        self.nodes.write_files(platform.raw_content, 'sw4lite_smpi/optimize/platform.xml')
        cmd = 'smpirun -np %d -hostfile ./hosts.txt -platform ./platform.xml --cfg=smpi/display-timing:yes ' % nb
        cmd += '-wrapper "perf record -F1000 --call-graph dwarf" --cfg=smpi/keep-temps:true '
        cmd += './sw4lite ../tests/pointsource/pointsource.in'
        simulation = self.director.run_unique(cmd, directory='sw4lite_smpi/optimize')
        flamedir = os.path.join(self.nodes.working_dir, 'FlameGraph')
        cmd = 'perf script | {0}/stackcollapse-perf.pl --kernel | {0}/flamegraph.pl > flame_graph.svg'
        cmd = cmd.format(flamedir)
        self.director.run_unique(cmd, directory='sw4lite_smpi/optimize')
        self.add_local_to_archive('sw4lite_smpi/optimize/flame_graph.svg')
        self.add_content_to_archive(reality.stdout, 'result_reality.stdout')
        self.add_content_to_archive(reality.stderr, 'result_reality.stderr')
        self.add_content_to_archive(simulation.stdout, 'result_simulation.stdout')
        self.add_content_to_archive(simulation.stderr, 'result_simulation.stderr')

    pointsource = '''
fileio verbose=1 path=pointsource-h0p04
grid x=8 y=8 z=4 h=0.04
time t=0.6
testpointsource rho=1 cp=1.6 cs=0.8 halfspace=1
supergrid gp=30
source x=4.0 y=4.0 z=1.34 Mxx=1 Myy=1 Mzz=1 Mxy=0 Mxz=0 Myz=0 t0=0 freq=1 type=C6SmoothBump
developer checkfornan=0 cfl=1.3 reporttiming=1 corder=0
'''
