import sys
import itertools
import random
from .peanut import Job


class MPICalibration(Job):
    expfile_types = {'size': int, 'operation': str}

    def setup(self):
        super().setup()
        self.apt_install(
            'build-essential',
            'python3',
            'python3-dev',
            'zip',
            'make',
            'git',
            'time',
            'libopenmpi-dev',
            'openmpi-bin',
            'libxml2',
            'libxml2-dev',
            'hwloc',
            'pciutils',
            'net-tools',
        )
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration')
        self.nodes.run('make', directory='platform-calibration/src/calibration')
        return self

    def run_exp(self):
        min_s = min(exp['size'] for exp in self.expfile)
        max_s = max(exp['size'] for exp in self.expfile)
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(self.expfile.raw_content, path + '/zoo_sizes')
        self.nodes.run('mkdir -p %s' % (path + '/exp'))
        host = ','.join([node.host for node in self.nodes])
        args = '-d exp -m %d -M %d -p exp -s zoo_sizes' % (min_s, max_s)
        self.director.run('mpirun --allow-run-as-root -np 2 -host %s ./calibrate %s' % (host, args),
                          directory=path)
        self.add_local_to_archive(path + '/exp')

    @classmethod
    def gen_exp(self):
        op = ['Recv', 'Isend', 'PingPong', 'Wtime', 'Iprobe', 'Test']
        sizes = {int(10**random.uniform(0, 6)) for _ in range(1000)}
        exp = list(itertools.product(op, sizes))
        exp *= 50
        random.shuffle(exp)
        return [{'operation': op, 'size': size} for op, size in exp]


def main():
    MPICalibration.main(sys.argv[1:])
