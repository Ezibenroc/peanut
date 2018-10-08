import itertools
import random
from .peanut import Job, logger


class MPICalibration(Job):
    expfile_types = {'operation': str, 'size': int}
    op_com = ['Recv', 'Isend', 'PingPong']
    op_test = ['Wtime', 'Iprobe', 'Test']
    all_op = op_com + op_test
    expfile_header_in_file = False
    expfile_header = ['operation', 'size']

    @classmethod
    def check_exp(cls, exp):
        if exp['size'] < 0:
            raise ValueError('Error with experiment %s, negative size.' % exp)
        if exp['operation'] not in cls.all_op:
            raise ValueError('Error with experiment %s, unknown operation.' % exp)

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
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        min_s = min(exp['size'] for exp in expfile)
        max_s = max(exp['size'] for exp in expfile)
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(expfile.raw_content, path + '/zoo_sizes')
        self.nodes.run('mkdir -p %s' % (path + '/exp'))
        host = self.hostnames
        if len(host) == 1:  # testing on localhost
            host = ['localhost']*2
        elif len(host) > 2:
            host = host[:2]
            logger.warning('Too much nodes for the MPI calibration, will only use %s and %s' % tuple(host))
        host = ','.join(host)
        output = self.director.run_unique('python3 find_breakpoints.py --allow-run-as-root -np 2 -host %s' % host,
                                          directory=path)
        self.add_content_to_archive(output.stdout, 'breakpoints')
        args = '-d exp -m %d -M %d -p exp -s zoo_sizes' % (min_s, max_s)
        self.director.run('mpirun --allow-run-as-root -np 2 -host %s ./calibrate %s' % (host, args),
                          directory=path)
        self.add_local_to_archive(path + '/exp')

    @classmethod
    def gen_exp(cls):
        sizes_com = {int(10**random.uniform(0, 6)) for _ in range(1000)}
        sizes_test = {int(10**random.uniform(0, 4)) for _ in range(50)}
        exp = list(itertools.product(cls.op_com, sizes_com)) + list(itertools.product(cls.op_test, sizes_test))
        exp *= 50
        random.shuffle(exp)
        return [{'operation': op, 'size': size} for op, size in exp]
