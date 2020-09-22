import itertools
import random
import os
from .peanut import Job, logger


class MemoryCalibration(Job):
    installfile_types = {'monitoring': int,
            'hyperthreading': bool, 'perf_pct': int, 'idle_state': bool, 'turboboost': bool}
    expfile_types = {'operation': str, 'size': int}
    all_op = ['memcpy']
    expfile_header_in_file = False
    expfile_header = ['operation', 'size']

    @classmethod
    def check_exp(cls, exp):
        if exp['size'] < 0:
            raise ValueError('Error with experiment %s, negative size.' % exp)
        if exp['operation'] not in cls.all_op:
            raise ValueError('Error with experiment %s, unknown operation.' % exp)

    def setup(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        self.apt_install(
            'build-essential',
            'python3',
            'python3-dev',
            'zip',
            'make',
            'git',
            'time',
            'libxml2',
            'libxml2-dev',
            'hwloc',
            'pciutils',
            'net-tools',
        )
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration',
                checkout='88341bee7413eda570b7eff71f3182bc2cedc65e')
        # Then we compile our test program
        self.nodes.run('make test_memory', directory='platform-calibration/src/calibration')
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(expfile.raw_content, path + '/zoo_sizes')

        cmd = 'numactl --physcpubind=2 --localalloc '
        cmd += ' ./test_memory ./zoo_sizes result.csv'
        self.nodes.run(cmd, directory=path)
        # Adding a hostname column to each file
        result_files = []
        for node in self.nodes:
            name = node.hostnames[0]
            resfile = 'result_%s.csv' % name
            result_files.append(resfile)
            node.run('awk \'{print $0",%s"}\' result.csv > %s' % (name, resfile), directory=path)
            self.director.run("rsync -a '%s:%s' ." % (name, path + '/' + resfile), directory=path)
        self.director.run('cat %s > ./result.csv' % (' '.join(result_files)), directory=path)
        # Adding a header to the file
        self.nodes.run("sed -i '1s/^/function,size,timestamp,duration,hostname\\n/' ./result.csv", directory=path)
        self.add_local_to_archive(path + '/result.csv')

    @classmethod
    def gen_exp(cls):
        sizes_com = {int(10**random.uniform(0, 9)) for _ in range(1000)}
        exp = list(itertools.product(['memcpy'], sizes_com))
        exp *= 50
        random.shuffle(exp)
        return [{'operation': op, 'size': size} for op, size in exp]
