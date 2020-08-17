import itertools
import random
import os
from .peanut import Job, logger


class MPIRing(Job):
    installfile_types = {'monitoring': int}
    expfile_types = {'operation': str, 'size': int}
    all_op = ['ring']
    expfile_header_in_file = False
    expfile_header = ['operation', 'size']

    @classmethod
    def check_exp(cls, exp):
        if exp['size'] < 0:
            raise ValueError('Error with experiment %s, negative size.' % exp)
        if exp['operation'] not in cls.all_op:
            raise ValueError('Error with experiment %s, unknown operation.' % exp)

    def setup(self):
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
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration',
                checkout='7c1897a53c8bc204cd5e20fe0f0a269f35f18eb8')
        self.nodes.run('make test_ring', directory='platform-calibration/src/calibration')
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        path = '/tmp/platform-calibration/src/calibration'
        self.nodes.write_files(expfile.raw_content, path + '/zoo_sizes')
        expdir = path + '/exp'
        self.nodes.run('mkdir -p %s' % (expdir))
        host = self.hostnames
        nb_cores = len(self.nodes.cores)
        nb_ranks = len(host)*nb_cores

        mapping = []
        for rank in range(nb_ranks):
            host = self.hostnames[rank // nb_cores]
            core = rank % nb_cores
            mapping.append('rank %d=%s slot=%d' % (rank, host, core))
        mapping = '\n'.join(mapping)
        hosts = '\n'.join('%s slots=%d' % (host, nb_cores) for host in self.hostnames)
        hostfile = os.path.join('/tmp/hosts.txt')
        rankfile = os.path.join('/tmp/ranks.txt')
        self.nodes.write_files(hosts, hostfile)
        self.nodes.write_files(mapping, rankfile)
        cmd = 'mpirun --allow-run-as-root --report-bindings --timestamp-output -np %d' % nb_ranks
        cmd += ' -hostfile %s' % hostfile
        cmd += ' --rankfile %s' % rankfile
        cmd += ' ./test_ring ./zoo_sizes %s' % expdir
        self.director.run(cmd, directory=path)

        # Aggregating the results
        self.nodes.run('cat %s/*.csv > ./result.csv' % expdir, directory=path)
        # Adding a hostname column to each file
        result_files = []
        for node in self.nodes:
            name = node.hostnames[0]
            resfile = 'result_%s.csv' % name
            result_files.append(resfile)
            self.director.run("rsync -a '%s:%s/result.csv' %s" % (name, path, path + '/' + resfile), directory=path)
        self.director.run('cat %s > ./result.csv' % (' '.join(result_files)), directory=path)
        # Adding a header to the file
        self.nodes.run("sed -i '1s/^/function,rank,size,timestamp,duration\\n/' ./result.csv", directory=path)
        self.add_local_to_archive(path + '/result.csv')

    @classmethod
    def gen_exp(cls):
        sizes_com = {int(10**random.uniform(0, 9)) for _ in range(100)}
        exp = list(itertools.product(cls.all_op, sizes_com))
        exp *= 5
        random.shuffle(exp)
        return [{'operation': op, 'size': size} for op, size in exp]
