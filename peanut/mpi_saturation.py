from .peanut import Job


class MPISaturation(Job):
    expfile_types = {}
    expfile_header_in_file = False
    expfile_header = []

    def setup(self):
        super().setup()
        self.apt_install(
            'build-essential',
            'zip',
            'make',
            'git',
            'time',
            'libopenmpi-dev',
            'openmpi-bin',
            'hwloc',
            'pciutils',
            'net-tools',
        )
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration')
        self.nodes.run('mpicc alltoall_loadtest.c -o alltoall_loadtest',
                       directory='platform-calibration/src/saturation')
        return self

    def run_exp(self):
        path = '/tmp/platform-calibration/src/saturation'
        self.nodes.run('mkdir -p %s' % (path + '/exp_monocore'))
        self.nodes.run('mkdir -p %s' % (path + '/exp_multicore'))
        host = self.hostnames
        host_str = ','.join(host)
        self.director.run('mpirun --allow-run-as-root -np %d -host %s ./alltoall_loadtest' % (len(host), host_str),
                          directory=path)
        self.director.run('mv *.csv exp_monocore', directory=path)
        nb_cores = len(self.nodes.cores)
        host = [host for host in self.hostnames for _ in range(nb_cores)]
        host_str = ','.join(host)
        self.director.run('mpirun --allow-run-as-root -np %d -host %s ./alltoall_loadtest' % (len(host), host_str),
                          directory=path)
        self.director.run('mv *.csv exp_multicore', directory=path)
        self.add_local_to_archive(path + '/exp_monocore')
        self.add_local_to_archive(path + '/exp_multicore')
