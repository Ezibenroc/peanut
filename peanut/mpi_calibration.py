import itertools
import random
from .peanut import Job, logger


class MPICalibration(Job):
    installfile_types = {'monitoring': int, 'background_stress': bool}
    expfile_types = {'operation': str, 'size': int}
    op_com = ['Recv', 'Isend', 'PingPong']
    op_test = ['Wtime', 'Iprobe', 'Test']
    all_op = op_com + op_test
    expfile_header_in_file = False
    expfile_header = ['operation', 'size']
    path = '/tmp/platform-calibration/src/calibration'
    calibration_cores = {0, 1}

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
            'libopenmpi-dev',
            'openmpi-bin',
            'libxml2',
            'libxml2-dev',
            'hwloc',
            'pciutils',
            'net-tools',
        )
        self.git_clone('https://github.com/Ezibenroc/platform-calibration.git', 'platform-calibration')
        self.nodes.run('make', directory=self.path)
        if self.nodes.frequency_information.active_driver == 'intel_pstate':
            self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
            self.nodes.disable_hyperthreading()
            self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        else:
            self.nodes.disable_hyperthreading()
            self.nodes.set_frequency_performance()
        self.nodes.disable_idle_state()
        self.nodes.enable_turboboost()
        if install_options['background_stress']:
            self.start_stress()
        return self

    def start_stress(self):
        # First, we install OpenBLAS and compile the BLAS calibration program
        self.git_clone('https://github.com/xianyi/OpenBLAS.git', 'openblas', checkout='v0.3.1')
        self.nodes.run('make -j 64', directory='openblas')
        self.nodes.run('make install PREFIX=%s' % self.nodes.working_dir, directory='openblas')
        self.nodes.run('ln -s libopenblas.so libblas.so', directory='lib')
        make_var = 'CFLAGS="-DMASK_SIZE=0"'
        self.nodes.run('BLAS_INSTALLATION=%s make calibrate_blas %s' % (self.nodes.working_dir, make_var),
                       directory=self.path)
        # Then, we run the BLAS calibration in some background processes
        dgemm_expfile = self.path + '/dgemm_exp.csv'
        self.nodes.write_files(('dgemm' + ',128'*6 + '\n')*1000, dgemm_expfile)
        numactl_str = 'numactl --physcpubind=%d --localalloc'
        nb_cores = len(self.nodes.cores)
        ldlib = 'LD_LIBRARY_PATH=%s/lib' % self.nodes.working_dir
        for i in range(nb_cores):
            if i in self.calibration_cores:
                continue
            numactl = numactl_str % i
            filename = 'result_monocore_%d.csv' % i
            command = 'tmux new-session -d -s tmux_blas_%d "OMP_NUM_THREADS=1' % i
            cmd = './calibrate_blas -s ./dgemm_exp.csv'
            command += ' %s %s %s -l 1000000000 -o %s"' % (ldlib, numactl, cmd, filename)
            self.nodes.run(command, directory=self.path)

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        min_s = min(exp['size'] for exp in expfile)
        max_s = max(exp['size'] for exp in expfile)
        self.nodes.write_files(expfile.raw_content, self.path + '/zoo_sizes')
        self.nodes.run('mkdir -p %s' % (self.path + '/exp'))

        host = self.hostnames[0] + ',' + self.hostnames[-1]
        output = self.director.run_unique('python3 find_breakpoints.py --allow-run-as-root -np 2 -host %s' % host,
                                          directory=self.path)
        self.add_content_to_archive(output.stdout, 'breakpoints')

        mapping = []
        cores = list(self.calibration_cores)
        for rank, i in enumerate([0, -1]):
            host = self.hostnames[i]
            core = cores[i]
            mapping.append('rank %d=%s slot=%d' % (rank, host, core))
        mapping = '\n'.join(mapping)
        hosts = '\n'.join('%s slots=%d' % (host, len(self.nodes.cores)) for host in self.hostnames)
        hostfile = '/tmp/hosts.txt'
        rankfile = '/tmp/ranks.txt'
        self.nodes.write_files(hosts, hostfile)
        self.nodes.write_files(mapping, rankfile)
        cmd = 'mpirun --mca rmaps_rank_file_physical 1 --allow-run-as-root --report-bindings --timestamp-output -np 2'
        cmd += ' -hostfile %s' % hostfile
        cmd += ' --rankfile %s' % rankfile
        args = '-d exp -m %d -M %d -p exp -s zoo_sizes' % (min_s, max_s)
        cmd += ' ./calibrate %s' % args
        self.director.run(cmd, directory=self.path)
        self.add_local_to_archive(self.path + '/exp')

    @classmethod
    def gen_exp(cls):
        sizes_com = {int(10**random.uniform(0, 6)) for _ in range(1000)}
        sizes_test = {int(10**random.uniform(0, 4)) for _ in range(50)}
        exp = list(itertools.product(cls.op_com, sizes_com)) + list(itertools.product(cls.op_test, sizes_test))
        exp *= 50
        random.shuffle(exp)
        return [{'operation': op, 'size': size} for op, size in exp]
