import os
import time
from .peanut import Job, logger, RunError


class BitFlips(Job):
    expfile_types = {'mask_size': int, 'outer_loop': int, 'inner_loop': int, 'sleep_time': int,
                     'cores': str}
    expfile_header_in_file = True
    installfile_types = {'monitoring': int, 'AVX2': bool}

    @classmethod
    def check_exp(cls, exp):
        for k, v in exp.items():
            if k == 'cores':
                try:
                    [int(c) for c in v.split()]
                except ValueError:
                    raise ValueError('Error with experiment %s, unparsable core list %s' % (exp, k))
            elif k == 'mask_size':
                if v not in range(0, 54):
                    raise ValueError('The mask size must be in the interval [0, 53]')
            elif v < 0:
                raise ValueError('Error with experiment %s, negative %s' % (exp, k))

    def setup(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        self.git_clone('https://github.com/Ezibenroc/Stress-Test', 'stress-test',
                       checkout='3b4e66ff21d85b30b5687daed6647f65ca243ca9')
        if install_options['AVX2']:
            make_option = ' CFLAGS="-DAVX2"'
        else:
            make_option = ''
        self.nodes.run('make %s test_flips' % make_option, directory='stress-test')
        self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        self.nodes.disable_idle_state()
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        path = '/tmp/stress-test'
        for exp_id, exp in enumerate(expfile):
            cores = [int(c) for c in exp['cores'].split()]
            possible_cores = sum(self.nodes.cores, [])  # aggregating the list of lists
            diff = set(cores) - set(possible_cores)
            if len(diff) > 0:
                logger.error('The following cores are not available on the nodes: %s' % diff)

            numactl_str = 'numactl --physcpubind=%d --localalloc'
            monocore_files = []
            self.add_timestamp('sub_exp_start', exp_id)
            for core_id in cores:
                numactl = numactl_str % core_id
                filename = 'result_%d.csv' % core_id
                monocore_files.append(filename)
                cmd = './test_flips %s %d %d %d %d' % (filename, exp['mask_size'], exp['outer_loop'], exp['inner_loop'], core_id)
                command = 'tmux new-session -d -s tmux_exp_%d' % core_id
                command += ' "%s %s"' % (numactl, cmd)
                self.nodes.run(command, directory=path)
            # Waiting for all the commands to be finished
            waiting_nodes = list(self.nodes)
            while len(waiting_nodes) > 0:
                node = waiting_nodes[0]
                try:
                    node.run('tmux ls | grep tmux_exp')
                except RunError:  # this node has finished, let's remove it
                    waiting_nodes = waiting_nodes[1:]
                else:  # this node has not finished yet
                    time.sleep(60)

            self.add_timestamp('sub_exp_stop', exp_id)
            self.nodes.run('cat %s > ./result.csv' % (' '.join(monocore_files)), directory=path)
            # Adding a hostname column to each file
            result_files = []
            for node in self.nodes:
                name = node.hostnames[0]
                resfile = 'result_%s.csv' % name
                result_files.append(resfile)
                node.run('awk \'{print $0",%s"}\' result.csv > %s' % (name, resfile), directory=path)
                self.director.run("rsync -a '%s:%s' ." % (name, path + '/' + resfile), directory=path)
            output_file = './result_%d.csv' % exp_id
            self.director.run('cat %s > %s' % (' '.join(result_files), output_file), directory=path)
            # Adding a header to the file
            self.nodes.run("sed -i '1s/^/timestamp,duration,core,hostname\\n/' %s" % output_file, directory=path)
            self.add_local_to_archive(path + '/%s' % output_file)
            time.sleep(exp['sleep_time'])


    @classmethod
    def gen_exp(cls):
        import random
        exp = {
            'mask_size':  -1,
            'outer_loop': 1000,
            'inner_loop': 250000000,
            'sleep_time': 0,
            'cores': ' '.join(str(n) for n in range(32))
        }
        experiment = []
#       for size in range(54):
        for size in [0, 10, 20, 30, 40, 53]:
            tmp = dict(exp)
            tmp['mask_size'] = size
            experiment.append(tmp)
        copy = list(experiment)
        random.shuffle(experiment)
        random.shuffle(copy)
        return experiment + copy
