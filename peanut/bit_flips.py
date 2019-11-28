import os
from .peanut import Job, logger


class BitFlips(Job):
    expfile_types = {'mask_size': int, 'outer_loop': int, 'inner_loop': int, 'monitoring': int, 'sleep_time': int,
                     'cores': str}
    expfile_header_in_file = True

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
        self.git_clone('https://github.com/Ezibenroc/Stress-Test', 'stress-test',
                       checkout='792cd73816889ff38f392529dd3786c6e93d4299')
        self.nodes.run('make test_flips', directory='stress-test')
        self.nodes.set_frequency_information_pstate(min_perf_pct=30, max_perf_pct=30)
        self.nodes.disable_hyperthreading()
        self.nodes.set_frequency_information_pstate(min_perf_pct=100, max_perf_pct=100)
        self.nodes.disable_idle_state()
        return self

    def run_exp(self):
        assert len(self.expfile) == 1
        expfile = self.expfile[0]
        for i, exp in enumerate(expfile):
            cores = [int(c) for c in exp['cores'].split()]
            possible_cores = sum(self.nodes.cores, [])  # aggregating the list of lists
            diff = set(cores) - set(possible_cores)
            if len(diff) > 0:
                logger.error('The following cores are not available on the nodes: %s' % diff)

            numactl_str = 'numactl --physcpubind=%d --localalloc'
            monocore_files = []
            for i in cores:
                numactl = numactl_str % i
                filename = 'result_%d.csv' % i
                monocore_files.append(filename)
                cmd = './test_flips %s %d %d %d"' % (filename, exp['outer_loop'], exp['inner_loop'], i)
                command = 'tmux new-session -d -s tmux_exp_%d' % i
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

            self.nodes.run('cat %s > ./result.csv' % (' '.join(monocore_files)), directory=path)
            # Adding a hostname column to each file
            result_files = []
            for node in self.nodes:
                name = node.hostnames[0]
                resfile = 'result_%s.csv' % name
                result_files.append(resfile)
                node.run('awk \'{print $0",%s"}\' result.csv > %s' % (name, resfile), directory=path)
                self.director.run("rsync -a '%s:%s' ." % (name, path + '/' + resfile), directory=path)
            output_file = './result_%d.csv' % i
            self.director.run('cat %s > %s' % (' '.join(result_files), output_file), directory=path)
            # Adding a header to the file
            self.nodes.run("sed -i '1s/^/timestamp,duration,core,hostname\\n/' %s" % output_file, directory=path)
            self.add_local_to_archive(path + '/%s' % output_file)


    @classmethod
    def gen_exp(cls):
        import random
        exp = {
            'mask_size':  -1,
            'outer_loop': 1000,
            'inner_loop': 10000000,
            'sleep_time': 1,
            'monitoring': 1,
            'cores': ' '.join(str(n) for n in range(32))
        }
        experiment = []
        for size in range(54):
            tmp = dict(exp)
            tmp['mask_size'] = size
            experiment.append(tmp)
        random.shuffle(experiment)
        return experiment
