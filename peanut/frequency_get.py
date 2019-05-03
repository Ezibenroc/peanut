import time
from .peanut import Job, logger


class FrequencyGet(Job):
    sleep_time = 10
    percentages = list(range(10, 101, 10)) + [1]

    def setup(self):
        super().setup()
        self.apt_install(
            'build-essential',
            'zip',
            'make',
            'git',
            'time',
            'hwloc',
            'pciutils',
            'net-tools',
            'cpufrequtils',
            'linux-cpupower',
            'stress',
            'tmux',
        )
        return self

    def print_freq(self, name):
        time.sleep(self.sleep_time)
        name = name[:60].ljust(60)
        logger.info('%s # %s' % (name, self.nodes.pretty_frequency()))

    def test_frequencies(self, name):
        logger.info('#'*60)
        for pct in self.percentages:
            self.nodes.set_frequency_information_pstate(min_perf_pct=pct, max_perf_pct=pct)
            self.print_freq('%s, %3d%%' % (name, pct))

    def stress_all_cores(self):
        nb_proc = len(self.nodes.cores) * 4
        self.nodes.run('tmux new-session -d -s tmux_0 "stress -c %d -t 60000s"' % nb_proc)

    def run_exp(self):
        self.stress_all_cores()
        self.print_freq('Initial state')
        self.test_frequencies('Initial state')
        self.nodes.disable_idle_state()
        self.test_frequencies('C-states disabled')
        self.nodes.disable_turboboost()
        self.test_frequencies('Turboboost disabled')
        self.nodes.disable_hyperthreading()
        self.test_frequencies('hyperthreading disabled')
