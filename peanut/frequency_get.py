import time
from .peanut import Job, logger


class FrequencyGet(Job):
    sleep_time = 10
    frequencies = [1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]

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
        for freq in self.frequencies:
            self.nodes.set_frequency_information(governor='performance', min_freq=freq*1000000, max_freq=freq*1000000)
            self.print_freq('%s, frequency fixed to %.2f GHz' % (name, freq))

    def run_exp(self):
        self.nodes.run('tmux new-session -d -s tmux_0 "stress -c 32 -t 60000s"')  # load all the cores at 100%
        self.print_freq('Initial state')
        self.test_frequencies('Initial state')
        self.nodes.disable_idle_state()
        self.test_frequencies('C-states disabled')
        self.nodes.disable_turboboost()
        self.test_frequencies('Turboboost disabled')
        self.nodes.disable_hyperthreading()
        self.test_frequencies('hyperthreading disabled')
