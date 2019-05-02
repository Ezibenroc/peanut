import time
from .peanut import Job, logger


class FrequencyGet(Job):
    sleep_time = 10

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
        name = name[:40].ljust(40)
        logger.info('%s | %s' % (name, self.nodes.pretty_frequency()))

    def run_exp(self):
        self.nodes.run('tmux new-session -d -s tmux_0 "stress -c 32 -t 600s"')  # load all the cores at 100%
        self.print_freq('initial state')
        self.nodes.disable_hyperthreading()
        self.print_freq('hyperthreading disabled')
        self.nodes.disable_idle_state()
        self.print_freq('C-states disabled')
        self.nodes.disable_turboboost()
        self.print_freq('Turboboost disabled')
        for freq in [1.2, 1.4, 1.6, 1.8, 2, 2.2]:
            self.nodes.set_frequency_information(governor='performance', min_freq=freq*1000000, max_freq=freq*1000000)
            self.print_freq('Frequency fixed to %f GHz' % freq)
        self.nodes.set_frequency_performance()
        self.print_freq('Frequency fixed to the max')
