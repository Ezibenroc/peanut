import os
import time
from .peanut import Job, logger


class Empty(Job):

    def setup(self):
        pass

    def run_exp(self):
        logger.info(f'Job is ready, node(s):')
        for host in self.hostnames:
            logger.info(f'\t{host}')
        self.nodes.run('sleep 100d')
