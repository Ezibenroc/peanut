#! /usr/bin/env python3

import re
import datetime
import fabric
import logging
import collections
import colorlog
import time
import os
import sys
import socket
import tempfile
import yaml
import random
import json
import io
import copy
import argparse
import lxml.etree
import csv
import hashlib
from .version import __version__, __git_version__

handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    '%(log_color)s[%(asctime)s][%(levelname)s] %(message_log_color)s%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    secondary_log_colors={
        'message': {
            'DEBUG': 'white',
            'INFO': 'white',
            'WARNING': 'white',
            'ERROR': 'white',
            'CRITICAL': 'white',
        }
    }
)
handler.setFormatter(formatter)
logger = colorlog.getLogger(__name__)
log_stream = io.StringIO()
io_handler = logging.StreamHandler(log_stream)
io_handler.setFormatter(logging.Formatter('[%(asctime)s][%(levelname)s] %(message)s'))
logger.addHandler(handler)
logger.addHandler(io_handler)
logger.setLevel(logging.DEBUG)


class PeanutError(Exception):
    pass


class Time:
    def __init__(self, hours=None, minutes=None, seconds=None):
        assert hours or minutes or seconds
        self.hours = hours or 0
        self.minutes = minutes or 0
        self.seconds = seconds or 0

    def __repr__(self):
        return '%.2d:%.2d:%.2d' % (self.hours, self.minutes, self.seconds)

    @classmethod
    def parse(cls, val):
        regex = '(\d+):(\d\d):(\d\d)'
        match = re.fullmatch(regex, val)
        if match is None:
            raise ValueError('Wrong format for time %s' % val)
        h, m, s = match.groups()
        h, m, s = int(h), int(m), int(s)
        if m >= 60 or s >= 60:
            raise ValueError('Wrong format for time %s' % val)
        return cls(hours=h, minutes=m, seconds=s)

    @classmethod
    def from_seconds(cls, seconds):
        minutes = seconds // 60
        seconds %= 60
        hours = minutes // 60
        minutes %= 60
        return cls(hours=hours, minutes=minutes, seconds=seconds)


class RunError(Exception):
    pass


class Nodes:

    def __init__(self, nodes, name, working_dir, parent_nodes=None):
        self.nodes = fabric.ThreadingGroup.from_connections(nodes)
        self.name = name
        self.working_dir = working_dir
        if parent_nodes is None:
            self.__history = []
        else:
            self.__history = parent_nodes.__history

    def __iter__(self):
        yield from self.nodes

    @property
    def history(self):
        return copy.deepcopy(self.__history)

    def run(self, command, **kwargs):
        def clean_dict(d, k):
            values = set(d[k].values())
            if len(values) == 0:
                del d[k]
            elif len(values) == 1:
                val = values.pop()
                if val != '':
                    d[k] = val
                else:
                    del d[k]
        error = False
        if 'directory' in kwargs:
            directory = os.path.join(self.working_dir, kwargs['directory'])
            del kwargs['directory']
        else:
            directory = self.working_dir
        logger.debug('[%s | %s] %s' % (self.name, directory, command))
        if 'hide' not in kwargs:
            kwargs['hide'] = True
        real_command = 'cd %s && %s' % (directory, command)
        start = datetime.datetime.now()
        try:
            output = self.nodes.run(real_command, **kwargs)
        except fabric.exceptions.GroupException as e:
            output = {}
            for node, res in e.result.items():
                try:
                    output[node] = res.result
                except AttributeError:
                    output[node] = res
            error = True
            error_msg = ''
        stop = datetime.datetime.now()
        hist_entry = {
                'directory': directory,
                'nodes_type': self.name,
                'hostnames': self.hostnames,
                'command': command,
                'stdout': {},
                'stderr': {},
                'return_code': {},
                'date': {
                    'start': str(start),
                    'stop': str(stop),
                    'duration': (stop-start).total_seconds()
                }
            }
        error_msg_stdout = ''
        for node, node_output in output.items():
            hist_entry['stdout'][node.host] = node_output.stdout.strip()
            hist_entry['stderr'][node.host] = node_output.stderr.strip()
            hist_entry['return_code'][node.host] = node_output.return_code
            if node_output.return_code != 0:
                if node_output.stderr.strip():
                    error_msg = node_output.stderr
                if node_output.stdout.strip():
                    error_msg_stdout = node_output.stdout
        clean_dict(hist_entry, 'stdout')
        clean_dict(hist_entry, 'stderr')
        clean_dict(hist_entry, 'return_code')
        self.__history.append(hist_entry)
        if error:
            error_msg = error_msg or error_msg_stdout
            raise RunError(error_msg)
        return output

    def run_unique(self, *args, **kwargs):
        result = list(self.run(*args, **kwargs).values())
        for res in result[1:]:
            assert res.stdout == result[0].stdout
            assert res.stderr == result[1].stderr
        return result[0]

    def put(self, origin_file, target_file):
        target_file = os.path.join(self.working_dir, target_file)
        logger.debug('[%s] put: %s → %s' % (self.name, origin_file, target_file))
        for node in self.nodes:
            node.put(origin_file, target_file)

    def get(self, origin_file, target_file):
        assert len(self.nodes) == 1
        origin_file = os.path.join(self.working_dir, origin_file)
        logger.debug('[%s] get: %s → %s' % (self.name, origin_file, target_file))
        for node in self.nodes:
            node.get(origin_file, target_file)

    @property
    def hostnames(self):
        return [node.host for node in self.nodes]

    def __write_large_file(self, content, target_file):
        tmp_file = tempfile.NamedTemporaryFile(dir='.')
        with open(tmp_file.name, 'w') as f:
            f.write(content)
        self.put(tmp_file.name, target_file)
        tmp_file.close()

    def write_files(self, content, *target_files, avoid_copy=False):
        if avoid_copy:
            hash_origin = hashlib.sha512(content.encode()).hexdigest()
            try:
                hash_target = self.run_unique('sha512sum %s' % ' '.join(target_files)).stdout.strip()
                all_hashes = set()
                for target in hash_target.split('\n'):
                    h, f = target.split()
                    all_hashes.add(h.strip())
                if len(all_hashes) != 1:
                    hash_target = ''
                else:
                    hash_target = all_hashes.pop()
            except RunError:
                hash_target = ''
            if hash_target == hash_origin:
                logger.info('File(s) already available on node: %s' % ' '.join(target_files))
                return
            else:
                logger.info('File(s) not available on node, need to copy: %s' % ' '.join(target_files))
        target_files = [os.path.join(self.working_dir, target) for target in target_files]
        if len(content) < 80:  # arbitrary threshold...
            cmd = "echo -n '%s' | tee %s" % (content, ' '.join(target_files))
            self.run(cmd)
        else:
            self.__write_large_file(content, target_files[0])
            if len(target_files) > 1:
                remaining_files = ' '.join(target_files[1:])
                cmd = 'cat %s | tee %s' % (target_files[0], remaining_files)
                self.run(cmd)

    @property
    def cores(self):
        cores = {}
        keys = list(self.cores_full[0].keys())
        for PU, info in self.cores_full.items():
            cores[tuple(info[k] for k in keys if k != 'PU')] = []
        for PU, info in self.cores_full.items():
            cores[tuple(info[k] for k in keys if k != 'PU')].append(PU)
        return list(cores.values())

    @property
    def cores_full(self):
        try:
            return self.__cores
        except AttributeError:
            self.__cores = self._get_all_cores()
            return self.__cores

    @property
    def hyperthreads(self):
        try:
            return self.__hyperthreads
        except AttributeError:
            self.__hyperthreads = [group[1:] for group in self.cores]
            self.__hyperthreads = sum(self.__hyperthreads, [])
            return self.__hyperthreads

    def enable_hyperthreading(self):
        self.__set_hyperthreads(1)

    def disable_hyperthreading(self):
        self.__set_hyperthreads(0)

    def __set_hyperthreads(self, value):
        assert value in (0, 1)
        filenames = ['/sys/devices/system/cpu/cpu%d/online' % core_id for core_id in self.hyperthreads]
        self.write_files(str(value), *filenames)

    def _get_all_cores(self):
        ref_cores = None
        all_xml = self.__get_platform_xml()
        for node, xml in all_xml.items():
            cores = self.get_all_cores(xml)
            if ref_cores is None:
                ref_cores = cores
                ref_node = node
            elif cores != ref_cores:
                raise ValueError('Got different topologies for nodes %s and %s' % (ref_node.host, node.host))
        return ref_cores

    @classmethod
    def get_all_cores(cls, xml):
        xml = xml.findall('object')[0]
        result = cls.__process_cache(xml, {})
        return {info['PU']: info for info in result}

    def __get_platform_xml(self):
        result = self.run('lstopo -f topology.xml && cat topology.xml')
        xml = {}
        for node, output in result.items():
            xml[node] = lxml.etree.fromstring(output.stdout.encode('utf8'))
        return xml

    @classmethod
    def __process_cache(cls, xml, info):
        if 'cache' not in xml.get('type').lower():
            info[xml.get('type')] = int(xml.get('os_index'))
        cache = xml.findall('object')
        result = []
        for obj in cache:
            if obj.get('type') == 'Core':
                result.extend(cls.__process_core(obj, dict(info)))
            elif obj.get('type') in ('Machine', 'NUMANode', 'Package', 'Cache', 'L3Cache',
                                     'L2Cache', 'L1Cache', 'L1iCache'):
                result.extend(cls.__process_cache(obj, dict(info)))
        return result

    @classmethod
    def __process_core(cls, xml, info):
        info[xml.get('type')] = int(xml.get('os_index'))
        result = []
        for pu in xml.findall('object'):
            assert pu.get('type') == 'PU'
            tmp = dict(info)
            tmp[pu.get('type')] = int(pu.get('os_index'))
            result.append(tmp)
        return result

    @property
    def frequency_information(self):
        try:
            return self.__frequency_information
        except AttributeError:
            freq = self.run_unique('cpufreq-info -l').stdout
            min_f, max_f = [int(f) for f in freq.split()]
            governors = self.run_unique('cpufreq-info -g').stdout.split()
            active_driver = self.run_unique('cpufreq-info -d').stdout.strip()
            idle_driver = self.run_unique('cat /sys/devices/system/cpu/cpuidle/current_driver').stdout.strip()
            idle_files = self.run_unique('ls /sys/devices/system/cpu/cpu0/cpuidle').stdout.split()
            nb_states = len(idle_files)
            assert idle_files == ['state%d' % i for i in range(nb_states)]
            tuple_cls = collections.namedtuple('frequency_information', ['active_driver', 'idle_driver', 'governor',
                                                                         'idle_states', 'min_freq', 'max_freq'])
            self.__frequency_information = tuple_cls(active_driver, idle_driver, tuple(governors), range(nb_states)[1:],
                                                     min_f, max_f)
            return self.__frequency_information

    @property
    def current_frequency_information(self):
        info = self.run_unique('cpufreq-info -p').stdout.split()
        tuple_cls = collections.namedtuple('frequency_information', ['governor', 'min_freq', 'max_freq'])
        return tuple_cls(info[2], int(info[0]), int(info[1]))

    def set_frequency_information(self, governor=None, min_freq=None, max_freq=None):
        if not governor and not min_freq and not max_freq:
            raise ValueError('At least one of governor, min_freq and max_freq must be given.')
        cmd = 'cpupower -c all frequency-set'
        if governor:
            all_gov = self.frequency_information.governor
            if governor not in all_gov:
                raise ValueError('Governor %s not in the allowed governors %s' % (governor, all_gov))
            cmd += ' -g %s' % governor
        hw_min = self.frequency_information.min_freq
        hw_max = self.frequency_information.max_freq
        hw_range = range(hw_min, hw_max+1)
        if min_freq:
            if min_freq not in hw_range:
                raise ValueError('Minimum frequency %d not in the allowed frequency range [%d, %d]' % (min_freq,
                                                                                                       hw_min, hw_max))
            cmd += ' -d %d' % min_freq
        if max_freq:
            if max_freq not in hw_range:
                raise ValueError('Maximum frequency %d not in the allowed frequency range [%d, %d]' % (max_freq,
                                                                                                       hw_min, hw_max))
            if max_freq < min_freq:
                raise ValueError('Minimum frequency %d should be lower or equal to maximum frequency %d' % (min_freq,
                                                                                                            max_freq))
            cmd += ' -u %d' % max_freq
        self.run(cmd)

    def set_frequency_information_pstate(self, min_perf_pct=None, max_perf_pct=None):
        assert self.frequency_information.active_driver == 'intel_pstate'
        if min_perf_pct is None and max_perf_pct is None:
            raise ValueError('At least one of min_perf_pct and max_perf_pct must be given.')
        for val in [min_perf_pct, max_perf_pct]:
            if val is not None and val not in range(0, 101):
                raise ValueError('Percentage value %s must be an integer between 0 and 100' % val)
        if min_perf_pct is not None and max_perf_pct is not None and min_perf_pct > max_perf_pct:
            raise ValueError('The min_perf_pct(=%d) must be higher than the max_perf_pct(=%d)' % (min_perf_pct,
                                                                                                  max_perf_pct))
        if min_perf_pct is not None:
            self.write_files(str(min_perf_pct), '/sys/devices/system/cpu/intel_pstate/min_perf_pct')
        if max_perf_pct is not None:
            self.write_files(str(max_perf_pct), '/sys/devices/system/cpu/intel_pstate/max_perf_pct')

    def reset_frequency_information(self):
        info = self.frequency_information
        self.set_frequency_information('powersave', info.min_freq, info.max_freq)

    def set_frequency_performance(self):
        max_f = self.frequency_information.max_freq
        self.set_frequency_information('performance', max_f, max_f)

    def get_temperature(self):
        '''
        By default, lm-sensors is not installed, and there would be some configuration to do...
        '''
        result = self.run('cat /sys/class/thermal/thermal_zone*/temp')
        result = {k.host: [float(x)/1000 for x in v.stdout.split()] for (k, v) in result.items()}
        return result

    def get_frequency(self):
        cores = [c[0] for c in self.cores]
        files = ['/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq' % c for c in cores]
        result = self.run('cat %s' % ' '.join(files))
        result = {k.host: [int(x)*1000 for x in v.stdout.split()] for (k, v) in result.items()}
        return result

    def pretty_frequency(self):
        def mean(l): return sum(l)/len(l)
        frequencies = self.get_frequency()
        freq = sum(list(frequencies.values()), [])
        return 'min=%.2fGHz | max=%.2fGHz | mean=%.2fGHz' % (min(freq)*1e-9, max(freq)*1e-9, mean(freq)*1e-9)

    def __set_turboboost(self, value):
        assert value in (0, 1)
        if self.frequency_information.active_driver == 'intel_pstate':
            self.write_files(str(1-value), '/sys/devices/system/cpu/intel_pstate/no_turbo')
        else:
            self.write_files(str(value), '/sys/devices/system/cpu/cpufreq/boost')

    def enable_turboboost(self):
        self.__set_turboboost(1)

    def disable_turboboost(self):
        self.__set_turboboost(0)

    def __set_idle_state(self, value):
        cores = [c[0] for c in self.cores]
        assert value in (0, 1)
        files = ['/sys/devices/system/cpu/cpu%d/cpuidle/state%d/disable' % (core, state)
                 for core in cores for state in self.frequency_information.idle_states]
        self.write_files(str(1-value), *files)

    def enable_idle_state(self):
        self.__set_idle_state(1)

    def disable_idle_state(self):
        self.__set_idle_state(0)


class GitError(Exception):
    pass


class Job:
    install_path = '~/.local/bin/peanut'
    expfile_types = {}
    expfile_header = None
    expfile_header_in_file = True
    installfile_types = {}
    auto_oardel = False
    deployment_images = ['debian9-x64-%s' % mode for mode in ['min', 'base', 'nfs', 'big']]
    clusters = {
        'grenoble': ['dahu', 'yeti'],
        'lyon': ['sagittaire', 'hercule', 'orion', 'taurus', 'nova'],
        'nancy': ['griffon', 'graphene', 'graphite', 'grimoire', 'grisou', 'graphique', 'graoully', 'grimani', 'grele',
                  'grvingt', 'gros'],
        'rennes': ['parapide', 'parapluie', 'paranoia', 'parasilo', 'paravance'],
        'sophia': ['suno', 'uvb'],
        'lille': ['chetemi', 'chifflet', 'chiclet', 'chifflot'],
        'luxembourg': ['granduc', 'petitprince'],
        'nantes': ['econome', 'ecotype']
    }
    sites = {cluster: site for site, cluster_list in clusters.items() for cluster in cluster_list}
    special_clusters = {
        'testing': [],
        'production': ['graphique', 'graoully', 'grimani', 'grele', 'grvingt'],
    }
    queues = {cluster: queue for queue, cluster_list in special_clusters.items() for cluster in cluster_list}
    queues = collections.defaultdict(lambda: 'default', queues)

    def __init__(self, jobid, frontend, deploy=False):
        self.start_time = time.time()
        self.jobid = jobid
        self.frontend = frontend
        if deploy and deploy not in self.deployment_images:
            choices = ', '.join(self.deployment_images)
            raise ValueError('Unknown deployment image %s, possible choices are: %s' % (deploy, choices))
        self.deploy = deploy
        self.user = frontend.nodes[0].user
        self.site = frontend.nodes[0].host
        self.archive_name = '%s_%s_%d.zip' % (self.site,
                                              datetime.date.today(),
                                              self.jobid)
        self.information = {
                    'peanut_version': __version__,
                    'peanut_git_version': __git_version__,
                }

    def __del__(self):
        if self.auto_oardel:
            try:
                self.oardel()
            except Exception:
                pass

    @staticmethod
    def expandg5k(host, site):
        if 'grid5000' not in host:
            host = '%s.%s.grid5000.fr' % (host, site)
        return host

    @staticmethod
    def split_hostname(hostname):
        regex = '(?P<cluster>[a-z]+)-(?P<host>\d+)(.(?P<site>[a-z]+)(\.grid5000\.fr)?)?'
        match = re.fullmatch(regex, hostname)
        if match is None:
            raise ValueError('Wrong hostname format %s' % hostname)
        match = match.groupdict()
        match['host'] = int(match['host'])
        return collections.namedtuple('hostname', sorted(match),)(**match)

    def oardel(self):
        self.frontend.run('oardel %d' % self.jobid)

    @property
    def oar_node_file(self):
        return '/var/lib/oar/%d' % self.jobid

    def oarstat(self):
        result = self.frontend.run_unique('oarstat -fJ -j %d' % self.jobid)
        return json.loads(result.stdout)[str(self.jobid)]

    @classmethod
    def _oarstat_user(cls, frontend):
        try:
            result = frontend.run_unique('oarstat -J -u')
        except RunError as e:  # no job
            return {}
        return json.loads(result.stdout)

    @classmethod
    def get_jobs(cls, site, username):
        frontend = cls.g5k_frontend(site, username)
        stat = cls._oarstat_user(frontend)
        jobs = []
        for jobid, job_stat in stat.items():
            if job_stat['state'] in ('Running', 'Waiting'):
                job = int(jobid)
                deploy = None
                if 'deploy' in job_stat['types']:
                    deploy = 'debian9-x64-base'
                jobs.append(cls(job, frontend, deploy=deploy))
        if len(jobs) == 0:
            raise ValueError('No jobs were found for user %s on site %s' % (username, site))
        return jobs

    def set_expfile(self, filename):
        self.expfile = [ExpFile(filename=filename, types=self.expfile_types, header=self.expfile_header,
                                header_in_file=self.expfile_header_in_file)]

    def set_installfile(self, filename):
        self.installfile = InstallFile(filename=filename, types=self.installfile_types)

    def __find_hostnames(self):
        sleep_time = 5
        while True:
            stat = self.oarstat()
            hostnames = stat['assigned_network_address']
            if not hostnames:
                time.sleep(sleep_time + random.uniform(0, sleep_time/5))
                sleep_time = min(sleep_time*2, 60)
            else:
                break
        hostnames.sort(key=lambda host: self.split_hostname(host).host)
        self.__hostnames = hostnames
        self.cluster = self._check_hostnames(hostnames)

    @property
    def hostnames(self):
        try:
            return list(self.__hostnames)
        except AttributeError:
            self.__find_hostnames()
            return list(self.__hostnames)

    def kadeploy(self, env=None):
        sleep_time = 15
        assert self.deploy
        env = env or self.deploy
        self.hostnames
        while True:
            try:
                self.frontend.run('kadeploy3 -k -f %s -e %s' % (self.oar_node_file, env))
            except RunError as e:
                t = sleep_time + random.uniform(0, sleep_time/2)
                logger.warning('Kadeploy error, sleeping for %2.f seconds:\n%s' % (t, e))
                time.sleep(t)
                sleep_time = min(sleep_time*2, 120)
            else:
                break
        return self

    def __repr__(self):
        return '%s(%d)' % (self.__class__.__name__, self.jobid)

    @classmethod
    def _check_install(cls, frontend):
        name = 'frontend %s' % frontend.hostnames[0]
        try:
            version = frontend.run_unique('%s --git-version' % cls.install_path).stdout.strip()
        except RunError:
            raise PeanutError('Peanut is not installed on the %s' % name)
        version = version.split()[1]
        if version != __git_version__:
            err = '%s != %s' % (version[:5], __git_version__[:5])
            raise PeanutError('Peanut version mismatch between the %s and the client: %s' % (name, err))

    @classmethod
    def oarsub(cls, frontend, constraint, walltime, nb_nodes, *,
               deploy=False, queue=None, script=None, container=None):
        name = random.choice('☕🥐')
        constraint = '%s/nodes=%d,walltime=%s' % (
            constraint, nb_nodes, walltime)
        deploy_str = '-t deploy ' if deploy else '-t allow_classic_ssh'
        queue_str = '-q %s ' % queue if queue else ''
        cmd = 'oarsub -n "%s" %s%s -l "%s"' % (name, queue_str, deploy_str, constraint)
        if script:
            cmd += " '%s'" % script
        else:
            date = frontend.run_unique('date "+%Y-%m-%d %H:%M:%S"').stdout.strip()
            cmd += ' -r "%s"' % date
        if container:
            cmd += ' -t inner=%d' % container
        result = frontend.run_unique(cmd)
        regex = re.compile('OAR_JOB_ID=(\d+)')
        jobid = int(regex.search(result.stdout).groups()[0])
        return cls(jobid, frontend=frontend, deploy=deploy)

    @classmethod
    def oarsub_cluster(cls, username, cluster, walltime, nb_nodes, *,
                       deploy=False, script=None, container=None):
        site = cls.sites[cluster]
        queue = cls.queues[cluster]
        frontend = cls.g5k_frontend(site, username)
        constraint = "{cluster in ('%s')}" % cluster
        return cls.oarsub(frontend, constraint, walltime, nb_nodes, deploy=deploy,
                          queue=queue, script=script, container=container)

    @classmethod
    def _check_hostnames(cls, hostnames):
        clusters = set([cls.split_hostname(host).cluster for host in hostnames])
        if len(clusters) != 1:
            clusters = ', '.join(clusters)
            raise ValueError('Can only use nodes from a single cluster. Found clusters %s.' % clusters)
        cluster = clusters.pop()
        return cluster

    @classmethod
    def oarsub_hostnames(cls, username, hostnames, walltime, nb_nodes=None, *,
                         deploy=False, script=None, container=None):
        cluster = cls._check_hostnames(hostnames)
        site = cls.sites[cluster]
        queue = cls.queues[cluster]
        frontend = cls.g5k_frontend(site, username)
        hostnames = ["'%s'" % cls.expandg5k(host, site) for host in hostnames]
        constraint = "{network_address in (%s)}" % ', '.join(hostnames)
        if nb_nodes is None:
            nb_nodes = len(hostnames)
        return cls.oarsub(frontend, constraint, walltime, nb_nodes, deploy=deploy,
                          queue=queue, script=script, container=container)

    @classmethod
    def g5k_connection(cls, site, username):
        # socket.getfqdn() does not work anymore since the Debian 10 update in G5K
        fqdn = socket.getaddrinfo(socket.gethostname(), 0, flags=socket.AI_CANONNAME)[0][3]
        if 'grid5000' in fqdn:  # already inside G5K, no need for a gateway
            connection = fabric.Connection(site, user=username)
        else:
            gateway = fabric.Connection('access.grid5000.fr', user=username)
            connection = fabric.Connection(site, user=username, gateway=gateway)
        return connection

    @classmethod
    def g5k_frontend(cls, site, username):
        connection = cls.g5k_connection(site, username)
        frontend = Nodes([connection], name='frontend', working_dir='/home/%s' % username)
        return frontend

    def __open_nodes_connection(self):
        sleep_time = 5
        while True:
            try:
                self.nodes.run('echo "hello world"')
            except RunError:
                time.sleep(sleep_time + random.uniform(0, sleep_time/5))
                sleep_time = min(sleep_time*2, 60)
            else:
                break

    @property
    def nodes(self):
        try:
            return self.__nodes
        except AttributeError:
            if self.deploy:
                user = 'root'
            else:
                user = self.user
            connections = [fabric.Connection(host, user=user, gateway=self.frontend.nodes[0])
                           for host in self.hostnames]
            self.__nodes = Nodes(connections, name='allnodes', working_dir='/tmp', parent_nodes=self.frontend)
            self.orchestra = Nodes(connections[1:], name='orchestra', working_dir='/tmp', parent_nodes=self.__nodes)
            self.director = Nodes([connections[0]], name='director', working_dir='/tmp', parent_nodes=self.__nodes)
            self.__open_nodes_connection()
            return self.__nodes

    @property
    def history(self):
        return self.frontend.history

    def apt_install(self, *packages):
        sudo = 'sudo-g5k ' if not self.deploy else ''
        cmd = '{0}apt update -qq && {0}DEBIAN_FRONTEND=noninteractive apt upgrade -qq -y'.format(sudo)
        self.nodes.run(cmd)
        cmd = sudo + 'DEBIAN_FRONTEND=noninteractive apt install -qq -y %s' % ' '.join(packages)
        self.nodes.run(cmd)
        return self

    def __add_raw_information_to_archive(self, filename, command):
        if not self.deploy:
            command = 'sudo-g5k %s' % command
        self.nodes.run(command)
        self.director.run('cp %s information/%s' % (filename, self.director.hostnames[0]))
        for host in self.hostnames:
            if host == self.director.hostnames[0]:
                continue
            self.director.run('scp %s:/tmp/%s information/%s' % (host, filename, host))

    def get_timestamp(self):
        return str(datetime.datetime.now())

    def register_temperature(self):
        new_temp = self.nodes.get_temperature()
        entry = {'timestamp': self.get_timestamp(), 'temperatures': new_temp}
        try:
            self._temperatures.append(entry)
        except AttributeError:
            self._temperatures = [entry]

    def dump_temperatures(self, filename):
        if not hasattr(self, '_temperatures'):
            logger.warning('No temperature information recorded.')
            return
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerow(('timestamp', 'hostname', 'temperature', 'sensor_id'))
            for entry in self._temperatures:
                timestamp = entry['timestamp']
                for host, temperatures in entry['temperatures'].items():
                    for i, temp in enumerate(temperatures):
                        writer.writerow((timestamp, host, temp, i))
        self.add_file_to_archive(filename, filename)

    def start_monitoring(self, period=1):
        self.git_clone('https://github.com/Ezibenroc/Stress-Test.git', 'monitoring')
        command = 'python3 monitoring/basic_monitoring.py --period %d ' % period
        command += '--temp_output monitoring_temp.csv --freq_output monitoring_freq.csv '
        command += '--pid_file monitoring_pid'
        command = 'tmux new-session -d -s tmux_monitoring "%s"' % command
        self.nodes.run(command)

    def stop_monitoring(self):
        self.nodes.run('kill -SIGINT $(cat monitoring_pid)')
        for filename in ['monitoring_temp.csv', 'monitoring_freq.csv']:
            if len(self.orchestra.hostnames) > 0:
                self.orchestra.run('sed -i "1d" %s' % filename)  # removing the header
                remote_file = os.path.join(self.orchestra.working_dir, filename)
                all_files = []
                for i, node in enumerate(self.orchestra.hostnames):
                    local_file = '%s%d' % (filename, i)
                    all_files.append(local_file)
                    self.director.run("rsync -a '%s:%s' %s" % (node, remote_file, local_file))
                    self.director.run('cat %s >> %s' % (' '.join(all_files), filename))
            self.add_local_to_archive(filename)
        self.nodes.run('rm -rf monitoring')

    def add_raw_information_to_archive(self):
        for host in self.hostnames:
            self.director.run('mkdir -p information/%s' % host)
        commands_with_files = {
                    'cpuinfo.txt': 'cp /proc/cpuinfo cpuinfo.txt',
                    'environment.txt': 'env > environment.txt',
                    'topology.xml': 'lstopo -f topology.xml',
                    'topology.pdf': 'lstopo -f topology.pdf',
                    'lspci.txt': 'lspci -v > lspci.txt',
                    'dmidecode.txt': 'dmidecode > dmidecode.txt',
                    'lsmod.txt': 'lsmod > lsmod.txt',
                    }
        for filename, command in commands_with_files.items():
            self.__add_raw_information_to_archive(filename, command)
        self.director.run('zip -ru %s information' % self.archive_name)
        self.director.run('rm -rf information')

    def _arp_information(self):
        result = {host: {} for host in self.hostnames}
        arp_cmd = 'arp -a' if self.deploy else 'sudo-g5k arp -a'
        arp_output = self.nodes.run(arp_cmd)
        for node, arp in arp_output.items():
            arp_dict = {}
            for line in arp.stdout.strip().split('\n'):
                hostname, *rest = line.split()
                try:
                    arp_dict[hostname].append(rest)
                except KeyError:
                    arp_dict[hostname] = [rest]
            origin = node.host
            res = result[origin]
#            res['ip_address'] = node.run('hostname -I').stdout.strip()
            res['arp'] = {}
            res = res['arp']
            for hostname, interfaces in arp_dict.items():
                res[hostname] = []
                for line in interfaces:
                    res[hostname].append(' '.join(line))
        return result

    def platform_information(self):
        commands = {'kernel': 'uname -r',
                    'version': 'cat /proc/version',
                    'gcc': 'gcc -dumpversion',
                    'mpi': 'mpirun --version | head -n 1',
                    'cpu': 'cat /proc/cpuinfo  | grep "name"| uniq | cut -d: -f2 ',
                    }
        result = {host: {} for host in self.hostnames}
        for cmd_name, cmd in commands.items():
            try:
                output = self.nodes.run(cmd)
            except RunError:
                logger.warning('Could not get information about %s, the command errored.' % cmd_name)
                continue
            for host, res in output.items():
                result[host.host][cmd_name] = res.stdout.strip()
            if len(set([result[h][cmd_name] for h in self.hostnames])) != 1:
                logger.warning('Different settings found for %s (command %s)' % (cmd_name, cmd))
        try:
            arp_result = self._arp_information()
        except RunError:
            logger.warning('Could not get information about arp, the command errored.')
        else:
            for key, value in arp_result.items():
                result[key].update(value)
        result['site'] = self.site
        result['cluster'] = self.cluster
        result['jobid'] = self.jobid
        result['deployment'] = self.deploy
        result['command'] = ' '.join(sys.argv)
        result['replay_command'] = self.replay_command
        try:
            result['expfile'] = [f.basename for f in self.expfile]
        except AttributeError:
            pass
        try:
            result['installfile'] = self.installfile.basename
        except AttributeError:
            pass
        result.update(self.information)
        return result

    def add_local_to_archive(self, target):
        target = os.path.normpath(target)
        target_name = os.path.basename(target)
        target_dir = os.path.dirname(target)
        if target_dir and target_dir != self.director.working_dir:
            self.director.run('cp -r %s %s' % (target, target_name))
        self.director.run('zip -ru %s %s' % (self.archive_name, target_name))

    def add_file_to_archive(self, origin, target):
        self.director.put(origin, target)
        self.add_local_to_archive(target)

    def add_content_to_archive(self, content, filename):
        self.director.write_files(content, filename)
        self.add_local_to_archive(filename)

    def add_information_to_archive(self):
        self.add_raw_information_to_archive()
        job_info = self.platform_information()
        self.add_content_to_archive(yaml.dump(job_info, default_flow_style=False), 'info.yaml')
        self.add_content_to_archive(yaml.dump(self.oarstat(), default_flow_style=False), 'oarstat.yaml')
        self.dump_temperatures('temperatures.csv')
        try:
            expfile = self.expfile
        except AttributeError:  # no expfile
            pass
        else:
            for f in expfile:
                self.add_content_to_archive(f.raw_content, f.basename)
        try:
            installfile = self.installfile
            self.add_content_to_archive(installfile.raw_content, installfile.basename)
        except AttributeError:  # no installfile
            pass
        log = log_stream.getvalue()
        log = log.encode('ascii', 'ignore').decode()  # removing any non-ascii character
        self.add_content_to_archive(log, 'commands.log')
        history = json.dumps(self.history, indent=2, sort_keys=True)
        self.add_content_to_archive(history, 'history.json')

    def __commit_message(self):
        msg = '[AUTOMATIC COMMIT] %s\n\n' % self.__class__.__name__
        msg += 'user: %s\n' % self.user
        msg += 'site: %s\n' % self.site
        msg += 'cluster: %s\n' % self.cluster
        msg += 'nodes: [%s]\n' % ','.join(self.hostnames)
        msg += 'jobid: %d\n' % self.jobid
        msg += 'deployment: %s\n' % self.deploy
        msg += 'duration: %.0f seconds' % (time.time() - self.start_time)
        return msg

    def _git_push_archive(self, remote_url, path_in_repo, branch_name, max_tentatives=5):
        repository_path = self.director.run_unique('mktemp -d').stdout.strip()
        # we purposely do not use the git_clone command, this repository should not be added in the info.yaml file
        nb_tentatives = 0
        sleep_time = 15
        while True:
            try:
                nb_tentatives += 1
                self.director.run('git clone --depth 1 %s %s' % (remote_url, repository_path))
            except RunError as e:
                if nb_tentatives == max_tentatives:
                    raise GitError('Git clone failed\n%s' % e)
                t = random.uniform(sleep_time*2**nb_tentatives, sleep_time*2**(nb_tentatives+1))
                logger.warning('Previous command failed, sleeping for %.2f seconds' % t)
                time.sleep(t)
            else:
                break
        self.director.run('git checkout -b %s' % branch_name, directory=repository_path)
        self.director.run('mkdir -p %s' % path_in_repo, directory=repository_path)
        self.director.run('cp %s %s' % (self.archive_name, os.path.join(repository_path, path_in_repo)))
        self.director.run('git add .', directory=repository_path)
        author = 'peanut <%s>' % self.director.hostnames[0]
        self.director.run('git commit --author "%s" -m"$(echo -e "%s")"' % (author, self.__commit_message()),
                          directory=repository_path)
        while True:
            try:
                nb_tentatives += 1
                self.director.run('git push --set-upstream origin %s' % branch_name, directory=repository_path)
            except RunError as e:
                if nb_tentatives == max_tentatives:
                    raise GitError('Git push failed\n%s' % e)
                t = random.uniform(sleep_time*2**nb_tentatives, sleep_time*2**(nb_tentatives+1))
                logger.warning('Previous command failed, sleeping for %.2e seconds' % t)
                time.sleep(t)
            else:
                break

    def git_push_archive(self):
        assert self.installfile is not None
        install_options = self.installfile.content
        remote_url = install_options['remote_url']
        path_in_repo = install_options['path_in_repo']
        token_path = install_options['token_path']
        if not remote_url.startswith('https://'):
            raise GitError('Invalid remote URL, can only clone with HTTPS (got %s)' % remote_url)
        try:
            with open(token_path) as f:
                token = f.read().strip()
        except FileNotFoundError:
            raise GitError('Could not find token file %s' % token_path)
        # https://stackoverflow.com/a/29570677/4110059
        remote_url = 'https://oauth2:%s@%s' % (token, remote_url[8:])
        branch_name = 'exp_%d' % self.jobid
        self._git_push_archive(remote_url, path_in_repo, branch_name)

    def get_archive(self):
        #  If an installfile with a remote_url is specified, try to push the archive.
        #  Otherwise (or if the push fails), copy the archive locally.
        try:
            self.installfile.content['remote_url']
        except (AttributeError, KeyError):
            self.director.get(self.archive_name, self.archive_name)
        else:
            try:
                self.git_push_archive()
            except GitError as e:
                logger.warning(e)
                self.director.get(self.archive_name, self.archive_name)

    def send_key(self):
        if not self.deploy:  # no need for that if this is not a fresh deploy
            return
        self.director.run('ssh-keygen -b 2048 -t rsa -f .ssh/id_rsa -q -N ""', directory='/root')
        tmp_file = tempfile.NamedTemporaryFile(dir='.')
        self.director.get('/root/.ssh/id_rsa.pub', tmp_file.name)
        self.orchestra.put(tmp_file.name, '/tmp/id_rsa.pub')
        tmp_file.close()
        self.orchestra.run('cat /tmp/id_rsa.pub >> .ssh/authorized_keys', directory='/root')
        for host in self.orchestra.hostnames:
            self.director.run('ssh -o "StrictHostKeyChecking no" %s hostname' % host, directory='/root')
            short_target = host[:host.find('.')]
            self.director.run('ssh -o "StrictHostKeyChecking no" %s hostname' % short_target, directory='/root')

    def git_clone(self, url, repository_path, checkout=None, recursive=False, patch=None):
        cmd = 'git clone'
        if recursive:
            cmd += ' --recursive'
        self.nodes.run('%s %s %s' % (cmd, url, repository_path))
        if checkout:
            self.nodes.run('git checkout %s' % checkout, directory=repository_path)
        git_hash = self.nodes.run_unique('git rev-parse HEAD', directory=repository_path)
        git_hash = git_hash.stdout.strip()
        git_info = {'path': repository_path, 'url': url, 'hash': git_hash}
        if patch:
            self.nodes.write_files(patch, '/tmp/patch.diff')
            self.nodes.run('git apply --whitespace=fix /tmp/patch.diff', directory=repository_path)
            git_info['patch'] = patch
        key_name = 'git_repositories'
        if key_name not in self.information:
            self.information[key_name] = []
        self.information[key_name].append(git_info)

    @classmethod
    def parse_jobid(cls, jobid):
        regex = 'f?([a-z]+):(\d+)'
        match = re.fullmatch(regex, jobid)
        if match is None:
            raise ValueError('Wrong format for jobid %s' % jobid)
        site, jobid = match.groups()
        if site not in cls.clusters.keys():
            raise ValueError('Unknown site %s' % site)
        return site, int(jobid)

    @classmethod
    def _get_parser(cls):
        def positive_int(n):
            n = int(n)
            if n <= 0:
                raise ValueError('Not a positive integer.')
            return n
        parser = argparse.ArgumentParser(description='Peanut, the tiny job runner')
        parser.add_argument('--version', action='version',
                            version='%(prog)s {version}'.format(version=__version__))
        parser.add_argument('--git-version', action='version',
                            version='%(prog)s {version}'.format(version=__git_version__))
        sp = parser.add_subparsers(dest='command')
        sp.required = True
        sp_run = sp.add_parser('run', help='Run an experiment.')
        sp_run.add_argument('username', type=str, help='username to use for the experiment.')
        sp_run.add_argument('--deploy', choices=cls.deployment_images, default=False, help='Do a full node deployment.')
        group = sp_run.add_mutually_exclusive_group(required=True)
        group.add_argument('--cluster', help='Cluster for the experiment.', choices=cls.sites.keys())
        group.add_argument('--nodes', help='Nodes for the experiment.', type=str, nargs='+')
        group.add_argument('--jobid', help='Job ID for the experiment, of the form site:ID.', type=cls.parse_jobid)
        sp_run.add_argument('--walltime', help='Duration of the experiment.', type=Time.parse, default=Time(hours=1))
        sp_run.add_argument('--nbnodes', help='Number of nodes for the experiment.', type=positive_int, default=1)
        sp_run.add_argument('--container', help='Container job for this sub-job.', type=positive_int, default=None)
        sp_run.add_argument('--expfile', help='File which describes the experiment.',
                            nargs='+', type=lambda f: ExpFile(filename=f, types=cls.expfile_types,
                                                              header=cls.expfile_header,
                                                              header_in_file=cls.expfile_header_in_file))
        sp_run.add_argument('--installfile', help='File whith options regarding the installation.',
                            type=lambda f: InstallFile(filename=f, types=cls.installfile_types))
        sp_run.add_argument('--batch', help='Whether to run this as a batch job or not.',
                            action='store_true', default=False)
        sp_gen = sp.add_parser('generate', help='Generate an experiment file.')
        sp_gen.add_argument('filename', type=str, help='File to write the experiment.')
        return parser

    @classmethod
    def parse_args(cls, args):
        parser = cls._get_parser()
        args = vars(parser.parse_args(args))
        if args['command'] == 'generate':
            return args
        if args['nodes'] is not None:
            try:
                cls._check_hostnames(args['nodes'])
            except ValueError as e:
                parser.error(e)
        if args['jobid'] is not None:  # walltime and nb_nodes do not make sense
            del args['walltime']
            del args['nbnodes']
        for key, val in list(args.items()):
            if val is None:
                del args[key]
        return args

    @property
    def replay_command(self):
        stat = self.oarstat()
        walltime = Time.from_seconds(stat['walltime'])
        cmd = 'peanut %s run %s --batch ' % (self.__class__.__name__, self.user)
        if self.deploy:
            cmd += '--deploy %s ' % self.deploy
        cmd += '--nodes %s ' % ' '.join(self.hostnames)
        cmd += '--nbnodes %d ' % len(self.hostnames)
        cmd += '--walltime %s ' % walltime
        try:
            cmd += '--expfile %s' % ' '.join([f.basename for f in self.expfile])
        except AttributeError:
            pass
        try:
            cmd += '--installfile %s' % self.installfile.basename
        except AttributeError:
            pass
        return cmd.strip()

    @classmethod
    def job_from_args(cls, args):
        user = args['username']
        deploy = args['deploy']
        try:
            expfile = args['expfile']
        except KeyError:
            expfile = []
        try:
            cls.check_expfile(expfile)
        except ValueError as e:
            sys.exit(e)
        try:
            installfile = args['installfile']
        except KeyError:
            installfile = None
        if 'cluster' in args:
            cluster = args['cluster']
            site = cls.sites[cluster]
        elif 'nodes' in args:
            hostnames = args['nodes']
            cluster = cls._check_hostnames(hostnames)
            site = cls.sites[cluster]
        else:
            site, jobid = args['jobid']
        frontend = cls.g5k_frontend(site, user)
        if args['batch']:
            script = cls.generate_batch_command(args, site)
            cls._check_install(frontend)
            for f in expfile:
                frontend.write_files(f.raw_content, f.basename, avoid_copy=True)
            if installfile:
                frontend.write_files(installfile.raw_content, installfile.basename, avoid_copy=True)
        else:
            script = None
        try:
            container = args['container']
        except KeyError:
            container = None
        if 'cluster' in args:
            job = cls.oarsub_cluster(user, cluster=cluster, walltime=args['walltime'], nb_nodes=args['nbnodes'],
                                     deploy=deploy, script=script, container=container)
        elif 'nodes' in args:
            job = cls.oarsub_hostnames(user, hostnames=hostnames, walltime=args['walltime'],
                                       nb_nodes=args['nbnodes'], deploy=deploy, script=script, container=container)
        else:
            job = cls(jobid, frontend, deploy=deploy)
        job.expfile = expfile
        job.installfile = installfile
        return job

    def setup(self):
        if self.deploy:
            self.kadeploy()
        try:
            self.register_temperature()
        except RunError:
            logger.warning('No temperature information available.')
        self.nodes.run('rm -rf /tmp/*')
        # Creating an empty zip archive on the director node
        # See https://stackoverflow.com/a/50091682/4110059
        self.director.run('echo UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA== | base64 -d > %s' % self.archive_name)
        if self.deploy:
            self.send_key()

    def teardown(self):
        try:
            self.register_temperature()
        except RunError:
            logger.warning('No temperature information available.')
        self.add_information_to_archive()
        self.get_archive()
        self.oardel()

    def run_exp(self):
        raise NotImplementedError()

    @classmethod
    def gen_exp(cls):
        raise NotImplementedError()

    @classmethod
    def main(cls, args):
        try:
            args = cls.parse_args(args)
            if args['command'] == 'generate':
                cls._main_generate(args)
            else:
                assert args['command'] == 'run'
                cls._main_run(args)
        except PeanutError as e:
            logger.critical(e)
            sys.exit(-1)

    @classmethod
    def _main_generate(cls, args):
        exp = cls.gen_exp()
        ExpFile(filename=args['filename'], content=exp, types=cls.expfile_types,
                header_in_file=cls.expfile_header_in_file)

    @classmethod
    def _main_run(cls, args):
        logger.info('Starting a new job, args = %s' % args)
        job = cls.job_from_args(args)
        if args['batch']:
            logger.info('%s submitted' % job)
        else:
            logger.info('%s with %d nodes' % (job, len(job.hostnames)))
            logger.info('Setting up')
            job.setup()
            logger.info('Running the experiment')
            job.run_exp()
            logger.info('Tearing down')
            job.teardown()
        logger.info('Total time: %.0f seconds' % (time.time() - job.start_time))

    @classmethod
    def generate_batch_command(cls, args, site):
        cmd = '%s %s run %s ' % (cls.install_path, cls.__name__, args['username'])
        cmd += '--jobid %s:$OAR_JOB_ID ' % site
        if args['deploy']:
            cmd += '--deploy %s ' % args['deploy']
        try:
            cmd += '--expfile %s ' % ' '.join([f.basename for f in args['expfile']])
        except KeyError:
            pass
        try:
            cmd += '--installfile %s' % args['installfile'].basename
        except KeyError:
            pass
        return cmd

    @classmethod
    def check_exp(cls, exp):
        pass

    @classmethod
    def check_expfile(cls, expfile):
        for f in expfile:
            if not f.content:
                continue
            for exp in f:
                cls.check_exp(exp)


class AbstractFile:
    def __init__(self, filename, types):
        self.filename = filename
        self.extension = os.path.splitext(filename)[1]
        if not self.extension:
            raise ValueError('File %s has no extension' % filename)
        self.extension = self.extension[1:]
        self.types = types
        self.basename = os.path.basename(filename)

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.filename)

    def read_content(self):
        try:
            f = open(self.filename, 'r')
        except FileNotFoundError:
            raise ValueError('File %s does not exist.' % self.filename)
        self.raw_content = f.read()
        f.close()
        self.parse_content()


class InstallFile(AbstractFile):
    def __init__(self, *, filename, types):
        super().__init__(filename, types)
        self.read_content()

    def parse_content(self):
        assert self.extension == 'yaml'
        self.content = yaml.load(self.raw_content)
        for h in self.types:
            if h not in self.content:
                raise ValueError('Key "%s" not found in file %s' % (h, self.filename))
        for h, val in self.content.items():
            if h not in self.types:
                raise ValueError('Unknown key "%s" in file %s' % (h, self.filename))
            if not isinstance(val, self.types[h]):
                raise ValueError('Wrong type for key "%s" in file %s' % (h, self.filename))


class ExpFile(AbstractFile):
    def __init__(self, *, filename, content=None, types=None, header=None, header_in_file=True):
        super().__init__(filename, types)
        self.header = header
        self.header_in_file = header_in_file
        self.content = content
        self.raw_content = None
        if not content:
            self.read_content()
        else:
            if not self.types:
                self.types = {key: type(val) for key, val in self.content[0].items()}
            self.write_content()

    def write_content(self):
        self.check_types()
        raw = io.StringIO()
        writer = csv.writer(raw, lineterminator='\n')  # default seems to be '\r\n'
        if self.header is None:
            self.header = list(self.content[0].keys())
        if self.header_in_file:
            writer.writerow(self.header)
        for row in self.content:
            writer.writerow([row[key] for key in self.header])
        self.raw_content = raw.getvalue()
        with open(self.filename, 'w') as f:
            f.write(self.raw_content)

    def parse_content(self):
        if self.extension != 'csv':
            return
        assert self.header or self.header_in_file
        reader = csv.reader(io.StringIO(self.raw_content))
        if self.header_in_file:
            self.header = [h.strip() for h in next(reader)]
        if self.types:
            expected_header = set(self.types)
            real_header = set(self.header)
            if expected_header != real_header:
                raise ValueError('Wrong format with file %s, expected header %s, got %s.' % (self.filename,
                                                                                             expected_header,
                                                                                             real_header))
        self.content = []
        for i, row in enumerate(reader):
            row = [val.strip() for val in row]
            if len(row) != len(self.header):
                raise ValueError('Wrong format with file %s, row %d.' % (self.filename, i+2))
            new_row = {key: val for key, val in zip(self.header, row)}
            if self.types:
                for key, val in new_row.items():
                    cls = self.types[key]
                    try:
                        new_row[key] = cls(new_row[key])
                    except ValueError:
                        raise ValueError('Wrong format with file %s for key %s, expected a %s value, got "%s"' %
                                         (self.filename, key, cls.__name__, val))
            self.content.append(new_row)

    def check_types(self):
        for row in self.content:
            row_keys = set(row.keys())
            types_keys = set(self.types.keys())
            if row_keys != types_keys:
                raise ValueError('Mismatching keys in content, got %s and %s' % (row_keys, types_keys))
            types = {key: type(val) for key, val in row.items()}
            for key in types:
                real = types[key]
                expected = self.types[key]
                if real != expected:
                    raise ValueError('Wrong type with file %s for key %s, expected %s, got "%s"' %
                                     (self.filename, key, expected.__name__, real.__name__))

    def __iter__(self):
        yield from self.content

    def __len__(self):
        return len(self.content)
