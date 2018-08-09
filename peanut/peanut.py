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
        regex = '(\d\d):(\d\d):(\d\d)'
        match = re.fullmatch(regex, val)
        if match is None:
            raise ValueError('Wrong format for time %s' % val)
        h, m, s = match.groups()
        return cls(hours=int(h), minutes=int(m), seconds=int(s))

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

    def write_files(self, content, *target_files):
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
            cores = self.__get_all_cores(xml)
            if ref_cores is None:
                ref_cores = cores
                ref_node = node
            elif cores != ref_cores:
                raise ValueError('Got different topologies for nodes %s and %s' % (ref_node.host, node.host))
        return ref_cores

    def __get_all_cores(self, xml):
        xml = xml.findall('object')[0]
        return self.__process_cache(xml)

    def __get_platform_xml(self):
        result = self.run('lstopo -f topology.xml && cat topology.xml')
        xml = {}
        for node, output in result.items():
            xml[node] = lxml.etree.fromstring(output.stdout.encode('utf8'))
        return xml

    def __process_cache(self, xml):
        cache = xml.findall('object')
        result = []
        for obj in cache:
            if obj.get('type') == 'Core':
                result.append(self.__process_core(obj))
            elif obj.get('type') in ('Machine', 'NUMANode', 'Package', 'Cache', 'L3Cache',
                                     'L2Cache', 'L1Cache', 'L1iCache'):
                result.extend(self.__process_cache(obj))
        return result

    def __process_core(self, xml):
        result = []
        for pu in xml.findall('object'):
            assert pu.get('type') == 'PU'
            result.append(int(pu.get('os_index')))
        return result

    @property
    def frequency_information(self):
        try:
            return self.__frequency_information
        except AttributeError:
            freq = self.run_unique('cpufreq-info -l').stdout
            min_f, max_f = [int(f) for f in freq.split()]
            governors = self.run_unique('cpufreq-info -g').stdout.split()
            tuple_cls = collections.namedtuple('frequency_information', ['governor', 'min_freq', 'max_freq'])
            self.__frequency_information = tuple_cls(tuple(governors), min_f, max_f)
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

    def reset_frequency_information(self):
        info = self.frequency_information
        self.set_frequency_information('powersave', info.min_freq, info.max_freq)

    def set_frequency_performance(self):
        max_f = self.frequency_information.max_freq
        self.set_frequency_information('performance', max_f, max_f)


class Job:
    install_path = '~/.local/bin/peanut'
    expfile_types = {}
    auto_oardel = False
    deployment_images = ['debian9-x64-%s' % mode for mode in ['min', 'base', 'nfs', 'big']]
    clusters = {
        'grenoble': ['dahu', 'yeti'],
        'lyon': ['sagittaire', 'hercule', 'orion', 'taurus', 'nova'],
        'nancy': ['griffon', 'graphene', 'graphite', 'grimoire', 'grisou', 'graphique', 'graoully', 'grimani', 'grele',
                  'grvingt'],
        'rennes': ['parapide', 'parapluie', 'paranoia', 'parasilo', 'paravance'],
        'sophia': ['suno', 'uvb'],
        'lille': ['chetemi', 'chifflet'],
        'luxembourg': ['granduc', 'petitprince'],
        'nantes': ['econome', 'ecotype']
    }
    sites = {cluster: site for site, cluster_list in clusters.items() for cluster in cluster_list}
    special_clusters = {
        'testing': ['dahu', 'yeti'],
        'production': ['graphique', 'graoully', 'grimani', 'grele', 'grvingt'],
    }
    queues = {cluster: queue for queue, cluster_list in special_clusters.items() for cluster in cluster_list}
    queues = collections.defaultdict(lambda: 'default', queues)

    def __init__(self, jobid, frontend, deploy=False):
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
        hostnames.sort()
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
        assert self.deploy
        env = env or self.deploy
        self.hostnames
        self.frontend.run('kadeploy3 -k -f %s -e %s' % (self.oar_node_file, env))
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
               deploy=False, queue=None, script=None):
        name = random.choice('☕🥐')
        constraint = '%s/nodes=%d,walltime=%s' % (
            constraint, nb_nodes, walltime)
        deploy_str = '-t deploy ' if deploy else '-t allow_classic_ssh'
        queue_str = '-q %s ' % queue if queue else ''
        cmd = 'oarsub -n "%s" %s%s -l "%s"' % (name, queue_str, deploy_str, constraint)
        if script:
            cmd += " '%s'" % script
        else:
            date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cmd += ' -r "%s"' % date
        result = frontend.run_unique(cmd)
        regex = re.compile('OAR_JOB_ID=(\d+)')
        jobid = int(regex.search(result.stdout).groups()[0])
        return cls(jobid, frontend=frontend, deploy=deploy)

    @classmethod
    def oarsub_cluster(cls, username, cluster, walltime, nb_nodes, *,
                       deploy=False, script=None):
        site = cls.sites[cluster]
        queue = cls.queues[cluster]
        frontend = cls.g5k_frontend(site, username)
        constraint = "{cluster in ('%s')}" % cluster
        return cls.oarsub(frontend, constraint, walltime, nb_nodes, deploy=deploy,
                          queue=queue, script=script)

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
                         deploy=False, script=None):
        cluster = cls._check_hostnames(hostnames)
        site = cls.sites[cluster]
        queue = cls.queues[cluster]
        frontend = cls.g5k_frontend(site, username)
        hostnames = ["'%s'" % cls.expandg5k(host, site) for host in hostnames]
        constraint = "{network_address in (%s)}" % ', '.join(hostnames)
        if nb_nodes is None:
            nb_nodes = len(hostnames)
        return cls.oarsub(frontend, constraint, walltime, nb_nodes, deploy=deploy,
                          queue=queue, script=script)

    @classmethod
    def g5k_connection(cls, site, username):
        if 'grid5000' in socket.getfqdn():  # already inside G5K, no need for a gateway
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
        result['expfile'] = self.expfile.basename
        result.update(self.information)
        return result

    def add_local_to_archive(self, target):
        target_name = os.path.basename(os.path.normpath(target))
        if target_name != target:
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
        try:
            expfile = self.expfile
        except AttributeError:  # no expfile
            pass
        else:
            self.add_content_to_archive(expfile.raw_content, expfile.basename)
        log = log_stream.getvalue()
        log = log.encode('ascii', 'ignore').decode()  # removing any non-ascii character
        self.add_content_to_archive(log, 'commands.log')
        history = json.dumps(self.history, indent=2, sort_keys=True)
        self.add_content_to_archive(history, 'history.json')

    def get_archive(self):
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

    def git_clone(self, url, repository_path, checkout=None):
        self.nodes.run('git clone %s %s' % (url, repository_path))
        if checkout:
            self.nodes.run('git checkout %s' % checkout, directory=repository_path)
        git_hash = self.nodes.run_unique('git rev-parse HEAD', directory=repository_path)
        git_hash = git_hash.stdout.strip()
        key_name = 'git_repositories'
        if key_name not in self.information:
            self.information[key_name] = []
        self.information[key_name].append({'path': repository_path, 'url': url, 'hash': git_hash})

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
        sp_run.add_argument('--expfile', help='File which describes the experiment.', required=True,
                            type=lambda f: ExpFile(filename=f, types=cls.expfile_types))
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
            cmd += '--expfile %s' % self.expfile.basename
        except AttributeError:
            pass
        return cmd.strip()

    @classmethod
    def job_from_args(cls, args):
        user = args['username']
        deploy = args['deploy']
        expfile = args['expfile']
        try:
            cls.check_expfile(expfile)
        except ValueError as e:
            sys.exit(e)
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
            frontend.write_files(expfile.raw_content, expfile.basename)
        else:
            script = None
        if 'cluster' in args:
            job = cls.oarsub_cluster(user, cluster=cluster, walltime=args['walltime'], nb_nodes=args['nbnodes'],
                                     deploy=deploy, script=script)
        elif 'nodes' in args:
            job = cls.oarsub_hostnames(user, hostnames=hostnames, walltime=args['walltime'],
                                       nb_nodes=args['nbnodes'], deploy=deploy, script=script)
        else:
            job = cls(jobid, frontend, deploy=deploy)
        job.expfile = expfile
        return job

    def setup(self):
        if self.deploy:
            self.kadeploy()
        self.nodes  # triggering nodes instanciation, a bit dirty FIXME
        # Creating an empty zip archive on the director node
        # See https://stackoverflow.com/a/50091682/4110059
        self.director.run('echo UEsFBgAAAAAAAAAAAAAAAAAAAAAAAA== | base64 -d > %s' % self.archive_name)
        if self.deploy:
            self.send_key()

    def teardown(self):
        self.add_information_to_archive()
        self.get_archive()
        self.oardel()

    def run_exp(self, expfile):
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
        ExpFile(filename=args['filename'], content=exp)

    @classmethod
    def _main_run(cls, args):
        start = time.time()
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
        logger.info('Total time: %.2f seconds' % (time.time() - start))

    @classmethod
    def generate_batch_command(cls, args, site):
        cmd = '%s %s run %s ' % (cls.install_path, cls.__name__, args['username'])
        cmd += '--jobid %s:$OAR_JOB_ID ' % site
        if args['deploy']:
            cmd += '--deploy %s ' % args['deploy']
        cmd += '--expfile %s' % args['expfile'].basename
        return cmd

    @classmethod
    def check_exp(cls, exp):
        pass

    @classmethod
    def check_expfile(cls, expfile):
        for exp in expfile:
            cls.check_exp(exp)


class ExpFile:
    def __init__(self, *, filename, content=None, types=None):
        self.filename = filename
        self.types = types
        self.content = content
        self.raw_content = None
        self.basename = os.path.basename(filename)
        if not content:
            self.read_content()
        else:
            if not self.types:
                self.types = {key: type(val) for key, val in self.content[0].items()}
            self.write_content()

    def read_content(self):
        try:
            f = open(self.filename, 'r')
        except FileNotFoundError:
            raise ValueError('File %s does not exist.' % self.filename)
        self.raw_content = f.read()
        f.close()
        self.parse_content()

    def write_content(self):
        self.check_types()
        raw = io.StringIO()
        writer = csv.writer(raw)
        header = set(self.content[0].keys())
        writer.writerow(header)
        for row in self.content:
            writer.writerow([row[key] for key in header])
        self.raw_content = raw.getvalue()
        with open(self.filename, 'w') as f:
            f.write(self.raw_content)

    def parse_content(self):
        reader = csv.reader(io.StringIO(self.raw_content))
        header = [h.strip() for h in next(reader)]
        if self.types:
            expected_header = set(self.types)
            real_header = set(header)
            if expected_header != real_header:
                raise ValueError('Wrong format with file %s, expected header %s, got %s.' % (self.filename,
                                                                                             expected_header,
                                                                                             real_header))
        self.content = []
        for i, row in enumerate(reader):
            row = [val.strip() for val in row]
            if len(row) != len(header):
                raise ValueError('Wrong format with file %s, row %d.' % (self.filename, i+2))
            new_row = {key: val for key, val in zip(header, row)}
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

    def __repr__(self):
        return '%s(%s)' % (self.__class__.__name__, self.filename)

    def __iter__(self):
        yield from self.content

    def __len__(self):
        return len(self.content)
