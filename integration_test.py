import unittest
import tempfile
import random
import string
import hashlib
import os
import zipfile
from datetime import datetime
import itertools
from peanut import Job, Time, Nodes, ExpFile
from peanut.peanut import RunError

Job.auto_oardel = True


class Util(unittest.TestCase):
    cluster = 'taurus'
    site = Job.sites[cluster]
    nb_nodes = 3
    user = 'tocornebize'


class TestBasic(Util):
    def test_frontend(self):
        frontend = Job.g5k_connection(self.site, self.user)
        result = frontend.run('hostname -f', hide=True).stdout.strip()
        self.assertEqual(result, 'f%s.%s.grid5000.fr' % (self.site, self.site))


class TestNodes(Util):
    def assert_run(self, expected_result, *args, **kwargs):
        result = self.node.run(*args, **kwargs)
        self.assertEqual(len(result), 1)
        result = list(result.values())[0].stdout.strip()
        self.assertEqual(result, expected_result)

    def test_run(self):
        frontend = Job.g5k_connection(self.site, self.user)
        self.node = Nodes([frontend], name='foo', working_dir='/tmp')
        self.assert_run('f%s.%s.grid5000.fr' % (self.site, self.site), 'hostname -f')
        self.assert_run('', 'mkdir -p foo/bar')
        self.assert_run('/tmp/foo/bar', 'pwd', directory='foo/bar')
        directory = '/home/%s' % self.user
        self.assert_run(directory, 'pwd', directory=directory)

    def test_put_get(self):
        frontend = Job.g5k_connection(self.site, self.user)
        self.node = Nodes([frontend], name='foo', working_dir='/home/%s' % self.user)
        tmp_file = tempfile.NamedTemporaryFile(dir='.')
        with open(tmp_file.name, 'w') as f:
            f.write('hello, world!\n')
        filename = 'test_fabfile'
        self.node.put(tmp_file.name, filename)
        self.assert_run('4dca0fd5f424a31b03ab807cbae77eb32bf2d089eed1cee154b3afed458de0dc  %s' % filename,
                        'sha256sum %s' % filename)
        tmp_new = tempfile.NamedTemporaryFile(dir='.')
        self.node.get(filename, tmp_new.name)
        with open(tmp_new.name, 'r') as f:
            content = f.read()
        self.assertEqual(content, 'hello, world!\n')
        self.assert_run('', 'rm -f %s' % filename)

    def test_write_files(self):
        frontend = Job.g5k_connection(self.site, self.user)
        self.node = Nodes([frontend], name='foo', working_dir='/home/%s' % self.user)
        for size in [1, 10, 50, 100, 1000, 10000]:
            for filenames in [['test_fabfile'], ['test_fabfile%d' % i for i in range(10)]]:
                content = ''.join(random.choices(string.ascii_lowercase + '\n\t ', k=size))
                content_hash = hashlib.sha256(content.encode('ascii')).hexdigest()
                self.node.write_files(content, *filenames)
                for filename in filenames:
                    self.assert_run('%s  %s' % (content_hash, filename), 'sha256sum %s' % filename)
                self.assert_run('', 'rm -f %s' % ' '.join(filenames))

    def test_cores(self):
        frontend = Job.g5k_connection(self.site, self.user)
        self.node = Nodes([frontend], name='foo', working_dir='/home/%s' % self.user)
        self.assertEqual(self.node.cores, [[0, 1, 2, 3]])  # might change if the frontend server of Lyon changes

    def test_hyperthreads(self):
        frontend = Job.g5k_connection(self.site, self.user)
        self.node = Nodes([frontend], name='foo', working_dir='/home/%s' % self.user)
        self.assertEqual(self.node.hyperthreads, [1, 2, 3])  # might change if the frontend server of Lyon changes

    def test_history(self):
        lyon = Job.g5k_connection('lyon', self.user)
        nancy = Job.g5k_connection('nancy', self.user)
        rennes = Job.g5k_connection('rennes', self.user)
        allnodes = Nodes([lyon, nancy, rennes], name='nodes', working_dir='/home/%s' % self.user)
        other = Nodes([lyon], name='other', working_dir='/home/%s' % self.user, parent_nodes=allnodes)
        commands = {'pwd': (lambda nodes: nodes.working_dir),
                    'hostname': (lambda nodes: {n.host: 'f' + n.host for n in nodes})}

        tests = list(itertools.product([allnodes, other], commands.keys(), [True, False]))
        random.shuffle(tests)
        history = allnodes.history
        for i, (nodes, command, switch_std) in enumerate(tests):
            real_command = command
            if switch_std:
                real_command = command + ' 3>&2 2>&1 1>&3'
            start = datetime.now()
            nodes.run(real_command)
            stop = datetime.now()
            old_history = history
            history = nodes.history
            self.assertEqual(len(history), i+1)
            self.assertEqual(history[:i], old_history)
            hist_entry = history[-1]
            self.assertEqual(hist_entry['command'], real_command)
            expected_output = commands[command](nodes)
            if len(expected_output) == 1:
                expected_output = list(expected_output.values())[0]
            if not switch_std:
                self.assertEqual(hist_entry['stdout'], expected_output)
                self.assertNotIn('stderr', hist_entry)
            else:
                self.assertEqual(hist_entry['stderr'], expected_output)
                self.assertNotIn('stdout', hist_entry)
            date = hist_entry['date']
            real_start = datetime.strptime(date['start'], '%Y-%m-%d %H:%M:%S.%f')
            real_stop = datetime.strptime(date['stop'], '%Y-%m-%d %H:%M:%S.%f')
            self.assertEqual(date['duration'], (real_stop - real_start).total_seconds())
            self.assertAlmostEqual(0, (real_start - start).total_seconds(), delta=1e-3)
            self.assertAlmostEqual(0, (real_stop - stop).total_seconds(), delta=1e-3)
            self.assertEqual(0, hist_entry['return_code'])

    def test_errored_history(self):
        lyon = Job.g5k_connection('lyon', self.user)
        nancy = Job.g5k_connection('nancy', self.user)
        rennes = Job.g5k_connection('rennes', self.user)
        allnodes = Nodes([lyon, nancy, rennes], name='nodes', working_dir='/home/%s' % self.user)
        other = Nodes([lyon], name='other', working_dir='/home/%s' % self.user, parent_nodes=allnodes)
        allnodes.run('rm -rf bla')
        with self.assertRaises(RunError):
            allnodes.run('ls bla')
        hist_entry = allnodes.history[-1]
        self.assertEqual(hist_entry['return_code'], 2)
        msg = "ls: cannot access 'bla': No such file or directory"
        self.assertEqual(hist_entry['stderr'], msg)
        other.run('touch bla')
        with self.assertRaises(RunError):
            allnodes.run('ls bla')
        hist_entry = allnodes.history[-1]
        self.assertEqual(hist_entry['return_code'], {'lyon': 0, 'nancy': 2, 'rennes': 2})
        expected = {n: msg for n in ['nancy', 'rennes']}
        expected['lyon'] = ''
        self.assertEqual(hist_entry['stderr'], expected)
        allnodes.run('rm -rf bla')


class TestJob(Util):

    def test_job(self):
        job = Job.oarsub_cluster(username=self.user,
                                 cluster=self.cluster,
                                 walltime=Time(minutes=5),
                                 nb_nodes=self.nb_nodes,
                                 deploy=False,
                                 )
        result = job.frontend.run_unique('hostname -f').stdout.strip()
        self.assertEqual(result, 'f%s.%s.grid5000.fr' % (self.site, self.site))
        hosts = job.hostnames
        jobs = Job.get_jobs(self.site, self.user)
        self.assertTrue(any(j.hostnames == hosts for j in jobs))
        self.assertEqual(len(set(hosts)), self.nb_nodes)
        for host in hosts:
            self.assertEqual(host[:len(self.cluster)], self.cluster)
        self.assertEqual(set(job.hostnames), set(job.nodes.hostnames))
        result = job.nodes.run('hostname -f')
        for node, res in result.items():
            self.assertEqual(node.host, res.stdout.strip())
        result = job.nodes.run_unique('pwd')
        self.assertEqual(result.stdout.strip(), '/tmp')
        expected_cores = [[i, i+12] for i in list(range(0, 12, 2)) + list(range(1, 12, 2))]
        self.assertEqual(job.nodes.cores, expected_cores)
        self.assertEqual(set(job.nodes.hyperthreads), set(range(12, 24)))
        job.add_content_to_archive('hello, world!', 'file_hello')
        job.add_content_to_archive('foo bar'*50, 'file_foo')
        with open('/tmp/bla', 'w') as f:
            f.write('this is a test')
        job.add_file_to_archive('/tmp/bla', 'file_test')
        job.get_archive()
        self.assertTrue(os.path.isfile(job.archive_name))
        archive = zipfile.ZipFile(job.archive_name)
        self.assertEqual(set(archive.namelist()), {'file_hello', 'file_foo', 'file_test'})
        self.assertEqual(archive.read('file_hello').decode('ascii'), 'hello, world!')
        self.assertEqual(archive.read('file_foo').decode('ascii'), 'foo bar'*50)
        self.assertEqual(archive.read('file_test').decode('ascii'), 'this is a test')
        job.add_information_to_archive()
        job.get_archive()
        archive = zipfile.ZipFile(job.archive_name)
        for name in ['info.yaml', 'oarstat.yaml', 'commands.log', 'history.json']:
            self.assertIn(name, archive.namelist())
        for host in job.hostnames:
            for name in ['cpuinfo.txt', 'environment.txt', 'topology.xml', 'topology.pdf', 'lspci.txt',
                         'dmidecode.txt']:
                name = os.path.join('information', host, name)
                self.assertIn(name, archive.namelist())
        os.remove(job.archive_name)

    def test_info(self):
        job = Job.oarsub_cluster(username=self.user,
                                 cluster=self.cluster,
                                 walltime=Time(minutes=5),
                                 nb_nodes=self.nb_nodes,
                                 deploy=False,
                                 )
        info = job.platform_information()
        self.assertEqual(info['deployment'], False)
        self.assertIn('command', info)
        self.assertEqual(info['jobid'], job.jobid)
        self.assertEqual(info['site'], self.site)
        self.assertEqual(info['cluster'], self.cluster)
        for host in job.hostnames:
            self.assertIn(host, info)
            for key in ['arp', 'cpu', 'kernel', 'mpi', 'version']:
                self.assertIn(key, info[host])
        git_hash = '61888b8576a7913c9fa7c40c2918f92bcc1f5c17'
        url = 'https://github.com/dylanbeattie/rockstar.git'
        job.git_clone(url, 'rockstar', checkout=git_hash)
        new_info = job.platform_information()
        self.assertEqual(new_info['git_repositories'], [{'path': 'rockstar', 'url': url, 'hash': git_hash}])
        del new_info['git_repositories']
        self.assertEqual(new_info, info)

    def assert_run(self, expected_result, cmd):
        result = self.job.nodes.run_unique(cmd).stdout.strip()
        self.assertEqual(result, expected_result)

    def test_job_deploy(self):
        self.job = Job.oarsub_cluster(username=self.user,
                                      cluster=self.cluster,
                                      walltime=Time(minutes=15),
                                      nb_nodes=self.nb_nodes,
                                      deploy='debian9-x64-min',
                                      )
        self.job.kadeploy().apt_install('hwloc', 'cpufrequtils', 'linux-cpupower')
        nb = len(self.job.nodes.hyperthreads)
        self.assert_run(str(nb*2), 'grep processor /proc/cpuinfo | wc -l')
        self.job.nodes.disable_hyperthreading()
        self.assert_run(str(nb), 'grep processor /proc/cpuinfo | wc -l')
        self.job.nodes.enable_hyperthreading()
        self.assert_run(str(nb*2), 'grep processor /proc/cpuinfo | wc -l')
        freq_info = self.job.nodes.frequency_information
        self.assertEqual(freq_info.governor, ('performance', 'powersave'))
        self.assertEqual(freq_info.min_freq, 1200000)
        self.assertEqual(freq_info.max_freq, 2800000)
        for _ in range(2):
            for governor in freq_info.governor:
                min_freq = random.randint(freq_info.min_freq, freq_info.max_freq)
                max_freq = random.randint(min_freq, freq_info.max_freq)
                self.job.nodes.set_frequency_information(governor, min_freq, max_freq)
                self.assertEqual(self.job.nodes.current_frequency_information, (governor, min_freq, max_freq))
        self.job.nodes.reset_frequency_information()
        self.assertEqual(self.job.nodes.current_frequency_information,
                         ('powersave', freq_info.min_freq, freq_info.max_freq))
        self.job.nodes.set_frequency_performance()
        self.assertEqual(self.job.nodes.current_frequency_information,
                         ('performance', freq_info.max_freq, freq_info.max_freq))
        with self.assertRaises(ValueError):
            self.job.nodes.set_frequency_information(governor='bla')
        with self.assertRaises(ValueError):
            self.job.nodes.set_frequency_information(min_freq=42)
        with self.assertRaises(ValueError):
            self.job.nodes.set_frequency_information(max_freq=42)
        with self.assertRaises(ValueError):
            self.job.nodes.set_frequency_information(min_freq=freq_info.min_freq + 100, max_freq=freq_info.min_freq)


class ContentFile:
    def __init__(self, content, extension=None):
        self.content = content
        self.extension = extension

    def __enter__(self):
        suffix = '.' + self.extension if self.extension else None
        self.tmp_file = tempfile.NamedTemporaryFile(dir='.', suffix=suffix)
        with open(self.tmp_file.name, 'w') as f:
            f.write(self.content)
        return self.tmp_file.name

    def __exit__(self, type, value, traceback):
        self.tmp_file.close()


class TestExpFile(unittest.TestCase):
    def test_read_csv_untyped(self):
        content = '\n'.join(['aaa, bbb, ccc',
                             'foo, 18, 3.14',
                             'bar, 5, 0.111'])
        with ContentFile(content, 'csv') as filename:
            expfile = ExpFile(filename=filename)
            self.assertEqual(expfile.raw_content, content)
            self.assertEqual(expfile.content, [{'aaa': 'foo', 'bbb': '18', 'ccc': '3.14'},
                                               {'aaa': 'bar', 'bbb':  '5', 'ccc': '0.111'}])

    def test_read_csv_typed(self):
        content = '\n'.join(['aaa, bbb, ccc',
                             'foo, 18, 3.14',
                             'bar, 5, 0.111'])
        with ContentFile(content, 'csv') as filename:
            expfile = ExpFile(filename=filename, types={'aaa': str, 'bbb': int, 'ccc': float})
            self.assertEqual(expfile.raw_content, content)
            self.assertEqual(expfile.content, [{'aaa': 'foo', 'bbb': 18, 'ccc': 3.14},
                                               {'aaa': 'bar', 'bbb':  5, 'ccc': 0.111}])

    def test_read_csv_wrong_header(self):
        content = '\n'.join(['aaa, bbb, ccc',
                             'foo, 18, 3.14',
                             'bar, 5, 0.111'])
        with ContentFile(content, 'csv') as filename:
            with self.assertRaises(ValueError):
                ExpFile(filename=filename, types={'aaa': str, 'bbb': int, 'xxx': float})
            with self.assertRaises(ValueError):
                ExpFile(filename=filename, types={'aaa': str, 'bbb': int})
            with self.assertRaises(ValueError):
                ExpFile(filename=filename, types={'aaa': str, 'bbb': int, 'ccc': float, 'ddd': int})

    def test_read_csv_wrong_number_columns(self):
        content = '\n'.join(['aaa, bbb, ccc',
                             'foo, 18',
                             'bar, 5, 0.111'])
        with ContentFile(content, 'csv') as filename:
            with self.assertRaises(ValueError):
                ExpFile(filename=filename)

    def test_read_csv_wrong_type(self):
        content = '\n'.join(['aaa, bbb, ccc',
                             'foo, 18, 3.14',
                             'bar, 5, 0.111'])
        with ContentFile(content, 'csv') as filename:
            with self.assertRaises(ValueError):
                ExpFile(filename=filename, types={'aaa': str, 'bbb': int, 'ccc': int})
            with self.assertRaises(ValueError):
                ExpFile(filename=filename, types={'aaa': float, 'bbb': int, 'ccc': float})

    def test_write_csv_untyped(self):
        content = [{'aaa': 'foo', 'bbb': 18, 'ccc': 3.14},
                   {'aaa': 'bar', 'bbb':  5, 'ccc': 0.111}]
        with ContentFile('', 'csv') as filename:
            expfile = ExpFile(content=content, filename=filename)
            self.assertEqual(expfile.types, {'aaa': str, 'bbb': int, 'ccc': float})
            new_expfile = ExpFile(filename=filename, types=expfile.types)
            self.assertEqual(new_expfile.content, content)


if __name__ == '__main__':
    unittest.main()
