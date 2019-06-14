#!/usr/bin/env python3

import sys
from setuptools import setup
import subprocess

VERSION = '0.0.0'


class CommandError(Exception):
    pass


def run(args):
    proc = subprocess.Popen(args, stdout=subprocess.PIPE)
    stdout, stderr = proc.communicate()
    if proc.returncode != 0:
        raise CommandError('Error with the command %s.\n' % ' '.join(args))
    return stdout.decode('ascii').strip()


def git_version():
    return run(['git', 'rev-parse', 'HEAD'])


def git_tag():
    return run(['git', 'describe', '--always', '--dirty'])


def write_version(filename, version_dict):
    with open(filename, 'w') as f:
        for version_name in version_dict:
            f.write('%s = "%s"\n' % (version_name, version_dict[version_name]))


if __name__ == '__main__':
    try:
        write_version('peanut/version.py', {
                '__version__': VERSION,
                '__git_version__': git_version(),
            })
    except CommandError as e:
        if sys.argv[0] != '-c':
            sys.exit(e)
    setup(name='peanut',
          version=VERSION,
          description="Experiment engine for Grid'5000",
          author='Tom Cornebize',
          author_email='tom.cornebize@gmail.com',
          packages=['peanut'],
          entry_points={
              'console_scripts': ['peanut = peanut.__main__:main',
                                  ]
          },
          install_requires=[
              'fabric',
              'colorlog',
              'pyyaml',
              'lxml',
          ],
          url='https://github.com/Ezibenroc/peanut',
          license='MIT',
          classifiers=[
              'License :: OSI Approved :: MIT License',
              'Intended Audience :: Developers',
              'Operating System :: POSIX :: Linux',
              'Operating System :: MacOS :: MacOS X',
              'Programming Language :: Python :: 3.6',
          ],
          )
