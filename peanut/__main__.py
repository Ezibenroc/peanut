import argparse
import zipfile
import yaml
import os
from .mpi_calibration import MPICalibration
from .mpi_saturation import MPISaturation
from .hpl import HPL
from .blas_calibration import BLASCalibration
from .stress_test import StressTest
from .bit_flips import BitFlips
from .version import __version__, __git_version__
from .smpi_hpl import SMPIHPL
from .frequency_get import FrequencyGet
from .sw4lite import SW4lite
from .simdjson import Simdjson

classes = [
    MPICalibration,
    MPISaturation,
    BLASCalibration,
    StressTest,
    HPL,
    SMPIHPL,
    FrequencyGet,
    SW4lite,
    BitFlips,
    Simdjson,
]

entry_points = {cls.__name__: cls.main for cls in classes}


def replay(args):
    parser = argparse.ArgumentParser(description='Peanut, the tiny job runner')
    parser.add_argument('zip_name', type=str, help='Zip file of the experiment to replay.')
    args = parser.parse_args(args)
    try:
        input_zip = zipfile.ZipFile(args.zip_name)
    except FileNotFoundError:
        parser.error('File %s does not exist' % args.zip_name)
    try:
        info = yaml.load(input_zip.read('info.yaml'))
        replay_command = info['replay_command']
        expfile_name = info['expfile']
        expfile_content = input_zip.read(expfile_name)
        git_version = info['peanut_git_version']
    except KeyError:
        parser.error('Wrong format for archive %s' % args.zip_name)
    if git_version != __git_version__:
        parser.error('Mismatch between the peanut versions. To replay this experiment, checkout %s' % git_version)
    with open(expfile_name, 'wb') as f:
        f.write(expfile_content)
    args = replay_command.split(' ')
    assert args[0] == 'peanut'
    entry_points[args[1]](args[2:])
    os.remove(expfile_name)


entry_points['replay'] = replay


def main():
    parser = argparse.ArgumentParser(description='Peanut, the tiny job runner')
    parser.add_argument('command', choices=entry_points.keys(), help='Experiment to run.')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=__version__))
    parser.add_argument('--git-version', action='version',
                        version='%(prog)s {version}'.format(version=__git_version__))
    args, command_args = parser.parse_known_args()
    entry_points[args.command](command_args)


if __name__ == '__main__':
    main()
