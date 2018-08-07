import argparse
from .mpi_calibration import MPICalibration
from .version import __version__, __git_version__


def main():
    entry_points = {cls.__name__: cls.main for cls in [MPICalibration]}
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
