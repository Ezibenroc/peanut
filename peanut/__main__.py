import argparse
from .mpi_calibration import MPICalibration


def main():
    entry_points = {
        'mpi_calibration': MPICalibration.main,
    }
    parser = argparse.ArgumentParser(description='Peanut, the tiny job runner')
    parser.add_argument('command', choices=entry_points.keys(), help='Experiment to run.')
    args, command_args = parser.parse_known_args()
    entry_points[args.command](command_args)


if __name__ == '__main__':
    main()
