# Installation

A quick way to install `peanut` is with `pip`:
```py
pip install git@github.com:Ezibenroc/peanut.git
```
(this is a Python 3 package, so you have to change `pip` by `pip3` if your system defaults to Python 2, you may also
need to either pass the `--user` option or prepend this command with `sudo`)

You need to install `peanut` on the Grid'5000 frontend of the site you want to use. If you want to launch jobs from your
PC, you also need to install it here.

# Usage

This experiment engine can run several kinds of experiments. As a first example, we will take the HPL benchmark.

First, you need an install file, name it `install.yaml`, it contains the various options you can use to tune the
preamble of this experiment (how HPL is installed, warmup, monitoring...):
```yaml
trace_execution: False        # True to trace all the MPI and kernel calls
terminate_early: False        # True to terminate the execution after only 5 iterations
insert_bcast: False           # True to insert a small broadcast at the start and end of the main function
trace_dgemm: False            # True to tracc all the dgemm calls
monitoring: 0                 # number of seconds between each probe of the monitoring script, 0 to disable
warmup_time: 600              # number of seconds of the warmup time
openblas: v0.3.13             # string representing the OpenBLAS version to install
openmpi: distribution_package # distribution_package for a 'apt install openmpi', a version string (like "4.1.0") for an installation from source (warning: experimental)
```

You also need an experiment file, name it `exp.csv`, each row represent a run of HPL, each column is a parameter (e.g.
matrix size, broadcast algorithm...):
```csv
matrix_size,block_size,proc_p,proc_q,pfact,rfact,bcast,depth,swap,mem_align,process_per_node,thread_per_process
250000,128,32,32,1,2,2,0,0,8,32,1
250000,256,32,32,1,2,5,1,1,8,32,1
250000,128,32,32,1,2,5,1,1,8,32,1
250000,128,32,32,1,2,1,0,0,8,32,1
```

Then, from your PC or from the frontend, you can launch a Grid'5000 batch job that will run this experiment on the
cluster Dahu with 32 nodes (don't forget to replace my login by yours):
```sh
peanut HPL run tocornebize --batch --deploy debian9-x64-base --cluster dahu --nbnodes 32 \
    --walltime 01:00:00 --expfile exp.csv--installfile install.yaml
```

At the end of the job, you will get a `*.zip` archive in your Grid'5000 home directory.


# Available experiments

A dozen of experiments are currently available in peanut:
- MPICalibration: calibrate the performance of point-to-point MPI communications
- MPIRing: advanced calibration method for point-to-point MPI communications, where all the MPI ranks are spamming the
  MPI_Iprobe function (and optionnally the dgemm function)
- MPISaturation: run point-to-point MPI communications between an increasing number of node pairs, to estimate the
  maximum aggregated bandwidth
- MemoryCalibration: calibrate the performance of memory writes, akin to the MPI calibration script
- BLASCalibration: calibrate the performance of the dgemm function, either in single-threaded or in multi-threaded mode
- StressTest: run some basic stress on the nodes (legacy script)
- HPL: perform real executions of HPL
- SMPIHPL: perform simulated executions of HPL
- FrequencyGet: collect and display the core frequency when under stress, with various settings (turboboost,
  hyperthreading and C-states enabled or not)
- Simdjson: run some basic benchmarks for the SimdJSON library
- SW4lite: tentative to run the SW4lite benchmark both in reality and in simulation (unfinished work)
- BitFlips: tentative to reproduce the "bit-flip" performance annomaly with a custom code (unfinished work)

# Adding a new experiment

To add a new experiment, the "best" way is to add a Python file in the `peanut` directory where a new subclass of
`peanut.Job` is declared, defining the `setup` and `run_exp` methods, then add the said class to the `classes` list in
file `peanut/__main__.py`.

Alternatively, the aforementioned Python file can be directly executed by peanut (provided it contains the subclass of
`Job` with the required methods, as described above):
```sh
peanut my_script.py run tocornebize --deploy debian9-x64-base --cluster dahu --nbnodes 2 --walltime 00:20:00
```
