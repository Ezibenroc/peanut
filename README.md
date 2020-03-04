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

First, you need an install file, name it `install.yaml`:
```yaml
trace_execution: False  # True to trace all the MPI and kernel calls
terminate_early: False  # True to terminate the execution after only 5 iterations
insert_bcast: False     # True to insert a small broadcast at the start and end of the main function
trace_dgemm: False      # True to tracc all the dgemm calls
monitoring: 0           # number of seconds between each probe of the monitoring script, 0 to disable
warmup_time: 600        # number of seconds of the warmup time
```

You also need an experiment file, name it `exp.csv`
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
