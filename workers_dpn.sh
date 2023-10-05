#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH --gres=gpu:4
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -c 4 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J fairnas # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
DASK_DISTRIBUTED__WORKER__DAEMON=False dask-worker --nthreads 1 --lifetime 10000000000000000000000000 --memory-limit 0 --scheduler-file "scheduler-dpn-file.json"
