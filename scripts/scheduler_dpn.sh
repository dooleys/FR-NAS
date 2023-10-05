#!/bin/bash
#!/bin/bash
#SBATCH -p mldlc_gpu-rtx2080
#SBATCH -t 6-00:00:00 # time (D-HH:MM)
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J fairnas # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
dask-scheduler --scheduler-file  "scheduler-dpn-file.json" --idle-timeout 1000000000000000000000000 --port 1796
