#!/bin/bash
#SBATCH -p bosch_gpu-rtx2080 #teslaP100
#SBATCH --gres=gpu:4
#SBATCH -t 4-00:00:00 # time (D-HH:MM)
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J fairnas # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
python src/search/train_from_scratch.py
