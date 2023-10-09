#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=mscelebsh
#SBATCH --array=1-1

#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger






#SBATCH --gres=gpu:rtxa5000:8
#SBATCH --cpus-per-task=32
#SBATCH --mem=31gb
#SBATCH --time=48:00:00


#SBATCH --output=log/%x_%A_%a.log
#SBATCH --error=log/%x_%A_%a.log
#SBATCH --mail-user=sdooley1@umiacs.umd.edu
#SBATCH --mail-type=FAIL,ARRAY_TASKS






export MASTER_PORT=$(shuf -i 2000-65000 -n 1) # Remember that these are fixed across the entire array
export MASTER_ADDR=`/bin/hostname -s`

srun $(head -n $((1*(SLURM_ARRAY_TASK_ID+0)-1+1)) .job_list_Kjrswlk.temp.sh | tail -n 1) 