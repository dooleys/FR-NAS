#!/bin/bash

# Lines that begin with #SBATCH specify commands to be used by SLURM for scheduling
#SBATCH --job-name=jupyter

#SBATCH --qos=scavenger
#SBATCH --partition=scavenger
#SBATCH --account=scavenger

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=31gb
#SBATCH --time=48:00:00


#SBATCH --output=.notebook_rzoSwmM.log
#SBATCH --error=.notebook_errors_rzoSwmM.log







export JUPYTER_PORT=$(shuf -i 2000-65000 -n 1)
export HOSTNAME=`/bin/hostname -s`

printf "
Run this command for the ssh connection:
ssh -N -f -L localhost:${JUPYTER_PORT}:${HOSTNAME}:${JUPYTER_PORT} sdooley1@nexuscml01.umiacs.umd.edu

and open the following web adress in your local browser:
http://localhost:${JUPYTER_PORT}/?token=MGP2rzABUhE9dA
" >> .notebook_rzoSwmM.log

jupyter notebook --no-browser --port=${JUPYTER_PORT} --ip ${HOSTNAME} --NotebookApp.token=MGP2rzABUhE9dA

