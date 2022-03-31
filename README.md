# FR-NAS

The training and testing scripts are located here:
<face.evoLVe.PyTorch/fairness_test_Celeba_timm.py> and <face.evoLVe.PyTorch/fairness_train_timm.py>

To create the list of commands (one command per model), there is a hacky script <fix.py> which I use to create the command scripts (like in <commands/scratch.sh>. Generally I pass this bash file to slurm to queue one job for each line. 

The results of the commands are put in the files in the <results_nooversampling> folder. 

I am using python 3.9.7 and I output the packages from my conda environment into the requirements.txt file. I imagine there are more packages in here than necessary, but it is what is working for me currently on my system. 
