# FR-NAS

The training and testing scripts are located here:
<code>src/fairness_test_Celeba_timm.py</code> and <code>src/fairness_train_timm.py</code>

To create the list of commands (one command per model), there is a hacky script <code>fix.py</code> which I use to create the command scripts (like in <code>commands/scratch.sh</code>. Generally I pass this bash file to slurm to queue one job for each line. 

The results of the commands are put in the files in the <code>results_nooversampling</code> folder. 

I am using python 3.9.7 and I output the packages from my conda environment into the requirements.txt file. I imagine there are more packages in here than necessary, but it is what is working for me currently on my system. 
