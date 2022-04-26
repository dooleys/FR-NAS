# FR-NAS
## Create Configs
### To create config files for a model execute the following command. Make sure to pass your chosen hyperparams as command line arguments as described in the example below:
<code> python create_configs.py --user_config <path_to_user_config> --backbone <backbone> --batch_size <batch_size> </code> 
 
<code> python create_configs.py --user_config config_user.yaml --backbone  ghostnet_100 --batch_size 64</code>
 
<code> python create_configs.py --user_config config_user.yaml --backbone vgg19 --batch_size 64 --lr 0.01 --momentum 0.9 --weight-decay 1e-4 --sched step --lr-cycle-decay 0.1 </code>
 
## Training
### To train a model based on the created configs execute the following command
<code>python src/fairness_train_timm.py --config_path <your_config_path> </code> 
 
<code>python src/fairness_train_timm.py --config_path configs/ghostnet_100/config_ghostnet_100_MagFace_Adam.yaml </code>
 
<code> python src/fairness_train_timm.py --config_path configs/vgg19/config_vgg19_MagFace_SGD.yaml </code> 

## Create list of things to train
To create the config files from a list of commands, run:

 <code> bash ./make_phase1bii.sh > phase1bii.sh</code>

To then create the Phase1b(ii) and Phase1b(iii) training commands, run:

 <code>bash ./make_phase1bii.sh > phase1bii.sh</code>
 
<code>bash ./make_phase1biii.sh > phase1biii.sh</code>
