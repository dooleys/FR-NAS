# FR-NAS
## Create Configs
### To create config files for a model execute the following command. Make sure to pass your chosen hyperparams as command line arguments as described in the example below:
<code> python create_configs.py --backbone <backbone> --batch_size <batch_size> </code> 
 
<code> python create_configs.py --backbone ghostnet_100 --batch_size 64 </code>
 
## Training
### To train a model based on the created configs execute the following command
<code>python src/fairness_train_timm.py --config_path <your_config_path> </code> 
 
<code>python src/fairness_train_timm.py --config_path configs/ghostnet_100/config_ghostnet_100_MagFace_Adam.yaml </code> 
