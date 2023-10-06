#!/bin/bash
for f in vgg_configs/configs_default/**/*.yaml; do
  python src/train/fairness_train_vgg.py --config_path $f
done 