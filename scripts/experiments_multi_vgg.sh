#!/bin/bash
for f in vgg_configs/configs_multi/**/*.yaml; do
  python src/fairness_train_vgg.py --config_path $f
done 