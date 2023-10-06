#!/bin/bash
for f in celeba_configs/configs_multi/**/*.yaml; do
  python src/train/fairness_train_celeba.py --config_path $f
done 