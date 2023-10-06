#!/bin/bash
for f in celeba_configs/configs_default/**/*.yaml; do
  python src/train/fairness_train_celeba.py --config_path $f
done 