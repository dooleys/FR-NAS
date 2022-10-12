#!/bin/bash
for f in configs_multi/**/*.yaml; do
  echo "python src/fairness_train_timm.py --config_path $f"
done 