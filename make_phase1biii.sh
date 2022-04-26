
for f in configs_multi/**/*.yaml; do
  echo "python python src/fairness_train_timm.py --config_path $f"
done 
