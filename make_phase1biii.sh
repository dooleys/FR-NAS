
for f in configs_multi/**/*.yaml; do
  default=$(echo $f | sed 's/configs_multi/configs/g')
  if ! [[ -f $default ]];then
    echo "python python src/fairness_train_timm.py --config_path $f"
  fi
done 
