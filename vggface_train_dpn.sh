python src/fairness_train_timm-vggface2.py --config_path configs/dpn107/config_dpn107_CosFace_sgd.yaml
python src/fairness_train_timm-vggface2.py --config_path configs_multi/dpn107/config_dpn107_ArcFace_SGD.yaml
python src/fairness_train_timm-vggface2.py --config_path configs_multi/dpn107/config_dpn107_CosFace_AdamW.yaml
python src/fairness_train_timm-vggface2.py --config_path configs_multi/dpn107/config_dpn107_MagFace_AdamW.yaml
python src/fairness_train_timm-vggface2.py --config_path configs_multi/dpn107/config_dpn107_MagFace_SGD.yaml
python src/fairness_train_timm-vggface2.py --config_path configs_unified_lr/dpn107/config_dpn107_MagFace_SGD_0.1_cosine.yaml
