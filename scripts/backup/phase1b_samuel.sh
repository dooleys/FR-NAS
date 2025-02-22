python create_phase1b_configs.py --user_config config_user.yaml --backbone gluon_inception_v3 --batch_size 64 --lr 0.4 --warmup-epochs 5 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone tnt_s_patch16_224 --batch_size 64 --sched cosine --opt AdamW  --warmup-lr 1e-6 --model-ema --model-ema-decay 0.99996 --warmup-epochs 5 --opt-eps 1e-8 --lr 1e-3 --weight-decay .05 --drop 0 --drop-path .1 --input_size 224 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone inception_resnet_v2 --batch_size 64 --lr 0.045 --lr-cycle-decay 0.94 --momentum 0.9 --opt RMSProp --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone rexnet_200 --batch_size 64 --lr 0.5 --momentum 0.9 --weight-decay 1e-5 --opt SGD --input_size 224 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone ig_resnext101_32x8d --batch_size 64 --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --sched step --opt SGD --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone resnetrs101 --batch_size 64 --momentum 0.9 --lr 1.6 --opt SGD --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone selecsls60b --batch_size 64 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone fbnetv3_g --batch_size 64 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone legacy_senet154 --batch_size 64 --lr 0.6 --momentum 0.9 --weight-decay 1e-5 --opt SGD --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone vgg19_bn --batch_size 64 --lr 0.1 --momentum 0.9 --weight-decay 1e-4 --sched step --opt SGD --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone vgg19 --batch_size 64 --lr 0.01 --momentum 0.9 --weight-decay 1e-4 --sched step --opt SGD --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone vit_large_patch16_224 --batch_size 64 --lr 0.5 --warmup-epochs 5 --weight-decay 0.00002 --clip-grad 1 --model-ema --opt AdamW --input_size 224 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone mobilenetv3_large_100 --batch_size 64 --sched step --decay-epochs 2.4 --decay-rate .973 --opt-eps .001 --warmup-lr 1e-6 --weight-decay 1e-5 --drop 0.2 --drop-connect 0.2 --model-ema --model-ema-decay 0.9999 --lr .064 --lr-noise 0.42 0.9 --opt RMSpropTF --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone inception_v4 --batch_size 64 --lr 0.045 --lr-cycle-decay 0.94 --momentum 0.9 --opt RMSProp  --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone resmlp_big_24_224_in22ft1k --lr 5e-3 --weight-decay 0.2 --opt AdamW --warmup-epochs 5 --input_size 224 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone jx_nest_base --batch_size 64 --lr 2.5e-4 --opt AdamW --weight-decay 0.05 --warmup-epoch 7 --input_size 224 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone visformer_small --batch_size 64 --lr 5e-4 --opt AdamW --weight-decay 0.05 --warmup-epoch 5 --opt-eps 1e-8 --warmup-lr 0.0001 --min-lr 1e-5 --input_size 224 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone ese_vovnet39b --batch_size 64 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone xception --batch_size 64 --lr 0.1 --warmup-epochs 5 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone gluon_xception65 --batch_size 64 --lr 0.1 --warmup-epochs 5 --save_freq 20
python create_phase1b_configs.py --user_config config_user.yaml --backbone xception65 --batch_size 64 --lr 0.1 --warmup-epochs 5 --save_freq 20