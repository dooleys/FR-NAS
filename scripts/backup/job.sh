#!/bin/bash
#SBATCH -p ml_gpu-teslaP100#,ml_gpu-rtx2080,bosch_gpu-rtx2080
#SBATCH --gres=gpu:2
#SBATCH -t 4-00:00 # time (D-HH:MM)
#SBATCH -c 64 # number of cores
#SBATCH -o logs/%j.%x.%N.out # STDOUT  (the folder log has to be created prior to running or this won't work)
#SBATCH -e logs/%j.%x.%N.err # STDERR  (the folder log has to be created prior to running or this won't work)
#SBATCH -J fairnas_dpns # sets the job name. If not specified, the file name will be used as job name
#SBATCH --mail-type=END,FAIL # (recive mails about end and timeouts/crashes of your job)
#Print some information about the job to STDOUT

# Missing args: --auto-augment ra --epochs 300 and resume, use resume with \' otherwise sbatch tries to read the file
python src/fairness_train_timm.py --config_path configs_unified_lr/dpn107/config_dpn107_CosFace_SGD_0.1_cosine.yaml #twins_svt_large/config_twins_svt_large_ArcFace_AdamW.yaml
#python src/fairness_train_timm.py --config_path configs/cait_xs24_384/config_cait_xs24_384_CosFace_adamw.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cait_xs24_384/config_cait_xs24_384_ArcFace_AdamW.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cait_xs24_384/config_cait_xs24_384_ArcFace_SGD.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cait_xs24_384/config_cait_xs24_384_CosFace_SGD.yaml 
#python src/fairness_train_timm.py --config_path configs_multi/cait_xs24_384/config_cait_xs24_384_MagFace_AdamW.yaml
#python src/fairness_train_timm.py --config_path configs_multi/cait_xs24_384/config_cait_xs24_384_MagFace_SGD.yaml
#python src/fairness_train_timm.py --config_path configs/ghostnet_100/config_ghostnet_100_CosFace_sgd.yaml
#python src/fairness_train_timm.py --config_path configs_multi/ghostnet_100/config_ghostnet_100_ArcFace_AdamW.yaml
#python src/fairness_train_timm.py --config_path configs_multi/ghostnet_100/config_ghostnet_100_ArcFace_SGD.yaml
#python src/fairness_train_timm.py --config_path configs_multi/ghostnet_100/config_ghostnet_100_CosFace_AdamW.yaml
#python src/fairness_train_timm.py --config_path configs_multi/ghostnet_100/config_ghostnet_100_MagFace_AdamW.yaml
#python src/fairness_train_timm.py --config_path configs_multi/ghostnet_100/config_ghostnet_100_MagFace_SGD.yaml 
#python src/fairness_train_timm.py --config_path configs/cspdarknet53/config_cspdarknet53_CosFace_adam.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cspdarknet53/config_cspdarknet53_ArcFace_AdamW.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cspdarknet53/config_cspdarknet53_ArcFace_SGD.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cspdarknet53/config_cspdarknet53_CosFace_AdamW.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cspdarknet53/config_cspdarknet53_CosFace_SGD.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cspdarknet53/config_cspdarknet53_MagFace_AdamW.yaml started
#python src/fairness_train_timm.py --config_path configs_multi/cspdarknet53/config_cspdarknet53_MagFace_SGD.yaml started
#python src/fairness_train_timm.py --config_path configs/resnet50/config_resnet50_MagFace_Adam.yaml
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_xs24_384  --input_size 384  --batch_size 64 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name efficientnet_b0  --input_size 112  --batch_size 64 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swin_large_patch4_window12_384_in22k --input_size 384 --batch_size 4 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name efficientnet_b1_pruned --input_size 112 --batch_size 250  --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name beit_large_patch16_384  --input_size 384 --batch_size 2  --seed 222
#python face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resnest269e --input_size 416 --batch_size 2 --seed 222
#end=$(date +%s.%N)
#runtime=$(python -c "print(${end} - ${start})")
#echo "Runtime was $runtime"
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name ig_resnext101_32x32d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name ig_resnext101_32x48d --input_size 112 --batch_size 2 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name legacy_seresnext26_32x4d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name seresnext50_32x4d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name ssl_resnext101_32x16d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name ssl_resnext101_32x4d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swsl_resnext101_32x16d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swsl_resnext101_32x8d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swsl_resnext50_32x4d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name tv_resnext50_32x4d --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resnetv2_152x2_bitm_in21k --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resnetv2_152x4_bitm --input_size 112 --batch_size 4 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resnetv2_152x4_bitm_in21k --input_size 112 --batch_size 4 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name dm_nfnet_f5 --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name dm_nfnet_f6 --input_size 112 --batch_size 4 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name tf_efficientnet_l2_ns --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name tf_efficientnet_l2_ns_475 --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_large_24_p8_224 --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_large_24_p8_224_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_large_24_p8_384_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_medium_24_p8_224 --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_medium_24_p8_224_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_medium_24_p8_384_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_small_12_p8_224_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_small_24_p8_224 --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_small_24_p8_224_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_small_24_p8_384_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name xcit_tiny_24_p8_384_dist --input_size 112 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name beit_large_patch16_224 --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name beit_large_patch16_224_in22k --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resmlp_big_24_224 --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resmlp_big_24_224_in22ft1k --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resmlp_big_24_distilled_224 --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_base_patch8_224 --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_base_patch8_224_in21k --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_huge_patch14_224_in21k --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_large_patch16_224 --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_large_r50_s32_224 --input_size 224 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name beit_large_patch16_384 --input_size 384 --batch_size 2 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_m36_384 --input_size 384 --batch_size 2 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_s24_384 --input_size 384 --batch_size 2 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_s36_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_xs24_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_xxs36_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swin_large_patch4_window12_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swin_large_patch4_window12_384_in22k --input_size 384 --batch_size 2 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_base_r50_s16_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_large_patch16_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_large_r50_s32_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name vit_small_r26_s32_384 --input_size 384 --batch_size 8 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resnest269e --input_size 416 --batch_size 2 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name beit_large_patch16_512 --input_size 512 --batch_size 4 --seed 222
#python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 KAIR/main_train_psnr.py --opt KAIR/options/swinir/train_hnas_classical.json  --dist True
