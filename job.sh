#!/bin/bash
export PYTHONPATH=.
start=$(date +%s.%N)
python src/fairness_train_timm.py --config_path configs/resnet50/config_resnet50_MagFace_Adam.yaml
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name cait_xs24_384  --input_size 384  --batch_size 64 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name efficientnet_b0  --input_size 112  --batch_size 64 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name swin_large_patch4_window12_384_in22k --input_size 384 --batch_size 4 --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name efficientnet_b1_pruned --input_size 112 --batch_size 250  --seed 222
#python3 face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name beit_large_patch16_384  --input_size 384 --batch_size 2  --seed 222
#python face.evoLVe.PyTorch/fairness_train_timm.py --backbone_name resnest269e --input_size 416 --batch_size 2 --seed 222
end=$(date +%s.%N)
runtime=$(python -c "print(${end} - ${start})")
echo "Runtime was $runtime"
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
