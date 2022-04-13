import argparse
import os
import random
import yaml

import numpy as np


def main(args):
    folder = f"{args.out_dir}/{args.backbone}/configs_single/"
    os.makedirs(folder, exist_ok=True)
    with open(folder + f"/config.yaml", "w") as fh:
        config=vars(args)
        print(config)
        yaml.dump(config, fh)
        print("Config created at:",folder + f"/config.yaml")
    #Test to check if we can read the written file
    with open(folder + f"/config.yaml", "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print(cfg)




if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--project_name', default="from-scratch_no-resampling_adam")
    parser.add_argument('--head_name', default='CosFace')
    parser.add_argument('--train_loss', default='Focal', type=str)
    parser.add_argument('--groups_to_modify', default= ['male', 'female'], type=str, nargs='+')
    parser.add_argument('--p_identities', default=[1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--p_images', default=[1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--min_num_images', default=3, type=int)
    parser.add_argument('--batch_size', default=250, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--scheduler', default="CosineAnnealingLR", type=str)
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--stages', default=[35, 65, 95], type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=3, type=int)
    parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='gpu id')
    parser.add_argument('--name', default='CelebA', type=str)
    parser.add_argument('--dataset', default='CelebA', type=str)
    parser.add_argument('--file_name', default='timm_from-scratch.csv', type=str)
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--comet_api_key',default="D1J58R7hYXPZzqZhrTIOe6GGQ",type=str)
    parser.add_argument('--comet_workspace',default="samueld")
    parser.add_argument('--checkpoints_root',default="/cmlscratch/sdooley1/rhea/FR-NAS/Checkpoints/timm_explore_few_epochs/",type=str)
    parser.add_argument('--metadata_file',default="/cmlscratch/sdooley1/timm_model_metadata.csv", type=str)
    parser.add_argument('--demographics_file',default="/cmlscratch/sdooley1/data/CelebA/CelebA_demographics.txt", type=str)
    parser.add_argument('--default_train_root',default="/cmlscratch/sdooley1/data/CelebA/Img/img_align_celeba_splits/train/", type=str)
    parser.add_argument('--default_test_root',default="/cmlscratch/sdooley1/data/CelebA/Img/img_align_celeba_splits/test/", type=str)
    parser.add_argument('--out_dir',default=".", type=str)
    args = parser.parse_args()
    main(args)
