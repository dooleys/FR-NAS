import argparse
import os
import random
import yaml

import numpy as np
import itertools


def main(args):
    folder = f"{args.out_dir}/configs/{args.backbone}/"
    os.makedirs(folder, exist_ok=True)
    config = vars(args)
    hp_grid = {}
    for x in config.keys():
        if x == "head" or x == "optimizer":
            hp_grid[x] = config[x]
    for k in hp_grid.keys():
        del config[k]
    all_configs = list(itertools.product(*hp_grid.values()))
    print("Number of configs to create: ", len(all_configs))
    keys = hp_grid.keys()
    for c in all_configs:
        counter = 0
        for k in keys:
            config[k] = c[counter]
            counter = counter + 1
        backbone = config["backbone"]
        head = config["head"]
        optimizer = config["optimizer"]
        with open(folder + f"/config_{backbone}_{head}_{optimizer}.yaml",
                  "w") as fh:
            yaml.dump(config, fh)
        with open(folder + f"/config_{backbone}_{head}_{optimizer}.yaml",
                  "r") as ymlfile:
            cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)

if __name__ == "__main__":
    """This is executed when run from the command line"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone', default='resnet50')
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--project_name',
                        default="from-scratch_no-resampling_adam",
                        type=str)
    parser.add_argument('--head',
                        default=['CosFace', 'ArcFace', 'MagFace'],
                        type=list)
    parser.add_argument('--train_loss', default='Focal', type=str)
    parser.add_argument('--min_num_images', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--dropout', default=0.3, type=float)
    parser.add_argument('--optimizer', default=["Adam",'AdamW', 'SGD'], type=list)
    parser.add_argument('--scheduler', default='CosineAnnealingLR', type=str)
    parser.add_argument('--rho', default=0.9, type=float)
    parser.add_argument(
        '--eps', default=1e-06,
        type=float)  #differnt for diff opts, ReduceLROnPlateau has this too
    parser.add_argument('--initial_accumulator_value', default=0, type=float)
    parser.add_argument('--groups_to_modify', default= ['male', 'female'], type=str, nargs='+')
    parser.add_argument('--p_identities', default=[1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--p_images', default=[1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--betas', default=(0.9, 0.999), type=tuple)
    parser.add_argument('--etas', default=(0.5, 1.2), type=tuple)
    parser.add_argument('--step_sizes', default=(1e-06, 50), type=tuple)
    parser.add_argument('--dampening', default=0, type=float)
    parser.add_argument('--nesterov', default=False, type=bool)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--lr_decay', default=0, type=float)
    parser.add_argument('--lambd', default=0.0001, type=float)
    parser.add_argument('--alpha', default=0.75, type=float)
    parser.add_argument('--t0', default=1000000.0, type=float)
    parser.add_argument('--max_iter', default=20, type=int)
    parser.add_argument('--max_eval', default=None, type=int)
    parser.add_argument('--tolerance_grad', default=1e-07, type=float)
    parser.add_argument('--tolerance_change', default=1e-09, type=float)
    parser.add_argument('--momentum_decay', default=0.004, type=float)
    parser.add_argument('--history_size', default=100, type=int)
    parser.add_argument('--line_search_fn', default=None, type=str)
    parser.add_argument('--centered', default=False, type=bool)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--verbose', default=False, type=bool)
    parser.add_argument('--last_epoch', default=-1, type=int)
    parser.add_argument('--lr_lambda', default=[0.01, 0.02],
                        type=list)  #discuss can be function too
    parser.add_argument('--step_size', default=30, type=int)
    parser.add_argument('--gamma', default=0.1,
                        type=int)  #Used many times: have dict?
    parser.add_argument('--milestones', default=[30, 80], type=list)
    parser.add_argument('--factor', default=0.3333333333333333, type=float)
    parser.add_argument('--start_factor',
                        default=0.3333333333333333,
                        type=float)
    parser.add_argument('--end_factor', default=1.0, type=float)
    parser.add_argument('--T_max', default=1000000, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--eta_min', default=0.0, type=float)
    parser.add_argument('--min_lr', default=0.0, type=float)
    parser.add_argument('--total_steps', default=None, type=int)
    parser.add_argument('--steps_per_epoch', default=1000, type=int)
    parser.add_argument('--epochs', default=None, type=int)
    parser.add_argument('--pct_start', default=0.3, type=float)
    parser.add_argument('--anneal_strategy', default='cos', type=str)
    parser.add_argument('--cycle_momentum', default=True, type=bool)
    parser.add_argument('--base_momentum', default=0.85, type=float)
    parser.add_argument('--max_momentum', default=0.95, type=float)
    parser.add_argument('--div_factor', default=25.0, type=float)
    parser.add_argument('--final_div_factor', default=10000.0, type=float)
    parser.add_argument('--three_phase', default=False, type=bool)
    parser.add_argument('--base_lr', default=1e-4,
                        type=float) 
    parser.add_argument('--max_lr', default=3, type=float)
    parser.add_argument('--step_size_up', default=2000, type=int)
    parser.add_argument('--step_size_down', default=None, type=int)
    parser.add_argument('--scale_mode', default='cycle', type=str)
    parser.add_argument('--T_0', default=1000, type=int)
    parser.add_argument('--T_mult', default=1, type=int)
    parser.add_argument('--threshold', default=0.0001, type=float)
    parser.add_argument('--threshold_mode', default='rel', type=str)
    parser.add_argument('--cooldown', default=0, type=int)
    parser.add_argument('--mode', default='min', type=str)
    parser.add_argument('--total_iters', default=5, type=int)
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--name', default='CelebA', type=str)
    parser.add_argument('--dataset', default='CelebA', type=str)
    parser.add_argument('--file_name',
                        default='timm_from-scratch.csv',
                        type=str)
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--comet_api_key',
                        default="KKiKMVZI9RCYowoKDZDS5Y2km",
                        type=str)
    parser.add_argument('--comet_workspace', default="rsukthanker")
    parser.add_argument(
        '--checkpoints_root',
        default=
        "Checkpoints/timm_explore_few_epochs/",
        type=str)
    parser.add_argument('--metadata_file',
                        default="timm_model_metadata.csv",
                        type=str)
    parser.add_argument(
        '--demographics_file',
        default="CelebA/CelebA_demographics.txt",
        type=str)
    parser.add_argument(
        '--default_train_root',
        default=
        "/work/dlclarge2/sukthank-ZCP_Competition/FairNAS/FR-NAS/data/CelebA/Img/img_align_celeba_splits/train/",
        type=str)
    parser.add_argument(
        '--default_test_root',
        default=
        "/work/dlclarge2/sukthank-ZCP_Competition/FairNAS/FR-NAS/data/CelebA/Img/img_align_celeba_splits/test/",
        type=str)
    parser.add_argument('--out_dir', default=".", type=str)
    args = parser.parse_args()
    main(args)
