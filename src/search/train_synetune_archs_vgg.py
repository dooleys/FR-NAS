

# from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.loss.focal import FocalLoss
from src.utils.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, \
    schedule_lr, AverageMeter, accuracy
from src.utils.fairness_utils import evaluate
from src.utils.data_utils_balanced import prepare_data
from src.utils.utils_train import Network
import numpy as np
import pandas as pd
import random
import timm
import math
from src.utils.utils import save_output_from_dict
from src.utils.utils_train import Network, get_head
from src.utils.fairness_utils import evaluate, add_column_to_file
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
import argparse
import argparse
import os
import time
from src.search.dpn107 import DPN
import numpy as np
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.search.operations import *
import os
import numpy as np
import random
device = torch.device("cuda")


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def rank_func(df):
    data = {}
    for g, g_df in df.groupby('gender_expression'):
        data[g] = (g_df['rank_by_id']).sum(axis=0)/g_df.shape[0]
    return abs(data['male']-data['female'])


def acc_by_gender(df):
    data = {}
    for g, g_df in df.groupby('gender_expression'):
        data[g] = g_df['rank_by_id'].sum(axis=0)/g_df.shape[0]
    return data['male'], data['female']


def acc_overall(df):
    return (df['rank_by_id'] == 0).sum(axis=0)/df.shape[0]


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def fairness_objective_dpn(config, seed, budget):
    with open("search_configs/config_vggface.yaml", "r") as ymlfile:
        args = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = dotdict(args)
    print(config)
    args.epochs = 10
    args.opt = config["optimizer"]
    args.head = config["head"]
    print(args)
    p_images = {
        args.groups_to_modify[i]: args.p_images[i]
        for i in range(len(args.groups_to_modify))
    }
    p_identities = {
        args.groups_to_modify[i]: args.p_identities[i]
        for i in range(len(args.groups_to_modify))
    }
    args.p_images = p_images
    args.p_identities = p_identities

    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))
    run_name = "Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_seed{}_2/".format(str(config["edge1"])+str(
        config["edge2"])+str(config["edge3"]), config["lr_adam"], config["head"], config["optimizer"], str(seed))
    directory = "Checkpoints_moasha/Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_seed_{}_2/".format(str(config["edge1"])+str(
        config["edge2"])+str(config["edge3"]), config["lr_adam"], config["head"], config["optimizer"], str(seed))
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_dir = "Checkpoints_moasha/"
    args.batch_size = 256
    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_val, demographic_to_labels_test = prepare_data(
        args)
    args.num_class = 7058

    edges = [int(config['edge1']), int(config['edge2']), int(config['edge3'])]
    # Build model
    backbone = DPN(edges, num_init_features=128, k_r=200,
                   groups=50, k_sec=(1, 1, 1, 1), inc_sec=(20, 64, 64, 128))
    input = torch.ones(4, 3, 32, 32)
    output = backbone(input)
    args.embedding_size = output.shape[-1]
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head, backbone = head.to(device), backbone.to(device)
    backbone = nn.DataParallel(backbone)
    model = Network(backbone, head)
    if (config["optimizer"] == "Adam") or (config["optimizer"] == "AdamW"):
        args.lr = config["lr_adam"]
    if config["optimizer"] == "SGD":
        args.lr = config["lr_sgd"]
    print(args.lr)
    print(args.opt)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, num_epochs = create_scheduler(args, optimizer)
    model.to(device)
    epoch = 0

    budget = args.epochs
    start = time.time()
    print('Start training')
    while epoch < int(budget):
        model.train()  # set to training mode
        meters = {}
        meters["loss"] = AverageMeter()
        meters["top5"] = AverageMeter()
        for inputs, labels, sens_attr, _ in tqdm(iter((dataloaders["train"]))):
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs, reg_loss = model(inputs, labels)
            loss = train_criterion(outputs, labels) + reg_loss
            loss = loss.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + 1, meters["top5"])
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            meters["loss"].update(loss.data.item(), inputs.size(0))
            meters["top5"].update(prec5.data.item(), inputs.size(0))
            # break
        backbone.eval()  # set to testing mode
        head.eval()
        k_accuracy = True
        multilabel_accuracy = True
        comp_rank = True
        loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all, rank = evaluate(
            dataloaders["test"],
            train_criterion,
            model,
            args.embedding_size,
            k_accuracy=k_accuracy,
            multilabel_accuracy=multilabel_accuracy,
            demographic_to_labels=demographic_to_labels_test,
            test=True, rank=comp_rank)
        rank_df = pd.DataFrame(np.insert(rank.numpy(), 0, indices_all, axis=1),
                                   columns=["index", "nearest_by_img", "nearest_by_id", "closest_point", "closest_same_id"])
        pickle_file = checkpoint_name_to_save = os.path.join(directory,
                                                                 "Checkpoint_Head_{}_Backbone_{}_Opt_{}_Dataset_{}_Epoch_{}_test.pkl"
                                                                 .format(args.head, args.backbone, args.opt, args.name,
                                                                         str(epoch+1)))
        rank_df.to_pickle(pickle_file)
        results = {}
        results['Model'] = args.backbone
        results['config_file'] = args.config_path
        results['seed'] = args.seed
        results['epoch'] = epoch
        for k in acc_k.keys():
            results['Acc multi '+k] = (round(acc[k].item()*100, 3))
            results['Acc k '+k] = (round(acc_k[k].item()*100, 3))
            results['Intra '+k] = (round(intra[k], 3))
            results['Inter '+k] = (round(inter[k], 3))

        #print(results)
        k_accuracy = True
        multilabel_accuracy = True
        comp_rank = True
        loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all, rank = evaluate(
            dataloaders["val"],
            train_criterion,
            model,
            args.embedding_size,
            k_accuracy=k_accuracy,
            multilabel_accuracy=multilabel_accuracy,
            demographic_to_labels=demographic_to_labels_val,
            test=True, rank=comp_rank)
        rank_df = pd.DataFrame(np.insert(rank.numpy(), 0, indices_all, axis=1),
                                   columns=["index", "nearest_by_img", "nearest_by_id", "closest_point", "closest_same_id"])
        pickle_file = checkpoint_name_to_save = os.path.join(directory,
                                                                 "Checkpoint_Head_{}_Backbone_{}_Opt_{}_Dataset_{}_Epoch_{}_val.pkl"
                                                                 .format(args.head, args.backbone, args.opt, args.name,
                                                                         str(epoch+1)))
        rank_df.to_pickle(pickle_file)
        results = {}
        results['Model'] = args.backbone
        results['config_file'] = args.config_path
        results['seed'] = args.seed
        results['epoch'] = epoch
        for k in acc_k.keys():
            results['Acc multi '+k] = (round(acc[k].item()*100, 3))
            results['Acc k '+k] = (round(acc_k[k].item()*100, 3))
            results['Intra '+k] = (round(intra[k], 3))
            results['Inter '+k] = (round(inter[k], 3))

        print(results)
#         save_output_from_dict(args.RFW_checkpoints_root, results, args.file_name)

        epoch += 1

        # save checkpoints per epoch
#             if (epoch == args.epochs) or (epoch % args.save_freq == 0):
        checkpoint_name_to_save = os.path.join(directory,
                                               "Checkpoint_Head_{}_Backbone_{}_Opt_{}_Dataset_{}_Epoch_{}.pth"
                                               .format(args.head, args.backbone, args.opt, args.name,
                                                       str(epoch)))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_name_to_save)
        # remove the previous checkpoint in certain instances
        if ((epoch-1)):
            prev_checkpoint_name_to_save = os.path.join(
                directory,
                "Checkpoint_Head_{}_Backbone_{}_Opt_{}_Dataset_{}_Epoch_{}.pth"
                .format(args.head, args.backbone, args.opt, args.name,
                        str(epoch-1)))
            os.remove(prev_checkpoint_name_to_save)


if __name__ == "__main__":

    # add config and seed as args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='moasha')
    parser.add_argument('--seed', type=int, default=111)
    args = parser.parse_args()
    if args.config == 'moasha':
        config = {
            "edge1": "1",
            "edge2": "0",
            "edge3": "8",
            "head": "CosFace",
            "optimizer": "Adam",
            "lr_adam": 0.000100
        }
    elif args.config == 'nsga2':
        config = {
            "edge1": "7",
            "edge2": "2",
            "edge3": "8",
            "head": "MagFace",
            "optimizer": "SGD",
            "lr_sgd": 0.5450347031001771}
    seed = args.seed
    set_seed(seed)
    fairness_objective_dpn(config, seed, 10)