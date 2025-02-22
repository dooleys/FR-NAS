#from comet_ml import Experiment
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
from timm.utils.model_ema import ModelEmaV2
import argparse
import argparse
import os
import pickle
import time
from src.search.dpn107 import DPN
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from distributed import Client
from torchsummary import summary
from torchvision import transforms
from collections import OrderedDict
from functools import partial
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.search.operations import *
from timm.data import IMAGENET_DPN_MEAN, IMAGENET_DPN_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import BatchNormAct2d, ConvNormAct, create_conv2d, create_classifier
from timm.models.registry import register_model
import os
import pickle
import numpy as np
import random
import itertools
import torch
import torch.nn as nn
import math
import time
from argparse import ArgumentParser

import numpy as np

from syne_tune import Reporter
device = torch.device("cuda")

def rank_func(df):
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = (g_df['rank_by_id']).sum(axis=0)/g_df.shape[0]
    return abs(data['male']-data['female'])

def acc_by_gender(df):
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = g_df['rank_by_id'].sum(axis=0)/g_df.shape[0]
    return data['male'],data['female']

def calculate_disparity(df):
    data = {}
    for g,g_df in df.groupby('gender_expression'):
        data[g] = g_df['rank_by_id'].sum(axis=0)/g_df.shape[0]
    return math.abs(data['male']-data['female'])/100.0

def acc_overall(df):
    return (df['rank_by_id'] == 0).sum(axis=0)/df.shape[0]

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def fairness_objective_dpn(lr, edge1, edge2, edge3, head, optimizer):
    with open("search_configs/config_vggface.yaml","r") as ymlfile:
        args = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = dotdict(args)
    args.epochs = 1
    args.opt = optimizer
    args.head = head
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
    directory ="Checkpoints_synetune/Checkpoints_Layers_{}_LR_{}_Head_{}_Optimizer_{}/".format(str(edge1)+str(edge2)+str(edge3), lr, head, optimizer)
    if not os.path.exists(directory):
       os.makedirs(directory)
    args.batch_size=64
    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_val, _ = prepare_data(
        args)
    args.num_class = 7058
    args.num_workers = 10
    edges=[edge1,edge2,edge3]
    # Build model
    backbone = DPN(edges,num_init_features=128, k_r=200, groups=50, k_sec=(1,1,1,1), inc_sec=(20, 64, 64, 128))
    input=torch.ones(4,3,32,32)
    output=backbone(input)
    args.embedding_size= output.shape[-1]
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head,backbone= head.to(device), backbone.to(device)
    backbone = nn.DataParallel(backbone)
    model = Network(backbone, head)
    if (optimizer == "Adam") or (optimizer == "AdamW"):
        args.lr=lr
    if optimizer == "SGD":
        args.lr=lr
    print(args.lr)
    print(args.opt)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, num_epochs = create_scheduler(args, optimizer)
    model.to(device)
    epoch=0
    start = time.time()
    print('Start training')


    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))
    while epoch < int(1):
            model.train()  # set to training mode
            meters = {}
            meters["loss"] = AverageMeter()
            meters["top5"] = AverageMeter()
            for inputs, labels, _, _ in tqdm(iter((dataloaders["train"]))):
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
                #break
            checkpoint_name_to_save=directory+"model_{}.pth".format(str(epoch))
            epoch=epoch+1
            #break
    backbone.eval()
    head.eval()
    k_accuracy = True
    multilabel_accuracy = True
    comp_rank = True
    _, acc_val, _, _, _, _, _, _, _, _, _, indices_all_val, demographic_all_val, rank_val = evaluate(dataloaders["val"],train_criterion,
                    model,
                    args.embedding_size,
                    k_accuracy=k_accuracy,
                    multilabel_accuracy=multilabel_accuracy,
                    demographic_to_labels=demographic_to_labels_val,
                    test=True, rank=comp_rank)
    print(len(list(indices_all_val)))
    print(len(list(rank_val[:,1])))
    print(len(list(demographic_all_val)))
    df = {"ids":list(indices_all_val), "rank_by_id":list(np.array(rank_val[:,1].cpu().detach().numpy())), "gender_expression": list(demographic_all_val)}
    df_val= pd.DataFrame(data=df)
    #rank_diff_val=rank_func(df_val)#/(1+rank_func(df_val))
    def calculate_disparity(df):
        data = {}
        for g,g_df in df.groupby('gender_expression'):
            data[g] = g_df['rank_by_id'].sum(axis=0)/g_df.shape[0]
        return np.abs(data['male']-data['female'])/100.0
    disparity = calculate_disparity(df_val)
    acc_val=acc_overall(df_val)
    del df
    del df_val
    del dataloaders
    del model
    return {
        "rev_acc": 1-(acc_val), 
        "disparity": disparity,
        }

    

if __name__ == "__main__":
    # plot_function()
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--head", type=str, default="ArcFace")
    parser.add_argument("--edge1", type=int, default=3)
    parser.add_argument("--edge2", type=int, default=4)
    parser.add_argument("--edge3", type=int, default=1)
    args, _ = parser.parse_known_args()
    reporter = Reporter()
    y = fairness_objective_dpn(lr=args.lr, optimizer=args.optimizer, head=args.head, edge1=args.edge1, edge2=args.edge2, edge3=args.edge3)
    reporter(step=1, **y)  