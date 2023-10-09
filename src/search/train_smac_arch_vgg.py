

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
    set_seed(seed)
    with open("search_configs/config_vggface.yaml", "r") as ymlfile:
        args = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = dotdict(args)
    print(config)
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
    run_name = "Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_{}/".format(str(config["edge1"])+str(
        config["edge2"])+str(config["edge3"]), config["lr_sgd"], config["head"], config["optimizer"], str(seed))
    directory = "Checkpoints_vggface2/Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_{}/".format(str(
        config["edge1"])+str(config["edge2"])+str(config["edge3"]), config["lr_sgd"], config["head"], config["optimizer"], str(seed))
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_dir = "Checkpoints_vggface2/"
    args.batch_size = 64
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
    args.epochs = 10
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
        checkpoint_name_to_save = directory+"model_{}.pth".format(str(epoch))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_name_to_save)
        epoch = epoch+1
        checkpoint_name_to_save = directory+"model_{}.pth".format(str(epoch))
        torch.save(
            {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config
            }, checkpoint_name_to_save)
        k_accuracy = True
        multilabel_accuracy = True
        comp_rank = True
        loss_val, acc_val, acc_k_val, predicted_all_val, intra_val, inter_val, angles_intra_val, angles_inter_val, correct_val, nearest_id_val, labels_all_val, indices_all_val, demographic_all_val, rank_val = evaluate(dataloaders["val"], train_criterion,
                                                                                                                                                                                                                          model,
                                                                                                                                                                                                                          args.embedding_size,
                                                                                                                                                                                                                          k_accuracy=k_accuracy,
                                                                                                                                                                                                                          multilabel_accuracy=multilabel_accuracy,
                                                                                                                                                                                                                          demographic_to_labels=demographic_to_labels_val,
                                                                                                                                                                                                                          test=True, rank=comp_rank)
        kacc_df, multi_df, rank_by_image_df, rank_by_id_df = None, None, None, None
        if k_accuracy:
            kacc_df = pd.DataFrame(np.array([list(indices_all_val),
                                             list(nearest_id_val)]).T,
                                   columns=['ids', 'epoch_'+str(epoch)]).astype(int)
        if multilabel_accuracy:
            multi_df = pd.DataFrame(np.array([list(indices_all_val),
                                              list(predicted_all_val)]).T,
                                    columns=['ids', 'epoch_'+str(epoch)]).astype(int)
        if comp_rank:
            rank_by_image_df = pd.DataFrame(np.array([list(indices_all_val),
                                                      list(rank_val[:, 0])]).T,
                                            columns=['ids', 'epoch_'+str(epoch)]).astype(int)
            rank_by_id_df = pd.DataFrame(np.array([list(indices_all_val),
                                                   list(rank_val[:, 1])]).T,
                                         columns=['ids', 'epoch_'+str(epoch)]).astype(int)
        add_column_to_file(output_dir,
                           "val",
                           run_name,
                           epoch,
                           multi_df=multi_df,
                           kacc_df=kacc_df,
                           rank_by_image_df=rank_by_image_df,
                           rank_by_id_df=rank_by_id_df)
        loss_test, acc_test, acc_k_test, predicted_all_test, intra_test, inter_test, angles_intra_test, angles_inter_test, correct_test, nearest_id_test, labels_all_test, indices_all_test, demographic_all_test, rank_test = evaluate(dataloaders["test"], train_criterion,
                                                                                                                                                                                                                                        model,
                                                                                                                                                                                                                                        args.embedding_size,
                                                                                                                                                                                                                                        k_accuracy=k_accuracy,
                                                                                                                                                                                                                                        multilabel_accuracy=multilabel_accuracy,
                                                                                                                                                                                                                                        demographic_to_labels=demographic_to_labels_test,
                                                                                                                                                                                                                                        test=True, rank=comp_rank)
        kacc_df, multi_df, rank_by_image_df, rank_by_id_df = None, None, None, None
        if k_accuracy:
            kacc_df = pd.DataFrame(np.array([list(indices_all_test),
                                             list(nearest_id_test)]).T,
                                   columns=['ids', 'epoch_'+str(epoch)]).astype(int)
        if multilabel_accuracy:
            multi_df = pd.DataFrame(np.array([list(indices_all_test),
                                              list(predicted_all_test)]).T,
                                    columns=['ids', 'epoch_'+str(epoch)]).astype(int)
        if comp_rank:
            rank_by_image_df = pd.DataFrame(np.array([list(indices_all_test),
                                                      list(rank_test[:, 0])]).T,
                                            columns=['ids', 'epoch_'+str(epoch)]).astype(int)
            rank_by_id_df = pd.DataFrame(np.array([list(indices_all_test),
                                                   list(rank_test[:, 1])]).T,
                                         columns=['ids', 'epoch_'+str(epoch)]).astype(int)
        add_column_to_file(output_dir,
                           "test",
                           run_name,
                           epoch,
                           multi_df=multi_df,
                           kacc_df=kacc_df,
                           rank_by_image_df=rank_by_image_df,
                           rank_by_id_df=rank_by_id_df)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--seed', type=int, default=111, help='seed')
    args = argparser.parse_args()
    config = {
        "edge1": "3",
        "edge2": "0",
        "edge3": "1",
        "head": "CosFace",
        "optimizer": "SGD",
        "lr_sgd": 0.13828312564892567
    }
    fairness_objective_dpn(config, args.seed, 10)
