#from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from loss.focal import FocalLoss
from utils.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, \
    schedule_lr, AverageMeter, accuracy
from utils.fairness_utils import evaluate
from utils.data_utils_balanced import prepare_data
from utils.utils_train import Network
import numpy as np
import pandas as pd
import random
import timm
import random
import math
from utils.utils import save_output_from_dict
from utils.utils_train_old import Network, get_head
from utils.fairness_utils import evaluate, add_column_to_file
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
import argparse
import argparse
import os
import time
from dpn107 import DPN
import numpy as np
import torch.optim as optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *
import os
import numpy as np
import random

device = torch.device("cuda")
seed = 555
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def count_parameters_in_MB(model):
  return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6

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



def acc_overall(df):
    return (df['rank_by_id'] == 0).sum(axis=0)/df.shape[0]

def get_identities_dict(dataloader):
    identities_dict = {"male":[],"female":[]}
    dict_dem ={"male":1,"female":-1}
    count = 0
    for inputs, labels, sens_attr, _ in tqdm(iter((dataloader))):
        sens_attr = [dict_dem[s] for s in sens_attr]
        for i in range(len(sens_attr)):
            if sens_attr[i]==1:
                if labels[i] not in identities_dict["male"]:
                   identities_dict["male"].append(int(labels[i]))
            else:
                if labels[i] not in identities_dict["female"]:
                   identities_dict["female"].append(int(labels[i]))
        #break
    return identities_dict

def flip(labels,to_labels,fraction,sense_attr):
    ids_male = []
    for i in range(len(sense_attr)):
        if sense_attr[i]==1:
           ids_male.append(i)
    n = int(fraction*len(ids_male))
    index = list(np.random.choice(ids_male, n, replace=False))
    for i in range(len(index)):
        while True:
           current_choice = random.choice(to_labels)
           if current_choice!=labels[index[i]]:
              break
        labels[index[i]] = current_choice
    return torch.Tensor(labels).long()

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def fairness_objective_dpn(config, seed, budget):
    with open("/work/dlclarge2/sukthank-ZCP_Competition/FR-NAS/configs/dpn107/config_dpn107_CosFace_sgd.yaml","r") as ymlfile:
        args = yaml.load(ymlfile, Loader=yaml.FullLoader)
    args = dotdict(args)
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
    print(p_identities)
    args.p_images = p_images
    args.p_identities = p_identities

    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))
    run_name = "Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_flipped/".format(str(config["edge1"])+str(config["edge2"])+str(config["edge3"]), config["lr_sgd"], config["head"],config["optimizer"])
    directory ="Checkpoints_scratch/Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_flipped/".format(str(config["edge1"])+str(config["edge2"])+str(config["edge3"]), config["lr_sgd"], config["head"],config["optimizer"])
    if not os.path.exists(directory):
       os.makedirs(directory)
    output_dir="Checkpoints_scratch/"
    args.batch_size=64
    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_test, demographic_to_labels_val = prepare_data(
        args)
    args.num_class = num_class
    edges=[config['edge1'],config['edge2'],config['edge3']]
    # Build model
    backbone = DPN(edges,num_init_features=128, k_r=200, groups=50, k_sec=(1,1,1,1), inc_sec=(20, 64, 64, 128))
    input=torch.ones(4,3,32,32)
    output=backbone(input)
    args.embedding_size= output.shape[-1]
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head,backbone= head.to(device), backbone.to(device)
    print(count_parameters_in_MB(head))
    print(count_parameters_in_MB(backbone))
    backbone = nn.DataParallel(backbone)
    model = Network(backbone, head)
    #model.load_state_dict(torch.load("/work/dlclarge2/sukthank-ZCP_Competition/models_pareto/680/model_99.pth")["model_state_dict"])
    print("model_loaded")
    if (config["optimizer"] == "Adam") or (config["optimizer"] == "AdamW"):
        args.lr=config["lr_adam"]
    if config["optimizer"] == "SGD":
        args.lr=config["lr_sgd"]
    print(args.lr)
    print(args.opt)
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, num_epochs = create_scheduler(args, optimizer)
    model.to(device)
    epoch=0
    start = time.time()
    dict_dem ={"male":1,"female":-1}
    identities_dict = get_identities_dict(dataloaders["train"])
    print('Start training')
    while epoch < int(budget):
            model.train()  # set to training mode
            meters = {}
            meters["loss"] = AverageMeter()
            meters["top5"] = AverageMeter()
            for inputs, labels, sens_attr, _ in tqdm(iter((dataloaders["train"]))):
                #print(labels.shape)
                sens_attr = [dict_dem[s] for s in list(sens_attr)]
                #print("labels before",labels)
                labels_after = flip(list(labels.long().numpy()),identities_dict["male"],0.3,sens_attr)
                #print("Flipped")
                #print("labels after",labels_after)
                #print(torch.sum(labels==labels_after))
                inputs, labels = inputs.to(device), labels_after.to(device).long()
                outputs, reg_loss = model(inputs, labels, torch.Tensor(sens_attr))
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

            if epoch==99:
             checkpoint_name_to_save=directory+"model_{}.pth".format(str(epoch))
             torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config
                }, checkpoint_name_to_save)
            #epoch=epoch+1
            #checkpoint_name_to_save=directory+"model_{}.pth".format(str(epoch))
            #torch.save(
            #    {
            #        'epoch': epoch,
            #        'model_state_dict': model.state_dict(),
            #        'optimizer_state_dict': optimizer.state_dict(),
            #        'config': config
            #    }, checkpoint_name_to_save)
            k_accuracy = True
            multilabel_accuracy = True
            comp_rank = True
            loss_val, acc_val, acc_k_val, predicted_all_val, intra_val, inter_val, angles_intra_val, angles_inter_val, correct_val, nearest_id_val, labels_all_val, indices_all_val, demographic_all_val, rank_val = evaluate(dataloaders["val"],train_criterion,
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
                                       columns=['ids','epoch_'+str(epoch)]).astype(int)
            if multilabel_accuracy:
                multi_df = pd.DataFrame(np.array([list(indices_all_val),
                                                  list(predicted_all_val)]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
            if comp_rank:
                rank_by_image_df = pd.DataFrame(np.array([list(indices_all_val),
                                                  list(rank_val[:,0])]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
                rank_by_id_df = pd.DataFrame(np.array([list(indices_all_val),
                                                  list(rank_val[:,1])]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
            add_column_to_file(output_dir,
                               "val",
                               run_name,
                               epoch,
                               multi_df = multi_df,
                               kacc_df = kacc_df,
                               rank_by_image_df = rank_by_image_df,
                               rank_by_id_df = rank_by_id_df)
            loss_test, acc_test, acc_k_test, predicted_all_test, intra_test, inter_test, angles_intra_test, angles_inter_test, correct_test, nearest_id_test, labels_all_test, indices_all_test, demographic_all_test, rank_test = evaluate(dataloaders["test"],train_criterion,
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
                                       columns=['ids','epoch_'+str(epoch)]).astype(int)
            if multilabel_accuracy:
                multi_df = pd.DataFrame(np.array([list(indices_all_test),
                                                  list(predicted_all_test)]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
            if comp_rank:
                rank_by_image_df = pd.DataFrame(np.array([list(indices_all_test),
                                                  list(rank_test[:,0])]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
                rank_by_id_df = pd.DataFrame(np.array([list(indices_all_test),
                                                  list(rank_test[:,1])]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
            add_column_to_file(output_dir,
                               "test",
                               run_name,
                               epoch,
                               multi_df = multi_df,
                               kacc_df = kacc_df,
                               rank_by_image_df = rank_by_image_df,
                               rank_by_id_df = rank_by_id_df)
            #break
            #if epoch==2:
            #    break
            epoch=epoch+1

if __name__ == "__main__":

 #SMAC config
 config ={
  'edge1': 0,
  'edge2': 0,
  'edge3': 0,
  'head': 'CosFace',
  'lr_sgd': 0.2813375341651194,
  'optimizer': 'SGD',
}
 '''config={
  'edge1': 6,
  'edge2': 8,
  'edge3': 0,
  'head': 'CosFace',
  'lr_adam': 0.0006048015915653069,
  'optimizer': 'Adam',
 }'''
 #SMAC config 2
 ''' config={
  'edge1': 0,
  'edge2': 1,
  'edge3': 0,
  'head': 'CosFace',
  'lr_sgd': 0.32348738788346576,
  'optimizer': 'SGD',
 }
 #SMAC config 3
 config={
  'edge1': 6,
  'edge2': 8,
  'edge3': 0,
  'head': 'CosFace',
  'lr_adam': 0.0006048015915653069,
  'optimizer': 'Adam',
 }
 config ={
  'edge1': 0,
  'edge2': 0,
  'edge3': 0,
  'head': 'CosFace',
  'lr_sgd': 0.2813375341651194,
  'optimizer': 'SGD',
}
 #Config 4
 config = {
  'edge1': 4,
  'edge2': 6,
  'edge3': 7,
  'head': 'MagFace',
  'lr_sgd': 0.21727296500399912,
  'optimizer': 'SGD',}
# config 5
 config =  {
      "edge1": 5,
      "edge2": 2,
      "edge3": 0,
      "head": "CosFace",
      "optimizer": "SGD",
      "lr_sgd": 0.12889657714325153
    }
# dpn smac acc 95.9
 config = {
      "edge1": 4,
      "edge2": 0,
      "edge3": 3,
      "head": "CosFace",
      "optimizer": "SGD",
      "lr_sgd": 0.2523152758955039
    }
 '''
 # dpn smac acc 96
 #config = {
 #      "edge1": 1,
 #      "edge2": 5,
 #      "edge3": 2,
 #      "head": "CosFace",
 #      "optimizer": "Adam",
 #      "lr_adam": 0.0006383109384330796
 #    }
 #config = {"edge1": 0, "edge2": 4, "edge3": 0, "head": "CosFace", "optimizer": "Adam", "lr_adam": 0.0005900596101948813}
 #config = {
 #    'edge1': 3,
 #  'edge2': 0,
 #  'edge3': 1,
 #  'head': 'CosFace',
 #  'lr_adam': 0.0002097247501603349,
 #  'optimizer': 'Adam',
 #}
 fairness_objective_dpn(config,0,100)
