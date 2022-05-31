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
import math
from utils.utils import save_output_from_dict
from utils.utils_train import Network, get_head
from utils.fairness_utils import evaluate, add_column_to_file
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from rexnet_graph import  ReXNetV1
from timm.utils.model_ema import ModelEmaV2
import argparse
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids=range(torch.cuda.device_count())
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
import time
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
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
#def acc_overall(df):
#    return df['rank_by_id'].sum(axis=0)/df.shape[0]
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
def fairness_objective_rexnet(config, budget, **kwargs):
    '''parser = argparse.ArgumentParser()
    #parser.add_argument("--user_config", type=str)
    parser.add_argument('--train_loss', default='Focal', type=str)
    parser.add_argument('--min_num_images', default=3, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--groups_to_modify',
                        default=['male', 'female'],
                        type=str,
                        nargs='+')
    parser.add_argument('--p_identities',
                        default=[1.0, 1.0],
                        type=float,
                        nargs='+')
    parser.add_argument('--p_images',
                        default=[1.0, 1.0],
                        type=float,
                        nargs='+')
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--name', default='CelebA', type=str)
    parser.add_argument('--dataset', default='CelebA', type=str)
    parser.add_argument('--seed', default=222, type=int)
    parser.add_argument('--out_dir', default=".", type=str)
    parser.add_argument('--save_freq', default=1, type=int)
    parser.add_argument('--torchscript', dest='torchscript', action='store_true',
                        help='torch.jit.script the full model')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')

    # Optimizer parameters
    parser.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: None, use opt default)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Optimizer momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=2e-5,
                        help='weight decay (default: 2e-5)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--clip-mode', type=str, default='norm',
                        help='Gradient clipping mode. One of ("norm", "value", "agc")')
    parser.add_argument('--layer-decay', type=float, default=None,
                        help='layer-wise learning rate decay (default: None)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "step"')
    parser.add_argument('--lr', type=float, default=0.05, metavar='LR',
                        help='learning rate (default: 0.05)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                        help='learning rate cycle len multiplier (default: 1.0)')
    parser.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                        help='amount to decay each learning rate cycle (default: 0.5)')
    parser.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                        help='learning rate cycle limit, cycles enabled if > 1')
    parser.add_argument('--lr-k-decay', type=float, default=1.0,
                        help='learning rate k-decay for cosine/poly (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=0.0001, metavar='LR',
                        help='warmup learning rate (default: 0.0001)')
    parser.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                        help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
    parser.add_argument('--start-epoch', default=None, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--decay-epochs', type=float, default=100, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=3, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Regularization parameters

    parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                        help='Dropout rate (default: 0.)')
    parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                        help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
    parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                        help='Drop path rate (default: None)')
    parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                        help='Drop block rate (default: None)')

    # Batch norm parameters (only works with gen_efficientnet based models currently)
    parser.add_argument('--bn-momentum', type=float, default=None,
                        help='BatchNorm momentum override (if not None)')
    parser.add_argument('--bn-eps', type=float, default=None,
                        help='BatchNorm epsilon override (if not None)')
    parser.add_argument('--sync-bn', action='store_true',
                        help='Enable NVIDIA Apex or Torch synchronized BatchNorm.')
    parser.add_argument('--dist-bn', type=str, default='reduce',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
    parser.add_argument('--split-bn', action='store_true',
                        help='Enable separate BN layers per augmentation split.')

    args = parser.parse_args()'''
    with open("/work/dlclarge2/sukthank-ZCP_Competition/FR-NAS/configs/resnet50/config_resnet50_CosFace_Adam.yaml","r") as ymlfile:
        args = yaml.load(ymlfile, Loader=yaml.FullLoader)
    #for key, value in user_config.items():
    #    setattr(args, key, value)
    args = dotdict(args)
    args.opt = config["optimizer"]
    args.head = config["head"]
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

    ####################################################################################################################################
    # ======= data, model and test data =======#

    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_test = prepare_data(
        args)
    args.num_class = num_class
    # get model's embedding size
    layers=[config['layer1'],config['layer2'],config['layer3'],config['layer4'],config['layer5'],config['layer6']]
    backbone = ReXNetV1(num_classes=0,width_mult=2,ch_div=8,layers=layers)
    input=torch.ones(2,3,32,32)
    output=backbone(input)
    args.embedding_size= output.shape[-1]
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head,backbone= head.to(device), backbone.to(device)
    backbone = nn.DataParallel(backbone)
    ####################################################################################################################
    # ======= argsimizer =======#
    model = Network(backbone, head)

    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
    scheduler, num_epochs = create_scheduler(args, optimizer)
    model = model.to(device)
    #print(model_ema)
    epoch=0
    print('Start training')
    start=time.time()
    while epoch < 1:#int(budget):
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
                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                meters["loss"].update(loss.data.item(), inputs.size(0))
                meters["top5"].update(prec5.data.item(), inputs.size(0))
                #batch += 1
                #print(loss)
                break
            epoch=epoch+1
    backbone.eval()
    head.eval()
    k_accuracy = True
    multilabel_accuracy = True
    comp_rank = True
    loss_val, acc_val, acc_k_val, predicted_all_val, intra_val, inter_val, angles_intra_val, angles_inter_val, correct_val, nearest_id_val, labels_all_val, indices_all_val, demographic_all_val, rank_val = evaluate(dataloaders["val"],train_criterion,
                    model,
                    args.embedding_size,
                    k_accuracy=k_accuracy,
                    multilabel_accuracy=multilabel_accuracy,
                    demographic_to_labels=demographic_to_labels_test,
                    test=True, rank=comp_rank)
    rank_by_id_val = pd.DataFrame(np.array([list(indices_all_val),
                list(rank_val[:,1])]).T,
                columns=['ids','rank_by_id']).astype(int)
    metadata_val= pd.read_csv('val_identities_gender-expression_seed_222.csv')
    df_val = rank_by_id_val.merge(metadata_val)
    rank_diff_val=rank_func(df_val)
    acc_val=acc_overall(df_val)
    k_accuracy = True
    multilabel_accuracy = True
    comp_rank = True
    loss_test, acc_test, acc_k_test, predicted_all_test, intra_test, inter_test, angles_intra_test, angles_inter_test, correct_test, nearest_id_test, labels_all_test, indices_all_test, demographic_all_test, rank_test = evaluate(dataloaders["test"],train_criterion,
                    model,
                    args.embedding_size,
                    k_accuracy=k_accuracy,
                    multilabel_accuracy=multilabel_accuracy,
                    demographic_to_labels=demographic_to_labels_test,
                    test=True, rank=comp_rank)
    rank_by_id_test = pd.DataFrame(np.array([list(indices_all_test),
                list(rank_test[:,1])]).T,
                columns=['ids','rank_by_id']).astype(int)
    metadata_test= pd.read_csv('test_identities_gender-expression_seed_222.csv')
    df_test = rank_by_id_test.merge(metadata_test)
    rank_diff_test=rank_func(df_test)
    acc_test=acc_overall(df_test)
    cost=time.time()-start
    print("Rank disparity",rank_diff_val)
    print("Overall Acc", acc_val)
    overall_all_val = 0
    print(acc_k_val)
    for k in acc_k_val.keys():
        overall_acc_val+=acc_k_val[k]
    overall_acc = overall_acc/2
    print("overall acc test", overall_acc)
    print("Rank disparity Test",rank_diff_test)
    print("Overall Acc Test", acc_test)
    res = {
        "fitness": [-acc_val, rank_diff_val],
        "cost": cost,
        "info": {"test_loss":-acc_test, "budget": budget}
    }
    return res
