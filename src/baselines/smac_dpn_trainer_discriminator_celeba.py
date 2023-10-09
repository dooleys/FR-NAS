# from comet_ml import Experiment
from src.utils.utils import get_val_data, separate_resnet_bn_paras, warm_up_lr, \
    schedule_lr, AverageMeter, accuracy, load_checkpoint
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import yaml
from src.loss.focal import FocalLoss
from src.utils.utils import AverageMeter, accuracy
from src.utils.fairness_utils import evaluate
from src.utils.data_utils_balanced import prepare_data
import numpy as np
import pandas as pd
import random
from src.utils.utils_train_angular import Network, get_head
from src.utils.fairness_utils import evaluate, add_column_to_file
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
import argparse
import os
import torch.optim as optim
import time
from src.search.dpn107 import DPN
import numpy as np
from src.head.metrics import CosFace, SphereFace, ArcFace, MagFace
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


class Network_Discriminator(torch.nn.Module):

    def __init__(self, backbone, head, head_sens):
        super(Network_Discriminator, self).__init__()
        self.head = head
        self.backbone = backbone
        self.head_sens = head_sens

    def forward(self, inputs, labels):
        features = self.head_sens(self.backbone(inputs))
        outputs, reg_loss = self.head(features, labels)

        return outputs, reg_loss


def get_head(args):
    if args.head == "CosFace":
        head = CosFace(in_features=args.embedding_size, out_features=args.num_class,
                       device_id=range(torch.cuda.device_count()))
    elif args.head == "ArcFace":
        head = ArcFace(in_features=args.embedding_size, out_features=args.num_class,
                       device_id=range(torch.cuda.device_count()))
    elif args.head == "SphereFace":
        head = SphereFace(in_features=args.embedding_size,
                          out_features=args.num_class, device_id=range(torch.cuda.device_count()))
    elif args.head == "MagFace":
        head = MagFace(in_features=args.embedding_size, out_features=args.num_class,
                       device_id=range(torch.cuda.device_count()))
    else:
        print("Head not supported")
        return
    return head


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)/1e6


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
    with open("search_configs/config_celeba.yaml", "r") as ymlfile:
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
    if (config["optimizer"] == "Adam") or (config["optimizer"] == "AdamW"):
        args.lr = config["lr_adam"]
        lr_key = "lr_adam"
    if config["optimizer"] == "SGD":
        args.lr = config["lr_sgd"]
        lr_key = "lr_sgd"
    run_name = "Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_/".format(str(config["edge1"])+str(
        config["edge2"])+str(config["edge3"]), config[lr_key], config["head"], config["optimizer"])
    directory = "Checkpoints_celeba/Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}_/".format(str(
        config["edge1"])+str(config["edge2"])+str(config["edge3"]), config[lr_key], config["head"], config["optimizer"])
    if not os.path.exists(directory):
        os.makedirs(directory)
    output_dir = "Checkpoints_celeba/"
    args.batch_size = 64
    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_val, demographic_to_labels_test = prepare_data(
        args)
    args.num_class = num_class
    edges = [config['edge1'], config['edge2'], config['edge3']]
    # Build model
    backbone = DPN(edges, num_init_features=128, k_r=200,
                   groups=50, k_sec=(1, 1, 1, 1), inc_sec=(20, 64, 64, 128))
    input = torch.ones(4, 3, 32, 32)
    output = backbone(input)
    args.embedding_size = output.shape[-1]
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    sens_criterion = torch.nn.CrossEntropyLoss()
    head, backbone = head.to(device), backbone.to(device)
    print(count_parameters_in_MB(head))
    print(count_parameters_in_MB(backbone))
    # model.load_state_dict(torch.load("/work/dlclarge2/sukthank-ZCP_Competition/models_pareto/680/model_99.pth")["model_state_dict"])
    # print("model_loaded")

    print(args.lr)
    print(args.opt)
    args.one_layer_sens = False
    args.only_discriminator = False
    args.manipulate = 1
    if args.one_layer_sens:
        head_sens = nn.Linear(in_features=args.embedding_size, out_features=2)
    else:
        head_sens = nn.Sequential(nn.Linear(in_features=args.embedding_size, out_features=args.embedding_size),
                                  nn.ReLU(),
                                  nn.Linear(in_features=args.embedding_size,
                                            out_features=args.embedding_size),
                                  nn.ReLU(),
                                  nn.Linear(in_features=args.embedding_size, out_features=2))

    _, head_paras_wo_bn = separate_resnet_bn_paras(head)

    if args.only_discriminator:
        optimizer = optim.SGD([{'params': head_sens.parameters(
        ), 'weight_decay': args.weight_decay, 'lr': args.lr_sens}], momentum=args.momentum)
    else:
        sensitive_net = nn.Sequential(nn.Linear(in_features=args.embedding_size, out_features=args.embedding_size),
                                      nn.ReLU(),
                                      nn.Linear(
                                          in_features=args.embedding_size, out_features=args.embedding_size),
                                      nn.ReLU(),
                                      nn.Linear(
                                          in_features=args.embedding_size, out_features=args.embedding_size),
                                      nn.ReLU(),
                                      nn.Linear(in_features=args.embedding_size, out_features=args.embedding_size))

        # backbone = nn.Sequential(backbone, sensitive_net)
        # optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

        if args.opt == "SGD":
            optimizer = optim.SGD([{'params': head_sens.parameters(), 'weight_decay': args.weight_decay, 'lr': 0.01},
                                   {'params': sensitive_net.parameters(
                                   ), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                   {'params': backbone.parameters(
                                   ), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                   {'params': head_paras_wo_bn, 'weight_decay': args.weight_decay, 'lr': args.lr}], momentum=args.momentum)
        elif args.opt == "Adam":
            optimizer = optim.Adam([{'params': head_sens.parameters(), 'weight_decay': args.weight_decay, 'lr': 0.001},
                                    {'params': sensitive_net.parameters(
                                    ), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                    {'params': backbone.parameters(
                                    ), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                    {'params': head_paras_wo_bn, 'weight_decay': args.weight_decay, 'lr': args.lr}])
        elif args.opt == "AdamW":
            optimizer = optim.Adam([{'params': head_sens.parameters(), 'weight_decay': args.weight_decay, 'lr': 0.001},
                                    {'params': sensitive_net.parameters(
                                    ), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                    {'params': backbone.parameters(
                                    ), 'weight_decay': args.weight_decay, 'lr': args.lr},
                                    {'params': head_paras_wo_bn, 'weight_decay': args.weight_decay, 'lr': args.lr}])
        scheduler, num_epochs = create_scheduler(args, optimizer)
    head_sens = head_sens.to(device)
    backbone = nn.DataParallel(backbone)
    backbone = backbone.to(device)
    model = Network_Discriminator(backbone, head, sensitive_net)
    model.to(device)
    epoch = 0
    start = time.time()
    dict_dem = {"male": 1, "female": -1}
    print('Start training')
    num_epoch_warm_up = budget // 25  # use the first 1/25 epochs to warm up
    # use the first 1/25 epochs to warm up
    args.alpha = 1
    num_batch_warm_up = len(dataloaders['train']) * num_epoch_warm_up
    while epoch < int(budget):
        model.train()  # set to training mode
        head_sens.train()
        meters = {}
        meters["loss"] = AverageMeter()
        meters["top5"] = AverageMeter()
        meters["sens_losses"] = AverageMeter()
        meters["sens_accs"] = AverageMeter()
        batch = 0
        for inputs, labels, sens_attr, _ in tqdm(iter((dataloaders["train"]))):
            batch = batch+1
            inputs, labels = inputs.to(device), labels.to(device).long()
            sens_attr = torch.Tensor([dict_dem[s] for s in list(sens_attr)])
            protected = torch.tensor(
                sens_attr == -1).type(torch.long).to(device)
            outputs, reg_loss = model(inputs, labels)
            features_sens = F.normalize(
                model.head_sens(model.backbone(inputs)))
            outputs_sens = head_sens(features_sens)
            softmax_scores = nn.Softmax()(outputs_sens)

            if args.manipulate == 1:
                # manipulate males
                regularization = torch.log(
                    1 + torch.abs(0.9 - softmax_scores[:, 1])).mean()
            elif args.manipulate == 0:
                # manipulate females
                regularization = torch.log(
                    1 + torch.abs(0.9 - softmax_scores[:, 0])).mean()
            else:
                print('Manipulate arg is wrong')

            outputs_sens_detached = head_sens(features_sens.detach())
            sens_loss = sens_criterion(outputs_sens_detached, protected)
            loss = train_criterion(outputs, labels) + reg_loss
            loss = loss.mean()
            if args.only_:
                optimizer.zero_grad()

            elif (batch + 1 <= num_batch_warm_up):
                total_loss = loss
                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
            else:
                total_loss = loss + args.alpha * regularization

                optimizer.zero_grad()
                total_loss.backward(retain_graph=True)
                for param in head_sens.parameters():
                    param.grad = torch.zeros_like(param.grad)

            sens_loss.backward()
            # optimizer.step()
            scheduler.step(epoch + 1, meters["top5"])
            prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
            meters["loss"].update(loss.data.item(), inputs.size(0))
            meters["top5"].update(prec5.data.item(), inputs.size(0))
            # break
        # break
        epoch = 1234
        if epoch == 1234:
            checkpoint_name_to_save = directory + \
                "model_{}.pth".format(str(epoch))
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': config
                }, checkpoint_name_to_save)
        # checkpoint_name_to_save=directory+"model_{}.pth".format(str(epoch))
        # torch.save(
        #    {
        #        'epoch': epoch,
        #        'model_state_dict': model.state_dict(),
        #        'optimizer_state_dict': optimizer.state_dict(),
        #        'config': config
        #    }, checkpoint_name_to_save)
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
        # break
        epoch = epoch+1


if __name__ == "__main__":
    # add argparser
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str,
                           default="config1", help='CelebA config name')
    argparser.add_argument('--seed', type=int, default=111, help='seed')
    args = argparser.parse_args()
    if args.config == "config1":
        config = {
            'edge1': 0,
            'edge2': 0,
            'edge3': 0,
            'head': 'CosFace',
            'lr_sgd': 0.2813375341651194,
            'optimizer': 'SGD', }
    else:
        config = {
            'edge1': 0,
            'edge2': 1,
            'edge3': 0,
            'head': 'CosFace',
            'lr_adam': 0.32348738788346576,
            'optimizer': 'SGD', }
    fairness_objective_dpn(config, 0, 100)
