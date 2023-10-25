import numpy as np
import bcolz
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from src.utils.utils_train import Network, get_head
from src.utils.utils import  get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
import timm
import argparse

from src.evaluation.model_factory import get_model_path
def get_pair(root, name):
    carray = bcolz.carray(rootdir = os.path.join(root, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(root, name))
    return carray, issame

def get_data(data_root):
    agedb_30, agedb_30_issame = get_pair("/work/dlclarge2/sukthank-ZCP_Competition/fairnas_backup/facerec_data/agedb_30/", 'agedb_30')
    cfp_fp, cfp_fp_issame = get_pair("/work/dlclarge2/sukthank-ZCP_Competition/fairnas_backup/facerec_data/cfp_fp", 'cfp_fp')
    lfw, lfw_issame = get_pair("/work/dlclarge2/sukthank-ZCP_Competition/fairnas_backup/facerec_data/lfw", 'lfw')
    calfw, calfw_issame = get_pair("/work/dlclarge2/sukthank-ZCP_Competition/fairnas_backup/facerec_data/calfw", 'calfw')
    cplfw, cplfw_issame = get_pair("/work/dlclarge2/sukthank-ZCP_Competition/fairnas_backup/facerec_data/cplfw", 'cplfw')
    cfp_ff, cfp_ff_issame = get_pair("/work/dlclarge2/sukthank-ZCP_Competition/fairnas_backup/facerec_data/cfp_ff", 'cfp_ff')
    return agedb_30, cfp_fp, cfp_ff, lfw, calfw, cplfw, agedb_30_issame, cfp_fp_issame, cfp_ff_issame, lfw_issame, calfw_issame, cplfw_issame

# define argparse
parser = argparse.ArgumentParser(description = 'Evaluation on different datasets')
parser.add_argument('--backbone', type = str, default = 'dpn107', help = 'model type to evaluate')
parser.add_argument('--head', type = str, default = 'CosFace', help = 'head type to evaluate')
parser.add_argument('--dataset', type = str, default = 'vggface2', help = 'dataset the model is trained on')
parser.add_argument('--optimizer', type = str, default = 'AdamW', help = 'optimzier used to change the model')
args = parser.parse_args()
if not "smac" in args.backbone:
    backbone = timm.create_model(args.backbone, pretrained=False, num_classes=0)
    backbone = backbone.to("cuda")
    input_dummy = torch.randn(4, 3, 112, 112).cuda()
    output_dummy = backbone(input_dummy)
    emb_size = output_dummy.shape[-1]
    args.embedding_size = emb_size
    args.num_class = 7058
    head = get_head(args)
    backbone = nn.DataParallel(backbone)
    model = Network(backbone, head)
else:
    from src.search.dpn107 import DPN
    if args.backbone == "smac_301":
        choices = [3,0,1]
    elif args.backbone == "smac_000":
        choices = [0,0,0]
    elif args.backbone == "smac_010":
        choices = [0,1,0]
    elif args.backbone == "smac_680":
        choices = [6,8,0]
    model = DPN(choices, num_init_features=128, k_r=200,
                   groups=50, k_sec=(1, 1, 1, 1), inc_sec=(20, 64, 64, 128))
    model = nn.DataParallel(model)
    model = model.to("cuda")
    input_dummy = torch.randn(4, 3, 112, 112).cuda()
    output_dummy = model(input_dummy)
    emb_size = output_dummy.shape[-1]
    args.embedding_size = emb_size
    args.num_class = 7058
    head = get_head(args)
    model = Network(model, head)

    

model_path = get_model_path(args.backbone,args.head,args.optimizer,args.dataset)
print("Model Path: ", model_path)
#print(list(torch.load(model_path)['model_state_dict'].keys()))

model.load_state_dict(torch.load(model_path)['model_state_dict'])

model.backbone.eval()

head = args.head
dataset = args.dataset
MULTI_GPU = False
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EMBEDDING_SIZE = emb_size
epoch = 0
BACKBONE = model.backbone
BATCH_SIZE = 128
writer = SummaryWriter("logs/{}_{}_{}_{}".format(head, dataset, args.backbone, args.optimizer))
agedb , cfp_fp, cfp_ff, lfw, calfw, cplfw, agedb_issame, cfp_fp_issame, cfp_ff_issame, lfw_issame, calfw_issame, cplfw_issame = get_data("/work/dlclarge2/sukthank-ZCP_Competition/NeurIPS2023/fr_datasets")
print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
print("LFW Accuracy: ", accuracy_lfw*100)
#buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
print("CFP_FF Accuracy: ", accuracy_cfp_ff*100)
#buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
print("CFP_FP Accuracy: ", accuracy_cfp_fp*100)
#buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
print("AgeDB Accuracy: ", accuracy_agedb*100)
#buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
print("CALFW Accuracy: ", accuracy_calfw*100)
#buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
print("CPLFW Accuracy: ", accuracy_cplfw*100)
#buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}".format(epoch + 1, 1, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw,))