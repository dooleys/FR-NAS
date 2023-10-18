import numpy as np
import bcolz
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy

from tensorboardX import SummaryWriter
from tqdm import tqdm
import os


def get_pair(root, name):
    carray = bcolz.carray(rootdir = os.path.join(root, name), mode='r')
    issame = np.load('{}/{}_list.npy'.format(root, name))
    return carray, issame

def get_data(data_root):
    agedb_30, agedb_30_issame = get_pair(data_root, 'agedb_30')
    cfp_fp, cfp_fp_issame = get_pair(data_root, 'cfp_fp')
    lfw, lfw_issame = get_pair(data_root, 'lfw')
    return agedb_30, cfp_fp, lfw, agedb_30_issame, cfp_fp_issame, lfw_issame
cfg = configurations[1]

SEED = 111 # random seed for reproduce results
torch.manual_seed(SEED)

DATA_ROOT = "/work/dlclarge2/sukthank-ZCP_Competition/NeurIPS2023/fr_datasets" # the parent root where your train/val/test data are stored
MODEL_ROOT = cfg['MODEL_ROOT'] # the root to buffer your checkpoints
LOG_ROOT = cfg['LOG_ROOT'] # the root to log your train/val status
BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
LOSS_NAME = cfg['LOSS_NAME'] # support: ['Focal', 'Softmax']

INPUT_SIZE = cfg['INPUT_SIZE']
RGB_MEAN = cfg['RGB_MEAN'] # for normalize inputs
RGB_STD = cfg['RGB_STD']
EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
BATCH_SIZE = cfg['BATCH_SIZE']
DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
LR = cfg['LR'] # initial LR
NUM_EPOCH = cfg['NUM_EPOCH']
WEIGHT_DECAY = cfg['WEIGHT_DECAY']
MOMENTUM = cfg['MOMENTUM']
STAGES = cfg['STAGES'] # epoch stages to decay learning rate

DEVICE = cfg['DEVICE']
MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
GPU_ID = cfg['GPU_ID'] # specify your GPU ids
PIN_MEMORY = cfg['PIN_MEMORY']
NUM_WORKERS = cfg['NUM_WORKERS']
print("=" * 60)
print("Overall Configurations:")
print(cfg)
print("=" * 60)

writer = SummaryWriter(LOG_ROOT) # writer for buffering intermedium results
BACKBONE_DICT = {'ResNet_50': ResNet_50(INPUT_SIZE), 
                 'ResNet_101': ResNet_101(INPUT_SIZE), 
                 'ResNet_152': ResNet_152(INPUT_SIZE),
                 'IR_50': IR_50(INPUT_SIZE), 
                 'IR_101': IR_101(INPUT_SIZE), 
                 'IR_152': IR_152(INPUT_SIZE),
                 'IR_SE_50': IR_SE_50(INPUT_SIZE), 
                 'IR_SE_101': IR_SE_101(INPUT_SIZE), 
                 'IR_SE_152': IR_SE_152(INPUT_SIZE)}
BACKBONE = BACKBONE_DICT[BACKBONE_NAME]
print("=" * 60)
print(BACKBONE)
print("{} Backbone Generated".format(BACKBONE_NAME))
print("=" * 60)

HEAD_DICT = {'ArcFace': ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
             'CosFace': CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
             'SphereFace': SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID),
             'Am_softmax': Am_softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)}
HEAD = HEAD_DICT[HEAD_NAME]
print("=" * 60)
print(HEAD)
print("{} Head Generated".format(HEAD_NAME))
print("=" * 60)
if BACKBONE_RESUME_ROOT and HEAD_RESUME_ROOT:
    print("=" * 60)
    if os.path.isfile(BACKBONE_RESUME_ROOT) and os.path.isfile(HEAD_RESUME_ROOT):
        print("Loading Backbone Checkpoint '{}'".format(BACKBONE_RESUME_ROOT))
        BACKBONE.load_state_dict(torch.load(BACKBONE_RESUME_ROOT))
        print("Loading Head Checkpoint '{}'".format(HEAD_RESUME_ROOT))
        HEAD.load_state_dict(torch.load(HEAD_RESUME_ROOT))
    else:
        print("No Checkpoint Found at '{}' and '{}'. Please Have a Check or Continue to Train from Scratch".format(BACKBONE_RESUME_ROOT, HEAD_RESUME_ROOT))
    print("=" * 60)
agedb_30, cfp_fp, lfw, vgg2_fp, agedb_30_issame, cfp_fp_issame, lfw_issame, vgg2_fp_issame = get_data(DATA_ROOT)
print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, lfw, lfw_issame)
buffer_val(writer, "LFW", accuracy_lfw, best_threshold_lfw, roc_curve_lfw, epoch + 1)
accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_ff, cfp_ff_issame)
buffer_val(writer, "CFP_FF", accuracy_cfp_ff, best_threshold_cfp_ff, roc_curve_cfp_ff, epoch + 1)
accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cfp_fp, cfp_fp_issame)
buffer_val(writer, "CFP_FP", accuracy_cfp_fp, best_threshold_cfp_fp, roc_curve_cfp_fp, epoch + 1)
accuracy_agedb, best_threshold_agedb, roc_curve_agedb = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, agedb, agedb_issame)
buffer_val(writer, "AgeDB", accuracy_agedb, best_threshold_agedb, roc_curve_agedb, epoch + 1)
accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, calfw, calfw_issame)
buffer_val(writer, "CALFW", accuracy_calfw, best_threshold_calfw, roc_curve_calfw, epoch + 1)
accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, cplfw, cplfw_issame)
buffer_val(writer, "CPLFW", accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw, epoch + 1)
accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, vgg2_fp, vgg2_fp_issame)
buffer_val(writer, "VGGFace2_FP", accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp, epoch + 1)
print("Epoch {}/{}, Evaluation: LFW Acc: {}, CFP_FF Acc: {}, CFP_FP Acc: {}, AgeDB Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1, NUM_EPOCH, accuracy_lfw, accuracy_cfp_ff, accuracy_cfp_fp, accuracy_agedb, accuracy_calfw, accuracy_cplfw, accuracy_vgg2_fp))