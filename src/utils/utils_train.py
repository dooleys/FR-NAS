import torch.optim as optim
import torch
from head.metrics import CosFace, SphereFace, ArcFace, MagFace
from utils.utils import separate_resnet_bn_paras
def get_optimizer(opt,backbone,head):
    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
    _, head_paras_wo_bn = separate_resnet_bn_paras(head)
    if opt["optimizer"]=="Adam":
        optimizer = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': opt["weight_decay"]},
                               {'params': backbone_paras_only_bn}], lr=opt["lr"], betas=opt["betas"], eps=opt["eps"])
    elif opt["optimizer"]=="SGD":
        optimizer = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': opt["weight_decay"]},
                               {'params': backbone_paras_only_bn}], lr=opt["lr"], momentum=opt["momentum"], dampening=opt["dampening"])     
    elif opt["optimizer"]=="AdamW":
        if opt["weight_decay"]==0:
           opt["weight_decay"]=0.01
        optimizer = optim.AdamW([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': opt["weight_decay"]},
                               {'params': backbone_paras_only_bn}], lr=opt["lr"], betas=opt["betas"], eps=opt["eps"])  
    else:
        print("Optimizer not supported") 
        return
    return optimizer

def get_scheduler(opt,optimizer):
    if opt["scheduler"]=="LinearLR":
        scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=opt["start_factor"], end_factor=opt["end_factor"], total_iters=opt["total_iters"], last_epoch=opt["last_epoch"])
    elif opt["scheduler"]=="CosineAnnealingLR":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt["T_max"], eta_min=opt["eta_min"], last_epoch=opt["last_epoch"])
    elif opt["scheduler"]=="CosineAnnealingWarmRestarts":
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, opt["T_0"], T_mult=opt["T_mult"], eta_min=opt["eta_min"], last_epoch=opt["last_epoch"])
    else:
        print("Scheduler not supported")
        return
    return scheduler

def get_head(opt):
    if opt["head"]=="CosFace":
       head = CosFace(in_features=opt["embedding_size"], out_features=opt["num_class"], device_id=range(torch.cuda.device_count()))
    elif opt["head"]=="ArcFace":
       head = ArcFace(in_features=opt["embedding_size"], out_features=opt["num_class"], device_id=range(torch.cuda.device_count()))
    elif opt["head"]=="SphereFace":
       head = SphereFace(in_features=opt["embedding_size"], out_features=opt["num_class"], device_id=range(torch.cuda.device_count()))
    elif opt["head"]=="MagFace":
       head = MagFace(in_features=opt["embedding_size"], out_features=opt["num_class"], device_id=range(torch.cuda.device_count()))
    else:
        print("Head not supported")
        return
    return head

