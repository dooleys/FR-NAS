import torch.optim as optim
import torch
from src.head.metrics_angular import CosFace, SphereFace, ArcFace, MagFace
from src.utils.utils import separate_resnet_bn_paras
import numpy as np
import random
class Network(torch.nn.Module):

    def __init__(self, backbone, head):
        super(Network, self).__init__()
        self.head=head
        self.backbone=backbone

    def forward(self, inputs, labels, sens):
        features = self.backbone(inputs)
        outputs, reg_loss = self.head(features, labels, sens)

        return outputs, reg_loss

def get_head(args):
    if args.head=="CosFace":
       head = CosFace(in_features=args.embedding_size, out_features=args.num_class, device_id=range(torch.cuda.device_count()))
    elif args.head=="ArcFace":
       head = ArcFace(in_features=args.embedding_size, out_features=args.num_class, device_id=range(torch.cuda.device_count()))
    elif args.head=="SphereFace":
       head = SphereFace(in_features=args.embedding_size, out_features=args.num_class, device_id=range(torch.cuda.device_count()))
    elif args.head=="MagFace":
       head = MagFace(in_features=args.embedding_size, out_features=args.num_class, device_id=range(torch.cuda.device_count()))
    else:
        print("Head not supported")
        return
    return head