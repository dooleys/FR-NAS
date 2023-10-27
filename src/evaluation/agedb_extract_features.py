# test the dataloader
from src.evaluation.dataloader_agedb import AgedbDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import pickle
from src.utils.utils_train import  get_head
from src.utils.utils_train import Network
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from src.evaluation.model_factory import get_model_path

def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5])
])
agedb_dataset = AgedbDataset(transform=transforms)
agedb_dataloader = DataLoader(agedb_dataset, batch_size=1, shuffle=False, num_workers=0)
identities = {}
id = 0
for i_batch, sample_batched in enumerate(agedb_dataloader):
    print(i_batch, sample_batched['image'].size(), sample_batched['identity'], sample_batched['age'], sample_batched['age_binned'])

# plot age distribution
import matplotlib.pyplot as plt
ages = []
for i_batch, sample_batched in enumerate(agedb_dataloader):
    ages.extend(sample_batched['age'].numpy())
plt.hist(ages, bins=100)
plt.savefig("age_distribution.png")

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
import yaml
import timm
import os
import json
import torch.nn as nn
from src.utils.data_utils_balanced import prepare_data
import yaml
import tqdm
with open("search_configs/config_vggface.yaml","r") as ymlfile:
    args = yaml.load(ymlfile, Loader=yaml.FullLoader)
args = dotdict(args)
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
feature_matrix = torch.empty(0, args.embedding_size)
label_matrix = []
age_matrix = []
age_binned_matrix = []
for i_batch, sample_batched in enumerate(agedb_dataloader):
    print(i_batch)
    inputs = sample_batched['image'].to(device)
    labels = sample_batched['identity'].to(device)
    inputs_flipped = torch.flip(inputs, [3])
    embed = model.backbone.module(inputs) + model.backbone.module(inputs_flipped)
    features_batch = l2_norm(embed)
    feature_matrix = torch.cat((feature_matrix, features_batch.detach().cpu()), dim = 0)
    label_matrix.append(sample_batched['identity'].detach().cpu().numpy())
    age_matrix.append(sample_batched['age'].detach().cpu().numpy())
    age_binned_matrix.append(sample_batched['age_binned'].detach().cpu().numpy())

with open("agedb_features_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","wb") as f:
    pickle.dump(feature_matrix, f)
with open("agedb_labels_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","wb") as f:
    pickle.dump(label_matrix, f)
with open("agedb_ages_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","wb") as f:
    pickle.dump(age_matrix, f)
with open("agedb_ages_binned_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","wb") as f:
    pickle.dump(age_binned_matrix, f)
