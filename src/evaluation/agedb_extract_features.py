# test the dataloader
from src.evaluation.dataloader_agedb import AgedbDataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
import pickle

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
from src.search.dpn107 import DPN

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
with open("/work/dlclarge2/sukthank-ZCP_Competition/FR-NAS/vgg_configs/configs_default/dpn107/config_dpn107_CosFace_sgd.yaml","r") as ymlfile:
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
args.num_class = 7058
from src.search.dpn107 import DPN
args.p_images = p_images
args.p_identities = p_identities
args.head = "CosFace"
edges=[0,1,0]
dpn_301 = DPN(edges,num_init_features=128, k_r=200, groups=50, k_sec=(1,1,1,1), inc_sec=(20, 64, 64, 128))
input=torch.ones(4,3,32,32)
output= dpn_301(input)
args.embedding_size= output.shape[-1]
head = get_head(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head,backbone= head.to(device), dpn_301.to(device)
backbone = nn.DataParallel(backbone)
model = Network(backbone, head)
model.load_state_dict(torch.load("/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/model_010/model_666.pth")['model_state_dict'])
model.eval()
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
with open("agedb_features_010.pkl","wb") as f:
    pickle.dump(feature_matrix, f)
with open("agedb_labels_010.pkl","wb") as f:
    pickle.dump(label_matrix, f)
with open("agedb_ages_010.pkl","wb") as f:
    pickle.dump(age_matrix, f)
with open("agedb_ages_binned_010.pkl","wb") as f:
    pickle.dump(age_binned_matrix, f)



'''dpn_cosface_sgd = timm.create_model('dpn107', pretrained=False, num_classes=0)
input=torch.ones(4,3,32,32)
output= dpn_cosface_sgd(input)
args.embedding_size= output.shape[-1]
args.head = "MagFace"
head = get_head(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head,backbone= head.to(device), dpn_cosface_sgd.to(device)
backbone = nn.DataParallel(backbone)
model = Network(backbone, head)

model.to(device)
model.load_state_dict(torch.load("/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/dpn107_MagFace_SGD_0.1/model_666.pth")['model_state_dict'])
model.eval()

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
with open("agedb_features_dpn_magface_celeba.pkl","wb") as f:
    pickle.dump(feature_matrix, f)
with open("agedb_labels_dpn_magface_celeba.pkl","wb") as f:
    pickle.dump(label_matrix, f)
with open("agedb_ages_dpn_magface_celeba.pkl","wb") as f:
    pickle.dump(age_matrix, f)
with open("agedb_ages_binned_dpn_magface_celeba.pkl","wb") as f:
    pickle.dump(age_binned_matrix, f)

dpn_magface_sgd = timm.create_model('dpn107', pretrained=False, num_classes=0)
args.head = "MagFace"
input=torch.ones(4,3,32,32)
output= dpn_magface_sgd(input)
args.embedding_size= output.shape[-1]
head = get_head(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head,backbone= head.to(device), dpn_magface_sgd.to(device)
backbone = nn.DataParallel(backbone)
model = Network(backbone, head)

model.to(device)
model.load_state_dict(torch.load("/work/dlclarge2/sukthank-ZCP_Competition/samuel_exps/FR-NAS/vggface2_train_111/dpn107_MagFace_SGD/Checkpoint_Head_MagFace_Backbone_dpn107_Opt_SGD_Dataset_CelebA_Epoch_11.pth/Checkpoint_Head_MagFace_Backbone_dpn107_Opt_SGD_Dataset_CelebA_Epoch_11.pth")['model_state_dict'])
model.eval()   
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
with open("agedb_features_dpn_magface_sgd.pkl","wb") as f:
    pickle.dump(feature_matrix, f)
with open("agedb_labels_dpn_magface_sgd.pkl","wb") as f:
    pickle.dump(label_matrix, f)
with open("agedb_ages_dpn_magface_sgd.pkl","wb") as f:
    pickle.dump(age_matrix, f)
with open("agedb_ages_binned_dpn_magface_sgd.pkl","wb") as f:
    pickle.dump(age_binned_matrix, f)'''

'''
dpn_cosface_adamw = timm.create_model('rexnet_200', pretrained=False, num_classes=0)
input=torch.ones(4,3,32,32)
args.head = "CosFace"
output= dpn_cosface_adamw(input)
args.embedding_size= output.shape[-1]
print(args)
head = get_head(args)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
head,backbone= head.to(device), dpn_cosface_adamw.to(device)
backbone = nn.DataParallel(backbone)
model = Network(backbone, head)

model.to(device)
model.load_state_dict(torch.load("/work/dlclarge2/sukthank-ZCP_Competition/samuel_exps/FR-NAS/vggface2_train_111/rexnet_200_CosFace_SGD_0.1_cosine/Checkpoint_Head_CosFace_Backbone_rexnet_200_Opt_SGD_Dataset_CelebA_Epoch_11.pth/Checkpoint_Head_CosFace_Backbone_rexnet_200_Opt_SGD_Dataset_CelebA_Epoch_11.pth")['model_state_dict'])
model.eval()

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
with open("agedb_features_rexnet_200_cosface.pkl","wb") as f:
    pickle.dump(feature_matrix, f)
with open("agedb_labels_rexnet_200_cosface.pkl","wb") as f:
    pickle.dump(label_matrix, f)
with open("agedb_ages_rexnet_200_cosface.pkl","wb") as f:
    pickle.dump(age_matrix, f)
with open("agedb_ages_binned_rexnet_200_cosface.pkl","wb") as f:
    pickle.dump(age_binned_matrix, f)'''
