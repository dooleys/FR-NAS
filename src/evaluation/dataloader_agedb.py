# write pytorch dataloader for agedb
import os
import torch
import numpy as np
import pickle
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from skimage import io, transform
import torch.nn as nn
from src.utils.utils_train import  get_head

def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output

class Network(torch.nn.Module):

    def __init__(self, backbone, head):
        super(Network, self).__init__()
        self.head=head
        self.backbone=backbone
        self.features_saved = []

    def reset(self):
        self.features_saved = []

    def forward(self, inputs, labels):
        #self.backbone.module.reset()
        #self.head.reset()
        features = self.backbone(inputs)
        outputs, reg_loss = self.head(features, labels)
        '''self.features_saved = self.backbone.module.features_saved
        for k in self.head.features_saved.keys():
            self.features_saved[k] = self.head.features_saved[k]'''
        #self.backbone.module.reset()
        #self.head.reset()
        return outputs, reg_loss
class AgedbDataset(Dataset):
    """Agedb dataset."""

    def __init__(self, root_dir="/work/dlclarge2/sukthank-ZCP_Competition/agedb_binned_dataset", transform=None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.len = self.__len__()
        print("Length of dataset: ", self.len)

    def __len__(self):
        self.image_paths = []
        for path, subdirs, files in os.walk(self.root_dir):
            for name in files:
                self.image_paths.append(os.path.join(path, name))
        return len(self.image_paths)
    
    def map_id(self,identity):
        with open("/work/dlclarge2/sukthank-ZCP_Competition/agedb_identities.pkl","rb") as f:
            identities = pickle.load(f)
        return identities[identity]
    
    def bin_ages(self, age):
        if age>0 and age<=25:
            return 0
        elif age>25 and age<=50:
            return 1
        elif age>50 and age<=75:
            return 2
        elif age>75 and age<=110:
            return 3
        
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.image_paths[idx].split("/")[-1]
        image = io.imread(self.image_paths[idx])
        identity = int(img_name.split("_")[0])
        identity = self.map_id(identity)
        if image.shape[-1] != 3:
            image = np.stack((image,)*3, axis=-1)
        age = int(img_name.split("_")[-1].split(".")[0])
        age_binned = self.bin_ages(age)
        sample = {'image': image, 'identity': identity, 'age': age, 'age_binned': age_binned}

        if self.transform:
            #print(sample['image'].shape)
            sample['image'] = self.transform(sample['image'])

        return sample

