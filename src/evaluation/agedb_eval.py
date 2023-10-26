import torch
from tqdm import tqdm
import numpy as np
import random
import os
import pandas as pd
import argparse
def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output
def l2_dist(feature_matrix, test_features):
    ''' computing distance matrix '''
    return torch.cdist(test_features, feature_matrix)
def predictions(feature_matrix, labels, demographic_to_labels, test_features, test_labels, test_demographic, rank):
    
    def process_row(row, distances, labels_np):
        """ 
        given:
        an array `row` where the row is a list of image ids where this list increases
        in distance in the featue space where the first point is the refernce point;
        and given an array `distance` which represents the distance to every point in increasing order,
        returns:
        a tuple (a,b,c,d), where
            a is the the index of the closest point with the same label
            b is the number of identities which are closer than the closest point of the same identity
            c is the distance of the closet point, and
            d is the distance of the closest image of the same identity
        """
        base_label = labels_np[row[0]]
        n_img = 1
        n_id = 1
        ids = set()
        while n_img < row.shape[0]:
            # add this id to the set of 
            ids.add(labels_np[row[n_img]])
            if labels_np[row[n_img]] == base_label:
                return n_img-1, len(list(ids))-1, distances[1], distances[n_img]
            n_img+=1
        return -1,-1,-1,-1

    # if rank is true, then compute the rank of the prediction
    if rank == True:
        k = test_features.shape[0]
    # otherwise, just compute the accuracy
    else:
        k = 2

    dist_matrix =  l2_dist(feature_matrix, test_features)
    labels_np = labels.numpy()
    distances, inc_dist = torch.topk(dist_matrix, dim=1, k = k, largest = False)
    nearest_same_label = torch.tensor([process_row(row[0],row[1], labels_np) for row in zip(inc_dist,distances)])

    correct = (nearest_same_label[:,0] == 0).long()
    print(correct)
    nearest_id = inc_dist[:,1].apply_(lambda x: labels_np[x])
    acc_k = {k:0 for k in demographic_to_labels.keys()}
    for k in acc_k.keys():
        acc_k[k] = (correct[test_demographic == int(k)]).float().mean()

    return acc_k, correct, nearest_id, nearest_same_label
parser = argparse.ArgumentParser(description = 'Evaluation on different datasets')
parser.add_argument('--backbone', type = str, default = 'dpn107', help = 'model type to evaluate')
parser.add_argument('--head', type = str, default = 'CosFace', help = 'head type to evaluate')
parser.add_argument('--dataset', type = str, default = 'vggface2', help = 'dataset the model is trained on')
parser.add_argument('--optimizer', type = str, default = 'AdamW', help = 'optimzier used to change the model')
import pickle
args = parser.parse_args()
with open("agedb_features_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","rb") as f:
    agedb_features = pickle.load(f)
with open("agedb_labels_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","rb") as f:
    agedb_labels = pickle.load(f)
with open("agedb_ages_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","rb") as f:
    agedb_ages = pickle.load(f)
with open("agedb_ages_binned_"+args.backbone+"_"+args.head+"_"+args.optimizer+"_"+args.dataset+".pkl","rb") as f:
    agedb_ages_binned = pickle.load(f)
agedb_ages_binned = [a[0] for a in agedb_ages_binned]
demographics = {}
for s in set(agedb_ages_binned):
    demographics[str(s)] = s
print(demographics)
print(agedb_features.shape)
agedb_labels = [a[0] for a in agedb_labels]
print(len(agedb_labels))
acc_k, correct, nearest_id, nearest_same_label = predictions(agedb_features, torch.tensor(agedb_labels), demographics, agedb_features, torch.tensor(agedb_labels), torch.tensor(agedb_ages_binned), True)
print(acc_k)
print(correct)
print(nearest_id)
print(nearest_same_label)

# compute max diff across accuracy of age groups
max_diff = 0
for i in range(4):
    for j in range(i+1,4):
        diff = abs(acc_k[str(i)] - acc_k[str(j)])
        if diff > max_diff:
            max_diff = diff
print("Max diff: ", max_diff*100)
