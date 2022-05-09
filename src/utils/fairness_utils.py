import torch
from tqdm import tqdm
import numpy as np
import random
import os
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def l2_norm(input, axis = 1):
    # normalizes input with respect to second norm
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output



def add_column_to_file(path, suffix, experiment_id, epoch, multi_df = None, kacc_df = None, rank_df = None):
    
    def _get_filename(path, suffix, tag):
        return os.path.join(path, '_'.join([suffix, tag])+'.csv')
    
    def _load_data(path):
        data = None
        if os.path.exists(path):
            data = pd.read_csv(path)
        return data
    
    def _check_epoch(df, epoch):
        # if this epoch shows up as a column name in the dataframe
        # throw an error
        columns = list(df.columns)
        assert 'epoch_'+str(epoch) not in columns
        return
        
    if multi_df is not None:
        fn = _get_filename(path, experiment_id, 'multi_'+suffix)
        print(fn)
        old_df = _load_data(fn)
        if old_df is None:
            multi_df.to_csv(fn,index=False)
        else:
            _check_epoch(old_df, epoch)
            old_df.merge(multi_df).to_csv(fn,index=False)
            
    if kacc_df is not None:
        fn = _get_filename(path, experiment_id, 'kacc_'+suffix)
        print(fn)
        old_df = _load_data(fn)
        if old_df is None:
            kacc_df.to_csv(fn,index=False)
        else:
            _check_epoch(old_df, epoch)
            old_df.merge(kacc_df).to_csv(fn,index=False)
    if rank_df is not None:
        fn = _get_filename(path, experiment_id, 'rank_'+suffix)
        print(fn)
        old_df = _load_data(fn)
        if old_df is None:
            rank_df.to_csv(fn,index=False)
        else:
            _check_epoch(old_df, epoch)
            old_df.merge(rank_df).to_csv(fn,index=False)
    return
    
    


def evaluate(dataloader, criterion, model, emb_size,  k_accuracy = False, multilabel_accuracy = False,
             demographic_to_labels = None, test = True, rank = False):

    loss = {k:torch.tensor(0.0) for k in demographic_to_labels.keys()}
    acc = {k:torch.tensor(0.0) for k in demographic_to_labels.keys()}
    count = {k:torch.tensor(0.0) for k in demographic_to_labels.keys()}
    acc_k = {k:torch.tensor(0.0) for k in demographic_to_labels.keys()}
    intra = {k:torch.tensor(0.0) for k in demographic_to_labels.keys()}
    inter = {k:torch.tensor(0.0) for k in demographic_to_labels.keys()}
    angles_intra, angles_inter, correct = 0, 0, 0
    
    #backbone.eval()
    #if multilabel_accuracy:
    #    head.eval()
    model.eval()
    # figure out embedding size
    if emb_size is None:
        inputs, _, _ = next(iter(dataloader))
        x = torch.randn(inputs.shape).to(device)
        emb_size = backbone(x).shape[1]


    feature_matrix = torch.empty(0, emb_size)
    labels_all = []
    indices_all = []
    demographic_all = []
    predicted_all = []

    for inputs, labels, sens_attr, indices in tqdm(iter(dataloader)):
        inputs = inputs.to(device)
        labels = labels.to(device).long()
        labels_all = labels_all + labels.cpu().tolist()
        indices_all = indices_all + indices.cpu().tolist()
        sens_attr = np.array(sens_attr)
        with torch.no_grad():

            if multilabel_accuracy:
                outputs, loss_g= model(inputs, labels)
                loss_value = criterion(outputs, labels)+loss_g

                # add sum of losses for female and male images
                for k in loss.keys():
                    loss[k] += loss_value[sens_attr == k].sum().cpu()

                # multiclass accuracy
                _, predicted = outputs.max(1)
                predicted_all = predicted_all + predicted.cpu().tolist()



                for k in acc.keys():
                    acc[k] +=  predicted[sens_attr == k].eq(labels[sens_attr == k]).sum().cpu().item()

                for k in count.keys():
                    count[k] += sum(sens_attr == k)

            if k_accuracy:
                #need to build feature matrix
                inputs_flipped = torch.flip(inputs, [3])
                try:
                    embed = model.module.backbone(inputs) + model.module.backbone(inputs_flipped)
                except AttributeError:
                    embed = model.backbone(inputs) + model.backbone(inputs_flipped)
                features_batch = l2_norm(embed)
                feature_matrix = torch.cat((feature_matrix, features_batch.detach().cpu()), dim = 0)

                demographic_all = demographic_all + sens_attr.tolist()


    if multilabel_accuracy:
        for k in acc.keys():
            acc[k] = acc[k]/count[k].item()
            loss[k] = loss[k]/count[k].item()

    if k_accuracy and test:
        intra, inter, angles_intra, angles_inter, class_centers = intra_inter_variance(feature_matrix, torch.tensor(labels_all), demographic_to_labels)
        acc_k, correct, nearest_id, rank = predictions(feature_matrix, torch.tensor(labels_all), demographic_to_labels, feature_matrix, torch.tensor(labels_all), np.array(demographic_all), rank)

    if k_accuracy and not test:
        m_labels = np.random.choice(demographic_to_labels['male'], 30)
        f_labels = np.random.choice(demographic_to_labels['female'], 30)
        m_idx = [idx for idx, label in enumerate(labels_all) if label in m_labels]
        f_idx = [idx for idx, label in enumerate(labels_all) if label in f_labels]

        test_labels = torch.cat((torch.tensor(labels_all)[m_idx], torch.tensor(labels_all)[f_idx]), dim=0)
        test_features = torch.cat((feature_matrix[m_idx], feature_matrix[f_idx]), dim=0)
        test_demographic = np.take(np.array(demographic_all), m_idx).tolist() +  np.take(np.array(demographic_all), f_idx).tolist()

        intra, inter, angles_intra, angles_inter, class_centers = intra_inter_variance(feature_matrix, torch.tensor(labels_all), demographic_to_labels)
        acc_k, correct, nearest_id, rank = predictions(feature_matrix, torch.tensor(labels_all), demographic_to_labels, test_features, torch.tensor(test_labels), np.array(test_demographic))

        
    return loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all, rank




def l2_dist(feature_matrix, test_features):
    ''' computing distance matrix '''
    return torch.cdist(test_features, feature_matrix)


def predictions(feature_matrix, labels, demographic_to_labels, test_features, test_labels, test_demographic, rank):
    
    def process_row(row, labels_np):
        """ 
        given a row where the row is a list of image ids where this list increases
        in distance in the featue space where the first point is the refernce point.
        returns the index of the closest point with the same label, if no such point, returns -1
        """
        base_label = labels_np[row[0]]
        i = 1
        while i < row.shape[0]:
            if labels_np[row[i]] == base_label:
                return i-1
            i+=1
        return -1

    # if rank is true, then compute the rank of the prediction
    if rank == True:
        k = test_features.shape[0]
    # otherwise, just compute the accuracy
    else:
        k = 2
            
    dist_matrix =  l2_dist(feature_matrix, test_features)
    labels_np = labels.numpy()
    inc_dist = torch.topk(dist_matrix, dim=1, k = k, largest = False)[1]
    nearest_same_label = torch.tensor([process_row(row, labels_np) for row in inc_dist])

    correct = (nearest_same_label == 0).long()
    nearest_id = inc_dist[:,1].apply_(lambda x: labels_np[x])
    acc_k = {}
    for k in acc_k.keys():
        acc_k[k] = (correct[test_demographic == k]).mean()

    return acc_k, correct, nearest_id, nearest_same_label


def intra_inter_variance(feature_matrix, labels, demographic_to_labels):
    classes = set(labels.tolist())
    angles_intra = {}
    angles_inter = {}
    class_centers = {}

    for i, idx in enumerate(classes):
        class_features = feature_matrix[labels == idx]
        class_centers[idx] = class_features.mean(0)
        class_centers_rep = class_centers[idx].repeat(class_features.shape[0],1)
        cos = torch.nn.CosineSimilarity()(class_features,class_centers_rep)
        angles_intra[idx] = ((torch.acos(cos)*180)/np.pi).mean()

    for i, idx in enumerate(classes):
        class_center_rep = class_centers[idx].repeat(len(classes), 1)
        cos = torch.nn.CosineSimilarity()(class_center_rep, torch.stack(list(class_centers.values())))
        min_angle = torch.sort(torch.acos(cos)*180/np.pi)[0][1]
        angles_inter[idx] = min_angle

    inter = {}
    intra = {}

    for k in demographic_to_labels.keys():
        inter[k] = torch.tensor([angles_inter[idx] for idx in demographic_to_labels[k] if idx in angles_inter.keys()]).mean().item()
        intra[k] = torch.tensor([angles_intra[idx] for idx in demographic_to_labels[k] if idx in angles_inter.keys()]).mean().item()

    return intra, inter, angles_intra, angles_inter, class_centers


def most_least_variant_classes(angles, idx_to_class):
    ''' Compute Most and Least variant identities'''
    angles_sorted = {k: v for k, v in sorted(angles.items(), key=lambda item: item[1])}

    least_var_labels = list(angles_sorted.keys())[0:10]
    most_var_labels = list(angles_sorted.keys())[-10:]
    least_var_classes = []
    most_var_classes = []
    for label in least_var_labels:
        least_var_classes.append(idx_to_class[label])
    for label in most_var_labels:
        most_var_classes.append(idx_to_class[label])
    return least_var_classes, most_var_classes
