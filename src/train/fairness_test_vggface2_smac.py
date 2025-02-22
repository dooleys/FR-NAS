from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os, sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import yaml
from src.head.metrics import CosFace
from src.loss.focal import FocalLoss
from src.utils.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, load_checkpoints_all, \
    schedule_lr, AverageMeter, accuracy
from src.utils.fairness_utils import evaluate
from src.utils.data_utils_balanced import prepare_data
from src.utils.utils_train import Network
import numpy as np
import pandas as pd
import random
import timm
from src.utils.utils import save_output_from_dict
from src.utils.utils_train import Network, get_head
from src.utils.fairness_utils import evaluate, add_column_to_file
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils.model_ema import ModelEmaV2
sys.path.append('/cmlscratch/sdooley1/merge_timm/FR-NAS/')
from src.search.dpn107 import DPN
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids=range(torch.cuda.device_count())
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def fairness_objective_dpn(config, seed, budget):

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str)

    args = parser.parse_args(['--config_path','/cmlscratch/sdooley1/merge_timm/FR-NAS/configs/dpn107/config_dpn107_CosFace_sgd.yaml'])

    
    with open(args.config_path, "r") as ymlfile:
        options = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print(options)
    for key, value in options.items():
        setattr(args, key, value)
        
    # dummy train example
    args.default_train_root = '/cmlscratch/sdooley1/data/vggface2/test/'
    args.default_test_root = '/cmlscratch/sdooley1/data/vggface2/test/'
    args.demographics_file = '/cmlscratch/sdooley1/data/vggface2/vggface2_demographics.txt'
    args.RFW_checkpoints_root = 'Checkpoints/vggface2/'
    args.dataset = 'vggface2'
    
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

    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))
    

    ####################################################################################################################################
    # ======= data, model and test data =======#
    run_name = "Checkpoints_Edges_{}_LR_{}_Head_{}_Optimizer_{}".format(str(config["edge1"])+str(config["edge2"])+str(config["edge3"]), config["lr"], config["head"],config["optimizer"])
    output_dir = os.path.join('/cmlscratch/sdooley1/merge_timm/FR-NAS/Checkpoints/models_pareto/', run_name)
    args.checkpoints_root = output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    output_dir = os.path.join('/cmlscratch/sdooley1/merge_timm/FR-NAS',args.RFW_checkpoints_root, run_name)
    args.RFW_checkpoints_root = output_dir
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        
    print(args.checkpoints_root)
    ckpts = os.listdir(args.checkpoints_root)
#     ckpts.sort(key = lambda x: int(x.split('Epoch_')[1].split('.')[0]))
    if not len(ckpts):
        sys.exit()


    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_test = prepare_data(
        args)

    args.num_class = 7058
    edges=[config['edge1'],config['edge2'],config['edge3']]
    # Build model
    backbone = DPN(edges,num_init_features=128, k_r=200, groups=50, k_sec=(1,1,1,1), inc_sec=(20, 64, 64, 128))
        
    config = timm.data.resolve_data_config({}, model=backbone)
    model_input_size = args.input_size

    # get model's embedding size
    meta = pd.read_csv(args.metadata_file)
    embedding_size = 1000
    args.embedding_size= embedding_size

    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head,backbone= head.to(device), backbone.to(device)
    backbone = nn.DataParallel(backbone)
    ####################################################################################################################
    # ======= argsimizer =======#
    model = Network(backbone, head)
   
    optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))
#     scheduler, num_epochs = create_scheduler(args, optimizer)

    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = ModelEmaV2(
            model, decay=args.model_ema_decay, device='cpu' if args.model_ema_force_cpu else None)
    model, model_ema, optimizer, epoch, batch, checkpoints_model_root = load_checkpoint(
        args, model, model_ema, optimizer, dataloaders["test"], p_identities,
        p_images)
    #model = nn.DataParallel(model)
    model = model.to(device)
    #print(model_ema)
    print('Start training')
    ckpts = os.listdir(args.checkpoints_root)
    #model = nn.DataParallel(model)
    model = model.to(device)
    checkpoints_model_root = args.checkpoints_root
    epoch_numbers = []

    for ckpt in ckpts:
        '''Adjust number of classes in the head'''
        args.num_class = 7058 #arbitrarily large number
        head = get_head(args)
        model.head = head.to(device)
        if model_ema is not None:
            model_ema.module.head = head.to(device)

        print(ckpts)
        checkpoint = torch.load(os.path.join(checkpoints_model_root, ckpt))
        epoch = checkpoint['epoch']
        if model_ema is not None:
            model_ema.load_state_dict(checkpoint['model_ema_state_dict'])
        model.load_state_dict(checkpoint['model_state_dict'])    

        '''Adjust number of classes in the head'''
        args.num_class = 10000 #arbitrarily large number
        head = get_head(args)
        model.head = head.to(device)
        if model_ema is not None:
            model_ema.module.head = head.to(device)
            
        '''For test data compute only k-neighbors accuracy and multi-accuracy'''
        k_accuracy = True
        multilabel_accuracy = True
        comp_rank = True
        loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all, rank = evaluate(
            dataloaders["test"],
            train_criterion,
            model,
            embedding_size,
            k_accuracy=k_accuracy,
            multilabel_accuracy=multilabel_accuracy,
            demographic_to_labels=demographic_to_labels_test,
            test=True, rank=comp_rank)
        if model_ema is not None:
            loss_ema, acc_ema, acc_k_ema, predicted_all_ema, intra_ema, inter_ema, angles_intra_ema, angles_inter_ema, correct_ema, nearest_id_ema, labels_all_ema, indices_all_ema, demographic_all_ema, rank = evaluate(
                    dataloaders["test"],
                    train_criterion,
                    model_ema.module,
                    embedding_size,
                    k_accuracy=k_accuracy,
                    multilabel_accuracy=multilabel_accuracy,
                    demographic_to_labels=demographic_to_labels_test,
                    test=True, rank=comp_rank)
        # save outputs
        # save outputs
        kacc_df, multi_df, rank_by_image_df, rank_by_id_df = None, None, None, None
        if k_accuracy:
            kacc_df = pd.DataFrame(np.array([list(indices_all),
                                             list(nearest_id)]).T,
                                   columns=['ids','epoch_'+str(epoch)]).astype(int)
        if multilabel_accuracy:
            multi_df = pd.DataFrame(np.array([list(indices_all),
                                              list(predicted_all)]).T,
                                    columns=['ids','epoch_'+str(epoch)]).astype(int)
        if comp_rank:
            rank_by_image_df = pd.DataFrame(np.array([list(indices_all),
                                              list(rank[:,0])]).T,
                                    columns=['ids','epoch_'+str(epoch)]).astype(int)
            rank_by_id_df = pd.DataFrame(np.array([list(indices_all),
                                              list(rank[:,1])]).T,
                                    columns=['ids','epoch_'+str(epoch)]).astype(int)
        print('before add_column_to_file')
        print(args.RFW_checkpoints_root, run_name, epoch)
        add_column_to_file(args.RFW_checkpoints_root,
                           "val",
                           run_name, 
                           epoch,
                           multi_df = multi_df, 
                           kacc_df = kacc_df, 
                           rank_by_image_df = rank_by_image_df,
                           rank_by_id_df = rank_by_id_df)
        results = {}
        results['Model'] = args.backbone
        results['config_file'] = args.config_path
        results['seed'] = args.seed
        results['epoch'] = epoch
        for k in acc_k.keys():

            results['Acc multi '+k] = (round(acc[k].item()*100, 3))
            results['Acc k '+k] = (round(acc_k[k].item()*100, 3))
            results['Intra '+k] = (round(intra[k], 3))
            results['Inter '+k] = (round(inter[k], 3))

        print(results)
        save_output_from_dict(args.RFW_checkpoints_root, results, args.file_name)
        if model_ema is not None:
            kacc_df_ema, multi_df_ema, rank_by_image_df_ema, rank_by_id_df_ema = None, None, None, None
            if comp_rank:
                rank_by_image_df_ema= pd.DataFrame(np.array([list(indices_all_ema),
                                             list(rank[:,0])]).T,
                                   columns=['ids','epoch_'+str(epoch)]).astype(int)
                rank_by_id_df_ema= pd.DataFrame(np.array([list(indices_all_ema),
                                             list(rank[:,1])]).T,
                                   columns=['ids','epoch_'+str(epoch)]).astype(int)
            if k_accuracy:
                kacc_df_ema= pd.DataFrame(np.array([list(indices_all_ema),
                                             list(nearest_id_ema)]).T,
                                   columns=['ids','epoch_'+str(epoch)]).astype(int)
            if multilabel_accuracy:
                multi_df_ema= pd.DataFrame(np.array([list(indices_all_ema),
                                              list(predicted_all_ema)]).T,
                                    columns=['ids','epoch_'+str(epoch)]).astype(int)
            add_column_to_file(args.RFW_checkpoints_root,
                           "ema_val",
                           run_name,
                           epoch,
                           multi_df = multi_df_ema,
                           kacc_df = kacc_df_ema,
                           rank_by_image_df = rank_by_image_df_ema,
                           rank_by_id_df = rank_by_id_df_ema)
            results_ema = {}
            results_ema['Model'] = args.backbone
            results_ema['config_file'] = args.config_path
            results_ema['seed'] = args.seed
            results_ema['epoch'] = epoch
            for k in acc_k_ema.keys():
                results_ema['Acc multi '+k] = (round(acc_ema[k].item()*100, 3))
                results_ema['Acc k '+k] = (round(acc_k_ema[k].item()*100, 3))
                results_ema['Intra '+k] = (round(intra_ema[k], 3))
                results_ema['Inter '+k] = (round(inter_ema[k], 3))

            print(results_ema)
            save_output_from_dict(args.RFW_checkpoints_root, results_ema, args.file_name_ema)


if __name__ == '__main__':
    #SMAC config 1
    config ={
    'edge1': 0,
    'edge2': 0,
    'edge3': 0,
    'head': 'CosFace',
    'lr': 0.2813375341651194,
    'optimizer': 'SGD',
    }
    fairness_objective_dpn(config,0,100)
    #SMAC config 2
    config={
    'edge1': 0,
    'edge2': 1,
    'edge3': 0,
    'head': 'CosFace',
    'lr': 0.32348738788346576,
    'optimizer': 'SGD',
    }
    fairness_objective_dpn(config,0,100)
    #SMAC config 3
    config={
    'edge1': 6,
    'edge2': 8,
    'edge3': 0,
    'head': 'CosFace',
    'lr': 0.0006048015915653069,
    'optimizer': 'Adam',
    }
    fairness_objective_dpn(config,0,100)
