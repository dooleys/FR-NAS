from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from src.head.metrics import CosFace
from src.loss.focal import FocalLoss
from src.utils.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, \
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
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device_ids=range(torch.cuda.device_count())
seed = 333
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str)

    args = parser.parse_args()
    with open(args.config_path, "r") as ymlfile:
        options = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print(options)
    for key, value in options.items():
        setattr(args, key, value)
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
    args.backbone = "tnt_s_patch16_224"
    ####################################################################################################################################
    # ======= data, model and test data =======#
    run_name = "tnt_444"
    output_dir = os.path.join(args.checkpoints_root, run_name)
    if not os.path.isdir(args.checkpoints_root):
        os.mkdir(args.checkpoints_root)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    experiment = Experiment(
        api_key=args.comet_api_key,
        project_name=args.project_name,
        workspace=args.comet_workspace,
    )
    experiment.add_tag(args.backbone)

    dataloaders, num_class, demographic_to_labels_train,demographic_to_labels_val, demographic_to_labels_test = prepare_data(
        args)
    args.num_class = num_class
    ''' Model '''
    if 'ghost' in args.backbone:
        backbone = timm.create_model(args.backbone,
                                     num_classes=0,
                                     pretrained=args.pretrained,
                                     drop_connect_rate=args.drop_connect,
                                     drop_path_rate=args.drop_path,
                                     drop_block_rate=args.drop_block,
                                     global_pool=args.gp,
                                     bn_momentum=args.bn_momentum,
                                     bn_eps=args.bn_eps,
                                     scriptable=args.torchscript,
                                     ).to(device)
    elif 'mobilenet' in args.backbone:
        backbone = timm.create_model(args.backbone,
                                     num_classes=0,
                                     pretrained=args.pretrained,
                                     drop_path_rate=args.drop_path,
                                     drop_block_rate=args.drop_block,
                                     global_pool=args.gp,
                                     bn_momentum=args.bn_momentum,
                                     bn_eps=args.bn_eps,
                                     scriptable=args.torchscript,
                                     ).to(device)
    else:
        backbone = timm.create_model(args.backbone,
                                     num_classes=0,
                                     pretrained=args.pretrained,
                                     drop_rate=args.drop,
                                     drop_connect_rate=args.drop_connect,
                                     drop_path_rate=args.drop_path,
                                     drop_block_rate=args.drop_block,
                                     global_pool=args.gp,
                                     bn_momentum=args.bn_momentum,
                                     bn_eps=args.bn_eps,
                                     scriptable=args.torchscript,
                                     ).to(device)
    config = timm.data.resolve_data_config({}, model=backbone)
    model_input_size = args.input_size

    # get model's embedding size
    meta = pd.read_csv(args.metadata_file)
    embedding_size = int(
        meta[meta.model_name == args.backbone].feature_dim)
    args.embedding_size= embedding_size
    #args.head = "MagFace"
    head = get_head(args)
    train_criterion = FocalLoss(elementwise=True)
    head,backbone= head.to(device), backbone.to(device)
    backbone = nn.DataParallel(backbone)
    ####################################################################################################################
    # ======= argsimizer =======#
    model = Network(backbone, head)
    checkpoint = torch.load("/work/dlclarge2/sukthank-ZCP_Competition/checkpoints_celeba/tnt/model_444.pth")
    #checkpoint = torch.load("")
    #model = nn.DataParallel(model)
    model = model.to(device)
    model.load_state_dict(checkpoint['model_state_dict'])  
    #print(model_ema)
    args.epochs = 100
    epoch = 100
    print('Start training')
    backbone.eval()  # set to testing mode
    head.eval()
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
    add_column_to_file(output_dir,
                               "test",
                               run_name, 
                               epoch,
                               multi_df = multi_df, 
                               kacc_df = kacc_df, 
                               rank_by_image_df = rank_by_image_df,
                               rank_by_id_df = rank_by_id_df)
    
    k_accuracy = True
    multilabel_accuracy = True
    comp_rank = True
    loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all, rank = evaluate(
                dataloaders["val"],
                train_criterion,
                model,
                embedding_size,
                k_accuracy=k_accuracy,
                multilabel_accuracy=multilabel_accuracy,
                demographic_to_labels=demographic_to_labels_val,
                test=True, rank=comp_rank)

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
    add_column_to_file(output_dir,
                               "val",
                               run_name, 
                               epoch,
                               multi_df = multi_df, 
                               kacc_df = kacc_df, 
                               rank_by_image_df = rank_by_image_df,
                               rank_by_id_df = rank_by_id_df)