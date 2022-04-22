from pathlib import Path
from comet_ml import Experiment
import argparse
from tqdm import tqdm
from config import user_configs
import os
import torch
import torch.nn as nn
import torch.optim as optim

from head.metrics import CosFace
from loss.focal import FocalLoss
from util.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, \
    schedule_lr, AverageMeter, accuracy
from util.fairness_utils import evaluate, add_column_to_file
from util.data_utils_balanced import prepare_data
import numpy as np
import pandas as pd
import random
import timm
from util.utils import save_output_from_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

user_cfg = user_configs[1]

default_test_root = user_cfg['default_test_root']
default_train_root = user_cfg['default_train_root']


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_test_root', default=default_test_root)
    parser.add_argument('--data_train_root', default=default_train_root)
    parser.add_argument('--demographics', default= user_cfg['demographics_file'])
    parser.add_argument('--backbone_name', default='resnet50')
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--project_name', default="from-scratch_no-resampling_adam")

    parser.add_argument('--checkpoints_root', default=user_cfg['checkpoints_root'])
    parser.add_argument('--head_name', default='CosFace')
    parser.add_argument('--train_loss', default='Focal', type=str)

    parser.add_argument('--groups_to_modify', default= ['male', 'female'], type=str, nargs='+')
    parser.add_argument('--p_identities', default=[1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--p_images', default=[1.0, 1.0], type=float, nargs='+')
    parser.add_argument('--min_num_images', default=3, type=int)

    parser.add_argument('--batch_size', default=250, type=int)
    parser.add_argument('--input_size', default=112, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--stages', default=[35, 65, 95], type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--num_epoch', default=3, type=int)
    parser.add_argument('--gpu_id', default=[0], type=int, nargs='+', help='gpu id')
    parser.add_argument('--name', default='CelebA', type=str)
    parser.add_argument('--dataset', default='CelebA', type=str)
    parser.add_argument('--file_name', default='timm_from-scratch.csv', type=str)
    parser.add_argument('--seed', default=222, type=int)



    args = parser.parse_args()
    checkpoint_directory = os.path.join(args.checkpoints_root, args.backbone_name + '_' + args.head_name)
    Path(checkpoint_directory).mkdir(parents=True, exist_ok=True)



    p_images = {args.groups_to_modify[i]:args.p_images[i] for i in range(len(args.groups_to_modify))}
    p_identities = {args.groups_to_modify[i]:args.p_identities[i] for i in range(len(args.groups_to_modify))}
    args.p_images = p_images
    args.p_identities = p_identities

    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))



    ####################################################################################################################################
    # ======= data, model and test data =======#

    experiment = Experiment(
        api_key=user_cfg['comet_api_key'],
        project_name=args.project_name,
        workspace=user_cfg['comet_workspace'],
    )
    experiment.add_tag(args.backbone_name)
    
    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_test = prepare_data(args)



    ''' Model '''
    backbone = timm.create_model(args.backbone_name, 
                                 num_classes=0,
                                 pretrained=args.pretrained).to(device)
    config = timm.data.resolve_data_config({}, model=backbone)
    model_input_size = config['input_size']
    
    # get model's embedding size
    meta = pd.read_csv(user_cfg['metadata_file'])
    embedding_size = int(meta[meta['model_name'] == args.backbone_name].feature_dim)


    
    head = CosFace(in_features=embedding_size, out_features=num_class, device_id=range(torch.cuda.device_count()))
    train_criterion = FocalLoss(elementwise=True)

    ####################################################################################################################
    # ======= optimizer =======#

    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
    _, head_paras_wo_bn = separate_resnet_bn_paras(head)
    optimizer = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': args.weight_decay},
                               {'params': backbone_paras_only_bn}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=100)

    backbone, head = backbone.to(device), head.to(device)
    backbone, head, optimizer, epoch, batch, checkpoints_model_root = load_checkpoint(args, backbone, head, optimizer, dataloaders['train'], p_identities, p_images)
    backbone = nn.DataParallel(backbone)
    backbone, head = backbone.to(device), head.to(device)


    ####################################################################################################################
    # ======= training =======#

    print('Start training')
    with experiment.train():
        while epoch <= args.num_epoch:

            experiment.log_current_epoch(epoch)
            backbone.train()  # set to training mode
            head.train()
            meters = {}
            meters['loss'] = AverageMeter()
            meters['top5'] = AverageMeter()

            for inputs, labels, sens_attr, _ in tqdm(iter(dataloaders['train'])):

                inputs, labels = inputs.to(device), labels.to(device).long()
                features = backbone(inputs)
                outputs = head(features,labels)
                loss = train_criterion(outputs, labels).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                meters['loss'].update(loss.data.item(), inputs.size(0))
                meters['top5'].update(prec5.data.item(), inputs.size(0))

                batch += 1


            backbone.eval()  # set to testing mode
            head.eval()
            experiment.log_metric("Training Loss", meters['loss'].avg, step=epoch)
            experiment.log_metric("Training Acc5", meters['top5'].avg, step=epoch)


            '''For test data compute k-neighbors accuracy and multi-accuracy'''
            k_accuracy = True
            multilabel_accuracy = True
            loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all = evaluate(dataloaders['test'], train_criterion, backbone, head, embedding_size,
                                       k_accuracy = k_accuracy, multilabel_accuracy = multilabel_accuracy,
                                       demographic_to_labels = demographic_to_labels_test, test = True)
            
            # save outputs
            kacc_df, multi_df = None, None
            if k_accuracy:
                kacc_df = pd.DataFrame(np.array([list(indices_all),
                                                 list(nearest_id)]).T, 
                                       columns=['ids','epoch_'+str(epoch)]).astype(int)
            if multilabel_accuracy:
                multi_df = pd.DataFrame(np.array([list(indices_all),
                                                  list(predicted_all)]).T,
                                        columns=['ids','epoch_'+str(epoch)]).astype(int)
            add_column_to_file(checkpoint_directory,
                               '', epoch, 
                               multi_df = multi_df, kacc_df = kacc_df)


            results = {}
            results['Model'] = args.backbone_name
            results['seed'] = args.seed
            results['epoch'] = epoch
            for k in acc_k.keys():
                experiment.log_metric("Acc multi Test " + k, acc[k], epoch=epoch)
                experiment.log_metric("Acc k Test " + k, acc_k[k], epoch=epoch)
                experiment.log_metric("Intra Test " + k, intra[k], epoch=epoch)
                experiment.log_metric("Inter Test " + k, inter[k], epoch=epoch)

                results['Acc multi '+k] = (round(acc[k].item()*100, 3))
                results['Acc k '+k] = (round(acc_k[k].item()*100, 3))
                results['Intra '+k] = (round(intra[k], 3))
                results['Inter '+k] = (round(inter[k], 3))

            print(results)
            save_output_from_dict(user_cfg['output_dir'], results, args.file_name)


            epoch += 1

            # save checkpoints per epoch

            if (epoch == args.num_epoch) or (epoch % 1 == 0):
                checkpoint_name_to_save = os.path.join(checkpoint_directory,
                            "Checkpoint_Head_{}_Backbone_{}_Dataset_{}_p_idx{}_p_img{}_Epoch_{}.pth".
                            format(args.head_name, args.backbone_name, args.name, str(args.p_identities), str(args.p_images), str(epoch)))



                torch.save({'epoch': epoch,
                            'backbone_state_dict': backbone.module.state_dict(),
                            'head_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                             checkpoint_name_to_save)
