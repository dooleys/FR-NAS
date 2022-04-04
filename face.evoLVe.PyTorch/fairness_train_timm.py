from comet_ml import Experiment
import argparse
from tqdm import tqdm
import os
import torch
import torch.nn as nn
import torch.optim as optim

from head.metrics import CosFace
from loss.focal import FocalLoss
from util.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, \
    schedule_lr, AverageMeter, accuracy
from util.fairness_utils import evaluate
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

default_test_root = '/cmlscratch/sdooley1/data/CelebA/Img/img_align_celeba_splits/test/'
default_train_root = '/cmlscratch/sdooley1/data/CelebA/Img/img_align_celeba_splits/train/'


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_test_root', default=default_test_root)
    parser.add_argument('--data_train_root', default=default_train_root)
    parser.add_argument('--demographics', default= '/cmlscratch/sdooley1/data/CelebA/CelebA_demographics.txt')
    parser.add_argument('--backbone_name', default='resnet50')
    parser.add_argument('--pretrained', default=False)
    parser.add_argument('--project_name', default="from-scratch_no-resampling_adam")

    parser.add_argument('--checkpoints_root', default='/cmlscratch/sdooley1/FR-NAS/Checkpoints/timm_explore_few_epochs/')
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
    if not os.path.isdir(os.path.join(args.checkpoints_root, args.backbone_name + '_' + args.head_name)):
        os.mkdir(os.path.join(args.checkpoints_root, args.backbone_name + '_' + args.head_name))



    p_images = {args.groups_to_modify[i]:args.p_images[i] for i in range(len(args.groups_to_modify))}
    p_identities = {args.groups_to_modify[i]:args.p_identities[i] for i in range(len(args.groups_to_modify))}
    args.p_images = p_images
    args.p_identities = p_identities

    print("P identities: {}".format(args.p_identities))
    print("P images: {}".format(args.p_images))



    ####################################################################################################################################
    # ======= data, model and test data =======#

    experiment = Experiment(
        api_key="D1J58R7hYXPZzqZhrTIOe6GGQ",
        project_name=args.project_name,
        workspace="samueld",
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
    meta = pd.read_csv('/cmlscratch/sdooley1/timm_model_metadata.csv')
    embedding_size = int(meta[meta['model_name'] == args.backbone_name].feature_dim)


    
    head = CosFace(in_features=embedding_size, out_features=num_class, device_id=args.gpu_id)
    train_criterion = FocalLoss(elementwise=True)

    ####################################################################################################################
    # ======= optimizer =======#

    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
    _, head_paras_wo_bn = separate_resnet_bn_paras(head)
    optimizer = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': args.weight_decay},
                               {'params': backbone_paras_only_bn}], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=100)

    backbone = nn.DataParallel(backbone)
    #head = nn.DataParallel(head)
    backbone, head = backbone.to(device), head.to(device)

    backbone, head, optimizer, epoch, batch, checkpoints_model_root = load_checkpoint(args, backbone, head, optimizer, dataloaders['train'], p_identities, p_images)

    ####################################################################################################################################
    # ======= train & validation & save checkpoint =======#
#     num_epoch_warm_up = args.num_epoch // 25  # use the first 1/25 epochs to warm up
#     num_batch_warm_up = len(dataloaders['train']) * num_epoch_warm_up  # use the first 1/25 epochs to warm up

    
#     _, acc_test, acc_k_test, intra_test, inter_test, _,_,_,_,_ = evaluate(dataloaders['test'], train_criterion, backbone, head, embedding_size, k_accuracy = True, multilabel_accuracy = True, demographic_to_labels = demographic_to_labels_test, test = True)

#     results = {}
#     results['Model'] = args.backbone_name
#     results['seed'] = args.seed
#     results['epoch'] = -1
#     for k in acc_k_test.keys():
# #                 experiment.log_metric("Loss Train " + k, loss_train[k], step=epoch)
# #                 experiment.log_metric("Acc Train " + k, acc_train[k], step=epoch)
#         experiment.log_metric("Acc multi Test " + k, acc_test[k], epoch=-1)
#         experiment.log_metric("Acc k Test " + k, acc_k_test[k], epoch=-1)
#         experiment.log_metric("Intra Test " + k, intra_test[k], epoch=-1)
#         experiment.log_metric("Inter Test " + k, inter_test[k], epoch=-1)

#         results['Acc multi '+k] = (round(acc_test[k].item()*100, 3))
#         results['Acc k '+k] = (round(acc_k_test[k].item()*100, 3))
#         results['Intra '+k] = (round(intra_test[k], 3))
#         results['Inter '+k] = (round(inter_test[k], 3))

#     print(results)
#     save_output_from_dict('results_nooversampling', results, args.file_name)
 ####################################################################################################################################
    # ======= training =======#

    print('Start training')
    with experiment.train():
        while epoch < args.num_epoch:

            experiment.log_current_epoch(epoch)
            backbone.train()  # set to training mode
            head.train()
            meters = {}
            meters['loss'] = AverageMeter()
            meters['top5'] = AverageMeter()

#             if epoch in args.stages:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
#                 schedule_lr(optimizer)

            for inputs, labels, sens_attr in tqdm(iter(dataloaders['train'])):

#                 if batch + 1 <= num_batch_warm_up:  # adjust LR for each training batch during warm up
#                     warm_up_lr(batch + 1, num_batch_warm_up, args.lr, optimizer)

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


            '''For train data compute only multilabel accuracy'''
#             loss_train, acc_train, _, _, _, _,_,_,_,_ = evaluate(dataloaders['train'], train_criterion, backbone,head, embedding_size,
#                                                 k_accuracy = False, multilabel_accuracy = True,
#                                                 demographic_to_labels = demographic_to_labels_train, test = False)

            '''For test data compute only k-neighbors accuracy'''
            _, acc_test, acc_k_test, intra_test, inter_test, _,_,_,_,_ = evaluate(dataloaders['test'], train_criterion, backbone, head, embedding_size,
                                       k_accuracy = True, multilabel_accuracy = True,
                                       demographic_to_labels = demographic_to_labels_test, test = True)

            results = {}
            results['Model'] = args.backbone_name
            results['seed'] = args.seed
            results['epoch'] = epoch
            for k in acc_k_test.keys():
#                 experiment.log_metric("Loss Train " + k, loss_train[k], step=epoch)
#                 experiment.log_metric("Acc Train " + k, acc_train[k], step=epoch)
                experiment.log_metric("Acc multi Test " + k, acc_test[k], epoch=epoch)
                experiment.log_metric("Acc k Test " + k, acc_k_test[k], epoch=epoch)
                experiment.log_metric("Intra Test " + k, intra_test[k], epoch=epoch)
                experiment.log_metric("Inter Test " + k, inter_test[k], epoch=epoch)

                results['Acc multi '+k] = (round(acc_test[k].item()*100, 3))
                results['Acc k '+k] = (round(acc_k_test[k].item()*100, 3))
                results['Intra '+k] = (round(intra_test[k], 3))
                results['Inter '+k] = (round(inter_test[k], 3))

            print(results)
            save_output_from_dict('results_nooversampling', results, args.file_name)


            epoch += 1

            # save checkpoints per epoch

            if (epoch == args.num_epoch):
                checkpoint_name_to_save = os.path.join(args.checkpoints_root, args.backbone_name + '_' + args.head_name,
                            "Checkpoint_Head_{}_Backbone_{}_Dataset_{}_p_idx{}_p_img{}_Epoch_{}.pth".
                            format(args.head_name, args.backbone_name, args.name, str(args.p_identities), str(args.p_images), str(epoch)))



                torch.save({'epoch': epoch,
                            'backbone_state_dict': backbone.module.state_dict(),
                            'head_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                             checkpoint_name_to_save)
