from comet_ml import Experiment
import argparse
from tqdm import tqdm
from config import user_configs
import os
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from head.metrics import CosFace
from loss.focal import FocalLoss
from utils.utils import separate_resnet_bn_paras, warm_up_lr, load_checkpoint, \
    schedule_lr, AverageMeter, accuracy
from utils.fairness_utils import evaluate
from utils.data_utils_balanced import prepare_data
import numpy as np
import pandas as pd
import random
import timm
from utils.utils import save_output_from_dict

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

    parser.add_argument('--config_path',type=str)

    args = parser.parse_args()
    with open(args.config_path, "r") as ymlfile:
        opt = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print(opt)

    if not os.path.isdir(os.path.join(opt['checkpoints_root'], opt['backbone'] + '_' + opt['head'] + '_' + opt['optimizer'])):
        os.mkdir(os.path.join(opt['checkpoints_root'], opt['backbone'] + '_' + opt['head']+ '_' + opt['optimizer']))
    p_images = {opt['groups_to_modify'][i]:opt['p_images'][i] for i in range(len(opt['groups_to_modify']))}
    p_identities = {opt['groups_to_modify'][i]:opt['p_identities'][i] for i in range(len(opt['groups_to_modify']))}
    opt['p_images'] = p_images
    opt['p_identities'] = p_identities

    print("P identities: {}".format(opt['p_identities']))
    print("P images: {}".format(opt['p_images']))



    ####################################################################################################################################
    # ======= data, model and test data =======#

    experiment = Experiment(
        api_key=opt['comet_api_key'],
        project_name=opt['project_name'],
        workspace=opt['comet_workspace'],
    )
    experiment.add_tag(opt['backbone'])
    
    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_test = prepare_data(opt)



    ''' Model '''
    backbone = timm.create_model(opt['backbone'], 
                                 num_classes=0,
                                 pretrained=opt['pretrained']).to(device)
    config = timm.data.resolve_data_config({}, model=backbone)
    model_input_size = config['input_size']
    
    # get model's embedding size
    meta = pd.read_csv(opt['metadata_file'])
    embedding_size = int(meta[meta['model_name'] == opt['backbone']].feature_dim)


    
    head = CosFace(in_features=embedding_size, out_features=num_class, device_id=range(torch.cuda.device_count()))
    train_criterion = FocalLoss(elementwise=True)

    ####################################################################################################################
    # ======= optimizer =======#

    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(backbone)
    _, head_paras_wo_bn = separate_resnet_bn_paras(head)
    if opt["optimizer"]=="Adam":
       optimizer = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': opt["weight_decay"]},
                               {'params': backbone_paras_only_bn}], lr=opt["lr"], betas=opt["betas"], eps=opt["eps"])
    #optimizer = optim.Adam([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': opt["weight_decay"]},
    #                           {'params': backbone_paras_only_bn}], lr=opt["lr"])
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, total_iters=100)

    backbone, head, optimizer, epoch, batch, checkpoints_model_root = load_checkpoint(opt, backbone, head, optimizer, dataloaders['train'], p_identities, p_images)
    backbone = nn.DataParallel(backbone)
    backbone, head = backbone.to(device), head.to(device)


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
        while epoch <= opt["num_epoch"]:

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
            results['Model'] = opt["backbone"]
            results['seed'] = opt["seed"]
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
            save_output_from_dict(opt['out_dir'], results, opt["file_name"])


            epoch += 1

            # save checkpoints per epoch

            if (epoch == opt["num_epoch"]) or (epoch % 20 == 0):
                checkpoint_name_to_save = os.path.join(opt["checkpoints_root"], opt["backbone"] + '_' + opt["head"],
                            "Checkpoint_Head_{}_Backbone_{}_Dataset_{}_p_idx{}_p_img{}_Epoch_{}.pth".
                            format(opts["head"], opt["backbone"], opt["name"], str(opt["p_identities"]), str(opt["p_images"]), str(epoch)))



                torch.save({'epoch': epoch,
                            'backbone_state_dict': backbone.module.state_dict(),
                            'head_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                             checkpoint_name_to_save)
