from comet_ml import Experiment
import argparse
from tqdm import tqdm
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
from utils.utils_train import get_optimizer, get_scheduler, get_head

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_path', type=str)

    args = parser.parse_args()
    with open(args.config_path, "r") as ymlfile:
        opt = yaml.load(ymlfile, Loader=yaml.FullLoader)
        print(opt)

    if not os.path.isdir(
            os.path.join(
                opt['checkpoints_root'],
                opt['backbone'] + '_' + opt['head'] + '_' + opt['optimizer'])):
        os.mkdir(
            os.path.join(
                opt['checkpoints_root'],
                opt['backbone'] + '_' + opt['head'] + '_' + opt['optimizer']))
    p_images = {
        opt['groups_to_modify'][i]: opt['p_images'][i]
        for i in range(len(opt['groups_to_modify']))
    }
    p_identities = {
        opt['groups_to_modify'][i]: opt['p_identities'][i]
        for i in range(len(opt['groups_to_modify']))
    }
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

    dataloaders, num_class, demographic_to_labels_train, demographic_to_labels_test = prepare_data(
        opt)
    opt["num_class"] = num_class
    ''' Model '''
    backbone = timm.create_model(opt['backbone'],
                                 num_classes=0,
                                 pretrained=opt['pretrained']).to(device)
    config = timm.data.resolve_data_config({}, model=backbone)
    model_input_size = config['input_size']

    # get model's embedding size
    meta = pd.read_csv(opt['metadata_file'])
    embedding_size = int(
        meta[meta['model_name'] == opt['backbone']].feature_dim)
    opt["embedding_size"] = embedding_size

    head = get_head(opt)
    train_criterion = FocalLoss(elementwise=True)

    ####################################################################################################################
    # ======= optimizer =======#

    backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(
        backbone)
    _, head_paras_wo_bn = separate_resnet_bn_paras(head)
    optimizer = get_optimizer(opt, backbone, head)
    scheduler = get_scheduler(opt, optimizer)

    backbone, head, optimizer, epoch, batch, checkpoints_model_root = load_checkpoint(
        opt, backbone, head, optimizer, dataloaders['train'], p_identities,
        p_images)
    backbone = nn.DataParallel(backbone)
    backbone, head = backbone.to(device), head.to(device)

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

            for inputs, labels, sens_attr, _ in tqdm(iter(
                    dataloaders['train'])):

                #                 if batch + 1 <= num_batch_warm_up:  # adjust LR for each training batch during warm up
                #                     warm_up_lr(batch + 1, num_batch_warm_up, args.lr, optimizer)

                inputs, labels = inputs.to(device), labels.to(device).long()
                features = backbone(inputs)
                outputs, reg_loss = head(features, labels)
                loss = train_criterion(outputs, labels) + reg_loss
                loss = loss.mean()
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
            experiment.log_metric("Training Loss",
                                  meters['loss'].avg,
                                  step=epoch)
            experiment.log_metric("Training Acc5",
                                  meters['top5'].avg,
                                  step=epoch)
            '''For train data compute only multilabel accuracy'''
            #             loss_train, acc_train, _, _, _, _,_,_,_,_ = evaluate(dataloaders['train'], train_criterion, backbone,head, embedding_size,
            #                                                 k_accuracy = False, multilabel_accuracy = True,
            #                                                 demographic_to_labels = demographic_to_labels_train, test = False)
            '''For test data compute only k-neighbors accuracy and multi-accuracy'''
            k_accuracy = True
            multilabel_accuracy = True
            loss, acc, acc_k, predicted_all, intra, inter, angles_intra, angles_inter, correct, nearest_id, labels_all, indices_all, demographic_all = evaluate(
                dataloaders['test'],
                train_criterion,
                backbone,
                head,
                embedding_size,
                k_accuracy=k_accuracy,
                multilabel_accuracy=multilabel_accuracy,
                demographic_to_labels=demographic_to_labels_test,
                test=True)

            # save outputs
            kacc_df, multi_df = None, None
            if k_accuracy:
                kacc_df = pd.DataFrame(np.array(
                    [list(indices_all), list(nearest_id)]).T,
                                       columns=['ids', 'epoch_' + str(epoch)])
            if multilabel_accuracy:
                multi_df = pd.DataFrame(np.array(
                    [list(indices_all), list(predicted_all)]).T,
                                        columns=['ids', 'epoch_' + str(epoch)])
            add_column_to_file(checkpoint_directory,
                               '',
                               epoch,
                               multi_df=multi_df,
                               kacc_df=kacc_df)

            results = {}
            results['Model'] = args.backbone_name
            results['seed'] = args.seed
            results['epoch'] = epoch
            for k in acc_k.keys():
                experiment.log_metric("Acc multi Test " + k,
                                      acc_k[k],
                                      epoch=epoch)
                experiment.log_metric("Acc k Test " + k, acc_k[k], epoch=epoch)
                experiment.log_metric("Intra Test " + k, intra[k], epoch=epoch)
                experiment.log_metric("Inter Test " + k, inter[k], epoch=epoch)

                results['Acc multi ' + k] = (round(acc_k[k].item() * 100, 3))
                results['Acc k ' + k] = (round(acc_k[k].item() * 100, 3))
                results['Intra ' + k] = (round(intra[k], 3))
                results['Inter ' + k] = (round(inter[k], 3))
            print(results)
            save_output_from_dict(opt['out_dir'], results, opt["file_name"])

            epoch += 1

            # save checkpoints per epoch

            if (epoch == opt["num_epoch"]) or (epoch % 20 == 0):
                checkpoint_name_to_save = os.path.join(
                    opt["checkpoints_root"],
                    opt["backbone"] + '_' + opt["head"],
                    "Checkpoint_Head_{}_Backbone_{}_Dataset_{}_p_idx{}_p_img{}_Epoch_{}.pth"
                    .format(opts["head"], opt["backbone"], opt["name"],
                            str(opt["p_identities"]), str(opt["p_images"]),
                            str(epoch)))

                torch.save(
                    {
                        'epoch': epoch,
                        'backbone_state_dict': backbone.module.state_dict(),
                        'head_state_dict': head.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()
                    }, checkpoint_name_to_save)
