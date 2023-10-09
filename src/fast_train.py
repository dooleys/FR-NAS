from comet_ml import Experiment
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from head.metrics import ArcFace, CosFace, SphereFace, Am_softmax, ArcFace_Half
from loss.focal import FocalLoss
from util.utils import make_weights_for_balanced_classes, get_val_data, separate_irse_bn_paras, \
    separate_resnet_bn_paras, warm_up_lr, schedule_lr, perform_val, get_time, buffer_val, AverageMeter, accuracy
import argparse
# from tensorboardX import SummaryWriter
from tqdm import tqdm
import os
from apex import amp

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_train_root', default='/fs/cml-datasets/MS_Celeb_aligned_112')
    parser.add_argument('--data_test_root', default='/cmlscratch/vcherepa/test_data')
    parser.add_argument('--checkpoints_root', default='/cmlscratch/sdooley1/checkpoints')

    parser.add_argument('--backbone_name', default='IR_SE_50')
    parser.add_argument('--head_name', default='ArcFace')
    parser.add_argument('--loss_name', default='Focal')

    parser.add_argument('--input_size', default=[112, 112], type=int)
    parser.add_argument('--rgb_mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--rgb_std', default=[0.5, 0.5, 0.5], type=int)

    parser.add_argument('--embedding_size', default=512, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--drop_last', default=True, type=bool)

    parser.add_argument('--lr', default=0.1, type=float)
    parser.add_argument('--num_epoch', default=125, type=int)
    parser.add_argument('--weight_decay', default=5e-4, type=float)

    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--multi_gpu', default=True, type=bool)
    parser.add_argument('--stages', default=[35, 65, 95], type=bool)

    parser.add_argument('--gpu_id', default=[0, 1, 2, 3], type=int, nargs='+', help='gpu id')
    parser.add_argument('--pin_memory', default=True, type=bool)
    parser.add_argument('--num_workers', default=0, type=int)
    parser.add_argument('--seed', default=1337, type=int)
    parser.add_argument('--disp_freq', default=None, type=int)

    args = parser.parse_args()

    seed = args.seed  # random seed for reproduce results
    torch.manual_seed(seed)

    data_train_root = args.data_train_root  # the parent root where your train/val/test data are stored
    data_test_root = args.data_test_root  # the parent root where your train/val/test data are stored
    checkpoints_root = args.checkpoints_root  # the root to resume training from a saved checkpoint

    backbone_name = args.backbone_name  # support: ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152']
    head_name = args.head_name  # support: ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']
    loss_name = args.loss_name  # support: ['Focal', 'Softmax']

    input_size = args.input_size
    rgb_mean = args.rgb_mean  # for normalize inputs
    rgb_std = args.rgb_std
    embedding_size = args.embedding_size  # feature dimension
    batch_size = args.batch_size
    drop_last = args.drop_last  # whether drop the last batch to ensure consistent batch_norm statistics
    lr = args.lr  # initial LR
    stages = args.stages
    num_epoch = args.num_epoch
    weight_decay = args.weight_decay
    momentum = args.momentum

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    multi_gpu = args.multi_gpu  # flag to use multiple GPUs
    gpu_id = args.gpu_id  # specify your GPU ids
    pin_memory = args.pin_memory
    num_workers = args.num_workers
    disp_freq = args.disp_freq
    print("=" * 60)

    ####################################################################################################################################
    # ======= data, weights and test data =======#

    experiment = Experiment(api_key='a60f8d43b0fdabea302e19e0ad7db2e2d77f4cf3',
                            project_name="FairNAS_MSCeleb" + str(backbone_name) + str(head_name))

    train_transform = transforms.Compose([
        # refer to https://pytorch.org/docs/stable/torchvision/transforms.html for more build-in online data augmentation
        transforms.Resize([int(128 * input_size[0] / 112), int(128 * input_size[0] / 112)]),  # smaller side resized
        transforms.RandomCrop([input_size[0], input_size[1]]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=rgb_mean,
                             std=rgb_std),
    ])

    dataset_train = datasets.ImageFolder(os.path.join(data_train_root, 'imgs'), train_transform)

    print("=" * 60)
    # create a weighted random sampler to process imbalanced data
    weights = make_weights_for_balanced_classes(dataset_train.imgs, len(dataset_train.classes))
    weights = torch.DoubleTensor(weights)
    sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))
    print("=" * 60)

    train_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=batch_size, sampler=sampler, pin_memory=pin_memory,
        num_workers=num_workers, drop_last=drop_last
    )

    num_class = len(train_loader.dataset.classes)
    print("Number of Training Classes: {}".format(num_class))

    lfw, calfw, cplfw, vgg2_fp, lfw_issame, calfw_issame, cplfw_issame, vgg2_fp_issame = get_val_data(data_test_root)

    ####################################################################################################################################
    # ======= model & loss & optimizer =======#
    backbone_dict = {'ResNet_50': ResNet_50(input_size),
                     'ResNet_18': ResNet_18(input_size),
                     'ResNet_101': ResNet_101(input_size),
                     'ResNet_152': ResNet_152(input_size),
                     'IR_50': IR_50(input_size),
                     'IR_101': IR_101(input_size),
                     'IR_152': IR_152(input_size),
                     'IR_SE_50': IR_SE_50(input_size),
                     'IR_SE_101': IR_SE_101(input_size),
                     'IR_SE_152': IR_SE_152(input_size)}
    backbone = backbone_dict[backbone_name]
    print("=" * 60)
    print(backbone)
    print("{} Backbone Generated".format(backbone_name))
    print("=" * 60)

    head_dict = {'ArcFace': ArcFace_Half(in_features=embedding_size, out_features=num_class, device_id=gpu_id),
                 'CosFace': CosFace(in_features=embedding_size, out_features=num_class, device_id=gpu_id),
                 'SphereFace': SphereFace(in_features=embedding_size, out_features=num_class, device_id=gpu_id),
                 'Am_softmax': Am_softmax(in_features=embedding_size, out_features=num_class, device_id=gpu_id)}
    head = head_dict[head_name]
    print("=" * 60)
    print(head)
    print("{} Head Generated".format(head_name))
    print("=" * 60)

    loss_dict = {'Focal': FocalLoss(),
                 'Softmax': nn.CrossEntropyLoss()}
    loss_criterion = loss_dict[loss_name]
    print("=" * 60)
    print(loss_criterion)
    print("{} Loss Generated".format(loss_name))
    print("=" * 60)

    if backbone_name.find("IR") >= 0:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_irse_bn_paras(
            backbone)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_irse_bn_paras(head)
    else:
        backbone_paras_only_bn, backbone_paras_wo_bn = separate_resnet_bn_paras(
            backbone)  # separate batch_norm parameters from others; do not do weight decay for batch_norm parameters to improve the generalizability
        _, head_paras_wo_bn = separate_resnet_bn_paras(head)

    optimizer = optim.SGD([{'params': backbone_paras_wo_bn + head_paras_wo_bn, 'weight_decay': weight_decay},
                           {'params': backbone_paras_only_bn}], lr=lr, momentum=momentum)
    [backbone, head], optimizer = amp.initialize([backbone.to(device), head.to(device)], optimizer, opt_level="O1",
                                                 loss_scale=128)

    print("=" * 60)
    print(optimizer)
    print("Optimizers Generated")
    print("=" * 60)

    ####################################################################################################################################
    # ======= load model weights and optimizers from checkpoints =======#

    # resume from a checkpoint
    name = "Checkpoint_Head_" + str(head_name) + "_Backbone_" + str(backbone_name) + '_Epoch_'
    checkpoints_model_root = os.path.join(checkpoints_root, str(backbone_name) + '_' + str(head_name))
    if not os.path.exists(checkpoints_model_root):
        os.mkdir(checkpoints_model_root)

    potential_checkpoints = [chckpt for chckpt in os.listdir(checkpoints_model_root) if chckpt.startswith(name)]
    print('Found checkpoints for this model:', potential_checkpoints)

    if len(potential_checkpoints) != 0:
        # find the latest checkpoint
        epoch_numbers = []
        for chckpt in potential_checkpoints:
            epoch_numbers.append([int(num) for num in chckpt[-8:].replace('.', '_').split('_') if num.isdigit()])
        last_checkpoint = name + str(max(epoch_numbers)[0]) + '.pth'

        print("Loading Checkpoint '{}'".format(os.path.join(checkpoints_model_root, last_checkpoint)))
        checkpoint = torch.load(os.path.join(checkpoints_model_root, last_checkpoint))
        backbone.load_state_dict(checkpoint['backbone_state_dict'])
        head.load_state_dict(checkpoint['head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        batch = len(train_loader) * epoch
    else:
        print("No Checkpoints Found at '{}'. Please Have a Check or Continue to Train from Scratch".format(
            checkpoints_model_root))
        epoch = 0
        batch = 0
    print("=" * 60)

    if multi_gpu:
        backbone = nn.DataParallel(backbone)
        backbone = backbone.to(device)
    else:
        backbone = backbone.to(device)

    ####################################################################################################################################
    # ======= train & validation & save checkpoint =======#
    if disp_freq == None:
        disp_freq = len(train_loader) // 1000  # frequency to display training loss & acc
    num_epoch_warm_up = num_epoch // 25  # use the first 1/25 epochs to warm up
    num_batch_warm_up = len(train_loader) * num_epoch_warm_up  # use the first 1/25 epochs to warm up

    ####################################################################################################################################
    # ======= training =======#

    with experiment.train():

        while epoch <= num_epoch:
            experiment.log_current_epoch(epoch)
            backbone.train()  # set to training mode
            head.train()

            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()

            if epoch == stages[
                0]:  # adjust LR for each training stage after warm up, you can also choose to adjust LR manually (with slight modification) once plaueau observed
                schedule_lr(optimizer)
            if epoch == stages[1]:
                schedule_lr(optimizer)
            if epoch == stages[2]:
                schedule_lr(optimizer)

            for inputs, labels in tqdm(iter(train_loader)):

                if batch + 1 <= num_batch_warm_up:  # adjust LR for each training batch during warm up
                    warm_up_lr(batch + 1, num_batch_warm_up, lr, optimizer)

                # compute output
                inputs = inputs.to(device)
                labels = labels.to(device).long()
                features = backbone(inputs)
                outputs = head(features, labels)
                loss = loss_criterion(outputs, labels)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(outputs.data, labels, topk=(1, 5))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step()

                # dispaly training loss & acc every DISP_FREQ
                if ((batch + 1) % disp_freq == 0) and batch != 0:
                    print("=" * 60)
                    print('Epoch {}/{} Batch {}/{}\t'
                          'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                          'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                          'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                        epoch + 1, num_epoch, batch + 1, len(train_loader) * num_epoch, loss=losses, top1=top1,
                        top5=top5))

                    experiment.log_metric("training loss", losses.avg, step=batch)
                    experiment.log_metric("training prec1", top1.avg, step=batch)
                    experiment.log_metric("training prec5", top5.avg, step=batch)
                    experiment.log_metric("lr", optimizer.param_groups[0]['lr'], step=batch)

                batch += 1  # batch index

            # training statistics per epoch (buffer for visualization)
            epoch_loss = losses.avg
            epoch_acc = top1.avg
            print("=" * 60)
            print('Epoch: {}/{}\t'
                  'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Training Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch + 1, num_epoch, loss=losses, top1=top1, top5=top5))
            print("=" * 60)

            experiment.log_metric("epoch training loss", epoch_loss, step=epoch)
            experiment.log_metric("epoch training acc", epoch_acc, step=epoch)

            # perform validation & save checkpoints per epoch
            print("=" * 60)
            print("Perform Evaluation on LFW, CFP_FF, CFP_FP, AgeDB, CALFW, CPLFW and VGG2_FP, and Save Checkpoints...")
            accuracy_lfw, best_threshold_lfw, roc_curve_lfw = perform_val(multi_gpu, device, embedding_size, batch_size,
                                                                          backbone, lfw, lfw_issame)
            experiment.log_metric("accuracy lfw", accuracy_lfw, step=epoch)

            accuracy_calfw, best_threshold_calfw, roc_curve_calfw = perform_val(multi_gpu, device, embedding_size,
                                                                                batch_size, backbone, calfw,
                                                                                calfw_issame)
            experiment.log_metric("accuracy calfw", accuracy_calfw, step=epoch)

            accuracy_cplfw, best_threshold_cplfw, roc_curve_cplfw = perform_val(multi_gpu, device, embedding_size,
                                                                                batch_size, backbone, cplfw,
                                                                                cplfw_issame)
            experiment.log_metric("accuracy cplfw", accuracy_cplfw, step=epoch)

            accuracy_vgg2_fp, best_threshold_vgg2_fp, roc_curve_vgg2_fp = perform_val(multi_gpu, device, embedding_size,
                                                                                      batch_size, backbone, vgg2_fp,
                                                                                      vgg2_fp_issame)
            experiment.log_metric("accuracy vgg2", accuracy_vgg2_fp, step=epoch)

            print(
                "Epoch {}/{}, Evaluation: LFW Acc: {}, CALFW Acc: {}, CPLFW Acc: {}, VGG2_FP Acc: {}".format(epoch + 1,
                                                                                                             num_epoch,
                                                                                                             accuracy_lfw,
                                                                                                             accuracy_calfw,
                                                                                                             accuracy_cplfw,
                                                                                                             accuracy_vgg2_fp))
            print("=" * 60)
            epoch += 1

            # save checkpoints per epoch

            if multi_gpu:
                torch.save({'epoch': epoch,
                            'backbone_state_dict': backbone.module.state_dict(),
                            'head_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoints_model_root,
                                        "Checkpoint_Head_{}_Backbone_{}_Epoch_{}.pth".format(head_name, backbone_name,
                                                                                             epoch)))
            else:
                torch.save({'epoch': epoch,
                            'backbone_state_dict': backbone.state_dict(),
                            'head_state_dict': head.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(checkpoints_model_root,
                                        "Checkpoint_Head_{}_Backbone_{}_Epoch_{}.pth".format(head_name, backbone_name,
                                                                                             epoch)))