import argparse
import torch
import os
import torch.nn as nn
print('Imported torch')
from util.fairness_utils import evaluate, most_least_variant_classes
from util.data_utils_balanced import load_dict_as_str
from util.data_utils_balanced import ImageFolderWithProtectedAttributes
from backbone.model_resnet import ResNet_50, ResNet_101, ResNet_152#, ResNet_18
from backbone.model_irse import IR_50, IR_101, IR_152, IR_SE_50, IR_SE_101, IR_SE_152
from backbone.MobileFaceNets import MobileFaceNet
from backbone.GhostNet import GhostNet
import numpy as np
import torchvision.transforms as transforms
import random
import json
import ast
from util.utils import save_output_from_dict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(222)
torch.cuda.manual_seed_all(222)
np.random.seed(222)
random.seed(222)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

default_test_root = '/cmlscratch/sdooley1/data/CelebA/Img/img_align_celeba_splits/test/'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_test_root', default=default_test_root)
    parser.add_argument('--checkpoint', default="backbone_ir50_ms1m_epoch63.pth")
    parser.add_argument('--demographics', default= '/cmlscratch/sdooley1/data/CelebA/CelebA_demographics.txt')
    parser.add_argument('--backbone_name', default='IR_50')

    parser.add_argument('--groups_to_modify', default=['male', 'female'], type=str, nargs = '+')
    #parser.add_argument('--p_identities', default=[1.0, 0.724], type=float, nargs = '+')
    parser.add_argument('--p_identities', default=[1.0, 1.0], type=float, nargs = '+')
    #parser.add_argument('--p_images', default=[1.0, 0.893], type=float,  nargs = '+')
    parser.add_argument('--p_images', default=[1.0, 1.0], type=float,  nargs = '+')

    parser.add_argument('--batch_size', default=250, type=int)
    parser.add_argument('--input_size', default=[112,112], type=int)
    parser.add_argument('--embedding_size', default=512, type=int)
    parser.add_argument('--mean', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--std', default=[0.5, 0.5, 0.5], type=int)
    parser.add_argument('--seed', default=[222], type=int, nargs = '+')
    parser.add_argument('--file_name', default='bias_celeba_balanced_ratio_cos.csv', type=str)


    args = parser.parse_args()
    p_images = {args.groups_to_modify[i]:args.p_images[i] for i in range(len(args.groups_to_modify))}
    p_identities = {args.groups_to_modify[i]:args.p_identities[i] for i in range(len(args.groups_to_modify))}
    print('We are here')
    test_transform = transforms.Compose([
        transforms.Resize([int(128 * args.input_size[0] / 112), int(128 * args.input_size[1] / 112)]),
        transforms.CenterCrop([args.input_size[0], args.input_size[1]]),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean,
                             std=args.std)])

    backbone_dict = {#'ResNet_18': ResNet_18(args.input_size),
                     'MobileFaceNet': MobileFaceNet(embedding_size=512, out_h=7, out_w = 7),
                     'GhostNet': GhostNet(),
                     'ResNet_50': ResNet_50(args.input_size),
                     'ResNet_101': ResNet_101(args.input_size),
                     'ResNet_152': ResNet_152(args.input_size),
                     'IR_50': IR_50(args.input_size),
                     'IR_101': IR_101(args.input_size),
                     'IR_152': IR_152(args.input_size),
                     'IR_SE_50': IR_SE_50(args.input_size),
                     'IR_SE_101': IR_SE_101(args.input_size),
                     'IR_SE_152': IR_SE_152(args.input_size)}


    '''Load Data'''
    # two dictionaries mapping demographic to classes
    demographic_to_classes = load_dict_as_str(args.demographics)
    classes_to_demographic = {cl: dem for dem, classes in demographic_to_classes.items() for cl in classes}

    results = {}
    results['Model'] = args.checkpoint
    results['seed'] = args.seed
    results['Idx training ratio male'] = 1.0
    results['Idx training ratio female'] = 1.0
    results['Img training ratio male'] = 1.0
    results['Img training ratio female'] = 1.0

    results['Data'] = args.data_test_root
    results['P identities male'] = p_identities['male']
    results['P identities female'] = p_identities['female']
    results['P images male'] = p_images['male']
    results['P images female'] = p_images['female']

    results['Acc m'] = []
    results['Acc f'] = []
    results['Intra m'] = []
    results['Intra f'] = []
    results['Inter m'] = []
    results['Inter f'] = []
    results['Ratio m'] = []
    results['Ratio f'] = []

    for s in args.seed:

        data = ImageFolderWithProtectedAttributes(args.data_test_root, transform=test_transform,
                                                                     demographic_to_all_classes=demographic_to_classes,
                                                                     all_classes_to_demographic = classes_to_demographic,
                                                                     p_identities = p_identities,
                                                                     p_images = p_images,
                                                                     min_num = 3,
                                                                     ref_num_images = 7000,
                                                                     seed = s)


        ''' Labels <-> Demographics'''
        demographic_to_labels = data.demographic_to_idx
        samples = data.samples
        label_to_demographic = {label: dem for dem, labels in demographic_to_labels.items() for label in labels}
        ''' Class <-> Label'''
        class_to_idx = data.class_to_idx
        idx_to_class = {idx: cl for cl, idx in class_to_idx.items()}


        ''' DataLoader + Model'''
        dataloader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)
        backbone = backbone_dict[args.backbone_name]
        model_checkpoint_path = os.path.join('/cmlscratch/sdooley1/models/', args.checkpoint)
        backbone.load_state_dict(torch.load(model_checkpoint_path))
        backbone = nn.DataParallel(backbone).to(device)

        if args.data_test_root == default_test_root:
            _,_, acc_k, intra, inter, angles_intra, angles_inter, correct, labels_all, demographic_all = evaluate(
                dataloader, None, backbone, None, args.embedding_size,
                k_accuracy = True, multilabel_accuracy = False,  demographic_to_labels = demographic_to_labels)

        correct, labels_all, demographic_all = np.array(correct), np.array(labels_all), np.array(demographic_all)


        ''' Identity Statistics'''
        num_male_identities = len(demographic_to_labels['male'])
        num_female_identities = len(demographic_to_labels['female'])
        print('Num of male identities is {}'.format(num_male_identities))
        print('Num of female identities is {}'.format(num_female_identities))
        num_male_images = sum(demographic_all == 'male')
        num_female_images = sum(demographic_all == 'female')
        print('Num of male images is {}'.format(num_male_images))
        print('Num of female images is {}'.format(num_female_images))
        av_num_male_images = num_male_images/num_male_identities
        av_num_female_images = num_female_images/num_female_identities
        print('Av. Num of male images is {}'.format(av_num_male_images))
        print('Av. Num of female images is {}'.format(av_num_female_images))


        print('Accuracies are {}'.format(acc_k))
        print('Intra variances are {}'.format(intra))
        print('Inter variances are {}'.format(inter))
        #print('Classes with least variation are ', least_var_classes)
        #print('Classes with most variation are ', most_var_classes)

        results['Acc m'].append(round(acc_k['male'].item(), 3))
        results['Acc f'].append(round(acc_k['female'].item(), 3))
        results['Intra m'].append(round(intra['male'], 3))
        results['Intra f'].append(round(intra['female'], 3))
        results['Inter m'].append(round(inter['male'], 3))
        results['Inter f'].append(round(inter['female'], 3))
        results['Ratio m'].append(round(intra['male']/inter['male'], 3))
        results['Ratio f'].append(round(intra['female']/inter['female'], 3))

    results['Acc m'] = np.array(results['Acc m']).mean()
    results['Acc f'] = np.array(results['Acc f']).mean()
    results['Intra m'] = np.array(results['Intra m']).mean()
    results['Intra f'] = np.array(results['Intra f']).mean()
    results['Inter m'] = np.array(results['Inter m']).mean()
    results['Inter f'] = np.array(results['Inter f']).mean()
    results['Ratio m'] = np.array(results['Ratio m']).mean()
    results['Ratio f'] = np.array(results['Ratio f']).mean()
    print(results)
    save_output_from_dict('results_nooversampling', results, args.file_name)