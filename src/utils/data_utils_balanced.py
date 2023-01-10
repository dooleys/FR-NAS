import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import random
import json
import os
from PIL import Image
import numpy as np

seed = 666
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def load_dict(filename):
    with open(filename) as f:
        dict = json.loads(f.read())
    return dict

def load_dict_as_str(filename):
    with open(filename) as f:
        dict = json.loads(f.read())
        for k in dict.keys():
            for i in range(len(dict[k])):
                dict[k][i] = str(dict[k][i])
    return dict


def save_dict(dict, filename):
    with open(filename, 'w') as f:
        f.write(json.dumps(dict))


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# TODO: specify the return type
def accimage_loader(path: str) -> Any:
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path: str) -> Any:
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def has_file_allowed_extension(filename: str, extensions: Tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)


def make_dataset(
    directory: str,
    class_to_idx: Optional[Dict[str, int]] = None,
    extensions: Optional[Tuple[str, ...]] = None,
    is_valid_file: Optional[Callable[[str], bool]] = None,
    p_images = None,
    all_classes_to_demographic = None,
    min_num = 2,
    ref_num_images = 0,
    seed = 222,
) -> List[Tuple[str, int]]:
    """Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str): root dataset directory
        class_to_idx (Optional[Dict[str, int]]): Dictionary mapping class name to class index. If omitted, is generated
            by :func:`find_classes`.
        extensions (optional): A list of allowed extensions.
            Either extensions or is_valid_file should be passed. Defaults to None.
        is_valid_file (optional): A function that takes path of a file
            and checks if the file is a valid file
            (used to check of corrupt files) both extensions and
            is_valid_file should not be passed. Defaults to None.
    Raises:
        ValueError: In case ``class_to_idx`` is empty.
        ValueError: In case ``extensions`` and ``is_valid_file`` are None or both are not None.
        FileNotFoundError: In case no valid file was found for any class.
    Returns:
        List[Tuple[str, int]]: samples of a form (path_to_sample, class)
    """
    directory = os.path.expanduser(directory)

    if not class_to_idx:
        raise ValueError("'class_to_index' must have at least one entry to collect any samples.")

    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")

    if extensions is not None:

        def is_valid_file(x: str) -> bool:
            return has_file_allowed_extension(x, cast(Tuple[str, ...], extensions))

    is_valid_file = cast(Callable[[str], bool], is_valid_file)

    instances_all = {dem: [] for dem in p_images.keys()}
    instances_essential = {dem: [] for dem in p_images.keys()}
    instances_additional = {dem: [] for dem in p_images.keys()}
    available_classes = set()
    count = [0]*len(class_to_idx.keys())
    for target_class in sorted(class_to_idx.keys()):
        label = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        # find demographic group
        demographic = all_classes_to_demographic[target_class]
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):

                    item = path, label, demographic
                    if count[label] < min_num:
                        instances_essential[demographic].append(item)
                    else:
                        instances_additional[demographic].append(item)
                    count[label] += 1

                    if target_class not in available_classes:
                        available_classes.add(target_class)

    num_additional_images_to_keep = {dem: max(0,int(ref_num_images*p_images[dem])-len(instances_essential[dem])) for dem in p_images.keys()}
    for dem in list(num_additional_images_to_keep.keys()):
        random.seed(seed)
        print('Overall # of images for {} available is {}'.format(dem, len(instances_essential[dem] + instances_additional[dem])))
        instances_additional[dem] = random.sample(instances_additional[dem], k=num_additional_images_to_keep[dem])
        instances_all[dem] = instances_essential[dem] + instances_additional[dem]
        print('# images selected for {} is {}'.format(dem, len(instances_all[dem])))
    instances = []
    for dem in list(instances_all.keys()):
        instances += instances_all[dem]
    empty_classes = available_classes - set(list(class_to_idx.keys()))
    if empty_classes:
        msg = f"Found no valid file for the classes {', '.join(sorted(empty_classes))}. "
        if extensions is not None:
            msg += f"Supported extensions are: {', '.join(extensions)}"
        raise FileNotFoundError(msg)

    return instances




class ImageFolderWithProtectedAttributes(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    class: original name of the class
    idx: label
    """

    def __init__(self, root, loader=default_loader, extensions=IMG_EXTENSIONS, transform=None,
                 target_transform=None, is_valid_file=None, demographic_to_all_classes=None, all_classes_to_demographic = None,
                 p_identities = None, p_images = None, min_num = 2, ref_num_images = None, seed = 1):

        super(ImageFolderWithProtectedAttributes, self).__init__(root, transform=transform, target_transform=target_transform)

        classes, class_to_idx = self.classes, self.class_to_idx

        # create demographic to classes containing only train or test classes + we need to delete classes with few images
        self.demographic_to_classes = {}

        for dem in list(demographic_to_all_classes.keys()):
            self.demographic_to_classes[dem] = []
            for i, cl in enumerate(demographic_to_all_classes[dem]):
                if (cl in class_to_idx.keys()):
                    num_imgs = len(os.listdir(os.path.join(root, cl)))
                    if (num_imgs >= min_num):
                        self.demographic_to_classes[dem].append(cl)

        # getting the minimum number of identities
        ref_num_identities = min([len(self.demographic_to_classes[dem]) for dem in self.demographic_to_classes.keys()])
        # shuffle data
        random.seed(seed)

        for dem in list(self.demographic_to_classes.keys()):
            random.shuffle(self.demographic_to_classes[dem])

        # change classes and class_to_idx here based on balance ratio
        for dem in list(p_identities.keys()):
            desired_num = int(ref_num_identities * p_identities[dem])
            # change labels that we want to keep
            self.demographic_to_classes[dem] = self.demographic_to_classes[dem][0:desired_num]

        # update classes used for training/testing + update class_to_idx
        classes = sum(self.demographic_to_classes.values(), [])
        class_to_idx = {classes[i] : i for i in range(len(classes))}

        classes = class_to_idx.keys() # original classes not index

        # create demographic to idx dict
        self.demographic_to_idx = {}
        for dem in list(self.demographic_to_classes.keys()):
            self.demographic_to_idx[dem] = []
            for cl in self.demographic_to_classes[dem]:
                self.demographic_to_idx[dem].append(class_to_idx[cl])

        #####
        samples = make_dataset(self.root, class_to_idx, extensions, is_valid_file, p_images, all_classes_to_demographic, min_num, ref_num_images, seed)

        if len(samples) == 0:
            raise (RuntimeError("Found 0 files in subfolders of: " + self.root + "\n"
                                                                                 "Supported extensions are: " + ",".join(
                extensions)))

        self.loader = loader
        self.extensions = extensions

        self.classes = list(classes)
        #self.class_to_idx = dict(class_to_idx)
        self.samples = list(samples)
        #print(samples)
        self.targets = [s[1] for s in samples]
        self.imgs = list(self.samples)
        self.attributes = [s[2] for s in samples]
        self.transform = transform

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        #original_tuple = super(ImageFolderWithProtectedAttributesModify, self).__getitem__(index)
        #print("Index",index)
        path, target, sens_attr = self.samples[index]
        img = Image.open(path)
        img.convert('RGB')
        img = self.transform(img)
        return (img, target, sens_attr, index)


def balanced_weights(images, nclasses, attr=1):
    '''
        Make a vector of weights for each image in the dataset, based
        on class frequency. The returned vector of weights can be used
        to create a WeightedRandomSampler for a DataLoader to have
        class balancing when sampling for a training batch.
            images - torchvisionDataset.imgs
            nclasses - len(torchvisionDataset.classes)
        https://discuss.pytorch.org/t/balanced-sampling-between-classes-with-torchvision-dataloader/2703/3
    '''
    count = [0] * nclasses
    for item in images:
        # print(item[attr])
        count[item[attr]] += 1  # item is (img-data, label-id)
    weight_per_class = [0.] * nclasses
    N = float(sum(count))  # total number of images
    for i in range(nclasses):
        if float(count[i]) != 0:
            weight_per_class[i] = N / float(count[i])
        else:
            weight_per_class[i] = 0
    weight = [0] * len(images)
    for idx, val in enumerate(images):
        weight[idx] = weight_per_class[val[attr]]

    return weight

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def prepare_data(args):
    # function prepares data: loads images and prepares dataloaders
    # demographic classes is a dict containing classes corresponding to each demographic group

    train_transform = transforms.Compose([
        transforms.Resize([int(128 * args.input_size / 112), int(128 * args.input_size / 112)]),
        transforms.RandomCrop([args.input_size, args.input_size]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean,
                             std=args.std),
    ])
    test_transform = transforms.Compose([
        transforms.Resize([int(128 * args.input_size / 112), int(128 * args.input_size / 112)]),
        transforms.CenterCrop([args.input_size, args.input_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=args.mean,
                             std=args.std)])

    ###################################################################################################################
    # ======= data, loss, network =======#
    demographic_to_all_classes = load_dict_as_str(args.demographics_file)
    all_classes_to_demographic = {cl: dem for dem, classes in demographic_to_all_classes.items() for cl in classes}

    if args.dataset == 'InterRace':
        num_ref_images_train = 70000
        num_ref_images_test = 9897
    elif args.dataset == 'CelebA':
        num_ref_images_train = 67562
        num_ref_images_test = 7636
    elif args.dataset =='BUPT':
        num_ref_images_train = 250000
        num_ref_images_test = 18000
    elif args.dataset =='RFW':
        num_ref_images_train = 8327
        num_ref_images_test = 8327
    elif args.dataset =='vggface2':
        num_ref_images_train = 500000
        num_ref_images_test = 15300
    else:
        raise NameError('Wrong dataset')




    datasets = {}
    print('PREPARING TRAIN DATASET')

    datasets['train'] = ImageFolderWithProtectedAttributes(args.default_train_root, transform=train_transform,
                                                                 demographic_to_all_classes=demographic_to_all_classes,
                                                                 all_classes_to_demographic = all_classes_to_demographic,
                                                                 p_identities = args.p_identities,
                                                                 p_images = args.p_images,
                                                                 min_num = args.min_num_images,
                                                                 ref_num_images = num_ref_images_train,
                                                                 seed = args.seed
                                                          )
    for k in list(demographic_to_all_classes.keys()):
        print('Number of idx for {} is {}'.format(k, len(datasets['train'].demographic_to_classes[k])))

    print('PREPARING TEST DATASET')
    datasets['val'] = ImageFolderWithProtectedAttributes(args.default_val_root, transform=test_transform,
                                                                 demographic_to_all_classes=demographic_to_all_classes,
                                                                 all_classes_to_demographic = all_classes_to_demographic,
                                                                 p_identities = {dem: 1.0 for dem,_ in args.p_identities.items()},
                                                                 p_images = {dem: 1.0 for dem,_ in args.p_images.items()},
                                                                 min_num = args.min_num_images,
                                                                 ref_num_images = num_ref_images_test,
                                                                 seed = args.seed
                                                         )

    for k in list(demographic_to_all_classes.keys()):
        print('Number of idx for {} is {}'.format(k, len(datasets['val'].demographic_to_classes[k])))

#     demographic_to_idx_train = datasets['train'].demographic_to_idx
    demographic_to_idx_train = None
    demographic_to_idx_test = datasets['val'].demographic_to_idx
    ######################################################



    dataloaders = {}
    g = torch.Generator()
    g.manual_seed(0)
#     train_imgs = datasets['train'].imgs
#     weights_train = torch.DoubleTensor(balanced_weights(train_imgs, nclasses=len(datasets['train'].classes)))
#     train_sampler = torch.utils.data.sampler.WeightedRandomSampler(weights_train, len(weights_train))
    num_class = len(datasets['train'].classes)
    '''
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size,
                                                       sampler=train_sampler, num_workers=args.num_workers,
                                                       drop_last=True)
    '''
    dataloaders['train'] = torch.utils.data.DataLoader(datasets['train'], batch_size=args.batch_size,
                                                       shuffle = True, num_workers=args.num_workers,
                                                       worker_init_fn=seed_worker,generator=g,
                                                       drop_last=True)
    dataloaders['val'] = torch.utils.data.DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    for k in list(dataloaders.keys()):
        print('Len of {} dataloader is {}'.format(k, len(dataloaders[k])))

    return dataloaders, num_class, demographic_to_idx_train, demographic_to_idx_test
