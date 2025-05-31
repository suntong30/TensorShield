import os
import torch
import pandas
import torchvision
torch.manual_seed(0)
import torch.nn as nn
import PIL.Image as Image
import torchvision.transforms as transforms
import numpy as np

from models.resnet import create_model as resnet_create_model
from models.vgg import create_model as vgg_create_model
from models.alexnet import create_model as alexnet_create_model
from models.mobilenetv2 import create_model as mobilenet_create_model

from functools import partial
from typing import Any, Callable, List, Optional, Union, Tuple
from pdb import set_trace as st
from .downsampled_imagenet import DownsampledImagenet

import os.path as osp
from torchvision.datasets import ImageFolder
class TinyImageNet200(ImageFolder):
    """
    Dataset for TinyImageNet200

    Note: the directory structure slightly varies from original
    To get there, run these two commands:
    - From within tiny-images-200 directory
        for dr in train/*; do
            echo $dr;
            mv $dr/images/* $dr/;
            rmdir $dr/images;
        done
    - From within tiny-images-200/val directory
         while read -r fname label remainder; do
            mkdir -p val2/$label;
            mv images/$fname val2/$label/;
        done < val_annotations.txt

    """

    def __init__(self, data_root, train=True, transform=None, target_transform=None):
        root = osp.join(data_root ,'tiny-imagenet-200')
        if not osp.exists(root):
            raise ValueError('Dataset not found at {}. Please download it from {}.'.format(
                root, 'https://tiny-imagenet.herokuapp.com'
            ))

        # Initialize ImageFolder
        _root = osp.join(root, 'train' if train else 'val2')
        super().__init__(root=_root, transform=transform,
                         target_transform=target_transform)
        self.root = root

        print('=> done loading {} ({}) with {} examples'.format(self.__class__.__name__, 'train' if train else 'test',
                                                                len(self.samples)))

        self._load_meta()

    def _load_meta(self):
        """Replace class names (synsets) with more descriptive labels"""
        # Load mapping
        synset_to_desc = dict()
        fpath = osp.join(self.root, 'words.txt')
        with open(fpath, 'r') as rf:
            for line in rf:
                synset, desc = line.strip().split(maxsplit=1)
                synset_to_desc[synset] = desc

        # Replace
        for i in range(len(self.classes)):
            self.classes[i] = synset_to_desc[self.classes[i]]
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

class CNN(nn.Module):
    def __init__(self, input_channel=3, num_classes=10):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channel, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(128*6*6, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class CIFAR100Dataset(torchvision.datasets.CIFAR100):
    def __init__(self, data_root, train, transform=None, download=True):
        super(CIFAR100Dataset, self).__init__(root=data_root, train=train, transform=transform, download=download)
        self.sample_subset = False
        self.num_classes = 100
        
    def set_subset(self, ratio):
        self.sample_subset = True
        num_sample = int(ratio * len(self.data))
        raw_idxs = list(range(len(self.data)))
        self.sample_idxs = np.random.choice(raw_idxs, size=num_sample)
        
    def __len__(self):
        if self.sample_subset:
            return len(self.sample_idxs)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.sample_subset:
            img, target = self.data[self.sample_idxs[index]], int(self.targets[self.sample_idxs[index]])
            real_index = self.sample_idxs[index]
        else:
            img, target = self.data[index], int(self.targets[index])
            real_index = index
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # iid = '{}_{:06d}'.format(self.split, index)
        return img, target
    
class CIFAR10Dataset(torchvision.datasets.CIFAR10):
    def __init__(self, data_root, train, transform=None, download=True):
        super(CIFAR10Dataset, self).__init__(root=data_root, train=train, transform=transform, download=download)
        self.sample_subset = False
        self.num_classes = 10
        
    def set_subset(self, ratio):
        self.sample_subset = True
        num_sample = int(ratio * len(self.data))
        raw_idxs = list(range(len(self.data)))
        self.sample_idxs = np.random.choice(raw_idxs, size=num_sample)
        
    def __len__(self):
        if self.sample_subset:
            return len(self.sample_idxs)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.sample_subset:
            img, target = self.data[self.sample_idxs[index]], int(self.targets[self.sample_idxs[index]])
            real_index = self.sample_idxs[index]
        else:
            img, target = self.data[index], int(self.targets[index])
            real_index = index
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        # iid = '{}_{:06d}'.format(self.split, index)
        return img, target

class UTKFaceDataset(torch.utils.data.Dataset):
    def __init__(self, root, attr: Union[List[str], str] = "gender", transform=None, target_transform=None)-> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.files = os.listdir(root+'/UTKFace/processed/')
        if isinstance(attr, list):
            self.attr = attr
        else:
            self.attr = [attr]

        self.lines = []
        for txt_file in self.files:
            with open(self.root+'/UTKFace/processed/' + txt_file, 'r') as f:
                assert f is not None
                for i in f:
                    image_name = i.split('jpg ')[0]
                    attrs = image_name.split('_')
                    if len(attrs) < 4 or int(attrs[2]) >= 4:
                        continue
                    if attrs[1] == "":
                        continue
                    self.lines.append(image_name+'jpg')


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index:int)-> Tuple[Any, Any]:
        
        attrs = self.lines[index].split('_')
        # print(self.lines[index], attrs, attrs[0], attrs[1], attrs[2])

        age = int(attrs[0])
        gender = int(attrs[1])
        race = int(attrs[2])

        image_path = os.path.join(self.root+'/UTKFace/UTKface_aligned_cropped/UTKFace', self.lines[index]+'.chip.jpg').rstrip()

        image = Image.open(image_path).convert('RGB')

        target: Any = []
        for t in self.attr:
            if t == "age":
                target.append(age)
            elif t == "gender":
                target.append(gender)
            elif t == "race":
                target.append(race)
            
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform:
            image = self.transform(image)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return image, target

class CelebA(torch.utils.data.Dataset):
    base_folder = "celeba"

    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        attr = pandas.read_csv(fn("list_attr_celeba.txt"), delim_whitespace=True, header=1)

        mask = slice(None)

        self.filename = splits[mask].index.values
        self.attr = torch.as_tensor(attr[mask].values)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(attr.columns)

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

class LFW(torch.utils.data.Dataset):
    base_folder = "lfw"

    def __init__(
            self,
            root: str,
            attr_list: str,
            target_type: Union[List[str], str] = "attr",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:

        if isinstance(target_type, list):
            self.target_type = target_type
        else:
            self.target_type = [target_type]

        self.root = root
        self.transform = transform
        self.target_transform =target_transform
        self.attr_list = attr_list

        fn = partial(os.path.join, self.root)
        attr = pandas.read_csv(fn("lfw_attributes_preprocessed.txt"), sep='\t', header=1)

        mask = slice(None)
        attr_array =  attr[mask].values
        img_idxs = attr_array[:,:2]

        # for i in range(len(img_idxs)):
        #     name = f"{img_idxs[i][0]}_{img_idxs[i][1]}.jpg"
        img_names = [
            f"{img_idxs[i][0]}/{img_idxs[i][0]}_{img_idxs[i][1]:04d}.jpg"
            for i in range(len(img_idxs))
        ]

        
        for i in range(len(attr_array)):
            try:
                img_attrs = attr_array[i,2:].astype(np.float32)
            except:
                print(i, attr_array[i])
                st()
        img_attrs = (attr_array[:,2:].astype(np.float32) > 0).astype(np.int8) # map from R to {0, 1}

        self.attr = torch.as_tensor(img_attrs).long()
        self.img_names = img_names
        self.attr_names = list(attr.columns)



    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        X = Image.open(os.path.join(self.root, self.base_folder, self.img_names[index]))

        target: Any = []
        for t, nums in zip(self.target_type, self.attr_list):
            if t == "attr":
                final_attr = 0
                for i in range(len(nums)):
                    final_attr += 2 ** i * self.attr[index][nums[i]]
                target.append(final_attr)
            else:
                # TODO: refactor with utils.verify_str_arg
                raise ValueError("Target type \"{}\" is not recognized.".format(t))

        if self.transform is not None:
            X = self.transform(X)

        if target:
            target = tuple(target) if len(target) > 1 else target[0]

            if self.target_transform is not None:
                target = self.target_transform(target)
        else:
            target = None

        return X, target

    def __len__(self) -> int:
        return len(self.attr)

    def extra_repr(self) -> str:
        lines = ["Target type: {target_type}", "Split: {split}"]
        return '\n'.join(lines).format(**self.__dict__)

def prepare_dataset(dataset_name, model="resnet18", pretrained=False):
    

    attr = None
    root = "/home/gpu2/sunt/TEEAI/TEESlice_mia/membership-inference/data"
    num_classes, dataset, target_model, shadow_model, pretrained_model = get_model_dataset(dataset_name, attr=attr, root=root, model=model, pretrained=pretrained)
    length = len(dataset)
    each_length = length//4
    target_train, target_test, shadow_train, shadow_test, _ = torch.utils.data.random_split(dataset, [each_length, each_length, each_length, each_length, len(dataset)-(each_length*4)])
    return num_classes, target_train, target_test, shadow_train, shadow_test, target_model, shadow_model, pretrained_model

def prepare_proxy_dataset(dataset, attr, root):
    num_classes, dataset, target_model, shadow_model, pretrained_model = get_model_dataset(dataset, attr=attr, root=root)

    return num_classes, dataset, target_model, shadow_model


def get_model_dataset(dataset_name, attr, root, model="resnet18", pretrained=False):
    if dataset_name.lower() == "utkfacerace":
        assert attr == "race"
        if isinstance(attr, list):
            num_classes = []
            for a in attr:
                if a == "age":
                    num_classes.append(117)
                elif a == "gender":
                    num_classes.append(2)
                elif a == "race":
                    num_classes.append(4)
                else:
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))
        else:
            if attr == "age":
                num_classes = 117
            elif attr == "gender":
                num_classes = 2
            elif attr == "race":
                num_classes = 4
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = UTKFaceDataset(root=root, attr=attr, transform=transform)
        input_channel = 3
        
    elif dataset_name.lower() == "celeba":
        if isinstance(attr, list):
            for a in attr:
                if a != "attr":
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))

                num_classes = [4, 2]
                # heavyMakeup MouthSlightlyOpen Smiling, Male Young
                # attr_list = [[18, 21, 31], [20, 39]]
                attr_list = [[31, 39], [20]]
        else:
            if attr == "attr":
                num_classes = 4
                # attr_list = [[18, 21, 31]]
                attr_list = [[31, 39]]
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = CelebA(root=root, attr_list=attr_list, target_type=attr, transform=transform)
        input_channel = 3
        
    elif dataset_name.lower() == "lfw":
        if isinstance(attr, list):
            for a in attr:
                if a != "attr":
                    raise ValueError("Target type \"{}\" is not recognized.".format(a))

                num_classes = [4, 2]
                # Eyeglasse Smiling, Male
                attr_list = [[16, 19], [2]]
        else:
            if attr == "attr":
                num_classes = 4
                attr_list = [[16, 19]]
            else:
                raise ValueError("Target type \"{}\" is not recognized.".format(attr))
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = LFW(
            root=root, attr_list=attr_list, target_type=attr, transform=transform
        )

        input_channel = 3
    

    elif dataset_name.lower() == "stl10":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_set = torchvision.datasets.STL10(
                root=root, split='train', transform=transform, download=True)
            
        test_set = torchvision.datasets.STL10(
                root=root, split='test', transform=transform, download=True)

        dataset = train_set + test_set
        input_channel = 3
        
    elif dataset_name.lower() == 'cifar100':
        num_classes = 100
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = CIFAR100Dataset(data_root=root, train=True, transform=transform, download=True)
        test = CIFAR100Dataset(data_root=root, train=False, transform=transform, download=True)
        dataset = train + test
    elif dataset_name.lower() == 'cifar10':
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train = CIFAR10Dataset(data_root=root, train=True, transform=transform, download=True)
        test = CIFAR10Dataset(data_root=root, train=False, transform=transform, download=True)
        dataset = train + test
    
    elif dataset_name.lower() == "stl10-unlabeled":
        num_classes = 10
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        dataset = torchvision.datasets.STL10(
                root=root, split='unlabeled', transform=transform, download=True)

        input_channel = 3
        
    elif dataset_name.lower() == "downsampled-imagenet":
        num_classes = 1000
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        dataset = DownsampledImagenet(
            root=root, transform=transform
        )
        input_channel = 3
    elif dataset_name.lower() == "tinyimagenet200":
        num_classes = 200
        if  "resnet" in model  or model == "alexnet" :
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                    std=(0.2023, 0.1994, 0.2010)),
            ])
        else:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                    std=(0.2023, 0.1994, 0.2010)),
            ])
        train = TinyImageNet200(
            data_root=root, transform=transform
        )
        test = TinyImageNet200(
            data_root=root, train=False, transform=transform
        )
        dataset = train + test
        print(type(dataset))
    
    if model == "cnn":
        if isinstance(num_classes, int):
            target_model = CNN(input_channel=input_channel, num_classes=num_classes)
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes)
        else:
            target_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
            shadow_model = CNN(input_channel=input_channel, num_classes=num_classes[0])
    elif model.startswith("resnet") :
        if isinstance(num_classes, int):
            target_model = resnet_create_model(model, num_classes=num_classes, pretrained=pretrained)
            shadow_model = resnet_create_model(model, num_classes=num_classes, pretrained=pretrained)
            pretrained_model = resnet_create_model(model, num_classes=num_classes, pretrained=True)
        else:
            target_model = resnet_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            shadow_model = resnet_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            pretrained_model = resnet_create_model(model, num_classes=num_classes[0], pretrained=True)
    elif model.startswith("vgg") :
        if isinstance(num_classes, int):
            target_model = vgg_create_model(model, num_classes=num_classes, pretrained=pretrained)
            shadow_model = vgg_create_model(model, num_classes=num_classes, pretrained=pretrained)
            pretrained_model = vgg_create_model(model, num_classes=num_classes, pretrained=True)
        else:
            target_model = vgg_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            shadow_model = vgg_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            pretrained_model = vgg_create_model(model, num_classes=num_classes[0], pretrained=True)
    elif model.startswith("alexnet") :
        if isinstance(num_classes, int):
            target_model = alexnet_create_model(model, num_classes=num_classes, pretrained=pretrained)
            shadow_model = alexnet_create_model(model, num_classes=num_classes, pretrained=pretrained)
            pretrained_model = alexnet_create_model(model, num_classes=num_classes, pretrained=True)
        else:
            target_model = alexnet_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            shadow_model = alexnet_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            pretrained_model = alexnet_create_model(model, num_classes=num_classes[0], pretrained=True)
    elif model.startswith("mobilenet") :
        if isinstance(num_classes, int):
            target_model = mobilenet_create_model(model, num_classes=num_classes, pretrained=pretrained)
            shadow_model = mobilenet_create_model(model, num_classes=num_classes, pretrained=pretrained)
            pretrained_model = mobilenet_create_model(model, num_classes=num_classes, pretrained=True)
        else:
            target_model = mobilenet_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            shadow_model = mobilenet_create_model(model, num_classes=num_classes[0], pretrained=pretrained)
            pretrained_model = mobilenet_create_model(model, num_classes=num_classes[0], pretrained=True)
    else:
        raise NotImplementedError

    return num_classes, dataset, target_model, shadow_model, pretrained_model