#!/usr/bin/python
"""This is a short description.
Replace this with a more detailed description of what this file contains.
"""
import argparse
import os.path as osp
import os
from datetime import datetime
import json
from collections import defaultdict as dd

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils import data as torch_data
from torch.utils.data import Dataset, DataLoader, Subset
import sys
sys.path.append('/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets')
import knockoff.config as cfg
from knockoff import datasets
import knockoff.utils.transforms as transform_utils
import knockoff.utils.model as model_utils
import knockoff.utils.utils as knockoff_utils
import knockoff.models.zoo as zoo

__author__ = "Tribhuvanesh Orekondy"
__maintainer__ = "Tribhuvanesh Orekondy"
__email__ = "orekondy@mpi-inf.mpg.de"
__status__ = "Development"


def main():
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model',
                        default=cfg.MODEL_DIR)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('-w', '--num_workers', metavar='N', type=int, help='# Worker threads to load data', default=10)
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', type=str, help='Use pretrained network', default=None)
    parser.add_argument('--weighted-loss', action='store_true', help='Use a weighted loss', default=None)
    args = parser.parse_args()
    params = vars(args)

    # torch.manual_seed(cfg.DEFAULT_SEED)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # ----------- Set up dataset
    dataset_name = params['dataset']
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]

    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    if "vgg" in params['model_arch'] or "alex" in params['model_arch']:
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    else:  
        test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    trainset = dataset(train=True, transform=train_transform)
    testset = dataset(train=False, transform=test_transform)
    num_classes = len(trainset.classes)
    params['num_classes'] = num_classes

    if params['train_subset'] is not None:
        idxs = np.arange(len(trainset))
        ntrainsubset = params['train_subset']
        idxs = np.random.choice(idxs, size=ntrainsubset, replace=False)
        trainset = Subset(trainset, idxs)

    # ----------- Set up model
    model_name = params['model_arch']
    pretrained = params['pretrained']
    is_victim = True
    # model = model_utils.get_net(model_name, n_output_classes=num_classes, pretrained=pretrained)
    model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes , is_victim = is_victim)
    model = model.to(device)
    
    out_path = params['out_path']
    tensor_names_file = osp.join(out_path, '.importance.txt')
    with open(tensor_names_file, 'r') as file:
        tensor_names = file.read().strip().split(',')
    
    print(tensor_names)
    for name, param in model.named_parameters():
        print(f"Layer Name: {name}, Parameter Shape: {param.size()}")
        
    def get_nested_attr(obj, attr):
        """ 递归获取嵌套属性 """
        attrs = attr.split('.')
        for a in attrs:
            obj = getattr(obj, a)
        return obj
    
    
    coefficients = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    total_params = sum(p.numel() for p in model.parameters())
    # 遍历每个系数，计算累积参数量的占比
    for coeff in coefficients:
        # 根据系数确定累加的张量数量
        index = int(coeff * len(tensor_names))
        selected_tensors = tensor_names[:index]

        # 计算选定张量的累积参数量
        cumulative_params = 0
        for name in selected_tensors:
            tensor = get_nested_attr(model, name)
            if tensor is not None:
                cumulative_params += tensor.numel()

        # 计算累积参数量占模型总参数量的比例
        ratio = cumulative_params / total_params * 100
        print(f"Coefficient: {coeff}, Cumulative Parameter Ratio: {ratio:.2f}%")
    # total_params = sum(p.numel() for p in model.parameters())
    # # 遍历张量名称，计算累积参数量
    # cumulative_params = 0
    # count = 0
    # for name in tensor_names:
    #     tensor = get_nested_attr(model, name)
    #     if tensor is not None:
    #         count += 1
    #         # 计算当前张量的参数量
    #         tensor_params = tensor.numel()
    #         cumulative_params += tensor_params
    #         # 计算累积参数量占模型总参数量的比例
    #         ratio = cumulative_params / total_params * 100
    #         print(f"{count/len(tensor_names)} Tensor Name: {name}, Size: {tensor.size()}, Number of Parameters: {tensor_params}, Cumulative Parameter Ratio: {ratio:.2f}%")
    #     else:
    #         print(f"Tensor named {name} not found in the model !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


if __name__ == '__main__':
    main()
