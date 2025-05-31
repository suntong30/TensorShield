import os,sys,datetime
import torch
import sys
sys.path.append('/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets')
from knockoff import datasets
import pickle
import random
from knockoff.adversary.train import samples_to_transferset
class Logger(object):
    def __init__(self, log2file=False, mode='train', path=None):
        if log2file:
            assert path is not None
            fn = os.path.join(path, '{}-{}.log'.format(mode, datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
            self.fp = open(fn, 'w')
        else:
            self.fp = sys.stdout

    def add_line(self, content):
        self.fp.write(content+'\n')
        self.fp.flush()


def prepare_dataset(dataset_name, modelfamily, transferset_path, budget):
    
    need_train_transform = dataset_name == "TINYIMAGENET200" and  "resnet18" not in transferset_path
    print("need_train_transform: ", need_train_transform)
    shadow_test = get_shadow_testset( dataset_name , need_train_transform)
    target_train, target_test = get_victim_dataset(dataset_name)
    
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    num_classes = transferset_samples[0][1].size(0)
    
    if need_train_transform :
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    else:
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    shadow_train  = samples_to_transferset(transferset_samples, budget, transform=transform)
    
    
    return num_classes, target_train, target_test, shadow_train, shadow_test

CFG_ShadowTEST_NUM = 1000

def get_victim_dataset(dataset_name):
    if dataset_name == "TINYIMAGENET200":
        dataset_name = "TinyImageNet200"
    valid_datasets = datasets.__dict__.keys()
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    test_transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    trainset = dataset(train=True, transform=train_transform)
    # testset = dataset(train=False, transform=test_transform)

    full_testset = dataset(train=False, transform=train_transform)

    subset_indices = random.sample(range(len(full_testset)), CFG_ShadowTEST_NUM)
    # testset = torch.utils.data.Subset(full_testset, subset_indices) # shadow_test
    # 找出没有被包含在 testset 中的所有其它索引
    remaining_indices = list(set(range(len(full_testset))) - set(subset_indices)) # target_test
    # 创建第二个子集 subset2，包含剩余的样本
    testset = torch.utils.data.Subset(full_testset, remaining_indices)
    return trainset, testset

def get_shadow_testset(dataset_name, need_train_transform=False):
    random.seed(42)
    valid_datasets = datasets.__dict__.keys()
    if dataset_name == "TINYIMAGENET200":
        dataset_name = "TinyImageNet200"
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    ################## 
    if  need_train_transform :
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    ################
    if dataset_name not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    dataset = datasets.__dict__[dataset_name]
    # testset = dataset(train=False, transform=transform)
    full_testset = dataset(train=False, transform=transform)

    subset_indices = random.sample(range(len(full_testset)), CFG_ShadowTEST_NUM)
    
    testset = torch.utils.data.Subset(full_testset, subset_indices)
    return  testset