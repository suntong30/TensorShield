import torch
import torch.nn as nn
import os.path as osp

import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet


def get_net(modelname, modeltype, pretrained=None, **kwargs):
    pretrained = 'imagenet'
    assert modeltype in ('mnist', 'cifar', 'imagenet')
    # print('[DEBUG] pretrained={}\tnum_classes={}'.format(pretrained, kwargs['num_classes']))
    if pretrained and pretrained is not None:
        return get_pretrainednet(modelname, modeltype, pretrained, **kwargs)
    else:
        try:
            # This should have ideally worked:
            model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(**kwargs)
        except AssertionError:
            # But, there's a bug in pretrained models which ignores the num_classes attribute.
            # So, temporarily load the model and replace the last linear layer
            model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))()
            if 'num_classes' in kwargs:
                num_classes = kwargs['num_classes']
                in_feat = model.last_linear.in_features
                model.last_linear = nn.Linear(in_feat, num_classes)
        return model


def get_pretrainednet(modelname, modeltype, pretrained='imagenet', num_classes=1000, **kwargs):
    is_victim = kwargs['is_victim']
    print(f'is_victim: {is_victim}')
    if pretrained == 'imagenet' and is_victim == 0:
        return get_imagenet_pretrainednet(modelname, num_classes, **kwargs)
    elif pretrained == 'imagenet' and is_victim == 1:
        return get_imagenet_pretrainednet_vitim(modelname, num_classes, **kwargs)
    elif osp.exists(pretrained):
        try:
            # This should have ideally worked:
            model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))(num_classes=num_classes, **kwargs)
        except AssertionError:
            # print('[DEBUG] pretrained={}\tnum_classes={}'.format(pretrained, num_classes))
            # But, there's a bug in pretrained models which ignores the num_classes attribute.
            # So, temporarily load the model and replace the last linear layer
            model = eval('knockoff.models.{}.{}'.format(modeltype, modelname))()
            in_feat = model.last_linear.in_features
            model.last_linear = nn.Linear(in_feat, num_classes)
        checkpoint = torch.load(pretrained)
        pretrained_state_dict = checkpoint.get('state_dict', checkpoint)
        copy_weights_(pretrained_state_dict, model.state_dict())
        return model
    else:
        raise ValueError('Currently only supported for imagenet or existing pretrained models')


def get_imagenet_pretrainednet_vitim(modelname, num_classes=1000, **kwargs):
    valid_models = knockoff.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
    if num_classes != 1000:
        # Replace last linear layer
        in_features = model.last_linear.in_features
        out_features = num_classes
        model.last_linear = nn.Linear(in_features, out_features, bias=True)
    return model

def get_imagenet_pretrainednet(modelname, num_classes=1000, **kwargs):
    valid_models = knockoff.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized. Supported models = {}'.format(valid_models)

    if modelname == 'resnet18':
        checkpoint_path = './models/victim/cifar100-resnet18/checkpoint.pth.tar'  # Add the path to your victim model, e.g., /home/knockoffnets/models/victim/cifar100-resnet18/checkpoint.pth.tar
        trained_state_dict = torch.load(checkpoint_path)['state_dict']

        # Initialize a new ImageNet-pretrained ResNet-18 model
        new_model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
        duplicate_victim_model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')

        # If the model was trained using DataParallel, keys may start with 'module.', so we remove it
        trained_state_dict = {k.replace('module.', ''): v for k, v in trained_state_dict.items()}
        num_ftrs = new_model.last_linear.in_features  # Get input features of the last linear layer
        num_classes = 100  # Set the number of classes as per the trained model

        # Replace the last layer of the new model
        new_model.last_linear = torch.nn.Linear(num_ftrs, num_classes)
        duplicate_victim_model.last_linear = torch.nn.Linear(num_ftrs, num_classes)
        duplicate_victim_model.load_state_dict(trained_state_dict, strict=False)

        for key, param in trained_state_dict.items():
            parts = key.split('.')

            # NOTE:
            # To simulate a black-box attack (i.e., protect specific tensors), uncomment the line below and use it instead of 'if True'
    
            # This will skip copying weights of protected tensors specific for resnet18-cifar100
            
            if 'layer1.1.conv1.weight' not in key and 'layer2.0.conv1.weight' not in key and 'layer1.0.conv1.weight' not in key \
                and 'layer1.1.conv2.weight' not in key and 'layer1.0.conv2.weight' not in key and 'layer2.0.conv2.weight' not in key\
                     and 'layer2.1.conv1.weight' not in key and 'layer2.1.conv2.weight' not in key and 'layer3.0.conv1.weight' not in key and 'layer3.0.conv2.weight' not in key      and 'last_linear.weight' not in key and 'last_linear.bias' not in key:             

            # if False:  # Set to True for white-box attack (no protection), False for black-box attack (protect all parameters)
                # Navigate to the corresponding module in new_model
                attr = new_model
                for part in parts[:-1]:
                    attr = getattr(attr, part, None)
                    assert(attr is not None)
                print(key, 'stolen') # stole weights
                if attr is not None:
                    param_name = parts[-1]
                    if hasattr(attr, param_name):
                        existing_param = getattr(attr, param_name)
                        existing_param.data.copy_(param.data)
            else:
                continue

        model = new_model

        # if num_classes != 1000:
        #     # Replace the last linear layer again if needed
        #     in_features = model.last_linear.in_features
        #     out_features = num_classes
        #     model.last_linear = nn.Linear(in_features, out_features, bias=True)

    else:
        model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')

        if num_classes != 1000:
            # Replace the last linear layer
            in_features = model.last_linear.in_features
            out_features = num_classes
            model.last_linear = nn.Linear(in_features, out_features, bias=True)

    return model



def copy_weights_(src_state_dict, dst_state_dict):
    n_params = len(src_state_dict)
    n_success, n_skipped, n_shape_mismatch = 0, 0, 0

    for i, (src_param_name, src_param) in enumerate(src_state_dict.items()):
        if src_param_name in dst_state_dict:
            dst_param = dst_state_dict[src_param_name]
            if dst_param.data.shape == src_param.data.shape:
                dst_param.data.copy_(src_param.data)
                n_success += 1
            else:
                print('Mismatch: {} ({} != {})'.format(src_param_name, dst_param.data.shape, src_param.data.shape))
                n_shape_mismatch += 1
        else:
            n_skipped += 1
    print('=> # Success param blocks loaded = {}/{}, '
          '# Skipped = {}, # Shape-mismatch = {}'.format(n_success, n_params, n_skipped, n_shape_mismatch))
