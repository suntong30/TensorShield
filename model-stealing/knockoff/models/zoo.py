import torch
import torch.nn as nn
import os.path as osp
import knockoff.adversary.get_importance as get_importance
import sys
sys.path.append('/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets')
import knockoff.models.cifar
import knockoff.models.mnist
import knockoff.models.imagenet
from knockoff import datasets

def get_net(modelname, modeltype, pretrained=None, **kwargs):
    pretrained = 'imagenet'
    assert modeltype in ('mnist', 'cifar', 'imagenet', 'stl')
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
    if "is_victim" in kwargs  and kwargs["is_victim"]:
        return get_imagenet_victimnet(modelname, num_classes, **kwargs)
    if pretrained == 'imagenet':
        return get_imagenet_pretrainednet(modelname, num_classes, **kwargs)
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


def get_imagenet_victimnet(modelname, num_classes=1000, **kwargs):
    valid_models = knockoff.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    print(f"[DEBUG]  modelname = {modelname} num_class = {num_classes}")


    last_layer_name = "last_linear"
    if "vgg" in modelname:
        kwargs.clear()
        last_layer_name  = "classifier"
        # model = eval('knockoff.models.{}.{}'.format("cifar", modelname))(num_classes=num_classes, **kwargs)

        last_layer_name  = "last_linear"
        model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
        in_features = eval(f"model.{last_layer_name}.in_features")
        out_features = num_classes
        model.last_linear = nn.Linear(in_features, out_features, bias=True)


    elif "alexnet" in modelname:
        kwargs.clear()
        last_layer_name  = "classifier"
        model = eval('knockoff.models.{}.{}'.format("cifar", modelname))(num_classes=num_classes, **kwargs)
    else:
        model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')

    # Replace last linear layer
    if "resnet" in modelname:
        in_features = eval(f"model.{last_layer_name}.in_features")
        out_features = num_classes
        model.last_linear = nn.Linear(in_features, out_features, bias=True)

    elif "mobilenet" in modelname:
        in_features = eval(f"model.{last_layer_name}.in_features")
        out_features = num_classes
        model.last_linear = nn.Linear(in_features, out_features, bias=True)
    # if "vgg" in modelname:
    #     model.classifier = nn.Linear(in_features, out_features, bias=True)
    # elif "alexnet" in modelname:
    #     model.classifier = nn.Linear(in_features, out_features, bias=True)
    # else:
    #     model.last_linear = nn.Linear(in_features, out_features, bias=True)
    # print(model)
    return model

def get_imagenet_pretrainednet(modelname, num_classes=1000, **kwargs):
    valid_models = knockoff.models.imagenet.__dict__.keys()
    assert modelname in valid_models, 'Model not recognized, Supported models = {}'.format(valid_models)
    victim_dir = kwargs["victim_dir"]
    protect_percent = kwargs["protect_percent"]

    checkpoint_path =  osp.join(victim_dir ,"checkpoint.pth.tar" )
    trained_state_dict = torch.load(checkpoint_path)['state_dict']

    dataset_name = kwargs["get_importance_dataset"]
    if dataset_name == "TINYIMAGENET200":
        dataset_name = "TinyImageNet200"
    dataset = datasets.__dict__[dataset_name]
    modelfamily = datasets.dataset_to_modelfamily[dataset_name]

    channel_percent = None
    if kwargs.__contains__("channel_percent"):
        channel_percent = kwargs["channel_percent"]
        print(f"[INFO] channel_percent = {channel_percent}")

    # if modelname == 'resnet18':
    if 'resnet' in modelname:
        ################################### add st #################################
        train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        if dataset_name == 'STL10':
            trainset = dataset(split='train', transform=train_transform)
        else:
            trainset = dataset(train=True, transform=train_transform)

        new_model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
        trained_state_dict = {k.replace('module.', ''): v for k, v in trained_state_dict.items()}
        num_ftrs = new_model.last_linear.in_features

        new_model.last_linear = torch.nn.Linear(num_ftrs, num_classes)

        if num_classes != 1000:
            # Replace last linear layer
            in_features = new_model.last_linear.in_features
            out_features = num_classes
            new_model.last_linear = nn.Linear(in_features, out_features, bias=True)

        if channel_percent and channel_percent < 0 :

            layer_name_list = []
            for name, param in new_model.named_parameters():
                if name.endswith('weight'):
                    layer_name_list.append(name)
            print(f"{modelname}  protect len = {len(layer_name_list)}")
            protect_layers = layer_name_list[: int(protect_percent)] # protect shallow layers
            # protect_layers = layer_name_list[-int(protect_percent):] # protect deep layers

            channel_percent = None
        else :
            protect_layers = get_importance.get_protect_layers(trainset, new_model, protect_percent, False, victim_dir)
            # channel_importance = get_importance.get_protect_channel(trainset, new_model, protect_percent, False, victim_dir)
            channel_percent = None
        print(f"protect_percent = {protect_percent}"  ,len(protect_layers), protect_layers)
        # if len(protect_layers)>0:
        #     protect_layers.append('last_linear.weight')
        #     protect_layers.append('layer4.2.bn3.weight')
        # protect_layers.append('_features.17.conv.0.0.weight')
        # protect_layers.append('_features.17.conv.0.1.weight')
        # protect_layers.append('_features.17.conv.1.0.weight')
        # protect_layers.append('_features.17.conv.1.0.weight')
        # protect_layers.append('_features.17.conv.1.1.weight')
        # protect_layers.append('_features.17.conv.2.weight')
        # protect_layers.append('_features.18.1.weight')

        for key, param in trained_state_dict.items():

            parts = key.split('.')
            # print(key,end="  ")
            if key not in protect_layers:

                attr = new_model
                for part in parts[:-1]:
                    attr = getattr(attr, part, None)
                    assert(attr != None)
                if attr is not None:
                    # If the corresponding layer is successfully found, update its parameters
                    param_name = parts[-1]
                    if hasattr(attr, param_name):
                        existing_param = getattr(attr, param_name)
                        existing_param.data.copy_(param.data)
                        # print("protect layer")
            else:
                attr = new_model
                for part in parts[:-1]:
                    attr = getattr(attr, part, None)
                    assert(attr != None)
                if attr is not None:
                    # If the corresponding layer is successfully found, update its parameters
                    param_name = parts[-1]  # 参数名，如 'weight'
                    if hasattr(attr, param_name):
                        existing_param = getattr(attr, param_name)
                        ############## ADD channel protect ################
                        if channel_percent is not None and channel_importance.__contains__(key):
                            channel_importance_list = channel_importance[key]
                            if channel_percent == 0.01:
                                #
                                import random
                                # print("shadow random0.95 protect")
                                index_list = random.sample(range(len(channel_importance_list)), int(len(channel_importance_list)*( 1 - channel_percent)))
                            else:
                                #  inverse channel_importance_list
                                channel_importance_list = channel_importance_list[::-1]
                                index_list = channel_importance_list[:int(len(channel_importance_list)*(1-channel_percent))]
                            for i in range(len(index_list)):
                                existing_param.data[index_list [i] ].copy_(param.data[index_list [i]])
                            print(f"protect channel with {int(len(channel_importance_list)*channel_percent)}")

        # if num_classes != 1000:
        #     # Replace last linear layer
        #     in_features = new_model.last_linear.in_features
        #     out_features = num_classes
        #     new_model.last_linear = nn.Linear(in_features, out_features, bias=True)
        model = new_model

    else:

        kwargs.clear()
        train_transform = datasets.modelfamily_to_transforms[modelfamily]['train']
        if dataset_name == 'STL10':
            trainset = dataset(split='train', transform=train_transform)
        else:
            trainset = dataset(train=True, transform=train_transform)

        print(f"[DEBUG] " + 'knockoff.models.{}.{}'.format("cifar", modelname))
        if 'mobilenet' in modelname:
            victim_model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
            new_model = knockoff.models.imagenet.__dict__[modelname](pretrained='imagenet')
            last_layer_name = "last_linear"

            in_features = eval(f"victim_model.{last_layer_name}.in_features")
            out_features = num_classes
            victim_model.last_linear = nn.Linear(in_features, out_features, bias=True)

            in_features = eval(f"new_model.{last_layer_name}.in_features")
            out_features = num_classes
            new_model.last_linear = nn.Linear(in_features, out_features, bias=True)
            # victim_model = eval('knockoff.models.{}.{}'.format("imagenet", modelname))(num_classes=num_classes, **kwargs)
            # new_model = eval('knockoff.models.{}.{}'.format("imagenet", modelname))(num_classes=num_classes, **kwargs)

        else:
            victim_model = eval('knockoff.models.{}.{}'.format("cifar", modelname))(num_classes=num_classes, **kwargs)
            new_model = eval('knockoff.models.{}.{}'.format("cifar", modelname))(num_classes=num_classes, **kwargs)
        # print(new_model)

        if channel_percent and  channel_percent < 0 :
            layer_name_list = []
            for name, param in new_model.named_parameters():
                if name.endswith('weight'):
                    layer_name_list.append(name)
            print(f"{modelname}  protect len = {len(layer_name_list)}")
            
            protect_layers = layer_name_list[:int(protect_percent)] # protect shallow layers
            # protect_layers = layer_name_list[-int(protect_percent):] # protect deep layers
            
            channel_percent = None
        else :
            protect_layers = get_importance.get_protect_layers(trainset, victim_model, protect_percent, False, victim_dir)
            channel_percent = None
            # channel_importance = get_importance.get_protect_channel(trainset, new_model, protect_percent, False, victim_dir)
        print(len(protect_layers), protect_layers)
        # protect_layers.append('_features.18.1.weight')

        trained_state_dict = {k.replace('module.', ''): v for k, v in trained_state_dict.items()}

        for key, param in trained_state_dict.items():
            parts = key.split('.')
            if key not in protect_layers:
                attr = new_model
                for part in parts[:-1]:
                    attr = getattr(attr, part, None)
                    assert(attr != None)
                print(key, 'Stolen')
                if attr is not None:
                    param_name = parts[-1]
                    if hasattr(attr, param_name):
                        existing_param = getattr(attr, param_name)
                        existing_param.data.copy_(param.data)
            else:
                attr = new_model
                for part in parts[:-1]:
                    attr = getattr(attr, part, None)
                    assert(attr != None)
                if attr is not None:
                    param_name = parts[-1]
                    # print(attr, param_name)
                    if hasattr(attr, param_name):

                        existing_param = getattr(attr, param_name)
                        if channel_percent is not None and channel_importance.__contains__(key):
                            channel_importance_list = channel_importance[key]
                            if channel_percent == 0.01:
                                import random
                                print("shadow random0.95 protect")
                                index_list = random.sample(range(len(channel_importance_list)), int(len(channel_importance_list)*( 1 - channel_percent)))
                            else :

                                channel_importance_list = channel_importance_list[::-1]

                                index_list = channel_importance_list[:int(len(channel_importance_list)*(1-channel_percent))]

                            for i in range(len(index_list)):
                                existing_param.data[index_list[i]].copy_(param.data[index_list[i]])
        model = new_model
    # with open("protect_param_num.txt", "a") as f:
    #     f.write(f"{modelname} protect_percent={protect_percent} protect_param_num = {protect_param_num}\n")
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
