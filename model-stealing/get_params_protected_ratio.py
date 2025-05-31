import torch
import math

# modify the path!
# ~/jbw/knockoff/TensorShield-main/knockoffnets/models/victim/
CHECKPOINT_PATH_TEMPLATE = '/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets/models/victim/{}/checkpoint.pth.tar'

# modify the path!
# IMPORTANCE_PATH_TEMPLATE = '/home/gpu2/linhl/knockoffnets/protected_model_info/victim/{}/.importance.txt'
IMPORTANCE_PATH_TEMPLATE = '/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets/models/victim/{}/.importance.txt'
model_names = [
    #"cifar10-resnet18",
    #"cifar100-mobilenetv2",
    #"tinyimagenet200-resnet18",
    #"cifar10-vgg16_bn",
    #"cifar10-mobilenetv2",
    #"cifar100-vgg16_bn",
    #"tinyimagenet200-vgg16_bn",
    #"tinyimagenet200-mobilenetv2"
    "cifar100-resnet50",
    ]
protected_layer_ratios = {
    "cifar100-vgg16_bn": [0.1, 1],
    "cifar10-resnet18": [0.5, 1],
    "cifar100-mobilenetv2": [0.3, 1],
    "cifar10-vgg16_bn": [0.7, 1],
    "cifar10-mobilenetv2": [0.95, 1],
    "tinyimagenet200-resnet18": [0.6, 1],
    "tinyimagenet200-vgg16_bn": [0.4, 1],
    "tinyimagenet200-mobilenetv2": [0.2, 1],
    "cifar100-resnet50": [0.5, 1],
}
def get_importance(model_name):
    data = None
    path = IMPORTANCE_PATH_TEMPLATE.format(model_name)
    with open(path, 'r') as f:
        data = f.readlines()
    layer_name_list = data[0].split(',')
    return layer_name_list



def get_params_ratio(model_name, protected_layer_ratio):
    layer_name_list = get_importance(model_name)
    num_protected_layer = math.floor(len(layer_name_list) * protected_layer_ratio)
    # num_protected_layer = math.ceil(len(layer_name_list) * protected_layer_ratio)
    protected_layer_name_list = layer_name_list[:num_protected_layer]
    checkpoint_path = CHECKPOINT_PATH_TEMPLATE.format(model_name)
    trained_state_dict = torch.load(checkpoint_path)['state_dict']
    num_model_params = 0
    num_protected_layer_params = 0
    params_names = []
    protected_names = []
    for key, param in trained_state_dict.items():
        num_model_params += param.numel()
        params_names.append(key)
        if key in protected_layer_name_list:
            protected_names.append(key)
            num_protected_layer_params += param.numel()
    
    protected_params_ratio = num_protected_layer_params / num_model_params
    print("------------------------------")
    print(f"model_name:\t\t\t\t\t\t{model_name}")
    print(f"protected_params_ratio:\t\t\t{protected_params_ratio * 100}%")
    print(f"protected_layer_ratio:\t\t\t{protected_layer_ratio}")
    print(f"num_model_params:\t\t\t\t{num_model_params}")
    print(f"num_protected_layer_params:\t\t{num_protected_layer_params}")
    print(f"checkpoint_path:\t\t\t\t{checkpoint_path}")
    print(f"protected_layer_name_list:\t\t{protected_layer_name_list}")
    print(f"protected_names:\t\t\t\t{protected_names}")
    print(f"importance_name_list:\t\t\t{layer_name_list}")
    print(f"protected_layer_name_list_num:\t{len(protected_layer_name_list)}")
    print(f"importance_name_list_num:\t\t{len(layer_name_list)}")
    print(f"params_names:\t\t\t\t\t{params_names}")
    return protected_params_ratio

if __name__=="__main__":
    for _model_name in model_names:
        _protected_layer_ratio = protected_layer_ratios[_model_name][0]
        get_params_ratio(_model_name, _protected_layer_ratio)
