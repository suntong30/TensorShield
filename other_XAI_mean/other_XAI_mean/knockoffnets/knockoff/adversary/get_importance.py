import torch 
import copy
import os.path as osp
import os
import torch.optim as optim
import random
from knockoff import datasets

def get_sample_layers(sorted_importances, percent, is_random):
    n = len(sorted_importances)
    num_elements = int(n * percent)
    print("get_sample_layers n={}  num_select={}".format(n,num_elements))
    if is_random:
        print("get_sample_layers is_random")
        return random.sample(sorted_importances, num_elements)
    else :
        print("get_sample_layers not_random")
        return sorted_importances[:num_elements]

def load_target_model(target_model, victim_dir):
    target_path = osp.join(victim_dir, "checkpoint.pth.tar")
    target_model = target_model.cuda()
    print(f"Load model from {target_path}")
    target_model.load_state_dict(torch.load(target_path)['state_dict'])
    target_model.eval()
    return target_model


def create_dir(dir_path):
    if not osp.exists(dir_path):
        print('Path {} does not exist. Creating it...'.format(dir_path))
        os.makedirs(dir_path)


def get_protect_layers(train_dataset, target_model, percent, is_random, victim_dir):    

    info_path = osp.join(victim_dir, '.importance.txt')
    
    
    if osp.exists(info_path):
        print('Importance info exists. Loading...')
        with open(info_path, 'r') as f:
            protect_list = f.read().splitlines()[0].split(',')
        print('Importance info loaded.')
        return get_sample_layers(protect_list, percent, is_random)
    
    print('Importance info does not exist. Calculating...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # split validation
    total_train_examples = len(train_dataset)
    val_size = int(total_train_examples * 0.1)
    train_size = total_train_examples - val_size
    # Now split the dataset
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Then, you create your dataloaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    
    pretrained = copy.deepcopy(target_model)
    model = copy.deepcopy(pretrained)
    model = load_target_model(model, victim_dir)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device('cuda')
    loss_function = torch.nn.CrossEntropyLoss()
    
    
    weight_updates = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    weight_importance = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    
    for epoch in range(10):  # num_epochs是您设置的训练轮数
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            # 累计权重更新和重要性
            with torch.no_grad():  # 不追踪这部分的梯度
                for name, param in model.named_parameters():
                    # print(name)
                    # resnet= ['fc.weight', 'fc.bias']
                    # vgg = ['classifier.weight', 'classifier.bias']
                    # if 'weight' in name:  
                    #     # 计算本轮的权重更新
                    #     current_update = (-optimizer.param_groups[0]['lr'] * param.grad)
                    #     param_grad_abs = param.grad
                    #     weight_updates[name].add_(current_update)
                    #     weight_importance[name].addcmul_(current_update, param_grad_abs)
                    #     weight_importance[name].div_(torch.tensor(param.numel()))
                    alpha=0.5
                    if 'weight' in name:
                        # 计算本轮的权重更新
                        current_update = (-optimizer.param_groups[0]['lr'] * param.grad)
                        
                        # 直接乘积，对应公式的第一部分
                        direct_importance = current_update * param.grad
                        
                        # 绝对值乘积，然后除以参数数量，对应公式的第二部分
                        abs_importance = torch.abs(current_update) * torch.abs(param.grad)
                        normalized_importance = torch.sum(abs_importance) / param.numel()

                        # 累积重要性
                        if name in weight_importance:
                            # 混合两种重要性评分
                            weight_importance[name] += alpha * torch.sum(direct_importance) + (1 - alpha) * normalized_importance
                        else:
                            weight_importance[name] = alpha * torch.sum(direct_importance) + (1 - alpha) * normalized_importance
                   
        sorted_importances = {k: v.sum().item() for k, v in weight_importance.items()}
        sorted_importances = sorted(sorted_importances.items(), key=lambda item: item[1], reverse=True)
        
    # 训练结束后，评估所有参数的重要性
    sorted_importances = {k: v.sum().item() for k, v in weight_importance.items()}
    sorted_importances = sorted(sorted_importances.items(), key=lambda item: item[1], reverse=True)
    print('--------------------------------------------')
    print (f'sorted_importances: {sorted_importances}')
    print('--------------------------------------------')
    
    # 排除名字带有 bn 和 downsample 的tensor
    filtered_importances = {k: v.sum().item() for k, v in weight_importance.items() if 'bn' not in k and 'downsample' not in k}
    # filtered_importances = {k: v.sum().item() for k, v in weight_importance.items()}
    sorted_filtered_importances = sorted(filtered_importances.items(), key=lambda item: item[1], reverse=True)
    print('--------------------------------------------')
    print (f'sorted_filtered_importances: {sorted_filtered_importances}')
    print('--------------------------------------------')
    
    # 设定选取方案
    new_select = [k for k in sorted_importances if "weight" in k[0]]
    # 写入文件
    with open(info_path, 'w') as f:
        f.write(','.join([item[0] for item in new_select]))
    
    
    new_select = get_sample_layers(new_select , percent, is_random)
    protect_list = [item[0] for item in new_select]
    
    print('--------------------------------------------')
    print (f'protect_list: {protect_list}')
    print('--------------------------------------------')
    return protect_list



def get_protect_channel(train_dataset, target_model, percent, is_random, victim_dir):    

    info_path = osp.join(victim_dir, 'channel.importance.txt')
    
    
    if osp.exists(info_path):
        print('channel importance info exists. Loading...')
        with open(info_path, 'r') as f:
            channel_importance = {line.split(':')[0]: eval(line.split(':')[1]) for line in f.read().splitlines()}
            print('channel importance info loaded.')
            return channel_importance
    
    print('Importance info does not exist. Calculating...')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # split validation
    total_train_examples = len(train_dataset)
    val_size = int(total_train_examples * 0.1)
    train_size = total_train_examples - val_size
    # Now split the dataset
    train_subset, _ = torch.utils.data.random_split(train_dataset, [train_size, val_size])

    # Then, you create your dataloaders
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)
    
    pretrained = copy.deepcopy(target_model)
    model = copy.deepcopy(pretrained)
    model = load_target_model(model, victim_dir)
    
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    device = torch.device('cuda')
    loss_function = torch.nn.CrossEntropyLoss()
    
    
    channel_updates = {}
    channel_importance = {}

    for epoch in range(10):
        print(f"channel Epoch == {epoch}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if 'weight' in name and len(param.shape) == 4:  # 检查是否为卷积层权重
                        if name not in channel_updates:
                            channel_updates[name] = torch.zeros(param.shape[0], device=device)
                            channel_importance[name] = torch.zeros(param.shape[0], device=device)
                        
                        for i in range(param.shape[0]):  # 遍历每个输出通道
                            current_update = (-optimizer.param_groups[0]['lr'] * param.grad[i].abs()).sum()
                            param_grad_abs = param.grad[i].abs().sum()
                            channel_updates[name][i] += current_update
                            channel_importance[name][i] += current_update * param_grad_abs / param[i].numel()

    # 训练结束后，评估每个通道的重要性
    # for name in channel_importance:
    #     channel_importance[name] /= channel_updates[name]  # 归一化
    #     sorted_indices = torch.argsort(channel_importance[name], descending=True)
    #     print(f"Layer {name}: Channel importance in descending order: {sorted_indices}")
    # 将每个卷积层的重要性写入文件
    with open(info_path, 'w') as f:
        for name in channel_importance:
            channel_importance[name] /= channel_updates[name]  # 归一化
            sorted_indices = torch.argsort(channel_importance[name], descending=True)
            print(f"Layer {name}: Channel importance length: {len(sorted_indices)}")
            f.write(f"{name}:{sorted_indices.tolist()}\n")  
    return channel_importance
    