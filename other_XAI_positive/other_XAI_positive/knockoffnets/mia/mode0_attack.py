from copy import deepcopy
import os.path as osp
import json
import sys
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from attack_model import PartialAttackModel
from util import *
from sklearn.metrics import f1_score, roc_auc_score
import pickle
import glob
import argparse
from knockoff.models import zoo
from knockoff.utils.model import soft_cross_entropy
sys.path.append('/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets')
import knockoff.utils.model as model_utils

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        m.bias.data.fill_(0)
    elif isinstance(m,nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

class attack_for_blackbox():
    def __init__(
        self, SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_train_loader, attack_test_loader, 
        target_model, shadow_model, attack_model, device, logger
    ):
        self.device = device
        self.logger = logger

        self.TARGET_PATH = TARGET_PATH
        self.SHADOW_PATH = SHADOW_PATH
        self.ATTACK_SETS = ATTACK_SETS


        
        
        self.target_model = target_model.to(self.device)
        self.shadow_model = shadow_model.to(self.device)

        self.target_model.load_state_dict(torch.load(self.TARGET_PATH))
        print(f"Load target from {self.TARGET_PATH}")
        self.shadow_model.load_state_dict(torch.load(self.SHADOW_PATH))
        print(f"Load shadow from {self.SHADOW_PATH}")

        self.target_model.eval()
        self.shadow_model.eval()

        self.attack_train_loader = attack_train_loader
        self.attack_test_loader = attack_test_loader

        self.attack_model = attack_model.to(self.device)
        torch.manual_seed(0)
        self.attack_model.apply(weights_init)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.attack_model.parameters(), lr=1e-5)
        
        self.best_test_gndtrth = []
        self.best_test_predict = []
        self.best_test_probabe = []
        self.best_acc = -1
        self.best_state_dict = None

    def _get_data(self, model, inputs):
        result = model(inputs)
        output, _ = torch.sort(result, descending=True)
        _, predicts = result.max(1)
        
        prediction = []
        for predict in predicts:
            prediction.append([1,] if predict else [0,])

        prediction = torch.Tensor(prediction)
        return output, prediction
    
    def prepare_dataset(self):
        with open(self.ATTACK_SETS + "train.p", "wb") as f:
            for inputs, targets, members in self.attack_train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.shadow_model, inputs)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.add_line(f"Finished Saving Train Dataset to {self.ATTACK_SETS + 'train.p'}")

        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.add_line(f"Finished Saving Test Dataset to {self.ATTACK_SETS + 'test.p'}")
        
    def prepare_test_dataset(self):
        print(f"Preparing test dataset for {self.TARGET_PATH}")
        with open(self.ATTACK_SETS + "test.p", "wb") as f:
            for inputs, targets, members in self.attack_test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                output, prediction = self._get_data(self.target_model, inputs)
                # output = output.cpu().detach().numpy()
            
                pickle.dump((output, prediction, members), f)

        self.logger.add_line(f"Finished Saving Test Dataset to {self.ATTACK_SETS + 'test.p'}")

    def train(self, epoch, result_path):
        self.attack_model.train()
        batch_idx = 1
        train_loss = 0
        correct = 0
        total = 0

        final_train_gndtrth = []
        final_train_predict = []
        final_train_probabe = []

        final_result = []

        with open(self.ATTACK_SETS + "train.p", "rb") as f:
            while(True):
                try:
                    output, prediction, members = pickle.load(f)
                    output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                    results = self.attack_model(output, prediction)
                    results = F.softmax(results, dim=1)

                    losses = self.criterion(results, members)
                    losses.backward()
                    self.optimizer.step()

                    train_loss += losses.item()
                    _, predicted = results.max(1)
                    total += members.size(0)
                    correct += predicted.eq(members).sum().item()

                    if epoch:
                        final_train_gndtrth.append(members)
                        final_train_predict.append(predicted)
                        final_train_probabe.append(results[:, 1])

                    batch_idx += 1
                except EOFError:
                    break

        if epoch:
            final_train_gndtrth = torch.cat(final_train_gndtrth, dim=0).cpu().detach().numpy()
            final_train_predict = torch.cat(final_train_predict, dim=0).cpu().detach().numpy()
            final_train_probabe = torch.cat(final_train_probabe, dim=0).cpu().detach().numpy()

            train_f1_score = f1_score(final_train_gndtrth, final_train_predict)
            train_roc_auc_score = roc_auc_score(final_train_gndtrth, final_train_probabe)

            final_result.append(train_f1_score)
            final_result.append(train_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_train_gndtrth, final_train_predict, final_train_probabe), f)
            
            self.logger.add_line("Saved Attack Train Ground Truth and Predict Sets")
            self.logger.add_line("Train F1: %f\nAUC: %f" % (train_f1_score, train_roc_auc_score))

        final_result.append(1.*correct/total)
        self.logger.add_line( 'Train Acc: %.3f%% (%d/%d) | Loss: %.3f' % (100.*correct/total, correct, total, 1.*train_loss/batch_idx))

        return final_result

    def test(self, epoch, result_path, best_result_path):
        self.attack_model.eval()
        batch_idx = 1
        correct = 0
        total = 0

        final_test_gndtrth = []
        final_test_predict = []
        final_test_probabe = []

        final_result = []

        with torch.no_grad():
            with open(self.ATTACK_SETS + "test.p", "rb") as f:
                while(True):
                    try:
                        output, prediction, members = pickle.load(f)
                        output, prediction, members = output.to(self.device), prediction.to(self.device), members.to(self.device)

                        results = self.attack_model(output, prediction)
                        _, predicted = results.max(1)
                        total += members.size(0)
                        correct += predicted.eq(members).sum().item()
                        results = F.softmax(results, dim=1)

                        final_test_gndtrth.append(members.detach())
                        final_test_predict.append(predicted.detach())
                        final_test_probabe.append(results[:, 1].detach())

                        batch_idx += 1
                    except EOFError:
                        break
                    
        acc = correct/(1.0*total)

        if epoch or acc > self.best_acc:
            final_test_gndtrth = torch.cat(final_test_gndtrth, dim=0).cpu().numpy()
            final_test_predict = torch.cat(final_test_predict, dim=0).cpu().numpy()
            final_test_probabe = torch.cat(final_test_probabe, dim=0).cpu().numpy()

            test_f1_score = f1_score(final_test_gndtrth, final_test_predict)
            test_roc_auc_score = roc_auc_score(final_test_gndtrth, final_test_probabe)

            final_result.append(test_f1_score)
            final_result.append(test_roc_auc_score)

            with open(result_path, "wb") as f:
                pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

            self.logger.add_line("Saved Attack Test Ground Truth and Predict Sets")
            self.logger.add_line("Test F1: %f\nAUC: %f" % (test_f1_score, test_roc_auc_score))
            
            if acc > self.best_acc:
                self.best_acc = acc
                self.best_test_gndtrth = final_test_gndtrth
                self.best_test_predict = final_test_predict
                self.best_test_probabe = final_test_probabe
                self.best_state_dict = self.attack_model.state_dict()
                
                with open(best_result_path, "wb") as f:
                    pickle.dump((final_test_gndtrth, final_test_predict, final_test_probabe), f)

        final_result.append(1.*correct/total)
        self.logger.add_line( 'Test Acc: %.3f%% (%d/%d)' % (100.*correct/(1.0*total), correct, total))

        return final_result
    
    def eval_best_result(self):
        best_f1_score = f1_score(self.best_test_gndtrth, self.best_test_predict)
        best_roc_auc_score = roc_auc_score(self.best_test_gndtrth, self.best_test_probabe)
        self.logger.add_line("Best Acc: %f\n F1: %f\nAUC: %f" % (self.best_acc, best_f1_score, best_roc_auc_score))
        return best_f1_score, best_roc_auc_score, self.best_acc

    def delete_pickle(self):
        train_file = glob.glob(self.ATTACK_SETS +"train.p")
        for trf in train_file:
            os.remove(trf)

        test_file = glob.glob(self.ATTACK_SETS +"test.p")
        for tef in test_file:
            os.remove(tef)

    def saveModel(self, path):
        torch.save(self.best_state_dict, path)
        
    def loadModel(self, path):
        ckpt = torch.load(path)
        self.attack_model.load_state_dict(ckpt)


def get_optimizer(parameters, optimizer_type, lr=0.01, momentum=0.5, **kwargs):
    assert optimizer_type in ['sgd', 'sgdm', 'adam', 'adagrad']
    if optimizer_type == 'sgd':
        optimizer = optim.SGD(parameters, lr)
    elif optimizer_type == 'sgdm':
        optimizer = optim.SGD(parameters, lr, momentum=momentum)
    elif optimizer_type == 'adagrad':
        optimizer = optim.Adagrad(parameters)
    elif optimizer_type == 'adam':
        optimizer = optim.Adam(parameters)
    else:
        raise ValueError('Unrecognized optimizer type')
    return optimizer


def trian_shadow_model(model, dataset_name, b, model_dir, device, shadow_epochs):
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    model.to(device)
    modelfamily = 'cifar'
    if "vgg" in  model_dir or "alex" in model_dir:
        transform = datasets.modelfamily_to_transforms[modelfamily]['train']
    else:
        transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    dataset = datasets.__dict__[dataset_name]
    testset = dataset(train=False, transform=transform)
    
    
    
    transferset_path = osp.join(model_dir, 'transferset.pickle')
    with open(transferset_path, 'rb') as rf:
        transferset_samples = pickle.load(rf)
    transferset = samples_to_transferset(transferset_samples, budget=b, transform=transform)

    print('=> Training at budget = {}'.format(len(transferset)))

    optimizer = get_optimizer(model.parameters(), 'sgdm', **params)

    checkpoint_suffix = '.{}'.format(b)
    criterion_train = model_utils.soft_cross_entropy
    model_utils.train_model(model, transferset, model_dir, testset=testset, criterion_train=criterion_train,
                                checkpoint_suffix=checkpoint_suffix, device=device, optimizer=optimizer, epochs=shadow_epochs)
    
    shadow_path = osp.join(model_dir, 'mia_shadow.pth')
    print(f"Shadow model saved to {shadow_path}")
    torch.save(model.state_dict(), shadow_path)


def attack_mode0(
    TARGET_PATH, SHADOW_PATH, ATTACK_PATH, device, attack_trainloader, attack_testloader, 
    target_model, shadow_model, attack_model, get_attack_set, num_classes
):
    MODELS_PATH = osp.join(ATTACK_PATH, "meminf_attack0.pth")
    RESULT_PATH = osp.join(ATTACK_PATH, "meminf_attack0.p")
    BEST_RESULT_PATH = osp.join(ATTACK_PATH, "meminf_best_attack0.p")
    ATTACK_SETS = osp.join(ATTACK_PATH, "meminf_attack_mode0_")
    logger = Logger(log2file=True, mode=sys._getframe().f_code.co_name, path=ATTACK_PATH)

    epochs = 20
    attack = attack_for_blackbox(
        SHADOW_PATH, TARGET_PATH, ATTACK_SETS, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, device, logger
    )

    if get_attack_set:
        attack.delete_pickle()
        attack.prepare_dataset()

    for i in range(epochs):
        flag = 1 if i == epochs-1 else 0
        logger.add_line("Epoch %d :" % (i+1))
        res_train = attack.train(flag, RESULT_PATH)
        res_test = attack.test(flag, RESULT_PATH, BEST_RESULT_PATH)
        # if gol.get_value("debug"):
        #     break
        
    res_best = attack.eval_best_result()

    attack.saveModel(MODELS_PATH)
    logger.add_line(f"Saved Attack Model to {MODELS_PATH}")
    print(f"{sys._getframe().f_code.co_name} finished")

    return res_train, res_test, res_best



def get_attack_dataset_with_shadow(target_train, target_test, shadow_train, shadow_test, batch_size):
    '''
    准备攻击模型的数据集
    Args:
        target_train: 目标模型的训练集 -》 mem_test
        target_test: 目标模型的测试集 -》 nonmem_test
        shadow_train: 部分保护的模型的训练集 -》 mem_train
        shadow_test: 部分保护的模型的测试集 -》 nonmem_train
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test
    '''
    
    
    mem_train, nonmem_train, mem_test, nonmem_test = list(shadow_train), list(shadow_test), list(target_train), list(target_test)

    # shadow_trian 是transfer的数据集，从pickle加载的时候需要把第二个维度转换成标签
    for i in range(len(mem_train)):
        # mem_train[i][1] 是一个100维的tensor，表示了每个类别的概率
        # 需要获取这个tensor中最大的概率对应的类别
        target_labels = mem_train[i][1].argmax()
        # target_labels类型是一个tensor，需要转换成一个int
        target_labels = target_labels.item()
        mem_train[i] = (mem_train[i][0], target_labels)
        
    for i in range(len(mem_train)):
        mem_train[i] = mem_train[i] + (1,)
    for i in range(len(nonmem_train)):
        nonmem_train[i] = nonmem_train[i] + (0,)

    for i in range(len(nonmem_test)):
        nonmem_test[i] = nonmem_test[i] + (0,)
 
    for i in range(len(mem_test)):
        mem_test[i] = mem_test[i] + (1,)
    
    # print(f"mem_train: {mem_train[0]}\n nonmem_train: {nonmem_train[0]}\n mem_test: {mem_test[0]}\n nonmem_test: {nonmem_test[0]}")
    
    print(f'len(mem_train): {len(mem_train)}')
    print(f'len(nonmem_train): {len(nonmem_train)}')
    print(f'len(mem_test): {len(mem_test)}')
    print(f'len(nonmem_test): {len(nonmem_test)}')
    train_length = min(len(mem_train), len(nonmem_train))
    test_length = min(len(mem_test), len(nonmem_test))
    
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    mem_train, _ = torch.utils.data.random_split(mem_train, [train_length, len(mem_train) - train_length])
    non_mem_train, _ = torch.utils.data.random_split(nonmem_train, [train_length, len(nonmem_train) - train_length])
    mem_test, _ = torch.utils.data.random_split(mem_test, [test_length, len(mem_test) - test_length])
    non_mem_test, _ = torch.utils.data.random_split(nonmem_test, [test_length, len(nonmem_test) - test_length])
    
    attack_train = mem_train + non_mem_train
    attack_test = mem_test + non_mem_test
    
    print(f'len(attack_train): {len(attack_train)}')
    print(f'len(attack_test): {len(attack_test)}')

    attack_trainloader = torch.utils.data.DataLoader(
        attack_train, batch_size=batch_size, shuffle=True, num_workers=2)
    attack_testloader = torch.utils.data.DataLoader(
        attack_test, batch_size=batch_size, shuffle=True, num_workers=2)

    return attack_trainloader, attack_testloader


def compute_confidence_gap(model, model_path, mem_set, nonmem_set, device):
    print("compute_confidence_gap")
    model = model.to(device)
    print(f"Load model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mem_loader = torch.utils.data.DataLoader(
        mem_set, batch_size=64, shuffle=True, num_workers=8)
    nonmem_loader = torch.utils.data.DataLoader(
        nonmem_set, batch_size=64, shuffle=True, num_workers=8)
    
    total_confidence = []
    with torch.no_grad():
        for inputs, targets in mem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)
            
            confidence = [
                output[t].item() for output, t in zip(outputs, targets)
            ]
            total_confidence += confidence

    mem_mean_confidence = np.mean(total_confidence)
    
    total_confidence = []
    with torch.no_grad():
        for inputs, targets in nonmem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = F.softmax(outputs, dim=1)

            confidence = [
                output[t].item() for output, t in zip(outputs, targets)
            ]
            total_confidence += confidence

    nonmem_mean_confidence = np.mean(total_confidence)
    
    gap = mem_mean_confidence - nonmem_mean_confidence
    # 计算并打印出两个平均置信度之间的差值
    print(f"Confidence gap: member mean conf {mem_mean_confidence:.2f}, non member mean conf {nonmem_mean_confidence:.2f}, gap {gap:.2f}")
    return gap


def compute_generalization_gap(model, model_path, mem_set, nonmem_set, device):
    print("compute_generalization_gap")
    model = model.to(device)
    print(f"Load model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    model.eval()

    mem_loader = torch.utils.data.DataLoader(
        mem_set, batch_size=64, shuffle=True, num_workers=8)
    nonmem_loader = torch.utils.data.DataLoader(
        nonmem_set, batch_size=64, shuffle=True, num_workers=8)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in mem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
    mem_acc = 1.*correct/total
    
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in nonmem_loader:
            if isinstance(targets, list):
                targets = targets[0]

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            _, predicted = outputs.max(1)
            total += targets.size(0)

            correct += predicted.eq(targets).sum().item()
    nonmem_acc = 1.*correct/total
    
    gap = mem_acc - nonmem_acc
    # 计算并打印出成员准确率和非成员准确率之间的差距
    print(f"Generalization gap: member acc {mem_acc:.2f}, non member acc {nonmem_acc:.2f}, gap {gap:.2f}")
    return gap


def meminf_no_train(
    save_dir, target_model_dir, shadow_model_dir,
    device, num_classes, target_train, target_test, shadow_train, shadow_test, 
    target_model, shadow_model, args):
    '''
        target_model_dir = 初始的victim模型 （训练结果存放的路径）
        shadow_model_dir = 部分保护的模型(使用transfer数据集训练得到的)
        target_model, shadow_model需要满足字典的架构
    '''
    
    LOG_PATH = save_dir
    if not os.path.exists(LOG_PATH):
        os.makedirs(LOG_PATH)
        
    shadow_model_path = osp.join(shadow_model_dir, "mia_shadow.pth")
    target_model_path = osp.join(target_model_dir, "target.pth")

    if not os.path.exists(shadow_model_path):
        print(f"Shadow model not exists in {shadow_model_path}")
        return 
            
    last_results, best_results = {}, {}
    
    generalization_gap = compute_generalization_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    last_results['generalization_gap'] = generalization_gap
    best_results['generalization_gap'] = generalization_gap
    # print(generalization_gap)
    
    confidence_gap = compute_confidence_gap(
        target_model, target_model_path, target_train, target_test, device
    )
    last_results['confidence_gap'] = confidence_gap
    best_results['confidence_gap'] = confidence_gap
    # print(confidence_gap)
            
    # shadow dataset, output
    tag = 'mode0'
    print("Prepare mode0 attack dataset")
    attack_trainloader, attack_testloader = get_attack_dataset_with_shadow(
        target_train, target_test, shadow_train, shadow_test, args.batch_size)
    print("Finish mode0 attack dataset prepare")
    attack_model = PartialAttackModel(num_classes)
    print("Mode0 attack")
    _, mode0_res_last, mode0_res_best = attack_mode0(
        target_model_path, shadow_model_path, LOG_PATH, device, attack_trainloader, attack_testloader, 
        target_model, shadow_model, attack_model, 1, num_classes
    )
    last_results[f"{tag}_last_f1"] = eval(f"{tag}_res_last")[0]
    last_results[f"{tag}_last_roc_auc"] = eval(f"{tag}_res_last")[1]
    last_results[f"{tag}_last_acc"] = eval(f"{tag}_res_last")[2]
    best_results[f"{tag}_best_f1"] = eval(f"{tag}_res_best")[0]
    best_results[f"{tag}_best_roc_auc"] = eval(f"{tag}_res_best")[1]
    best_results[f"{tag}_best_acc"] = eval(f"{tag}_res_best")[2]
    print("Mode0 attack finished")

    last_path = osp.join(save_dir, "last.json")
    with open(last_path, 'w') as f:
        json.dump(last_results, f, indent=True)
    best_path = osp.join(save_dir, "best.json")
    with open(best_path, 'w') as f:
        json.dump(best_results, f, indent=True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a model')
    # Required arguments
    parser.add_argument('dataset', metavar='DS_NAME', type=str, help='Dataset name')
    parser.add_argument('model_arch', metavar='MODEL_ARCH', type=str, help='Model name')
    # Optional arguments
    parser.add_argument('-o', '--out_path', metavar='PATH', type=str, help='Output path for model')
    parser.add_argument('--victim_dir', type=str)
    parser.add_argument('-d', '--device_id', metavar='D', type=int, help='Device id. -1 for CPU.', default=0)
    parser.add_argument('-b', '--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--budgets', metavar='N', type=int, help='Size of transfer set to construct',
                        required=True)
    parser.add_argument('-e', '--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--shadow-epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-2, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--shadow-lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--lr-step', type=int, default=30, metavar='N',
                        help='Step sizes for LR')
    parser.add_argument('--lr-gamma', type=float, default=0.1, metavar='N',
                        help='LR Decay Rate')
    parser.add_argument('--train_subset', type=int, help='Use a subset of train set', default=None)
    parser.add_argument('--pretrained', action="store_true", default=False)
    ##################
    parser.add_argument('--protect_percent', type=float, default=0.5, metavar='N',
                        help='protection percent [0 white] < protect_percent < [1 black]')
    parser.add_argument('--channel_percent', type=float, default=None, metavar='N')
    #######################
    args = parser.parse_args()
    params = vars(args)
    dataset_name = params['dataset']
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    if params['device_id'] >= 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(params['device_id'])
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    model_name = params['model_arch']
    budget = params['budgets']
    pretrained = "imagenet"
    modelfamily = "cifar"
    transfer_set_path = f"/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets/models/adversary/{dataset_name.lower()}-{model_name.lower()}-random/transferset.pickle"
    transfer_set_dir = f"/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets/models/adversary/{dataset_name.lower()}-{model_name.lower()}-random/"
    
    num_classes, target_train, target_test, shadow_train, shadow_test = prepare_dataset(
        dataset_name.upper(), modelfamily, transfer_set_path, budget=budget
    )
    print(f"target_train: {len(target_train)}, target_test: {len(target_test)}, shadow_train: {len(shadow_train)}, shadow_test: {len(shadow_test)}")
    
    # target_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    # shadow_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes)
    
    ## add 719
    
    channel_percent = params['channel_percent']
    target_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes, 
                            is_victim=True)
    shadow_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes, 
                            is_victim=True)
    
    tmp_shadow_model = zoo.get_net(model_name, modelfamily, pretrained, num_classes=num_classes, 
                        victim_dir = params['victim_dir'], get_importance_dataset = dataset_name.upper(),
                        protect_percent = params['protect_percent'] , channel_percent = channel_percent)
    
    trian_shadow_model(tmp_shadow_model, dataset_name, budget, transfer_set_dir, device, params['shadow_epochs'])
    
    
    
    meminf_no_train(
        args.out_path, args.victim_dir, args.out_path,
        device, num_classes, target_train, target_test, shadow_train, shadow_test, 
        target_model, shadow_model, args)
    # meminf_no_train(
    #     args.out_path, args.victim_dir, args.out_path,
    #     device, num_classes, target_train, target_test, target_train, target_test, 
    #     target_model, shadow_model, args)