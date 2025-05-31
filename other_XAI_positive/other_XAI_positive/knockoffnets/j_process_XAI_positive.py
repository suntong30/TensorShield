layer_dic ={"resnet18":41,"alexnet":6,"vgg16_bn":27, "resnet50":107, "mobilenetv2":105}
import  os
def delete_channel_log():
    channel_path = os.path.join(os.getcwd(), 'models/adversary')
    for root, dirs, files in os.walk(channel_path):
        if root.endswith('channel'):
            for file in files:
                if file.endswith('tsv'):
                    os.remove(os.path.join(root, file))
                    print("Delete file: ", os.path.join(root, file))
    print("Delete all channel log files successfully!")
    
def run():
    # datasets = ['cifar10', 'cifar100', 'tinyimagenet200']
    datasets = ['cifar100', 'cifar10', 'tinyimagenet200']
    # datasets = ['cifar10']
    # archs = ['resnet18', 'alexnet', 'vgg16_bn']
    archs = ['resnet18', 'vgg16_bn', 'mobilenetv2']
    final_result_path = os.path.join(os.getcwd(), 'ms_elastictrainer_result_resnet18_vgg16_mobilenetv2.csv')
    # final_result_path = os.path.join(os.getcwd(), 'ms_ours_result_mobilenetv2_cifar10_b1000.csv')
    with open(final_result_path, 'a') as f:
        f.write("dataset,arch,layer_percent,channel_percent,epoch,final_acc,best_acc\n")
    print("result path: ", final_result_path)
    epoch = 100
    for dataset in datasets:
        for arch in archs:
            # for layer_percent in range(layer_dic[arch]):
            for layer_percent in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]:
                for channel_percent in [-1]:
                    if dataset == "cifar100" and arch == "resnet18":
                        continue
                    cmd = f"/home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 ./knockoff/adversary/train.py models/adversary/{dataset}-{arch}-channel {arch} {dataset.upper()} --budgets 1000 -d 1 --pretrained imagenet --log-interval 100 --epochs {epoch} --lr 0.1 --victim_dir models/victim/{dataset}-{arch} --protect_percent {layer_percent} --channel_percent {channel_percent}"
                    if dataset == "tinyimagenet200":
                        cmd = cmd.replace("TINYIMAGENET200", "TinyImageNet200")
                    print("===========start==========")
                    print(cmd)
                    os.system(cmd)
                    print("===========finish==========")
                    log_path = f"models/adversary/{dataset}-{arch}-channel/train.1000.log.tsv"
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        final_result = lines[-1]
                        with open(final_result_path, 'a') as f:
                            final_result = final_result.strip().replace("\t"," ")
                            final_result = final_result.split(" ")
                            print(final_result)
                            epo =   final_result[-5]
                            final_acc = final_result[-2]
                            best_acc = final_result[-1]
                            f.write(f"{dataset},{arch},{layer_percent},{channel_percent},{epo},{final_acc},{best_acc}\n")

if __name__ == '__main__':
    # delete_channel_log()
    run()