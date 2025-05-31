layer_dic = {
    "resnet18": 41,
    "alexnet": 6,
    "vgg16_bn": 27,
    "resnet50": 107,
    "mobilenetv2": 105
}

import os

def delete_channel_log():
    # Delete all .tsv log files under directories ending with 'channel' inside 'models/adversary'
    channel_path = os.path.join(os.getcwd(), 'models/adversary')
    for root, dirs, files in os.walk(channel_path):
        if root.endswith('channel'):
            for file in files:
                if file.endswith('tsv'):
                    os.remove(os.path.join(root, file))
                    print("Delete file: ", os.path.join(root, file))
    print("Delete all channel log files successfully!")

def run():
    # Datasets to evaluate
    datasets = ['stl10']
    # Model architectures to evaluate
    archs = ['resnet18']
    
    # Path to store final results
    final_result_path = os.path.join(os.getcwd(), 'ms_layers_result_resnet18_stl10_b1000.csv')
    # Write CSV header
    with open(final_result_path, 'a') as f:
        f.write("dataset,arch,layer_percent,channel_percent,epoch,final_acc,best_acc\n")
    print("result path: ", final_result_path)

    epoch = 100
    for dataset in datasets:
        for arch in archs:
            for layer_percent in range(layer_dic[arch]):
                for channel_percent in [-1]:
                    cmd = f"python3 ./knockoff/adversary/train.py models/adversary/{dataset}-{arch}-channel {arch} {dataset.upper()} --budgets 1000 -d 1 --pretrained imagenet --log-interval 100 --epochs {epoch} --lr 0.1 --victim_dir models/victim/{dataset}-{arch} --protect_percent {layer_percent} --channel_percent {channel_percent}"
                    if dataset == "tinyimagenet200":
                        cmd = cmd.replace("TINYIMAGENET200", "TinyImageNet200")
                    
                    print("===========start==========")
                    print(cmd)
                    # Execute the command
                    os.system(cmd)
                    print("===========finish==========")

                    # Get the last line of the training log
                    log_path = f"models/adversary/{dataset}-{arch}-channel/train.1000.log.tsv"
                    with open(log_path, 'r') as f:
                        lines = f.readlines()
                        final_result = lines[-1]
                        # Write final result to the CSV file
                        with open(final_result_path, 'a') as f:
                            final_result = final_result.strip().replace("\t", " ")
                            final_result = final_result.split(" ")
                            print(final_result)
                            epo = final_result[-5]
                            final_acc = final_result[-2]
                            best_acc = final_result[-1]
                            f.write(f"{dataset},{arch},{layer_percent},{channel_percent},{epo},{final_acc},{best_acc}\n")

if __name__ == '__main__':
    # delete_channel_log()
    run()
