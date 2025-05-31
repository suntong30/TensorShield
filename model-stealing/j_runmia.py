# Delete log files under directories ending with 'channel' in knockoff/models/adversary
layer_dic = {"resnet18": 41, "alexnet": 6, "vgg16_bn": 27}
import os

def delete_channel_log():
    channel_path = os.path.join(os.getcwd(), 'models/adversary')
    for root, dirs, files in os.walk(channel_path):
        # Check if the directory name ends with 'channel'
        if root.endswith('channel'):
            for file in files:
                if file.endswith('tsv'):
                    os.remove(os.path.join(root, file))
                    print("Delete file: ", os.path.join(root, file))
    print("Delete all channel log files successfully!")

def run():
    datasets = ['tinyimagenet200', 'cifar10', 'cifar100']
    archs = ['resnet18', 'alexnet', 'vgg16_bn']
    final_result_path = os.path.join(os.getcwd(), 'triansh_mia_result.txt')
    
    # Write header if needed
    # with open(final_result_path, 'a') as f:
    #     f.write("dataset,arch,layer_percent,channel_percent,epoch,final_acc,best_acc\n")

    print("Result path: ", final_result_path)

    sh_epoch = 100
    for dataset in datasets:
        for arch in archs:
            for layer_percent in range(layer_dic[arch]):
                for channel_percent in [-1]:
                    # Skip this specific combination
                    if dataset == "cifar100" and arch == "resnet18":
                        continue

                    # Convert dataset name to uppercase
                    cmd = (
                        f"/home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 "
                        f"./mia/mode0_attack.py {dataset.upper()} {arch} "
                        f"--out_path ./models/adversary/{dataset.lower()}-{arch.lower()}-random "
                        f"-d 0 --budgets 5000 -e 100 --shadow-epochs {sh_epoch} "
                        f"--lr 0.1 --shadow-lr 0.1 "
                        f"--victim_dir models/victim/{dataset.lower()}-{arch.lower()} "
                        f"--protect_percent {layer_percent} --channel_percent {channel_percent}"
                    )

                    # Fix casing for TinyImageNet200
                    if dataset == "tinyimagenet200":
                        cmd = cmd.replace("TINYIMAGENET200", "TinyImageNet200")

                    print("=========== start ===========")
                    print(cmd)
                    # Run the command
                    os.system(cmd)

                    out_dir = f"./models/adversary/{dataset.lower()}-{arch.lower()}-random"
                    # Delete all files in out_dir that start with "meminf"
                    for root, dirs, files in os.walk(out_dir):
                        for file in files:
                            if file.startswith("meminf"):
                                os.remove(os.path.join(root, file))
                                print("Delete file: ", os.path.join(root, file))
                            if file.startswith("attack"):
                                # Read content of attack result file
                                with open(os.path.join(root, file), 'r') as f:
                                    lines = f.readlines()
                                    last_lines = lines[-12:]  # Read the last 12 lines

                                    # Append results to final_result_path
                                    with open(final_result_path, 'a') as fr:
                                        fr.write(f"{dataset}=={arch}==-{layer_percent}=={channel_percent}\n")
                                        for line in last_lines:
                                            fr.write(line)
                                        fr.write("\n")

                                os.remove(os.path.join(root, file))
                                print("Delete file: ", os.path.join(root, file))

                    print("\n\n=========== finish ===========\n")

if __name__ == '__main__':
    # delete_channel_log()
    run()
