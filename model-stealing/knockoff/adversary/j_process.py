import subprocess
import os
if __name__ == '__main__':
    victim_dir_list = os.listdir("/home/gpu2/jbw/knockoff/TensorShield-main/knockoffnets/models/victim")
    noneed_list = ["cifar10-resnet18", ".gitignore"]
    victim_dir_list = [victim for victim in victim_dir_list if victim not in noneed_list]
    
    for victim in victim_dir_list:
        if victim == "cifar10-resnet18" or victim == ".gitignore":
            continue
        with open("shielding.txt", "a") as f:
            f.write("+" * 12 + victim + "="*12 +"\n")
            net_type = victim.split("-")[1]
            dataset = victim.split("-")[0]
            if dataset == "cifar10":
                dataset = "CIFAR10"
            elif dataset == "cifar100":
                dataset = "CIFAR100"
            else:
                dataset = "TinyImageNet200"
            
            coomand = f"python3 ./knockoff/adversary/train.py \
            models/adversary/{victim}-random {net_type} {dataset} \
            --budgets 1000 -d 0 --pretrained imagenet --log-interval 100 \
            --epochs 100 --lr 0.01 --victim_dir models/victim/{victim} "
            
            for protect_percent in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                f.write(f"protect_percent={protect_percent}\n")
                subprocess.run(coomand + f" --protect_percent {protect_percent}", shell=True)
                info_path = f"models/adversary/{victim}-random/train.1000.log.tsv"
                # 从info_path中读取最后一行的acc
                with open(info_path, "r") as rf:
                    lines = rf.readlines()
                    last_line = lines[-1]
                    f.write(last_line)
                f.write("\n")
                f.flush()
            