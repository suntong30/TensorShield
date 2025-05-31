# 删除konckoff/models/adversary 路径下结尾为channel文件夹下 log文件
layer_dic ={"resnet18":41,"alexnet":6,"vgg16_bn":27}
import  os
def delete_channel_log():
    channel_path = os.path.join(os.getcwd(), 'models/adversary')
    for root, dirs, files in os.walk(channel_path):
        #判断目录名称是否带有channel
        if root.endswith('channel'):
            for file in files:
                if file.endswith('tsv'):
                    os.remove(os.path.join(root, file))
                    print("Delete file: ", os.path.join(root, file))
    print("Delete all channel log files successfully!")
    
def run():
    datasets = ['tinyimagenet200', 'cifar10', 'cifar100']
    archs = ['resnet18' ,'alexnet', 'vgg16_bn']
    final_result_path = os.path.join(os.getcwd(), 'triansh_mia_result.txt')
    # 写入表头
    # with open(final_result_path, 'a') as f:
    #     f.write("dataset,arch,layer_percent,channel_percent,epoch,final_acc,best_acc\n")
    print("result path: ", final_result_path)
    sh_epoch = 100
    for dataset in datasets:
        for arch in archs:
            for layer_percent in range(layer_dic[arch]):
                for channel_percent in [-1]:
                    if dataset == "cifar100" and arch == "resnet18":
                        continue
                    # 将dataset转换为大写
                    cmd = f"/home/gpu2/miniconda3/envs/sunt_torch2.1.1/bin/python3 ./mia/mode0_attack.py {dataset.upper()} {arch} --out_path ./models/adversary/{dataset.lower()}-{arch.lower()}-random -d 0 --budgets 5000 -e 100 --shadow-epochs {sh_epoch} --lr 0.1 --shadow-lr 0.1 --victim_dir models/victim/{dataset.lower()}-{arch.lower()} --protect_percent {layer_percent} --channel_percent {channel_percent}"
                    
                    if dataset == "tinyimagenet200":
                        cmd = cmd.replace("TINYIMAGENET200", "TinyImageNet200")
                    print("===========start==========")
                    print(cmd)
                    # 等待cmd执行完毕
                    os.system(cmd)
                    
                    
                    out_dir = f"./models/adversary/{dataset.lower()}-{arch.lower()}-random"
                    # 删除out_dir目录下所有以meminf开头的文件
                    for root, dirs, files in os.walk(out_dir):
                        for file in files:
                            if file.startswith("meminf"):
                                os.remove(os.path.join(root, file))
                                print("Delete file: ", os.path.join(root, file))
                            if file.startswith("attack"):
                                # 读取文件内容
                                with open(os.path.join(root, file), 'r') as f:
                                    lines = f.readlines()  # 读取所有行到列表
                                    last_lines = lines[-12:]  # 假设我们需要最后3行，可以根据需要调整
                                    with open(final_result_path, 'a') as fr:
                                        fr.write(f"{dataset}=={arch}==-{layer_percent}=={channel_percent}\n")
                                        for line in last_lines:
                                            fr.write(line)
                                        fr.write("\n")
                                os.remove(os.path.join(root, file))
                                print("Delete file: ", os.path.join(root, file))
                    print("\n\n ===========finish========== \n")
                    
if __name__ == '__main__':
    # delete_channel_log()
    run()