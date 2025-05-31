import os
import sys


if __name__ == "__main__":
    
    del_dir = sys.argv[1]  # 删除这个文件夹下所有以meminf开头的文件
    for root, dirs, files in os.walk(del_dir):
        for name in files:
            if "log" in name  or "pth" in name:
                os.remove(os.path.join(root, name))
                print("Delete file: ", os.path.join(root, name))

