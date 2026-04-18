import os
import subprocess
from tqdm import tqdm


# 定义读取文件夹的图片，并递归处理
def process_images(root_folder, intermediate_folder, new_folder):
    # 确保中间文件夹和新文件夹存在

    os.makedirs(intermediate_folder, exist_ok=True)
    os.makedirs(new_folder, exist_ok=True)

    # 遍历根文件夹中的所有子文件夹和图片文件
    # 统计所有图片文件的总数
    all_images = []

    # 遍历根文件夹中的所有子文件夹和图片文件
    for subdir, _, files in os.walk(root_folder):
        # 只处理直接在 root_folder 下的文件和子文件夹中的文件
        if subdir == root_folder:
            # 根目录下的文件直接处理
            files_to_process = files
            relative_subdir = ''
        else:
            # 子目录中的文件处理
            files_to_process = files
            relative_subdir = os.path.relpath(subdir, root_folder)

        for file in files_to_process:
            # 检查文件是否是图片（jpg或png格式）
            if file.lower().endswith(('.jpg', '.png', 'jpeg')):
                all_images.append((subdir, file, relative_subdir))

    # 创建一个进度条来显示所有文件处理的进度
    with tqdm(total=len(all_images), desc="Processing Images", unit="file") as pbar:
        for subdir, file, relative_subdir in all_images:
            old_path = os.path.join(subdir, file)
            # 按照子文件夹名称来组织新文件夹结构
            if relative_subdir:
                new_subdir = os.path.join(new_folder, relative_subdir)
            else:
                new_subdir = new_folder
            os.makedirs(new_subdir, exist_ok=True)
            new_path = os.path.join(new_subdir, file)

            # 压缩文件
            intermediate_path = os.path.join(intermediate_folder, file.split('.')[0] + '.bin')

            compress_image(old_path, intermediate_path)

            # 解压缩文件
            decompress_image(intermediate_path, new_path)

            # 更新进度条
            pbar.update(1)

# 执行压缩操作
def compress_image(old_path, intermediate_path):
    command = [
        'python', 'AppEncDec.py', 'compress', old_path, intermediate_path,
        '--qp', '1', '--model_type', '0', '--device', 'cuda'
    ]
    print(f"Compressing {old_path} to {intermediate_path}...")
    subprocess.run(command)


# 执行解压操作
def decompress_image(intermediate_path, new_path):
    command = [
        'python', 'AppEncDec.py', 'decompress', intermediate_path, new_path,
        '--device', 'cuda'
    ]
    print(f"Decompressing {intermediate_path} to {new_path}...")
    subprocess.run(command)


# 主程序入口
if __name__ == "__main__":
    root_folder = '/share/project/xiaotongli/tta/data/imagenet-adversarial/imagenet-a'  # 输入的文件夹
    intermediate_folder = 'intermediate'  # 存放中间文件的文件夹
    new_folder = '/share/project/xiaotongli/tta/data/imagenet-adversarial/new'  # 输出解压后图片的文件夹

    process_images(root_folder, intermediate_folder, new_folder)
