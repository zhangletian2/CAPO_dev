from PIL import Image
import torchvision.transforms as transforms
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import numpy as np
import os
from tqdm import tqdm

input_directory = '../data/imagenetv2/imagenetv2-matched-frequency-format-val/'
output_directory = '../data/imagenetv2/npy/'
# input_directory = './example/images/'
# output_directory = './example/npy/'
resolution = 224

if __name__ == "__main__":
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Folder '{output_directory}' created.")
    else:
        print(f"Folder '{output_directory}' already exists.")

    base_transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=BICUBIC),
        transforms.CenterCrop(resolution)])
    # ImageNet-v2
    for root, dirs, files in tqdm(os.walk(input_directory)):
        for file in files:
            # 检查文件是否为图像文件（根据扩展名）
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                filename = os.path.join(root, file)
                parts = filename.split('/')
                sub_dir = parts[-2:-1][0]
                img_num = parts[-1].split('.')[0]
                # im = base_transform(Image.open(os.path.join(input_directory, filename)))
                im = Image.open(filename)

                # import pdb
                # pdb.set_trace()
                npy = np.array(im)  # image类 转 numpy
                sub_dir = os.path.join(output_directory, sub_dir)
                if not os.path.exists(sub_dir):
                    os.makedirs(sub_dir)
                np.save(os.path.join(sub_dir, '%s.npy' % img_num), npy)
                #
                # im = Image.fromarray(npy)  # numpy 转 image类
                # im.save(os.path.join(output_directory, '%s.jpg' % img_num))

    # for filename in tqdm(os.listdir(input_directory)):
    #     if filename.endswith('.jpg'):
    #         img_num = filename.split('.')[0]
    #         # im = base_transform(Image.open(os.path.join(input_directory, filename)))
    #         im = Image.open(os.path.join(input_directory, filename))
    #
    #         # import pdb
    #         # pdb.set_trace()
    #         npy = np.array(im)  # image类 转 numpy
    #         np.save(os.path.join(output_directory, '%s.npy' % img_num), npy)
    #         #
    #         # im = Image.fromarray(npy)  # numpy 转 image类
    #         # im.save(os.path.join(output_directory, '%s.jpg' % img_num))
