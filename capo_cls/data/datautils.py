import os
from typing import Tuple
from PIL import Image
import numpy as np
from typing import Callable, Optional
from pathlib import Path
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

ID_to_DIRNAME = {
    'I': 'imagenet',
    'A': 'imagenet-adversarial',
    'K': 'imagenet-sketch',
    'R': 'imagenet-rendition',
    'V': 'imagenetv2',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, base_transform, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False, cloud_use_ori=False, rec_dir='new'):
    if set_id == 'I':
        root1 = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'images/val')
        root2 = root1 if cloud_use_ori else os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'images', rec_dir)
        testset = DualRootImageFolder(root1, root2, base_transform=base_transform, transform=transform)
    elif set_id == 'V':
        root1 = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenetv2-matched-frequency-format-val')
        root2 = root1 if cloud_use_ori else os.path.join(data_root, ID_to_DIRNAME[set_id], rec_dir)
        testset = DualRootImageFolder(root1, root2, base_transform=base_transform, transform=transform)
    elif set_id == 'A':
        root1 = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-a')
        root2 = root1 if cloud_use_ori else os.path.join(data_root, ID_to_DIRNAME[set_id], rec_dir)
        testset = DualRootImageFolder(root1, root2, base_transform=base_transform, transform=transform)
    elif set_id == 'K':
        root1 = os.path.join(data_root, ID_to_DIRNAME[set_id], 'images')
        root2 = root1 if cloud_use_ori else os.path.join(data_root, ID_to_DIRNAME[set_id], rec_dir)
        testset = DualRootImageFolder(root1, root2, base_transform=base_transform, transform=transform)
    elif set_id == 'R':
        root1 = os.path.join(data_root, ID_to_DIRNAME[set_id], 'imagenet-r')
        root2 = root1 if cloud_use_ori else os.path.join(data_root, ID_to_DIRNAME[set_id], rec_dir)
        testset = DualRootImageFolder(root1, root2, base_transform=base_transform, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform,
                                            mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), base_transform, transform,
                                            mode=mode, cloud_use_ori=cloud_use_ori, rec_dir=rec_dir)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError

    return testset


class DualRootImageFolder(Dataset):
    def __init__(self, root1: str, root2: str, base_transform: Optional[Callable] = None,
                 transform: Optional[Callable] = None, target_transform: Optional[Callable] = None,
                 loader: Optional[Callable] = None, is_valid_file: Optional[Callable] = None, allow_empty: bool = False):
        """
        Args:
            root1 (str or pathlib.Path): The first root directory.
            root2 (str or pathlib.Path): The second root directory.
            base_transform (callable, optional): A function/transform that takes in a PIL image.
            transform (callable, optional): A function/transform that takes in a PIL image.
            target_transform (callable, optional): A function/transform that takes in the target.
            loader (callable, optional): A function to load an image.
            is_valid_file (callable, optional): A function to validate an image file.
            allow_empty (bool, optional): Whether to allow empty directories as classes.
        """
        self.root1 = Path(root1)
        self.root2 = Path(root2)
        self.base_transform = base_transform
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader if loader else self.default_loader
        self.is_valid_file = is_valid_file
        self.allow_empty = allow_empty

        # Load the class names from the directories
        self.classes = sorted([d.name for d in self.root1.iterdir() if d.is_dir()])
        if not self.allow_empty:
            self.classes = [cls for cls in self.classes if
                            (self.root1 / cls).iterdir() and (self.root2 / cls).iterdir()]

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Collect the file paths for all images
        self.samples = []
        for cls in self.classes:
            cls_path1 = self.root1 / cls
            cls_path2 = self.root2 / cls
            # Get the image files that are common between both directories
            files1 = sorted([f for f in cls_path1.glob("*") if
                             f.is_file() and (self.is_valid_file(f) if self.is_valid_file else True)])
            files2 = sorted([f for f in cls_path2.glob("*") if
                             f.is_file() and (self.is_valid_file(f) if self.is_valid_file else True)])
            # Make sure to match files by name
            files1_dict = {f.name: f for f in files1}
            files2_dict = {f.name: f for f in files2}
            common_files = set(files1_dict.keys()).intersection(files2_dict.keys())
            for filename in common_files:
                self.samples.append((files1_dict[filename], files2_dict[filename], self.class_to_idx[cls]))

    def __getitem__(self, index: int):
        """
        Args:
            index (int): The index of the sample to return.

        Returns:
            tuple: (image1, image2, target), where image1 and image2 are the transformed images from
                   the two roots, and target is the class index.
        """
        file1, file2, target = self.samples[index]

        # Load images from both directories
        image1 = self.loader(file1)  # image from root1 (original file)
        image2 = self.loader(file2)  # image from root2 (reconstructed file)

        # Apply the transforms (if any)
        if self.transform:
            image1 = self.base_transform(image1)
            image2 = self.transform(image2)

        # Apply target transform (if any)
        if self.target_transform:
            target = self.target_transform(target)

        return image1, image2, target

    def __len__(self):
        return len(self.samples)

    @staticmethod
    def default_loader(path):
        """Default loader to read images using PIL."""
        return Image.open(path).convert('RGB')


class ImageNet_V2(Dataset):
    """ imagenet-v2 dataset """
    def __init__(self, root, mode='train', n_shot=None, transform=None):
        self.transform = transform
        self.path = root
        self.mode = mode

        self.cname = []
        with open(os.path.join(self.path, "variants.txt"), 'r') as fp:
            self.cname = [l.replace("\n", "") for l in fp.readlines()]

        self.image_list = []
        self.label_list = []
        with open(os.path.join(self.path, 'images_variant_{:s}.txt'.format(self.mode)), 'r') as fp:
            lines = [s.replace("\n", "") for s in fp.readlines()]
            for l in lines:
                ls = l.split(" ")
                img = ls[0]
                label = " ".join(ls[1:])
                self.image_list.append("{}.jpg".format(img))
                self.label_list.append(self.cname.index(label))

        if n_shot is not None:
            few_shot_samples = []
            c_range = max(self.label_list) + 1
            for c in range(c_range):
                c_idx = [idx for idx, lable in enumerate(self.label_list) if lable == c]
                random.seed(0)
                few_shot_samples.extend(random.sample(c_idx, n_shot))
            self.image_list = [self.image_list[i] for i in few_shot_samples]
            self.label_list = [self.label_list[i] for i in few_shot_samples]

    def __len__(self):
        return len(self.image_list)

    # def __getitem__(self, idx):
    #     image_path = os.path.join(self.path, 'images', self.image_list[idx])
    #     # image_path = os.path.join(self.path, 'rec_img/bz2/0', self.image_list[idx])
    #
    #     image = Image.open(image_path).convert('RGB')
    #     label = self.label_list[idx]
    #     if self.transform:
    #         image = self.transform(image)
    #
    #     return image, torch.tensor(label).long()

    def __getitem__(self, idx):
        # image_path = os.path.join(self.path, 'images', self.image_list[idx])
        image_path = os.path.join(self.path, 'rec_img/bz2/0', self.image_list[idx])
        ori_path = os.path.join(self.path, 'images', self.image_list[idx])

        image = Image.open(image_path).convert('RGB')
        ori = Image.open(ori_path).convert('RGB')
        label = self.label_list[idx]
        if self.transform:
            image = self.transform(image)
            ori = self.transform(ori)

        return image, ori, torch.tensor(label).long()


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ])


def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False,
                 severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity

    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views
