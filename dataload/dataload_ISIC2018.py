# begin 加载 ISIC2018
#  7分类
from torch.utils.data import Dataset, DataLoader, get_worker_info
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
import math
import os
import torch
from pathlib import Path

#
class ISIC2018Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.images_path = []
        self.images_label = []
        supported_formats = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件类型

        # 类别名称及其对应的标签索引
        class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
        self.class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

        # 遍历所有类别文件夹，收集图片路径和标签
        for class_name in os.listdir(data_path):
            class_dir = os.path.join(data_path, class_name)
            if class_name in self.class_indices and os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if os.path.splitext(img_name)[-1] in supported_formats:
                        img_path = os.path.join(class_dir, img_name)
                        self.images_path.append(img_path)
                        self.images_label.append(self.class_indices[class_name])
            else:
                print(f"Warning: Unrecognized class folder {class_name}")

        print(f"{len(self.images_path)} images were found in the dataset.")

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        img_path = self.images_path[idx]
        label = self.images_label[idx]
        img = Image.open(img_path)

        # 转换为 RGB 模式
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            raise ValueError('Image is not preprocessed')

        return img, label

# import numpy as np

# mod 初始化阶段存储图片
# class ISIC2018Dataset(Dataset):
#     def __init__(self, data_path, transform=None):
#         self.data_path = data_path
#         self.transform = transform
#         self.images_path = []  # 存储所有图片路径
#         self.images_label = []  # 存储图片对应的标签索引
#         self.images_data = []  # 存储图像数据
#         supported_formats = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件类型
#
#         # 类别名称及其对应的标签索引
#         class_names = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
#         self.class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}
#
#         # 遍历所有类别文件夹，收集图片路径、标签和图像数据
#         for class_name in os.listdir(data_path):
#             class_dir = os.path.join(data_path, class_name)
#             if class_name in self.class_indices and os.path.isdir(class_dir):
#                 for img_name in os.listdir(class_dir):
#                     if os.path.splitext(img_name)[-1] in supported_formats:
#                         img_path = os.path.join(class_dir, img_name)
#                         self.images_path.append(img_path)
#                         self.images_label.append(self.class_indices[class_name])
#
#                         # 使用 PIL 打开图片并转换为 NumPy 数组
#                         img = Image.open(img_path)
#                         if img.mode != 'RGB':
#                             img = img.convert('RGB')
#                         img_array = np.array(img)
#
#                         # 将图像数据存储为 NumPy 数组
#                         self.images_data.append(img_array)
#
#         # 将所有图像数据转换为 numpy 数组并进行内存映射
#         self.images_data = np.array(self.images_data, dtype=np.uint8)
#         print(f"{len(self.images_path)} images were found in the dataset.")
#
#     def __len__(self):
#         return len(self.images_path)
#
#     def __getitem__(self, idx):
#         # 直接从内存中加载图像数据
#         img_data = self.images_data[idx]
#         label = self.images_label[idx]
#
#         # 将图像数据转换为 PIL 图像对象
#         img = Image.fromarray(img_data)
#
#         if self.transform:
#             img = self.transform(img)
#         else:
#             raise ValueError('Image is not preprocessed')
#
#         return img, label




class ResizeWithAspectRatio:
    def __init__(self, target_height):
        self.target_height = target_height

    def __call__(self, img):
        width, height = img.size
        target_width = int(width * self.target_height / height)
        return img.resize((target_width, self.target_height), resample=Image.Resampling.LANCZOS)



def load_isic2018_dataset(batch_size=64, num_workers=4, pin_memory=True):
    """
    加载 ISIC2018 数据集
    :param batch_size: 批处理大小
    :param num_workers: DataLoader 的 worker 数量
    :param pin_memory: 是否使用 pin_memory
    :return: 训练、验证和测试的 DataLoader
    """
    # data_path = "D:/WBY/dataset/ISIC2018"
    data_path = "E:/dataset/ISIC2018"

    # note 版本1
    # train_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.RandomHorizontalFlip(),
    #     # AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # note 数据增强
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.7481, 0.2223, 0.4616], std=[0.9403, 0.8514, 0.8798])
    # ])
    #
    # val_test_transform = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.8438, 0.3051, 0.5409], std=[0.7399, 0.7435, 0.7748])
    # ])

    train_transform = transforms.Compose([
        # ResizeWithAspectRatio(336),
        transforms.Resize((336, 336)),
        transforms.RandomCrop((256, 256), padding=(20, 20)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集路径
    train_path = os.path.join(data_path, "train_class_augmentation")
    val_path = os.path.join(data_path, "val_class")
    test_path = os.path.join(data_path, "test_class")
    print("准备进行数据加载")

    # 数据集加载
    train_dataset = ISIC2018Dataset(data_path=train_path, transform=train_transform)
    # val_dataset = ISIC2018Dataset(data_path=val_path, transform=val_test_transform)
    test_dataset = ISIC2018Dataset(data_path=test_path, transform=val_test_transform)
    print("准备进行数据加载")
    # DataLoader 创建
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    # val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
    #                         num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    print("数据加载完毕")
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader = load_isic2018_dataset(batch_size=32)

