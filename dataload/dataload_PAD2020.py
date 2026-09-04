# begin 加载 PAD2020
#  采用交叉验证的方式
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from glob import glob
import numpy as np
import torch
import os
from termcolor import colored
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


class PAD2020Dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

        self.images_path = []  # 存储所有图片路径
        self.images_label = []  # 存储图片对应的标签索引
        supported_formats = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件类型

        # 类别名称及其对应的标签索引
        class_names = ['ACK', 'BCC', 'MEL', 'NEV', 'SCC', 'SEK']
        self.class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

        # 遍历所有给定的文件夹，收集图片路径和标签
        for data_path in data_paths:
            assert os.path.exists(data_path), f"Data path {data_path} does not exist."

            # 遍历每个类别的文件夹
            for class_name in os.listdir(data_path):
                class_dir = os.path.join(data_path, class_name)
                if class_name in self.class_indices and os.path.isdir(class_dir):
                    # 收集该类别下的所有图片路径和标签
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

        # 检查图像模式，并转换为 RGB
        if img.mode != 'RGB':
            img = img.convert('RGB')

        if self.transform:
            img = self.transform(img)
        else:
            raise ValueError('Image is not preprocessed')

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels



def load_PAD2020_dataset_10(num_split=2, batch_size=32, num_workers=1, pin_memory=True):
    """
    加载用于交叉验证的数据集
    :param val_split: 用于验证的 split 文件夹编号（1 到 10）
    :param batch_size: 批处理大小
    :return: 训练和验证的 DataLoader
    """
    assert 1 <= num_split <= 10, "val_split should be between 1 and 10"
    data_path = "E:/dataset/skin01/data_fold"

    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((256, 256)),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),  # 转换为 Tensor
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 准备训练和验证数据路径
    val_path = os.path.join(data_path, f"fold_{num_split}")
    train_path = [os.path.join(data_path, f"fold_{i}") for i in range(1, 11) if i != num_split]

    # 加载训练和验证集
    train_dataset = PAD2020Dataset(data_paths=train_path, transform=train_transform)
    val_dataset = PAD2020Dataset(data_paths=[val_path], transform=val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print("数据加载完毕")
    return train_loader, val_loader



def load_PAD2020_dataset_5(num_split=2, batch_size=32,num_workers=1,pin_memory=True):
    """
    加载用于交叉验证的数据集
    :param val_split: 用于验证的 split 文件夹编号（1 到 5)
    :param batch_size: 批处理大小
    :return: 训练和验证的 DataLoader
    """
    assert 1 <= num_split <= 5, "val_split should be between 1 and 5"
    data_path = "E:/dataset/skin01/data_fold"

    train_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((256, 256)),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),  # 转换为 Tensor
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    val_transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.RandomCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # 准备训练和验证数据路径
    val_path = os.path.join(data_path, f"split{num_split}")
    train_path = [os.path.join(data_path, f"split{i}") for i in range(1, 6) if i != num_split]

    # 加载训练和验证集
    train_dataset = PAD2020Dataset(data_paths=train_path, transform=train_transform)
    val_dataset = PAD2020Dataset(data_paths=[val_path], transform=val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    print("数据加载完毕")
    return train_loader, val_loader





# if __name__ == "__main__":
#     train_loader, val_loader = load_skin01_crossval_dataset()
