# begin 加载 ISIC2018
#  7分类
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy
from PIL import Image
import os
import torch
from pathlib import Path


class ISIC2017Dataset(Dataset):
    def __init__(self, data_path, transform=None):
        """
        ISIC2017 数据集加载器
        :param data_path: 数据路径 (训练、验证、测试集所在的根目录)
        :param transform: 图像预处理 transform
        """
        self.data_path = data_path
        self.transform = transform
        self.images_path = []  # 存储所有图片路径
        self.images_label = []  # 存储图片对应的标签索引
        supported_formats = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件类型

        class_names = ['MEL', 'NV', 'SEK']
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

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack(images, dim=0)
        labels = torch.tensor(labels)
        return images, labels


def load_isic2017_dataset(batch_size=64, num_workers=4, pin_memory=True):
    """
    加载 ISIC2018 数据集
    :param batch_size: 批处理大小
    :param num_workers: DataLoader 的 worker 数量
    :param pin_memory: 是否使用 pin_memory
    :return: 训练、验证和测试的 DataLoader
    """
    data_path = "E:/dataset/ISIC2017"  # 修改为 ISIC2018 数据集路径


    # 数据增强和预处理的配置
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomCrop((256, 256), padding=(40, 40)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((284, 284)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集路径
    train_path = os.path.join(data_path, "class_data_trainA")
    test_path = os.path.join(data_path, "class_data_test")

    # 数据集加载
    train_dataset = ISIC2017Dataset(data_path=train_path, transform=train_transform)
    test_dataset = ISIC2017Dataset(data_path=test_path, transform=val_test_transform)

    # DataLoader 创建
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    print("数据加载完毕")
    return train_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader = load_isic2017_dataset(batch_size=32)

