# begin 加载 ImageNet100 数据集
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
# note 防止死锁的方式
import cv2
from torchvision.transforms import AutoAugment, AutoAugmentPolicy


class ImageNet100Loader(Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform

        assert all([os.path.exists(path) for path in data_paths]), "One or more dataset paths do not exist."

        self.images_path = []
        self.images_label = []
        supported_formats = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]

        self.class_indices = {}

        for data_path in data_paths:
            # 获取类别
            categories = [cat for cat in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, cat))]
            categories.sort()

            for cat in categories:
                if cat not in self.class_indices:
                    self.class_indices[cat] = len(self.class_indices)

            for cat in categories:
                cat_path = os.path.join(data_path, cat)
                images = [os.path.join(cat_path, img) for img in os.listdir(cat_path) if
                          os.path.splitext(img)[-1] in supported_formats]
                self.images_path.extend(images)
                self.images_label.extend([self.class_indices[cat]] * len(images))

        print(f"{len(self.images_path)} images were found in the dataset.")

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        img_path = self.images_path[idx]
        label = self.images_label[idx]
        img = Image.open(img_path)

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


def load_imageNet100(batch_size=64,num_workers = 1, pin_memory = False):
    # base_path = "E:/dataset/ImageNet100"
    base_path = "D:/WBY/dataset/ImageNet100"

    train_paths = [base_path + "/train.X1"]
    val_path = [base_path + "/val.X"]

    # note 使用了 AutoAugment
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),  # 使用 ImageNet 数据增强策略
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(        # 随机擦除
            p=0.25,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    train_dataset = ImageNet100Loader(data_paths=train_paths, transform=train_transform)
    val_dataset = ImageNet100Loader(data_paths=val_path, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    print("Data loading complete.")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = load_imageNet100()
