from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import cv2
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

ImageFile.LOAD_TRUNCATED_IMAGES = True

def worker_init_fn(worker_id):
    print(f"Worker {worker_id} PID: {os.getpid()}")

class ImageNet1000Loader(Dataset):
    def __init__(self, data_paths: list, transform=None):
        self.data_paths = data_paths
        self.transform = transform

        assert all([os.path.exists(path) for path in data_paths]), "One or more dataset paths do not exist."

        self.images_path = []
        self.images_label = []
        supported_formats = [".jpg", ".JPG", ".png", ".PNG", ".JPEG"]

        self.class_indices = {}

        for data_path in data_paths:
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

        print(f"{len(self.images_path)} images were found in the dataset with {len(self.class_indices)} classes.")

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        img_path = self.images_path[idx]
        label = self.images_label[idx]
        try:
            img = Image.open(img_path)
            if img.mode != 'RGB':
                img = img.convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            raise e

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


def load_imageNet1000(batch_size=128, num_workers=8, pin_memory=True):
    base_path = "D://WBY/dataset"

    train_paths = [os.path.join(base_path, "imagenet1k_train")]
    val_paths = [os.path.join(base_path, "imagenet1k_val")]

    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(
            p=0.25,
            scale=(0.02, 0.33),
            ratio=(0.3, 3.3),
            value='random'
        )
    ])
    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载训练集和验证集
    train_dataset = ImageNet1000Loader(data_paths=train_paths, transform=train_transform)
    val_dataset = ImageNet1000Loader(data_paths=val_paths, transform=val_transform)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,
                               num_workers=6, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=2, pin_memory=True, persistent_workers=True, worker_init_fn=worker_init_fn)

    print("ImageNet1000 data loading complete.")
    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = load_imageNet1000()
