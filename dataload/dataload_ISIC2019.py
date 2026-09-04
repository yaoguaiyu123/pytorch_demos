import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torchvision import transforms
from torchvision.transforms import AutoAugment, AutoAugmentPolicy

class ISIC2019Dataset(Dataset):
    def __init__(self, data_paths, transform=None):
        self.data_paths = data_paths
        self.transform = transform

        self.images_path = []  # 存储所有图片路径
        self.images_label = []  # 存储图片对应的标签索引
        supported_formats = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件类型

        # 类别名称及其对应的标签索引
        class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
        # {
        #     'ACK': 0,
        #     'BCC': 1,
        #     'BKL': 2,
        #     'DF': 3,
        #     'MEL': 4,
        #     'NV': 5,
        #     'SCC': 6,
        #     'VASC': 7
        # }
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


def load_isic2019_dataset_10(num_split=2, batch_size=64, num_workers=4, pin_memory=True):
    """
    加载 ISIC 2019 数据集，用于交叉验证
    :param num_split: 用于验证的 split 文件夹编号（1 到 10）
    :param batch_size: 批处理大小
    :return: 训练和验证的 DataLoad
    """
    assert 1 <= num_split <= 10, "num_split should be between 1 and 10"
    # data_path = "D:/WBY/dataset/ISIC2019/10split_data"

    data_path = "E:/dataset/ISIC2019/10split_data_au1"

    # train_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.RandomHorizontalFlip(),  # 水平翻转
    #     AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
    #     transforms.ToTensor(),  # 转换为 Tensor
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #
    # val_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])

    train_transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.RandomCrop((256, 256), padding=(45, 30)),
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        AutoAugment(policy=AutoAugmentPolicy.IMAGENET),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((336, 336)),
        transforms.CenterCrop((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 准备训练和验证数据路径
    val_path = os.path.join(data_path, f"split_{num_split}")
    train_path = [os.path.join(data_path, f"split_{i}") for i in range(1, 11) if i != num_split]

    # 加载训练和验证集
    train_dataset = ISIC2019Dataset(data_paths=train_path, transform=train_transform)
    val_dataset = ISIC2019Dataset(data_paths=[val_path], transform=val_transform)

    # 创建 DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    print("数据加载完毕")
    return train_loader, val_loader


# note 五折交叉验证
def load_isic2019_dataset_5(num_split=1, batch_size=32, num_workers=4, pin_memory=True):
    """
    加载 ISIC 2019 数据集，用于交叉验证
    :param num_split: 用于验证的 split 文件夹编号（1 到 5）
    :param batch_size: 批处理大小
    :return: 训练和验证的 DataLoad
    """
    assert 1 <= num_split <= 5, "num_split should be between 1 and 5"
    # data_path = "D:/WBY/dataset/ISIC2019/5split_data"

    data_path = "E:/dataset/ISIC2019/5split_data_OLD"

    # note 标准归一化
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10, expand=False),
        transforms.RandomHorizontalFlip(),  # 水平翻转
        transforms.ToTensor(),  # 转换为 Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])

    # note 适配ISIC2019的归一化
    # train_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.RandomRotation(10, expand=False),
    #     transforms.RandomHorizontalFlip(),  # 水平翻转
    #     transforms.ToTensor(),  # 转换为 Tensor
    #     transforms.Normalize(mean=[0.7481, 0.2223, 0.4616], std=[0.9403, 0.8514, 0.8798])  # 使用训练集的归一化参数
    # ])
    #
    # val_transform = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.8438, 0.3051, 0.5409], std=[0.7399, 0.7435, 0.7748])  # 使用验证集的归一化参数
    # ])

    # 准备训练和验证数据路径
    val_path = os.path.join(data_path, f"split_{num_split}")
    train_path = [os.path.join(data_path, f"split_{i}") for i in range(1, 6) if i != num_split]

    # 加载训练和验证集
    train_dataset = ISIC2019Dataset(data_paths=train_path, transform=train_transform)
    val_dataset = ISIC2019Dataset(data_paths=[val_path], transform=val_transform)

    # 创建 DataLoader
    # mod 修改 num_workers 为0
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory, persistent_workers=True)

    print("数据加载完毕")
    return train_loader, val_loader


def calculate_normalization(dataset):
    """
    计算数据集中每个通道的均值和标准差
    :param dataset: 数据集对象
    :return: 每个通道的均值和标准差
    """
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    mean = 0.0
    std = 0.0
    total_samples = 0

    for images, _ in dataloader:
        # 计算当前批次的样本数
        batch_samples = images.size(0)
        total_samples += batch_samples

        # 将每张图片的像素值展平到二维，再计算均值和标准差
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)

    mean /= total_samples
    std /= total_samples

    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    # 加载 ISIC 2019 数据集
    train_loader, val_loader = load_isic2019_dataset_10(num_split=2, batch_size=32)

    # 计算训练数据集的归一化系数
    train_dataset = train_loader.dataset  # 获取训练集对象
    print("开始计算训练集归一化系数...")
    train_mean, train_std = calculate_normalization(train_dataset)
    print(f"训练集的均值: {train_mean}")
    print(f"训练集的标准差: {train_std}")

    # 计算验证数据集的归一化系数
    val_dataset = val_loader.dataset  # 获取验证集对象
    print("开始计算验证集归一化系数...")
    val_mean, val_std = calculate_normalization(val_dataset)
    print(f"验证集的均值: {val_mean}")
    print(f"验证集的标准差: {val_std}")
