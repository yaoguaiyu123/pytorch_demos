import torch.optim as optim
from typing import Any
from dataload import (load_isic2017_dataset, load_isic2018_dataset, load_PAD2020_dataset_10, load_isic2019_dataset_10,
                      load_isic2019_dataset_5, load_imageNet100, load_imageNet1000)
from timm import create_model
from torchvision import models

from my_model import rdm_3
import torch.cuda
import multiprocessing
from math import ceil
import torch
import torch.nn as nn
import timm

Mode = 1 # 0, 1, 2
NUM_GPU = torch.cuda.device_count()
if Mode == 0:
    CUDA_AVAI = False
    DataParallel = False
    Device = torch.device('cpu')
elif Mode == 1:
    # GPU
    CUDA_AVAI = torch.cuda.is_available()
    DataParallel = NUM_GPU > 1
    Device = torch.device('cuda:0' if CUDA_AVAI else 'cpu')


# note 用于配置和优化多进程加载数据时的工作线程数量和内存管理策略
def workerManager(batch_size):
    NUM_PROC = multiprocessing.cpu_count()
    if CUDA_AVAI:
        if batch_size <= 32:
            WorkerKanban = batch_size // 4
        elif batch_size <= 64:
            WorkerKanban = batch_size // 8
        else:
            WorkerKanban = batch_size // 16
        NumWorkers = ceil(min(WorkerKanban, NUM_PROC - 2) / 2.) * 2
        PinMemory = True
    else:
        NumWorkers = ceil(min(4 * round(NUM_PROC / 8), NUM_PROC - 2) / 2.) * 2
        PinMemory = False
    print("We use [%d/%d] workers, and pin memory is %s" %(NumWorkers, NUM_PROC, str(PinMemory)))
    return PinMemory, NumWorkers

# note 控制学习率的类
class CustomCosineScheduler(object):
    def __init__(
            self,
            optimizer,
            milestones=30,
            maxEpochs=120,
            minLrRate=2e-4,
            **kwargs: Any,
    ) -> None:
        super(CustomCosineScheduler, self).__init__()
        self.EpochTemp = 0
        self.Optimizer = optimizer
        self.Milestones = milestones

        self.MinLrRate = minLrRate
        self.MaxLrRate = minLrRate * 10

        self.WarmupEpoches = max(milestones // 2, 0)
        if self.WarmupEpoches > 0:
            self.WarmupLrRate = minLrRate
            self.warmup_step = (self.MaxLrRate - self.WarmupLrRate) / self.WarmupEpoches

        self.Period = milestones
        self.MaxEpochs = maxEpochs

    def get_lr(self, Epoch: int) -> float:
        if Epoch == self.Milestones:
            self.EpochTemp = Epoch - 1
            self.WarmupEpoches += Epoch
            self.Period = self.MaxEpochs

        if Epoch < self.WarmupEpoches:
            Currlr = self.WarmupLrRate + (Epoch - self.EpochTemp) * self.warmup_step
        elif Epoch + 1 < self.Period:
            Currlr = self.MinLrRate + 0.5 * (self.MaxLrRate - self.MinLrRate) * (
                        1 + math.cos(math.pi * Epoch / self.Period))
        else:
            Currlr = self.MinLrRate
        return max(0.0, Currlr)

    def step(self, Epoch: int):
        Values = self.get_lr(Epoch)
        self.Optimizer.param_groups[0]['lr'] = Values

        self._last_lr = [group['lr'] for group in self.Optimizer.param_groups]



# note 控制学习率的类
import math
from typing import Any

class CustomLearningRateScheduler:
    def __init__(
        self,
        optimizer,
        initial_lr: float,
        min_lr: float = 8e-4,
        warmup_epochs: int = 5,
        cosine_epochs_1: int = 25,
        cosine_epochs_2: int = 80,
    ) -> None:

        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs_1 = cosine_epochs_1
        self.cosine_epochs_2 = cosine_epochs_2

        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.warmup_lr_start = initial_lr * 0.1

        self.cos2_start_lr = min_lr / 5
        self.cos2_min_lr = min_lr / 80  # note  最小的学习率

        self.total_epochs = warmup_epochs + cosine_epochs_1 + cosine_epochs_2

        # Calculate warmup increment per epoch
        if warmup_epochs > 0:
            self.warmup_step = (initial_lr - self.warmup_lr_start) / warmup_epochs

    def get_lr(self, epoch: int) -> float:

        # 预热
        if epoch < self.warmup_epochs:
            current_lr = self.warmup_lr_start + epoch * self.warmup_step
        # cos1
        elif epoch < self.warmup_epochs + self.cosine_epochs_1:
            phase_epoch = epoch - self.warmup_epochs
            current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                1 + math.cos(math.pi * phase_epoch / self.cosine_epochs_1)
            )
        # cos2
        elif epoch < self.total_epochs:
            phase_epoch = epoch - (self.warmup_epochs + self.cosine_epochs_1)
            current_lr = self.cos2_min_lr + 0.5 * (self.cos2_start_lr - self.cos2_min_lr) * (
                1 + math.cos(math.pi * phase_epoch / self.cosine_epochs_2)
            )
        # 剩余阶段
        else:
            current_lr = self.min_lr

        return max(0.0, current_lr)

    def step(self, epoch: int):
        current_lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]


class ThreeStageScheduler:
    def __init__(
            self,
            optimizer,
            initial_lr: float,
            min_lr: float = 1e-6,
            warmup_epochs: int = 5,
            cosine_epochs: int = 125,
            linear_epochs: int = 20,
    ) -> None:
        """
        三阶段学习率调度器：预热->余弦衰减->线性衰减
        Args:
            optimizer: 优化器
            initial_lr: 初始学习率
            min_lr: 最小学习率
            warmup_epochs: 预热轮数
            cosine_epochs: 余弦衰减轮数
            linear_epochs: 平稳轮数
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.cosine_epochs = cosine_epochs
        self.linear_epochs = linear_epochs

        self.min_lr = min_lr
        self.initial_lr = initial_lr
        self.warmup_lr_start = initial_lr * 0.1

        self.total_epochs = warmup_epochs + cosine_epochs + linear_epochs

        # 计算预热阶段每个epoch的学习率增量
        if warmup_epochs > 0:
            self.warmup_step = (initial_lr - self.warmup_lr_start) / warmup_epochs


    def get_lr(self, epoch: int) -> float:
        # 预热阶段：线性增加
        if epoch < self.warmup_epochs:
            current_lr = self.warmup_lr_start + epoch * self.warmup_step

        # 余弦衰减阶段
        elif epoch < self.warmup_epochs + self.cosine_epochs:
            phase_epoch = epoch - self.warmup_epochs
            current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
                    1 + math.cos(math.pi * phase_epoch / self.cosine_epochs)
            )

        # 平稳阶段
        elif epoch < self.total_epochs:
            current_lr = self.min_lr
        else:
            current_lr = self.min_lr

        return max(0.0, current_lr)

    def step(self, epoch: int):
        """更新优化器的学习率"""
        current_lr = self.get_lr(epoch)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr

        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    def get_last_lr(self):
        """返回最后一次设置的学习率"""
        return self._last_lr




# note 优化器选择函数
def optimizerChoice(NetParam, lr, Choice='Adam', **kwargs: Any):
    # OptimChoices = ['Adam', 'AdamW', 'Adamax', 'SparseAdam', 'SGD', 'ASGD',
    #                'RMSprop', 'Rprop', 'LBFGS', 'Adadelta', 'Adagrad']

    CallDict = {
        'adam': optim.Adam,
        'adamw': optim.AdamW,
        'adamax': optim.Adamax,
        'sparseadam': optim.SparseAdam,
        'sgd': optim.SGD,
        'asgd': optim.ASGD,
        'rmsprop': optim.RMSprop,
        'rprop': optim.Rprop,
        'lbfgs': optim.LBFGS,
        'adadelta': optim.Adadelta,
        'adagrad': optim.Adagrad,
    }

    Optimizer = CallDict[Choice](NetParam, lr=lr, **kwargs)

    return Optimizer


# note 选择模型的函数
def choose_model(data_choice,model_choice,model_options,pretrained_path = None):
    if model_choice in model_options:
        name, model_code, model_weights_path = model_options[model_choice]
        # begin 官方的模型
        if len(model_choice) >= 3:
            model = None
            # note 数据集检查
            num_classes = None
            if data_choice == '3':
                num_classes = 8
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_ISIC2019.{extension}"
            elif data_choice == '4':
                num_classes = 7
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_ISIC2018.{extension}"
            elif data_choice == '1':
                num_classes = 6
            elif data_choice == '5':
                num_classes = 3
            else:
                return None, None

            # note 选择不同的官方模型
            if model_code == "efficientnetV1_b1":
                model = models.efficientnet_b1(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(in_features, num_classes)
                )
            elif model_code == "regnet_x_800mf":
                model = models.regnet_x_800mf(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif model_code == "regnet_y_800mf":
                model = models.regnet_y_800mf(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilenetv4_conv_medium":
                model = create_model('mobilenetv4_conv_medium', pretrained=False, num_classes=num_classes)
            elif model_code == "regnet_y_1_6gf":
                model = models.regnet_y_1_6gf(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilevitv2_100":
                model = create_model('mobilevitv2_100', pretrained=False)
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilevitv2_075":
                model = create_model('mobilevitv2_075', pretrained=False)
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilevitv2_050":
                model = create_model('mobilevitv2_050', pretrained=False)
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            elif model_code == "efficientnetV2_b0":
                model = create_model('tf_efficientnetv2_b1', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif model_code == "efficientnetV2_b1":
                model = create_model('tf_efficientnetv2_b1', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif model_code == "efficientnetV2_b2":
                model = create_model('tf_efficientnetv2_b2', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif model_code == "mobilenetv4_conv_small":
                model = create_model('mobilenetv4_conv_small', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)

            else:
                raise ValueError(f"不支持的模型代码：{model_code}")


            # note 调整模型权重路径
            # model = load_library_weight(model, model_code)
            return model, model_weights_path

        # begin 自己的模型
        else:
            if data_choice =='1':
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_PAD2020.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=6")  # 6分类
            elif data_choice =='2':
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_imageNet1k.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=1000") # 500分类
            elif data_choice =='3':
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_ISIC2019.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=8")  # 8分类
            elif data_choice =='4':
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_ISIC2018.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=7")  # 7分类

            print(f"已选择 {model_code}")
            print(f"将使用 {model_weights_path} 作为权重保存的路径")
            model = eval(model_code)
            # sp 加载预训练权重并冻结
            if pretrained_path is not None:
                model = load_local_weight(model, weight_path=pretrained_path)
            return model, model_weights_path

    return None


# note 选择模型预训练的函数
def choose_model_to_pretrain(data_choice, model_choice, model_options):
    if model_choice in model_options:
        name, model_code, pretrained_path = model_options[model_choice]
        # note 官方提供的模型
        if len(model_choice) >= 3:
            model = eval(model_code)  # sp 获取 model
            if data_choice =='1':
                in_features = model.classifier[3].in_features
                num_classes = 1000     # sp imagenet1k_s 1000分类
                model.classifier[3] = nn.Linear(in_features, num_classes)
            elif data_choice == '2':
                in_features = model.classifier[3].in_features
                num_classes = 100   # sp ImageNet100 100分类
                model.classifier[3] = nn.Linear(in_features, num_classes)
            else:
                print("------------未知的data_choice选项------------")
            print(f"已选择 {model_code}")
            return model, None
        else:
            # note 自己的模型
            if data_choice =='1':
                name_part, extension = pretrained_path.rsplit('.', 1)
                pretrained_path = f"{name_part}_imageNet1k_s.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=1000")  # 1000分类
            elif data_choice =='2':
                name_part, extension = pretrained_path.rsplit('.', 1)
                pretrained_path = f"{name_part}_imageNet100.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=100") # 100分类
            elif data_choice =='3':
                name_part, extension = pretrained_path.rsplit('.', 1)
                pretrained_path = f"{name_part}_imageNet1000.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=1000") # 1000分类
            print(f"已选择 {model_code}")
            print(f"将使用 {pretrained_path} 作为权重保存的路径")
            return eval(model_code), pretrained_path
    else:
        print("选择 model 失败，请重新输入有效的数字")
        return None,None


# note 选择验证模型的函数
def choose_val_model(model_choice,model_options, data_choice, weight_path):
    if model_choice in model_options:
        name, model_code, model_weights_path = model_options[model_choice]
        # begin 官方的模型
        if len(model_choice) >= 3:
            model = None
            # note 数据集检查
            num_classes = None
            if data_choice == '3':
                num_classes = 8
            elif data_choice == '4':
                num_classes = 7
            else:
                return None, None

            # note 选择不同的官方模型
            if model_code == "efficientnetV1_b1":
                model = models.efficientnet_b1(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2, inplace=True),
                    nn.Linear(in_features, num_classes)
                )
            elif model_code == "regnet_x_800mf":
                model = models.regnet_x_800mf(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif model_code == "regnet_y_800mf":
                model = models.regnet_y_800mf(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilenetv4_conv_medium":
                model = create_model('mobilenetv4_conv_medium', pretrained=False, num_classes=num_classes)
            elif model_code == "regnet_y_1_6gf":
                model = models.regnet_y_1_6gf(weights=None)
                in_features = model.fc.in_features
                model.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilevitv2_100":
                model = create_model('mobilevitv2_100', pretrained=False)
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilevitv2_075":
                model = create_model('mobilevitv2_075', pretrained=False)
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            elif model_code == "mobilevitv2_050":
                model = create_model('mobilevitv2_050', pretrained=False)
                in_features = model.head.fc.in_features
                model.head.fc = nn.Linear(in_features, num_classes)
            elif model_code == "efficientnetV2_b0":
                model = create_model('tf_efficientnetv2_b1', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif model_code == "efficientnetV2_b1":
                model = create_model('tf_efficientnetv2_b1', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif model_code == "efficientnetV2_b2":
                model = create_model('tf_efficientnetv2_b2', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            elif model_code == "mobilenetv4_conv_small":
                model = create_model('mobilenetv4_conv_small', pretrained=False)
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)

            # note 加载模型权重
            model = load_local_weight(model, weight_path=weight_path, freeze=False, skip=False)
            return model, None

        # begin 自己的模型
        else:
            if data_choice =='3':
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_ISIC2019.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=8")  # 8分类
            elif data_choice =='4':
                name_part, extension = model_weights_path.rsplit('.', 1)
                model_weights_path = f"{name_part}_ISIC2018.{extension}"
                model_code = model_code.replace("num_classes=200", "num_classes=7")  # 7分类

            print(f"已选择 {model_code}")
            print(f"将使用 {model_weights_path} 作为权重保存的路径")
            model = eval(model_code)
            model = load_local_weight(model, weight_path=weight_path, freeze=False, skip=False)
            return model, None

    return None


# note 选择数据集的函数
def choose_dataset(choice='1', split=1):

    if choice == '1':
        print("加载 PAD2020 数据集...")
        pinMemory, numWorkers = workerManager(batch_size=16)
        train_loader, val_loader = load_PAD2020_dataset_10(num_split=split, batch_size=16, num_workers=numWorkers,pin_memory=pinMemory)
    elif choice == '2':
        print("加载 ImageNet1k 数据集...")
        train_loader, val_loader = load_imageNet1000(batch_size=64)
    elif choice == '3':
        print("加载 ISIC2019 数据集...")
        # train_loader, val_loader = load_isic2019_dataset_5(num_split=split, batch_size=16, num_workers=8,pin_memory=True)
        train_loader, val_loader = load_isic2019_dataset_10(num_split=2, batch_size=16, num_workers=8,pin_memory=True)
    elif choice == '4':
        print("加载 ISIC2018 数据集...")
        train_loader, val_loader = load_isic2018_dataset(batch_size=16, num_workers=8,pin_memory=True)
    elif choice == '5':
        print("加载 ISIC2017 数据集...")
        train_loader, val_loader = load_isic2017_dataset(batch_size=16, num_workers=8, pin_memory=True)
    else:
        print("输入有误，默认加载ISIC2019数据集")
        pinMemory, numWorkers = workerManager(batch_size=64)
        train_loader, val_loader = load_isic2019_dataset_5(num_split=split, batch_size=64, num_workers=8,pin_memory=pinMemory)
    return train_loader, val_loader


# note 选择预训练数据集的函数
def choose_pretrain_dataset(choice='1'):

    if choice == '1':
        print("加载 ImageNet100 数据集...")
        train_loader, val_loader = load_imageNet100(batch_size=128, pin_memory = True, num_workers=8)
    elif choice == '2':
        print(Fore.GREEN + "加载 ImageNet1000 数据集..." + Style.RESET_ALL)
        train_loader, val_loader = load_imageNet1000(batch_size=64, pin_memory = True, num_workers=8)
    else:
        print("输入有误，默认加载 ImageNet100 数据集...")
        train_loader, val_loader = load_imageNet100(batch_size=128, pin_memory = True, num_workers=8)
    return train_loader, val_loader

import os
# note 加载第三方库提供的预训练权重
def load_library_weight(model, model_code):
    print(f'\033[1;32m加载官方提供的权重: {model_code}\033[0m')

    # 根据模型代码，加载对应的预训练权重路径
    weight_path_map = {
        "efficientnetV1_b1": "../tools/download_weights/efficientnet_b1_weights.pth",
        "regnet_x_800mf": "../tools/download_weights/regnet_x_800mf_weights.pth",
        "regnet_y_800mf": r"E:/python_demos/super_demo01_2/tools/download_weights/regnet_y_800mf_weights.pth",
        "mobilenetv4_conv_medium": "E:/python_demos/super_demo01_2/tools/download_weights/mobilenetv4_conv_medium_weights.pth",
        "mobilenetv4_conv_small": "E:/python_demos/super_demo01_2/tools/download_weights/mobilenetv4_conv_small_weights.pth",
        "regnet_y_1_6gf":"E:/python_demos/super_demo01_2/tools/download_weights/regnet_y_1_6g_weights.pth",
        "mobilevitv2_100": "E:/python_demos/super_demo01_2/tools/download_weights/mobilevitv2_100_weights.pth",
        "mobilevitv2_075": "E:/python_demos/super_demo01_2/tools/download_weights/mobilevitv2_075_weights.pth",
        "mobilevitv2_050": "E:/python_demos/super_demo01_2/tools/download_weights/mobilevitv2_050_weights.pth",
        "efficientnetV2_b0": "E:/python_demos/super_demo01_2/tools/download_weights/tf_efficientnetV2_b0_weights.pth",
        "efficientnetV2_b1": "E:/python_demos/super_demo01_2/tools/download_weights/tf_efficientnetV2_b1_weights.pth",
        "efficientnetV2_b2": "E:/python_demos/super_demo01_2/tools/download_weights/tf_efficientnetV2_b2_weights.pth",
    }

    if model_code not in weight_path_map:
        print(f"\033[1;33m未找到 {model_code} 的预训练权重路径\033[0m")
        return model

    weight_path = weight_path_map[model_code]

    # 检查路径是否存在
    if not os.path.exists(weight_path):
        print(f"\033[1;31m在提供的权重路径: {weight_path}没有找到对应文件\033[0m")
        return model

    try:
        # 加载预训练权重
        pretrained_dict = torch.load(weight_path, map_location='cpu')
        model_dict = model.state_dict()

        # 要跳过的层和参数
        skip_layers = ['classifier', 'fc', 'head']
        ignore_str_list = ["running_mean", "running_var", "num_batches_tracked"]

        filtered_dict = {
            k: v for k, v in pretrained_dict.items()
            if not any(layer in k for layer in skip_layers)
               and not any(ignore_str in k for ignore_str in ignore_str_list)
        }

        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"\033[1;32m成功加载 {len(filtered_dict)} 层参数到 {model_code}\033[0m")

    except Exception as e:
        print(f"\033[1;31m加载预训练权重失败: {e}\033[0m")

    return model


# note 加载本地的预训练权重
def load_local_weight(model, weight_path, freeze=True, skip=True):

    print(f'\033[1;32m加载本地权重的路径: {weight_path}\033[0m')

    if weight_path is None:
        return model

    pretrained_dict = torch.load(weight_path, map_location=torch.device('cpu'))
    model_dict = model.state_dict()

    if skip:
        print("不加载指定的层")
        skip_layers = ['classifier', 'fc', 'head', 'linear4']
        ignore_str_list = ["running_mean", "running_var", "num_batches_tracked"]
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if not any(layer in k for layer in skip_layers)
               and not any(ignore_str in k for ignore_str in ignore_str_list)
        }
        model_dict.update(pretrained_dict)
    else:
        print("加载所有层")
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
        }
        model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict, strict=False)

    # 冻结权重逻辑
    if freeze:
        freeze_classifier = weight_path is not None
        model = weightFrozen(model, freeze_classifier)

    return model




# note 冻结或者解冻的函数
from colorama import Fore, Style, init
init(autoreset=True)

def weightFrozen(Model, freezeClassifier=True):
    if freezeClassifier:
        # 冻结除了 classifier、fc 和 linear4 的所有层
        for Name, Param in Model.named_parameters():
            if 'classifier' not in Name.lower() and 'fc' not in Name.lower() and 'linear4' not in Name.lower():
                Param.requires_grad = False
            else:
                Param.requires_grad = True
        print(f"{Fore.YELLOW}--------------已冻结除 classifier、fc 和 linear4 外的所有层--------------")
    else:
        # 解冻所有层
        for Name, Param in Model.named_parameters():
            Param.requires_grad = True
        print(f"{Fore.YELLOW}-------------已解冻所有层--------------")

    return Model

