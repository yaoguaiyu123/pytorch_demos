import torch.nn as nn
import numpy as np
from pytorch_toolbelt.losses import CrossEntropyFocalLoss
from utils.loss_func import ClassBalancedFocalLoss,SeesawCeLoss
import time
import torch
from timm.loss import SoftTargetCrossEntropy
from tqdm import tqdm
import copy
import torch.nn.functional as F
from timm.data import Mixup
import pandas as pd

mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    prob=1.0,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=100,
)


class SoftTargetFocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(SoftTargetFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):

        BCE_loss = F.log_softmax(inputs, dim=-1)
        loss = -targets * BCE_loss
        loss = loss.sum(dim=-1)
        p_t = torch.exp(-loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * loss

        return focal_loss.mean()


# begin NormalCnnTrainer
class CnnTrainer(object):
    def __init__(self, Device, NumClasses, Optim, Model, TrainDL, TestDL, loss_mode=0) -> None:

        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0

        TruePositive, FalsePositive, TrueNegative, FalseNegative = np.zeros((4, NumClasses), dtype=int)
        Recall, Precision, Specificity, F1Score = np.zeros((4, NumClasses), dtype=float)

        # 根据模式选择损失函数
        num_classes = 8
        samples_per_class = torch.tensor([260, 1000, 1000, 290, 2600, 2600, 450, 270], dtype=torch.float32).to(Device)
        cum_samples = torch.zeros(num_classes, dtype=torch.float32).to(Device)

        if loss_mode == 0:
            loss_func = nn.CrossEntropyLoss().to(Device)
        elif loss_mode == 1:
            loss_func = CrossEntropyFocalLoss().to(Device)
        elif loss_mode == 2:
            loss_func = ClassBalancedFocalLoss(num_classes, samples_per_class).to(Device)
        elif loss_mode == 3:
            loss_func = SeesawCeLoss(num_classes=num_classes, cum_samples=cum_samples, p=0.8, q=2.0).to(Device)
        else:
            loss_func = nn.CrossEntropyLoss().to(Device)

        # 将模型切换到训练模式
        Model.train()
        for x, y in tqdm(TrainDL, ncols=60, colour='blue'):
            Optim.zero_grad()

            x, y = x.to(Device), y.to(Device)
            y = torch.squeeze(y)

            YPred = Model(x)

            # 使用 SeesawLoss 时更新累计样本数量
            if loss_mode == 4:
                cum_samples = self.update_cumulative_samples(y, cum_samples)

            Loss = loss_func(YPred, y)

            Loss.backward()
            Optim.step()

            with torch.no_grad():
                YPred_class = torch.argmax(YPred, dim=1)
                NumCorrect += (YPred_class == y).sum().item()
                NumTotal += y.size(0)
                RunningLoss += Loss.item()

        with torch.no_grad():
            TrainLoss = RunningLoss / NumTotal
            TrainAcc = NumCorrect / NumTotal

        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0


        # begin ------------------模型的验证------------------
        # note 重新参数化
        re_model = Model
        if hasattr(Model, 'reparameterize_model'):
            re_model = Model.reparameterize_model()
            re_model.to(Device)

        re_model.eval()
        with torch.no_grad():
            for x, y in tqdm(TestDL, ncols=60, colour='blue'):
                x, y = x.to(Device), y.to(Device)
                y = torch.squeeze(y)

                YPred = re_model(x)
                Loss = loss_func(YPred, y)

                YPred_class = torch.argmax(YPred, dim=1)
                NumCorrect += (YPred_class == y).sum().item()
                BatchSize = y.size(0)
                NumTotal += BatchSize
                RunningLoss += Loss.item()

                for i in range(BatchSize):
                    for k in range(NumClasses):
                        if y[i].item() == k:
                            if YPred_class[i] == y[i]:
                                TruePositive[k] += 1
                            else:
                                FalseNegative[k] += 1
                        else:
                            if YPred_class[i].item() == k:
                                FalsePositive[k] += 1
                            else:
                                TrueNegative[k] += 1

        TestLoss = RunningLoss / NumTotal
        TestAcc = NumCorrect / NumTotal

        for k in range(NumClasses):
            PositiveAll = TruePositive[k] + FalseNegative[k]
            TPAndFP = TruePositive[k] + FalsePositive[k]
            NegativeAll = TrueNegative[k] + FalsePositive[k]
            if PositiveAll != 0:
                Recall[k] = TruePositive[k] / PositiveAll
            if TPAndFP != 0:
                Precision[k] = TruePositive[k] / TPAndFP
            if (Recall[k] + Precision[k]) != 0:
                F1Score[k] = 2 * Recall[k] * Precision[k] / (Recall[k] + Precision[k])
            if NegativeAll != 0:
                Specificity[k] = TrueNegative[k] / NegativeAll

        self.TrainLoss = TrainLoss
        self.TrainAcc = TrainAcc
        self.TestLoss = TestLoss
        self.TestAcc = TestAcc
        self.Recall = Recall
        self.Precision = Precision
        self.F1Score = F1Score
        self.Specificity = Specificity

    def update_cumulative_samples(self, labels, cum_samples):
        """更新累计样本数"""
        unique_labels, counts = labels.unique(return_counts=True)
        cum_samples[unique_labels] += counts
        return cum_samples


# begin CnnTrainerPro
class ModelEMA:
    def __init__(self, model, decay):
        self.ema_model = copy.deepcopy(model)
        self.decay = decay
        self.ema_model.eval()  # 设置 EMA 模型为评估模式

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            esd = self.ema_model.state_dict()
            for k in esd.keys():
                esd[k].copy_(esd[k] * self.decay + msd[k] * (1. - self.decay))

from sklearn.metrics import roc_auc_score
# begin 选择性进行 Rep 和 EMA 的类
class CnnTrainerProWithOptionalEMA(object):
    def __init__(self,
        device,
        num_classes,
        optim,
        model,
        train_dataloader,
        test_dataloader,
        rep=False,
        use_mixup_cutmix=True,
        use_ema=False,
        ema_decay=0.9999) -> None:

        # Mixup 和 CutMix 数据增强初始化
        mixup_fn = None
        if use_mixup_cutmix:
            mixup_fn = Mixup(
                mixup_alpha=0.6,
                cutmix_alpha=1,
                prob=1,  # mod 0.8 -> 1
                switch_prob=0,  # mod 修改为0
                mode='batch',
                label_smoothing=0.1,
                num_classes=num_classes
            )

        # 初始化 EMA 模型
        ema_model = None
        if use_ema:
            ema_model = ModelEMA(model, decay=ema_decay)

        # 模型训练和测试相关变量初始化
        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0

        # 初始化测试集的混淆矩阵
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

        # 初始化其他统计指标
        TruePositive, FalsePositive, TrueNegative, FalseNegative = np.zeros((4, num_classes), dtype=int)
        Recall, Precision, Specificity, F1Score = np.zeros((4, num_classes), dtype=float)

        # note 支持软标签的交叉熵损失函数
        # loss_func = SoftTargetCrossEntropy().to(device)
        loss_func = SoftTargetFocalLoss().to(device)

        # ------------------ 模型训练 ------------------
        model.train()
        for x, y in tqdm(train_dataloader, desc="Train", ncols=60, colour='blue'):
            optim.zero_grad()
            x = x.to(device)
            y = y.to(device)

            # 若批次大小为奇数，丢弃最后一个样本
            if x.size(0) % 2 != 0:
                x = x[:-1]
                y = y[:-1]

            # 应用 Mixup / CutMix
            if mixup_fn is not None:
                x, y = mixup_fn(x, y)
            else:
                # 若不使用 Mixup, 需将y转为 one-hot
                y = nn.functional.one_hot(y, num_classes=num_classes).float()

            YPred = model(x)
            Loss = loss_func(YPred, y)

            Loss.backward()
            optim.step()

            # 更新 EMA 模型
            if use_ema and ema_model is not None:
                ema_model.update(model)

            with torch.no_grad():
                YPred_class = torch.argmax(YPred, dim=1)
                y_class = torch.argmax(y, dim=1)
                NumCorrect += (YPred_class == y_class).sum().item()
                NumTotal += y.size(0)
                RunningLoss += Loss.item()

        with torch.no_grad():
            TrainLoss = RunningLoss / len(train_dataloader)
            TrainAcc = NumCorrect / NumTotal

        # ------------------ 模型验证 ------------------
        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0

        # 为计算 AUC 存储预测概率和真实标签
        all_test_probs = []
        all_test_labels = []

        if rep and hasattr(model, 'reparameterize_model'):
            re_model = model.reparameterize_model()  # 结构重参数化
        else:
            re_model = model

        re_model.to(device)
        re_model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_dataloader, desc="Val ", ncols=60, colour='cyan'):
                x, y = x.to(device), y.to(device)
                y = torch.squeeze(y)

                # 若批次大小为奇数，丢弃最后一个样本
                if x.size(0) % 2 != 0:
                    x = x[:-1]
                    y = y[:-1]

                # 将标签也转为 one-hot(只用于计算 loss); 但多分类 AUC 更常用原label
                y_one_hot = nn.functional.one_hot(y, num_classes=num_classes).float()

                def apply_softmax(tensor):
                    # 检查输入是否为 Tensor
                    if not isinstance(tensor, torch.Tensor):
                        raise ValueError("输入必须是一个 PyTorch Tensor。")

                    # 应用 Softmax（通常在最后一个维度进行归一化）
                    softmax_tensor = F.softmax(tensor, dim=-1)

                    return softmax_tensor

                YPred = re_model(x)
                YPred01 = apply_softmax(YPred)
                Loss = loss_func(YPred, y_one_hot)
                RunningLoss += Loss.item()

                # 预测类别
                YPred_class = torch.argmax(YPred, dim=1)
                NumCorrect += (YPred_class == y).sum().item()
                NumTotal += y.size(0)

                # 更新混淆矩阵
                for t, p in zip(y.cpu().numpy(), YPred_class.cpu().numpy()):
                    confusion_matrix[t, p] += 1

                # 计算真/假正例和真/假反例
                BatchSize = y.size(0)
                for i in range(BatchSize):
                    for k in range(num_classes):
                        if y[i].item() == k:
                            if YPred_class[i] == k:
                                TruePositive[k] += 1
                            else:
                                FalseNegative[k] += 1
                        else:
                            if YPred_class[i].item() == k:
                                FalsePositive[k] += 1
                            else:
                                TrueNegative[k] += 1

                # 收集真实标签(y)和模型输出的预测概率(YPred) 用于计算多分类 AUC
                all_test_labels.extend(y.detach().cpu().numpy())
                all_test_probs.extend(YPred01.detach().cpu().numpy())

        TestLoss = RunningLoss / len(test_dataloader)
        TestAcc = NumCorrect / NumTotal

        # 计算多分类 AUC (One-vs-Rest)
        # 注意：all_test_labels 为 shape (N, )
        #       all_test_probs  为 shape (N, num_classes)
        try:
            AUC = roc_auc_score(
                all_test_labels,
                all_test_probs,
                multi_class='ovr',
                average='macro'  # 你也可选择 'weighted'/'micro'/'macro'
            )
        except ValueError:
            print("all_test_labels 的大小: ", len(all_test_labels))
            print("all_test_probs 的大小: ", len(all_test_probs))
            print(all_test_probs)
            AUC = float('nan')

        # 计算其他指标: 召回率, 精确率, F1分数, 特异性
        for k in range(num_classes):
            PositiveAll = TruePositive[k] + FalseNegative[k]
            TPAndFP = TruePositive[k] + FalsePositive[k]
            NegativeAll = TrueNegative[k] + FalsePositive[k]
            if PositiveAll != 0:
                Recall[k] = TruePositive[k] / PositiveAll
            if TPAndFP != 0:
                Precision[k] = TruePositive[k] / TPAndFP
            if (Recall[k] + Precision[k]) != 0:
                F1Score[k] = 2 * Recall[k] * Precision[k] / (Recall[k] + Precision[k])
            if NegativeAll != 0:
                Specificity[k] = TrueNegative[k] / NegativeAll

        # 保存训练和测试结果
        self.TrainLoss = TrainLoss
        self.TrainAcc = TrainAcc
        self.TestLoss = TestLoss
        self.TestAcc = TestAcc
        self.Recall = Recall
        self.Precision = Precision
        self.F1Score = F1Score
        self.Specificity = Specificity
        self.confusion_matrix = confusion_matrix
        self.AUC = AUC  # 新增：多分类 AUC

        # ------------------ EMA 模型验证 ------------------
        if use_ema and ema_model is not None:
            ema_NumCorrect = 0
            ema_NumTotal = 0
            ema_RunningLoss = 0

            # 初始化 EMA 模型的混淆矩阵
            ema_confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)

            # 初始化 EMA 模型的统计指标
            ema_TruePositive, ema_FalsePositive, ema_TrueNegative, ema_FalseNegative = np.zeros((4, num_classes),
                                                                                                dtype=int)
            ema_Recall, ema_Precision, ema_Specificity, ema_F1Score = np.zeros((4, num_classes), dtype=float)

            # 也收集 EMA 输出的预测概率，用于计算 AUC
            ema_all_test_probs = []
            ema_all_test_labels = []

            ema_model.ema_model.to(device)
            ema_model.ema_model.eval()
            with torch.no_grad():
                for x, y in tqdm(test_dataloader, desc="EMA ", ncols=60, colour='green'):
                    x, y = x.to(device), y.to(device)
                    y = torch.squeeze(y)

                    # 若批次大小为奇数，丢弃最后一个样本
                    if x.size(0) % 2 != 0:
                        x = x[:-1]
                        y = y[:-1]

                    y_one_hot = nn.functional.one_hot(y, num_classes=num_classes).float()

                    YPred = ema_model.ema_model(x)
                    Loss = loss_func(YPred, y_one_hot)

                    ema_RunningLoss += Loss.item()

                    YPred_class = torch.argmax(YPred, dim=1)
                    ema_NumCorrect += (YPred_class == y).sum().item()
                    ema_NumTotal += y.size(0)

                    # 更新混淆矩阵
                    for t, p in zip(y.cpu().numpy(), YPred_class.cpu().numpy()):
                        ema_confusion_matrix[t, p] += 1

                    # 计算真/假正例和真/假反例
                    BatchSize = y.size(0)
                    for i in range(BatchSize):
                        for k in range(num_classes):
                            if y[i].item() == k:
                                if YPred_class[i] == k:
                                    ema_TruePositive[k] += 1
                                else:
                                    ema_FalseNegative[k] += 1
                            else:
                                if YPred_class[i].item() == k:
                                    ema_FalsePositive[k] += 1
                                else:
                                    ema_TrueNegative[k] += 1

                    # 收集用于计算多分类 AUC
                    ema_all_test_labels.extend(y.detach().cpu().numpy())
                    ema_all_test_probs.extend(YPred.detach().cpu().numpy())

            ema_TestLoss = ema_RunningLoss / len(test_dataloader)
            ema_TestAcc = ema_NumCorrect / ema_NumTotal

            try:
                emaAUC = roc_auc_score(
                    ema_all_test_labels,
                    ema_all_test_probs,
                    multi_class='ovr',
                    average='macro'
                )
            except ValueError:
                emaAUC = float('nan')

            for k in range(num_classes):
                PositiveAll = ema_TruePositive[k] + ema_FalseNegative[k]
                TPAndFP = ema_TruePositive[k] + ema_FalsePositive[k]
                NegativeAll = ema_TrueNegative[k] + ema_FalsePositive[k]
                if PositiveAll != 0:
                    ema_Recall[k] = ema_TruePositive[k] / PositiveAll
                if TPAndFP != 0:
                    ema_Precision[k] = ema_TruePositive[k] / TPAndFP
                if (ema_Recall[k] + ema_Precision[k]) != 0:
                    ema_F1Score[k] = 2 * ema_Recall[k] * ema_Precision[k] / (ema_Recall[k] + ema_Precision[k])
                if NegativeAll != 0:
                    ema_Specificity[k] = ema_TrueNegative[k] / NegativeAll

            # 保存 EMA 模型的结果
            self.ema_TestLoss = ema_TestLoss
            self.ema_TestAcc = ema_TestAcc
            self.ema_Recall = ema_Recall
            self.ema_Precision = ema_Precision
            self.ema_F1Score = ema_F1Score
            self.ema_Specificity = ema_Specificity
            self.ema_confusion_matrix = ema_confusion_matrix
            self.emaAUC = emaAUC  # 新增：EMA多分类 AUC


    def save_model_confusion_matrix(self, confusion_matrix, output_path="./model_confusion_matrix.xlsx"):
        if confusion_matrix is None:
            print("错误, 检测到 confusion_matrix 为 None")
            return

        from openpyxl import Workbook

        # 创建工作簿和工作表
        wb = Workbook()
        ws = wb.active

        # 创建表头
        header = [""] + [f"Pred_{i}" for i in range(confusion_matrix.shape[1])]
        ws.append(header)

        # 写入混淆矩阵数据
        for i in range(confusion_matrix.shape[0]):
            row = [f"True_{i}"] + confusion_matrix[i].tolist()
            ws.append(row)

        # 保存文件
        wb.save(output_path)
        print(f"原始模型混淆矩阵已保存至 {output_path}")

    def save_ema_confusion_matrix(self, confusion_matrix, output_path="./ema_confusion_matrix.xlsx"):
        if confusion_matrix is None:
            print("错误, 检测到 ema_confusion_matrix 为 None")
            return

        from openpyxl import Workbook

        # 创建工作簿和工作表
        wb = Workbook()
        ws = wb.active

        # 创建表头
        header = [""] + [f"Pred_{i}" for i in range(confusion_matrix.shape[1])]
        ws.append(header)

        # 写入混淆矩阵数据
        for i in range(confusion_matrix.shape[0]):
            row = [f"True_{i}"] + confusion_matrix[i].tolist()
            ws.append(row)

        # 保存文件
        wb.save(output_path)
        print(f"EMA 模型混淆矩阵已保存至 {output_path}")




# begin 尽量不进行验证的类
class CnnTrainerProConVal(object):
    def __init__(self, device, num_classes, optim, model, train_dataloader, test_dataloader, rep=False, use_mixup_cutmix=True) -> None:
        # Mixup 和 CutMix
        mixup_fn = None
        if use_mixup_cutmix:
            mixup_fn = Mixup(
                mixup_alpha=0.8,
                cutmix_alpha=1,
                prob=1,             # 应用 Mixup/CutMix 的概率
                switch_prob=0.5,    # 在 Mixup 和 CutMix 之间切换的概率
                mode='batch',       # 'batch' 模式对整个批次应用相同的混合
                label_smoothing=0.1,# 标签平滑值
                num_classes=num_classes
            )

        # 初始化统计变量
        self.TrainLoss = 0
        self.TrainAcc = 0
        self.TestLoss = 0
        self.TestAcc = 0
        self.Recall = np.zeros(num_classes, dtype=float)
        self.Precision = np.zeros(num_classes, dtype=float)
        self.F1Score = np.zeros(num_classes, dtype=float)
        self.Specificity = np.zeros(num_classes, dtype=float)

        # 初始化损失函数
        self.loss_func = SoftTargetCrossEntropy().to(device)

        # 保存必要的变量
        self.device = device
        self.num_classes = num_classes
        self.optim = optim
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.rep = rep
        self.mixup_fn = mixup_fn

    def train_one_epoch(self):
        """
        执行一个完整的训练过程。
        """
        self.model.train()
        NumCorrect, NumTotal, RunningLoss = 0, 0, 0

        for x, y in tqdm(self.train_dataloader, ncols=60, colour='blue'):
            self.optim.zero_grad()
            x = x.to(self.device)
            y = y.to(self.device)

            # 如果批次大小是奇数，丢弃最后一个样本
            if x.size(0) % 2 != 0:
                x = x[:-1]
                y = y[:-1]

            # 应用 Mixup 和 CutMix
            if self.mixup_fn is not None:
                x, y = self.mixup_fn(x, y)
            else:
                y = nn.functional.one_hot(y, num_classes=self.num_classes).float()

            # 前向传播和计算损失
            YPred = self.model(x)
            Loss = self.loss_func(YPred, y)
            Loss.backward()
            self.optim.step()

            # 统计准确率和损失
            with torch.no_grad():
                YPred_class = torch.argmax(YPred, dim=1)
                y_class = torch.argmax(y, dim=1)
                NumCorrect += (YPred_class == y_class).sum().item()
                NumTotal += y.size(0)
                RunningLoss += Loss.item()

        # 计算平均损失和准确率
        self.TrainLoss = RunningLoss / len(self.train_dataloader)
        self.TrainAcc = NumCorrect / NumTotal

    def validate(self, epoch):
        if epoch % 50 != 0:
            return

        print(f"Running validation for epoch {epoch}.")
        self.model.eval()
        if self.rep and hasattr(self.model, 'reparameterize_model'):
            re_model = self.model.reparameterize_model()  # 结构重参数化
        else:
            re_model = self.model

        re_model.to(self.device)
        NumCorrect, NumTotal, RunningLoss = 0, 0, 0
        TruePositive = np.zeros(self.num_classes, dtype=int)
        FalsePositive = np.zeros(self.num_classes, dtype=int)
        TrueNegative = np.zeros(self.num_classes, dtype=int)
        FalseNegative = np.zeros(self.num_classes, dtype=int)

        with torch.no_grad():
            for x, y in tqdm(self.test_dataloader, ncols=60, colour='blue'):
                x, y = x.to(self.device), y.to(self.device)
                y = torch.squeeze(y)

                # 如果批次大小是奇数，丢弃最后一个样本
                if x.size(0) % 2 != 0:
                    x = x[:-1]
                    y = y[:-1]

                y_one_hot = nn.functional.one_hot(y, num_classes=self.num_classes).float()

                # 前向传播和计算损失
                YPred = re_model(x)
                Loss = self.loss_func(YPred, y_one_hot)

                YPred_class = torch.argmax(YPred, dim=1)
                NumCorrect += (YPred_class == y).sum().item()
                BatchSize = y.size(0)
                NumTotal += BatchSize
                RunningLoss += Loss.item()

                # 统计真/假正例和真/假反例
                for i in range(BatchSize):
                    for k in range(self.num_classes):
                        if y[i].item() == k:
                            if YPred_class[i] == y[i]:
                                TruePositive[k] += 1
                            else:
                                FalseNegative[k] += 1
                        else:
                            if YPred_class[i].item() == k:
                                FalsePositive[k] += 1
                            else:
                                TrueNegative[k] += 1

        # 计算验证集的平均损失和准确率
        self.TestLoss = RunningLoss / len(self.test_dataloader)
        self.TestAcc = NumCorrect / NumTotal

        # 计算各类指标
        for k in range(self.num_classes):
            PositiveAll = TruePositive[k] + FalseNegative[k]
            TPAndFP = TruePositive[k] + FalsePositive[k]
            NegativeAll = TrueNegative[k] + FalsePositive[k]
            if PositiveAll != 0:
                self.Recall[k] = TruePositive[k] / PositiveAll
            if TPAndFP != 0:
                self.Precision[k] = TruePositive[k] / TPAndFP
            if (self.Recall[k] + self.Precision[k]) != 0:
                self.F1Score[k] = 2 * self.Recall[k] * self.Precision[k] / (self.Recall[k] + self.Precision[k])
            if NegativeAll != 0:
                self.Specificity[k] = TrueNegative[k] / NegativeAll



# note 一轮验证
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, roc_curve
from tqdm import tqdm
import matplotlib.pyplot as plt


class CnnValidator(object):
    def __init__(self, device, num_classes, model, test_loader) -> None:
        self.device = device
        self.orig_throughput = 0
        self.re_throughput = 0

        # 计算吞吐量
        x_sample, _ = next(iter(test_loader))
        self.orig_throughput = self._measure_throughput(model, x_sample)

        # 初始化原始模型的统计变量
        self.TruePositive = np.zeros(num_classes, dtype=int)
        self.FalsePositive = np.zeros(num_classes, dtype=int)
        self.TrueNegative = np.zeros(num_classes, dtype=int)
        self.FalseNegative = np.zeros(num_classes, dtype=int)

        self.Recall = np.zeros(num_classes, dtype=float)
        self.Precision = np.zeros(num_classes, dtype=float)
        self.Specificity = np.zeros(num_classes, dtype=float)
        self.F1Score = np.zeros(num_classes, dtype=float)

        self.TrueLabels = []
        self.PredLabels = []
        self.PredProbs = []

        # 验证原始模型
        self.validate_model(model, test_loader, num_classes)

        # 如果存在重参数化模型，进行重参数化模型的验证
        if hasattr(model, 'reparameterize_model'):
            re_model = model.reparameterize_model()
            re_model.eval()
            self.re_throughput = self._measure_throughput(re_model, x_sample)

            # 初始化重参数化模型的统计变量
            self.re_TruePositive = np.zeros(num_classes, dtype=int)
            self.re_FalsePositive = np.zeros(num_classes, dtype=int)
            self.re_TrueNegative = np.zeros(num_classes, dtype=int)
            self.re_FalseNegative = np.zeros(num_classes, dtype=int)

            self.re_Recall = np.zeros(num_classes, dtype=float)
            self.re_Precision = np.zeros(num_classes, dtype=float)
            self.re_Specificity = np.zeros(num_classes, dtype=float)
            self.re_F1Score = np.zeros(num_classes, dtype=float)

            self.re_TrueLabels = []
            self.re_PredLabels = []
            self.re_PredProbs = []

            # 验证重参数化模型
            self.validate_model(re_model, test_loader, num_classes, is_remodel=True)

    def validate_model(self, model, test_loader, num_classes, is_remodel=False):
        prefix = "re_" if is_remodel else ""
        NumCorrect = 0
        NumTotal = 0
        RunningLoss = 0
        loss_func = torch.nn.CrossEntropyLoss().to(self.device)

        model.eval()
        with torch.no_grad():
            for x, y in tqdm(test_loader, ncols=60, colour='blue'):
                x, y = x.to(self.device), y.to(self.device)
                y = torch.squeeze(y)

                # 模型预测
                YPred = model(x)
                Loss = loss_func(YPred, y)

                # 累计损失
                RunningLoss += Loss.item()

                # 预测概率和类别
                YPredProbs = torch.softmax(YPred, dim=1)
                if is_remodel:
                    self.re_PredProbs.extend(YPredProbs.cpu().numpy())
                else:
                    self.PredProbs.extend(YPredProbs.cpu().numpy())

                YPredLabels = torch.argmax(YPred, dim=1)

                # 计算准确率
                NumCorrect += (torch.eq(YPredLabels, y)).sum().item()
                BatchSize = y.size(0)
                NumTotal += BatchSize

                # 保存真实标签和预测标签
                if is_remodel:
                    self.re_TrueLabels.extend(y.cpu().numpy())
                    self.re_PredLabels.extend(YPredLabels.cpu().numpy())
                else:
                    self.TrueLabels.extend(y.cpu().numpy())
                    self.PredLabels.extend(YPredLabels.cpu().numpy())

                # 逐样本更新 TP, FP, TN, FN
                for i in range(BatchSize):
                    for k in range(num_classes):
                        if y[i].item() == k:
                            if YPredLabels[i] == y[i]:
                                if is_remodel:
                                    self.re_TruePositive[k] += 1
                                else:
                                    self.TruePositive[k] += 1
                            else:
                                if is_remodel:
                                    self.re_FalseNegative[k] += 1
                                else:
                                    self.FalseNegative[k] += 1
                        else:
                            if YPredLabels[i].item() == k:
                                if is_remodel:
                                    self.re_FalsePositive[k] += 1
                                else:
                                    self.FalsePositive[k] += 1
                            else:
                                if is_remodel:
                                    self.re_TrueNegative[k] += 1
                                else:
                                    self.TrueNegative[k] += 1

        # 计算验证损失和准确率
        if is_remodel:
            self.re_TestLoss = RunningLoss / NumTotal
            self.re_TestAcc = NumCorrect / NumTotal
        else:
            self.TestLoss = RunningLoss / NumTotal
            self.TestAcc = NumCorrect / NumTotal

        # 逐类别计算指标
        for k in range(num_classes):
            if is_remodel:
                TP, FP = self.re_TruePositive[k], self.re_FalsePositive[k]
                TN, FN = self.re_TrueNegative[k], self.re_FalseNegative[k]
            else:
                TP, FP = self.TruePositive[k], self.FalsePositive[k]
                TN, FN = self.TrueNegative[k], self.FalseNegative[k]

            PositiveAll = TP + FN
            TPAndFP = TP + FP
            NegativeAll = TN + FP

            if is_remodel:
                if PositiveAll != 0:
                    self.re_Recall[k] = TP / PositiveAll
                if TPAndFP != 0:
                    self.re_Precision[k] = TP / TPAndFP
                if self.re_Recall[k] + self.re_Precision[k] != 0:
                    self.re_F1Score[k] = 2 * self.re_Recall[k] * self.re_Precision[k] / (
                                self.re_Recall[k] + self.re_Precision[k])
                if NegativeAll != 0:
                    self.re_Specificity[k] = TN / NegativeAll
            else:
                if PositiveAll != 0:
                    self.Recall[k] = TP / PositiveAll
                if TPAndFP != 0:
                    self.Precision[k] = TP / TPAndFP
                if self.Recall[k] + self.Precision[k] != 0:
                    self.F1Score[k] = 2 * self.Recall[k] * self.Precision[k] / (self.Recall[k] + self.Precision[k])
                if NegativeAll != 0:
                    self.Specificity[k] = TN / NegativeAll

        # 计算 AUC
        if is_remodel:
            self.re_AUC = self.calculate_auc(num_classes, is_remodel=True)
        else:
            self.AUC = self.calculate_auc(num_classes)

        # 保存指标到文件
        self.save_metrics_to_csv(num_classes, is_remodel)

    # note 计算 AUC
    def calculate_auc(self, NumClasses, is_remodel=False):
        prefix = "re_" if is_remodel else ""

        # 根据是否是重参数化模型选择对应的标签和预测概率
        TrueLabels = getattr(self, f'{prefix}TrueLabels')
        PredProbs = getattr(self, f'{prefix}PredProbs')

        TrueLabelsOneHot = np.eye(NumClasses)[TrueLabels]
        PredProbs = np.array(PredProbs)

        # 保存输入数据时也添加前缀区分
        pd.DataFrame(TrueLabels).to_csv(f'{prefix}input1.csv', index=False, header=False)
        pd.DataFrame(PredProbs).to_csv(f'{prefix}input2.csv', index=False, header=False)

        # 检查预测概率范围并标准化
        if not np.all((PredProbs >= 0) & (PredProbs <= 1)):
            PredProbs = np.exp(PredProbs) / np.sum(np.exp(PredProbs), axis=1, keepdims=True)

        # 计算 AUC
        return roc_auc_score(TrueLabelsOneHot, PredProbs, average='macro', multi_class='ovr')

    # note 绘制 ROC 曲线
    def plot_roc_curve(self, NumClasses):
        TrueLabelsOneHot = np.eye(NumClasses)[self.TrueLabels]
        PredProbs = np.array(self.PredProbs)
        if not np.all((PredProbs >= 0) & (PredProbs <= 1)):
            PredProbs = np.exp(PredProbs) / np.sum(np.exp(PredProbs), axis=1, keepdims=True)

        plt.figure(figsize=(10, 8))
        for i in range(NumClasses):
            fpr, tpr, _ = roc_curve(TrueLabelsOneHot[:, i], PredProbs[:, i])
            plt.plot(fpr, tpr, label=f'Class {i}')
        plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()

    # note 保存每个类别的评估指标到CSV文件
    def save_metrics_to_csv(self, num_classes, is_remodel=False):
        import pandas as pd

        prefix = "re_" if is_remodel else ""

        # 根据是否是重参数化模型选择对应的属性
        metrics_dict = {
            'Class': list(range(num_classes)),
            'TP': getattr(self, f'{prefix}TruePositive'),
            'TN': getattr(self, f'{prefix}TrueNegative'),
            'FP': getattr(self, f'{prefix}FalsePositive'),
            'FN': getattr(self, f'{prefix}FalseNegative'),
            'Precision': getattr(self, f'{prefix}Precision'),
            'Recall': getattr(self, f'{prefix}Recall'),
            'F1-Score': getattr(self, f'{prefix}F1Score'),
            'Specificity': getattr(self, f'{prefix}Specificity')
        }

        # 创建DataFrame
        df = pd.DataFrame(metrics_dict)

        # 计算并添加平均值行
        means = df.mean(numeric_only=True)
        means_row = pd.DataFrame({
            'Class': ['Mean'],
            'TP': [means['TP']],
            'TN': [means['TN']],
            'FP': [means['FP']],
            'FN': [means['FN']],
            'Precision': [means['Precision']],
            'Recall': [means['Recall']],
            'F1-Score': [means['F1-Score']],
            'Specificity': [means['Specificity']]
        })

        # 保存第1组数据
        df = pd.concat([df, means_row], ignore_index=True)
        df.to_csv(f'{prefix}res1.csv', index=False)

        # 保存总体指标
        overall_metrics = {
            'Metric': ['Total Accuracy', 'Total AUC', 'Average Loss',
                       'Macro Precision', 'Macro Recall', 'Macro F1-Score', 'Macro Specificity'],
            'Value': [
                getattr(self, f'{prefix}TestAcc'),
                getattr(self, f'{prefix}AUC'),
                getattr(self, f'{prefix}TestLoss'),
                means['Precision'],
                means['Recall'],
                means['F1-Score'],
                means['Specificity']
            ]
        }
        # 保存第2组数据
        pd.DataFrame(overall_metrics).to_csv(f'{prefix}res2.csv', index=False)

    def _measure_throughput(self, model, x_sample, warm_up_steps=10, repeat_steps=100):
        # 确保模型在正确的设备上
        model = model.to(self.device)
        model.eval()
        total_time = 0
        total_images = 0

        x_sample = x_sample.to(self.device)
        batch_size = x_sample.size(0)

        with torch.no_grad():
            # GPU预热
            for _ in range(warm_up_steps):
                _ = model(x_sample)
                torch.cuda.synchronize()

            # 多次重复测试取平均
            for _ in range(repeat_steps):
                start_time = time.time()
                _ = model(x_sample)
                torch.cuda.synchronize()
                end_time = time.time()

                total_time += (end_time - start_time)
                total_images += batch_size

        return total_images / total_time

    def print_throughput(self):
        # ANSI颜色代码
        GREEN = '\033[92m'
        BLUE = '\033[94m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        ENDC = '\033[0m'

        print(f"\n{YELLOW}{'=' * 20} 模型吞吐量测试 {'=' * 20}{ENDC}")
        print(f"{GREEN}原始模型吞吐量: {self.orig_throughput:.2f} images/s{ENDC}")

        if self.re_throughput > 0:  # 如果存在重参数化模型
            print(f"{BLUE}重参数化模型吞吐量: {self.re_throughput:.2f} images/s{ENDC}")

            # 计算性能提升
            speedup = (self.re_throughput / self.orig_throughput - 1) * 100
            if speedup > 0:
                print(f"{YELLOW}性能提升: +{speedup:.2f}%{ENDC}")
            else:
                print(f"{RED}性能下降: {speedup:.2f}%{ENDC}")

            # 显示具体batch处理时间
            orig_batch_time = 1000 / self.orig_throughput  # 转换为毫秒
            re_batch_time = 1000 / self.re_throughput
            print(f"\n{GREEN}原始模型每批次处理时间: {orig_batch_time:.2f} ms{ENDC}")
            print(f"{BLUE}重参数化模型每批次处理时间: {re_batch_time:.2f} ms{ENDC}")