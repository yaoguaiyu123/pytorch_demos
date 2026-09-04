# begin 损失函数
from torch.autograd import Variable
from torch import Tensor
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



# note 标签平滑损失函数
class LabelSmoothing(nn.Module):
    def __init__(self, size, smoothing=0.1):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x表示输入 (N，M)N个样本，M表示总类数，每一个类的概率log P
        target表示label（M，）
        """
        assert x.size(1) == self.size
        true_dist = x.data.clone()  # 先深复制过来
        # print true_dist
        true_dist.fill_(self.smoothing / (self.size - 1))  # otherwise的公式
        # print true_dist
        # 变成one-hot编码，1表示按列填充，
        # target.data.unsqueeze(1)表示索引,confidence表示填充的数字
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        self.true_dist = true_dist

        return self.criterion(x, Variable(true_dist, requires_grad=False))



# note 焦点损失函数
class CustomFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(CustomFocalLoss, self).__init__()
        self.alpha = alpha  # 类别权重
        self.gamma = gamma  # 焦点参数
        self.reduction = reduction  # 损失的归约方式

    def forward(self, logits, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss



# mod 类平衡损失函数, beta 为固定值
# def CBLoss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
#     """计算 `logits` 和真实标签 `labels` 之间的类别平衡损失（Class Balanced Loss）。
#
#     类别平衡损失（Class Balanced Loss）公式：
#     ((1 - beta) / (1 - beta^n)) * Loss(labels, logits)
#     其中，Loss 是神经网络中常用的标准损失函数之一。
#
#     参数：
#       labels：大小为 [batch] 的整型张量。
#       logits：大小为 [batch, no_of_classes] 的浮点型张量。
#       samples_per_cls：大小为 [no_of_classes] 的 Python 列表。
#       no_of_classes：类别总数，整数。
#       loss_type：字符串，取值为 "sigmoid"、"focal" 或 "softmax" 之一。
#       beta：浮点数。类别平衡损失的超参数。
#       gamma：浮点数。Focal Loss 的超参数。
#
#     返回：
#       cbloss：表示类别平衡损失的浮点型张量。
#     """
#
#     effective_num = 1.0 - np.power(beta, samples_per_cls)  # [num_classes]
#     weights = (1.0 - beta) / np.array(effective_num)  # 得到权重
#     weights = weights / np.sum(weights) * no_of_classes  # 对权重进行归一化
#
#     labels_one_hot = F.one_hot(labels, no_of_classes).float()
#
#     # 创建张量
#     weights = torch.tensor(weights).float()  # [num_classes]
#     weights = weights.unsqueeze(0)  # [1 num_classes]
#     weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot  # [1 num_classes] -> [B num_classes]
#     weights = weights.sum(1)  # [B num_classes] -> [B]
#     weights = weights.unsqueeze(1)  # [B] -> [B 1]
#     weights = weights.repeat(1,no_of_classes)  # [B num_classes]
#
#     cbloss = None
#     # 使用类平衡函数嵌套别的函数
#     if loss_type == "focal":
#         cbloss = FocalLoss(labels_one_hot, logits, weights, gamma)
#     elif loss_type == "sigmoid":
#         cbloss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
#     elif loss_type == "softmax":
#         pred = logits.softmax(dim = 1)
#         cbloss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
#     return cbloss


# note 类平衡损失函数, beta 为数组
def CBLoss(labels, logits, samples_per_cls, no_of_classes, loss_type, beta, gamma):
    beta = beta.cpu()
    samples_per_cls = samples_per_cls.cpu()
    beta = np.array(beta)
    samples_per_cls = np.array(samples_per_cls)

    # 避免 beta 中出现 1 的值
    epsilon = 1e-5
    beta = np.clip(beta, 0, 1 - epsilon)

    effective_num = 1.0 - np.power(beta, samples_per_cls)  # [num_classes]
    weights = (1.0 - beta) / effective_num  # [num_classes]
    weights = weights / np.sum(weights) * no_of_classes  # 对权重进行归一化

    # labels_one_hot = F.one_hot(labels, no_of_classes).float()

    # 创建张量
    weights = torch.tensor(weights, dtype=torch.float32, device=logits.device)  # [num_classes]
    # weights = weights.unsqueeze(0)  # [1, num_classes]
    # weights = weights.repeat(labels_one_hot.shape[0], 1)  # [batch_size, num_classes]
    # weights = weights * labels_one_hot  # [batch_size, num_classes]
    # weights = weights.sum(1)  # [batch_size]
    # weights = weights.unsqueeze(1)  # [batch_size, 1]
    # weights = weights.repeat(1, no_of_classes)  # [batch_size, num_classes]

    loss_func = None
    # 使用类平衡函数嵌套别的函数
    if loss_type == "focal":
        loss_func = CustomFocalLoss(alpha=weights, gamma=gamma, reduction="mean")

    loss = loss_func(logits, labels)

    return loss



# note LDAMLoss 来源于 (https://github.com/kaidic/LDAM-DRW)
class LDAMLoss(nn.Module):

    def __init__(self, para_dict=None):
        super(LDAMLoss, self).__init__()
        s = 30
        self.num_class_list = para_dict["num_class_list"]
        self.device = para_dict["device"]

        cfg = para_dict["cfg"]
        max_m = cfg.LOSS.LDAM.MAX_MARGIN
        m_list = 1.0 / np.sqrt(np.sqrt(self.num_class_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(self.device)
        self.m_list = m_list
        assert s > 0

        self.s = s
        self.step_epoch = cfg.LOSS.LDAM.DRW_EPOCH
        self.weight = None

    def reset_epoch(self, epoch):
        idx = (epoch-1) // self.step_epoch
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], self.num_class_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(self.num_class_list)
        self.weight = torch.FloatTensor(per_cls_weights).to(self.device)

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        index_float = index_float.to(self.device)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight= self.weight)



# note 类平衡损失函数 + 焦点损失函数
class ClassBalancedFocalLoss(nn.Module):
    def __init__(self, num_classes, samples_per_class, gamma=2.0):
        super(ClassBalancedFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.gamma = gamma
        # 计算每个类的 beta: beta_i = (N_i - 1) / N_i
        self.betas = (samples_per_class - 1) / samples_per_class

    def forward(self, logits, labels):
        loss_type = "focal"
        cb_loss = CBLoss(labels, logits, self.samples_per_class, self.num_classes, loss_type, self.betas, self.gamma)
        return cb_loss




def reduce_loss(loss: Tensor, reduction: str) -> Tensor:
    """按指定方式减少损失。

    参数：
        loss (Tensor): 逐元素损失张量。
        reduction (str): 可选项为 "none"、"mean" 和 "sum"。

    返回：
        Tensor: 处理后的损失张量。
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean: 1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()



def weight_reduce_loss(loss: Tensor,
                       weight: Optional[Tensor] = None,
                       reduction: str = 'mean',
                       avg_factor: Optional[float] = None) -> Tensor:
    """应用逐元素权重并减少损失。

    参数：
        loss (Tensor): 逐元素损失张量。
        weight (Optional[Tensor], optional): 逐元素权重。默认为 None。
        reduction (str, optional): 与 PyTorch 内置损失函数中的 reduction 相同。默认为 'mean'。
        avg_factor (Optional[float], optional): 计算损失均值时的平均因子。默认为 None。

    返回：
        Tensor: 处理后的损失值。
    """
    # 如果指定了权重，应用逐元素权重
    if weight is not None:
        loss = loss * weight

    # 如果没有指定 avg_factor，直接减少损失
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # 如果 reduction 为 mean，则按 avg_factor 平均损失
        if reduction == 'mean':
            # 避免在 avg_factor 为 0.0 时引发 ZeroDivisionError，
            # 例如，当图像的所有标签都属于忽略索引时。
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # 如果 reduction 为 'none'，则不执行任何操作，否则抛出错误
        elif reduction != 'none':
            raise ValueError('avg_factor 不能与 reduction="sum" 一起使用')
    return loss





# note 分类任务的seesaw损失函数
# 更多参数:
#   use_sigmoid (bool, 可选): 预测是否使用 sigmoid 或 softmax。仅支持 False。
#   p (float, 可选): 缓解因子中的参数 "p"。默认为 0.8。
#   q (float, 可选): 补偿因子中的参数 "q"。默认为 2.0。
#   num_classes (int, 可选): 类别数量。默认为 LVIS v1 数据集中的 1203。
#   eps (float, 可选): 平滑补偿因子计算时的最小除数值。
#   reduction (str, 可选): 用于减少损失的方式。可选项为 "none"、"mean" 和 "sum"。
#   loss_weight (float, 可选): 损失的权重。默认为 1.0。

class SeesawCeLoss(nn.Module):
    def __init__(self,
                 num_classes: int,
                 cum_samples: Tensor,
                 p: float = 0.8,
                 q: float = 2.0,
                 eps: float = 1e-7,
                 reduction: str = 'mean'):
        """
        SeeSawLoss 损失函数的类封装

        参数：
            num_classes (int): 类别总数。
            cum_samples (Tensor): 每个类别的累积样本数量。
            p (float): 缓解因子中的参数 p，默认为 0.8。
            q (float): 补偿因子中的参数 q，默认为 2.0。
            eps (float): 用于平滑补偿因子计算的最小除数值，默认为 1e-7。
            reduction (str): 用于减少损失的方法，默认为 'mean'。
        """
        super(SeesawCeLoss, self).__init__()
        self.num_classes = num_classes
        self.cum_samples = cum_samples
        self.p = p
        self.q = q
        self.eps = eps
        self.reduction = reduction

    def forward(self, logits: Tensor, labels: Tensor) -> Tensor:
        """
        前向传播计算 SeeSawLoss

        参数：
            logits (Tensor): 预测值，形状为 (N, C)，其中 C 是类别数量。
            labels (Tensor): 预测的学习标签。

        返回：
            Tensor: 计算后的损失。
        """
        assert logits.size(-1) == self.num_classes
        assert len(self.cum_samples) == self.num_classes

        onehot_labels = F.one_hot(labels, self.num_classes)
        seesaw_weights = logits.new_ones(onehot_labels.size())

        # 缓解因子
        if self.p > 0:
            sample_ratio_matrix = self.cum_samples[None, :].clamp(min=1) / self.cum_samples[:, None].clamp(min=1)
            index = (sample_ratio_matrix < 1.0).float()
            sample_weights = sample_ratio_matrix.pow(self.p) * index + (1 - index)
            mitigation_factor = sample_weights[labels.long(), :]
            seesaw_weights = seesaw_weights * mitigation_factor

        # 补偿因子
        if self.q > 0:
            scores = F.softmax(logits.detach(), dim=1)
            self_scores = scores[torch.arange(0, len(scores)).to(scores.device).long(), labels.long()]
            score_matrix = scores / self_scores[:, None].clamp(min=self.eps)
            index = (score_matrix > 1.0).float()
            compensation_factor = score_matrix.pow(self.q) * index + (1 - index)
            seesaw_weights = seesaw_weights * compensation_factor

        logits = logits + (seesaw_weights.log() * (1 - onehot_labels))

        loss = F.cross_entropy(logits, labels, weight=None, reduction='none')

        # 默认 label_weights 设置为 None，所以没有传入标签权重
        loss = weight_reduce_loss(loss, weight=None, reduction=self.reduction)
        return loss




# 测试类平衡焦点损失函数
def func02():
    num_classes = 5
    samples_per_class = torch.tensor([100, 200, 300, 400, 500], dtype=torch.float32)   # 每个类别的样本数量
    gamma = 2.0  # 焦点损失中的 gamma 参数

    # 初始化类平衡焦点损失函数
    cb_focal_loss = ClassBalancedFocalLoss(num_classes, samples_per_class, gamma)

    # batch_size = 16
    logits = torch.randn(16, num_classes)  # 随机生成16个样本的logits
    labels = torch.randint(0, num_classes, (16,))  # 随机生成标签

    # 计算损失
    loss = cb_focal_loss(logits, labels)
    print(loss)




# 测试 SeeSaw 损失函数
# 超参数 p = 0.8, q = 2, and τ = 20.
def func03():
    # sp cum_samples: 累计样本数（cumulative samples） 是指到当前时刻为止，每个类别中已经看到的样本数量
    cum_samples = torch.tensor([100, 200, 300, 400, 500, 600,700 ,800])
    sesawloss = SeesawCeLoss(num_classes=8, cum_samples=cum_samples, p=0.8, q=2.0)
    logits = torch.randn(16, 8)
    labels = torch.randint(0, 8, (16,))
    loss = sesawloss(logits, labels)
    print(loss)



if __name__ == '__main__':
    # func01()
    func02()
    # func03()
    # draw_picture()
