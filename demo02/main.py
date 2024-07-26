# Author:妖怪鱼
# Date:2024/7/25
# Introduction:pytorch识别手写文字MNIST(卷积神经网络模型)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import cv2
import numpy as np

# note 定义超参数
# 超参数是在机器学习和深度学习模型训练过程中设置的参数，这些参数的值是在训练之前确定的，不会在训练过程中进行更新
BATCH_SIZE = 16 # 每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 选择使用GPU或者是CPU训练
print("最终选择的DEVICE:" ,DEVICE)
EPOCHS = 10  # 训练数据集的轮次

# note 构建pipeline，对图像做处理
pipeline = transforms.Compose([
    transforms.ToTensor(), # 将图片抓换成tensor格式
    transforms.Normalize((0.1307,),(0.3081,))  # 降低模型复杂度
])

# note 下载并加载数据集
# 这里使用了关键字参数的语法，增加可读性和易维护性
# 下载数据集的时候，其内部会自动做判断，如果发现已经下载好了数据，则不会再下载
train_set = datasets.MNIST(root="", train=True, download=True, transform=pipeline) # 下载训练集
test_set = datasets.MNIST(root="", train=False, download=True, transform=pipeline) # 下载测试集
print(f"训练集大小: {len(train_set)} 个样本")
print(f"测试集大小: {len(test_set)} 个样本")

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)   # 打乱16张训练集的数据
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

# note 显示第一张图片
# 打开文件
with open("./MNIST/raw/train-images-idx3-ubyte", "rb") as file:
    content = file.read() # 读取文件中的所有内容，并返回为一个字符串
    # [expression for item in iterable 为列表解析表达式
    # expression:表示生成列表中每个元素的表达式，可以是对 item 的操作或函数调用
    # for item in iterable:迭代 iterable 中的每个 item
    encodelist = [item for item in content[16:16 + 784]]  # 读取784个字节
    print("处理之后的ascii码列表:",encodelist)
    # np.array(encodelist, dtype=np.uint8)：将 encodelist转换为一个NumPy数组，数据类型为 uint8（无符号8位整数）
    # .reshape(28, 28, 1)：将一维数组重塑为 28x28x1 的三维数组
    imageMar = np.array(encodelist, dtype=np.uint8).reshape(28, 28, 1)
    print(imageMar.shape)
    cv2.imwrite("image.png", imageMar)

# note 构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1 ,10, 5)  # 1表示灰度图片的通道，10表示输出通道， 5表示卷积和
        self.conv2 = nn.Conv2d(10 ,20, 3)  # 10输入通道，20表示输出通道， 3表示卷积和
        self.fc1 = nn.Linear(20*10*10 ,500) # 20*10*10表示输入通道，500是输出通道
        self.fc2 = nn.Linear(500 ,10) # 500是输入通道，10是输出通道

    def forward(self ,x):
        input_size = x.size(0)  # batch_size   1   28   28
        x = self.conv1(x) # 输入:batcc_size * 1 * 28 * 28，输出:batch * 10 * 24 * 24 (28-5+1)
        x = F.relu(x) # 保持shape不变
        x = F.max_pool2d(x , 2, 2) # 输入:batch*10*24*24  输出:batch*10*12*12
        x = self.conv2(x) # 输入:batch*10*12*12  输出:batch*20*10*10
        x = F.relu(x)
        x = x.view(input_size, -1) # 拉平
        x = self.fc1(x) # 输入: batch*2000 输出:batch*500
        x = F.relu(x) # 保持形状不变
        x = self.fc2(x) # 输入: batch*500 输出:batch*10
        output = F.log_softmax(x ,dim=1) # 计算分类后，每个数字的概率值
        return output

# note 定义优化器
model = Digit().to(DEVICE)
optimizer = optim.Adam(model.parameters())


# note 定义训练方法
def train_model(model , device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE
        data, target = data.to(device),target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失
        loss = F.cross_entropy(output, target)
        # 找到概率值最大的下标
        pred = output.max(1 ,keepdim=True)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0:
            print("Train Epoch :{} \t Loss : {:.6f}".format(epoch, loss.item()))


# note 定义测试方法
def test_model(model, device ,test_loader):
    # 模型验证
    model.eval()
    # 正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad(): # 不会计算梯度，也不会进行反向传播
        for data, target in test_loader:
             # 部署到device
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率值最大的下标
            pred = output.max(1, keepdim=True)[1]
            # 累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("测试集 -- 平均损失值: {:.4f}, 准确率 : {:.3f}\n".format(
            test_loss,100.0 * correct / len(test_loader.dataset)
        ))


# note 调用方法训练和测试方法
for epoch in range(1 , EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)



