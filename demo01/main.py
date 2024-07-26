# Author:妖怪鱼
# Date:2024/7/25
# Introduction:pytorch识别手写文字demo(多层感知机)

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
1
# 这个类定义了一个简单的多层感知机（MLP），包含四个全连接层
# fc1：输入层到第一个隐藏层（64个神经元）
# fc2：第一个隐藏层到第二个隐藏层（64个神经元）
# fc3：第二个隐藏层到第三个隐藏层（64个神经元）
# fc4：第三个隐藏层到输出层（10个神经元，对应数字0到9）
# 每一层的输出都经过 ReLU 激活函数，最后一层经过 Log Softmax 函数计算概率
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x


# 使用 transforms.ToTensor() 将图像转换为张量。
# 从 MNIST 数据集中加载训练或测试数据。
# 返回一个数据加载器，每个批次包含15个样本，数据顺序随机打乱。
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 模型评估函数
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28 * 28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

# 主函数
def main():
    # 加载训练数据和测试数据
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    # 初始化神经网络模型
    net = Net()
    print("initial accuracy:", evaluate(test_data, net))
    # 定义优化器，使用 Adam 优化算法
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for epoch in range(2):
        for (x, y) in train_data:
            # 清零梯度
            net.zero_grad()
            # 将输入数据展平，并通过网络进行前向传播
            output = net.forward(x.view(-1, 28 * 28))
            # 计算负对数似然损失
            loss = torch.nn.functional.nll_loss(output, y)
            # 反向传播计算梯度
            loss.backward()
            # 更新模型参数
            optimizer.step()
        # 每个 epoch 结束后，评估模型在测试集上的准确率
        print("epoch", epoch, "准确率:", evaluate(test_data, net))
    # 可视化前几个测试样本的预测结果
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:  # 只显示前 4 个样本
            break
        # 对每个样本进行预测
        predict = torch.argmax(net.forward(x[0].view(-1, 28 * 28)))
        # 显示预测结果和对应的图像
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("预测的结果是: " + str(int(predict)))
    plt.show()


# 运行主函数
if __name__ == "__main__":
    main()
