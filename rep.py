# 结构重参数化的示例
import torch
import torch.nn as nn


class ReparamModule(nn.Module):
    def __init__(self):
        super(ReparamModule, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

    def reparameterize(self):
        new_conv = nn.Conv2d(3, 16, 3, padding=1)
        with torch.no_grad():
            gamma = self.bn.weight
            beta = self.bn.bias
            mean = self.bn.running_mean
            var = self.bn.running_var
            eps = self.bn.eps

            scale = gamma / torch.sqrt(var + eps)
            new_conv.weight.copy_(self.conv.weight * scale[:, None, None, None])
            new_conv.bias.copy_(beta - mean * scale)
        return new_conv


model = ReparamModule()
x = torch.randn(1, 3, 32, 32)
output = model(x)

reparam_model = model.reparameterize()
output_inference = reparam_model(x)
