# begin GoodNet  Test
import torch.nn as nn
from torch.nn import init
from my_model.net_blocks import (Block, PCBlock, SPADownBlock,MDCBlock)

class LDRMNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LDRMNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=3, bias=False)  # H/2
        self.bn1 = nn.BatchNorm2d(32)
        self.hs1 = nn.SiLU(inplace=True)


        # note Large
        self.bneck = nn.Sequential(
            Block(3, 32, 64, 32, nn.SiLU, False, 2),  # H/4(64)
            PCBlock(in_channels=32, atrous_rates=[2, 3, 5]),
            Block(3, 32, 144, 80, nn.SiLU, False, 2),  # H/8(32)
            MDCBlock(80, 224),
            Block(3, 80, 224, 80, nn.SiLU, True, 1),
            PCBlock(in_channels=80, atrous_rates=[2, 4, 6]),
            MDCBlock(80, 224),
            Block(3, 80, 224, 80, nn.SiLU, True, 1),
            SPADownBlock(3, 80, 448, 112, nn.SiLU, True, 2),  # H/16(16)
            MDCBlock(112, 448),
            Block(3, 112, 448, 112, nn.SiLU, True, 1),
            Block(3, 112, 672, 160, nn.SiLU, True, 1),
        )

        # note Small
        # self.bneck = nn.Sequential(
        #     Block(3, 32, 64, 32, nn.SiLU, False, 2),  # H/4(64)
        #     PCBlock(in_channels=32, atrous_rates=[2, 3, 5]),
        #     Block(3, 32, 144, 80, nn.SiLU, False, 2),  # H/8(32)
        #     MDCBlock(80, 192),
        #     PCBlock(in_channels=80, atrous_rates=[2, 4, 6]),
        #     MDCBlock(80, 192),
        #     Block(3, 80, 192, 80, nn.SiLU, True, 1),
        #     SPADownBlock(3, 80, 448, 112, nn.SiLU, True, 2),  # H/16(16)
        #     MDCBlock(112, 336),
        #     Block(3, 112, 512, 160, nn.SiLU, True, 1),
        # )


        self.conv2 = nn.Conv2d(160, 960, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(960)
        self.hs2 = nn.SiLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.linear3 = nn.Linear(960, 1280, bias=False)
        self.bn3 = nn.BatchNorm1d(1280)
        self.hs3 = nn.SiLU(inplace=True)
        self.drop = nn.Dropout(0.2)

        self.linear4 = nn.Linear(1280, num_classes)
        self.init_params()

    # 参数初始化
    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.hs1(self.bn1(self.conv1(x)))
        res = self.bneck(out)
        out = self.hs2(self.bn2(self.conv2(res)))  # 1x1卷积提高通道数
        out = self.gap(out).flatten(1)
        out = self.drop(self.hs3(self.bn3(self.linear3(out))))
        return self.linear4(out)

    def reparameterize_model(self):
        """对模型中的所有 MDCBlock 进行重参数化，返回一个新的模型实例。"""
        new_model = LDRMNet(num_classes=self.linear4.out_features)
        new_model.load_state_dict(self.state_dict())   # 加载参数

        for idx, module in enumerate(new_model.bneck):
            if isinstance(module, MDCBlock) or isinstance(module, Block) or isinstance(module, SPADownBlock):
                module.reparameterize()
        return new_model


def rdm_1(num_classes=1000):
    return LDRMNet(num_classes=num_classes)


if __name__ == '__main__':
    net = LDRMNet(num_classes=1000)
    net_m = net.reparameterize_model()
    print(net_m)