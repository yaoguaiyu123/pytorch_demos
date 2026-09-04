# begin GoodNet  Test
import torch.nn as nn
from torch.nn import init
from my_model.net_blocks import (DACB, DACB2, PCB, SPADB, MSDCB)

class LDRMNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(LDRMNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 36, kernel_size=3, stride=2, padding=3, bias=False)  # H/2
        self.bn1 = nn.BatchNorm2d(36)
        self.hs1 = nn.SiLU(inplace=True)


        # note Large
        # self.bneck = nn.Sequential(
        #     DACB(3, 32, 64, 32, nn.SiLU, False, 2),  # H/4(64)
        #     PCB(in_channels=32, atrous_rates=[2, 3, 5]),
        #     DACB(3, 32, 144, 80, nn.SiLU, False, 2),  # H/8(32)
        #     MSDCB(80, 224),
        #     DACB(3, 80, 224, 80, nn.SiLU, True, 1),
        #     PCB(in_channels=80, atrous_rates=[2, 4, 6]),
        #     MSDCB(80, 224),
        #     DACB(3, 80, 224, 80, nn.SiLU, True, 1),
        #     MSDCB(80, 448, stride=2, out_channels=112),  # H/16(16)
        #     DACB(3, 112, 448, 112, nn.SiLU, True, 1),
        #     MSDCB(112, 448),
        #     DACB(3, 112, 672, 160, nn.SiLU, True, 1),
        # )

        # note 66.58
        # self.bneck = nn.Sequential(
        #     DACB(3, 32, 64, 32, nn.SiLU, False, 2),  # H/4(64)
        #     PCB(in_channels=32, atrous_rates=[2, 3, 5]),
        #     DACB(3, 32, 144, 80, nn.SiLU, False, 2),  # H/8(32)
        #     MSDCB(80, 224),
        #     DACB(3, 80, 224, 80, nn.SiLU, True, 1),
        #     PCB(in_channels=80, atrous_rates=[2, 4, 6]),
        #     MSDCB(80, 224),
        #     DACB(3, 80, 224, 80, nn.SiLU, True, 1),
        #     DACB(3, 80, 448, 112, nn.SiLU, True, 2),    # H/16(16)
        #     MSDCB(112, 448),
        #     DACB(3, 112, 448, 112, nn.SiLU, True, 1),
        #     DACB(3, 112, 672, 160, nn.SiLU, True, 1),
        # )


        # note 69.539
        # self.bneck = nn.Sequential(
        #     DACB2(3, 32, 64, 32, nn.SiLU, False, 2),  # H/4(64)
        #     PCB(in_channels=32, atrous_rates=[2, 3, 5]),
        #     DACB2(3, 32, 144, 80, nn.SiLU, False, 2),  # H/8(32)
        #     MSDCB(80, 224),
        #     DACB(3, 80, 224, 80, nn.SiLU, True, 1),
        #     PCB(in_channels=80, atrous_rates=[2, 4, 6]),
        #     MSDCB(80, 224),
        #     DACB(3, 80, 224, 80, nn.SiLU, True, 1),
        #     DACB2(3, 80, 448, 112, nn.SiLU, True, 2),    # H/16(16)
        #     MSDCB(112, 448),
        #     DACB(3, 112, 448, 112, nn.SiLU, True, 1),
        #     DACB(3, 112, 672, 160, nn.SiLU, True, 1),
        # )


        # note 63
        self.bneck = nn.Sequential(
            DACB2(3, 36, 72, 36, nn.SiLU, False, 2),  # H/4(64)
            PCB(in_channels=36, atrous_rates=[2, 3, 5]),
            DACB2(3, 36, 160, 88, nn.SiLU, False, 2),  # H/8(32)
            MSDCB(88, 248),
            DACB(3, 88, 248, 88, nn.SiLU, True, 1),
            PCB(in_channels=88, atrous_rates=[2, 4, 6]),
            MSDCB(88, 248),
            DACB(3, 88, 248, 88, nn.SiLU, True, 1),
            DACB2(3, 88, 492, 124, nn.SiLU, True, 2),  # H/16(16)
            MSDCB(124, 493),
            DACB(3, 124, 492, 124, nn.SiLU, True, 1),
            DACB(3, 124, 492, 124, nn.SiLU, True, 1),
            DACB(3, 124, 744, 176, nn.SiLU, True, 1),
        )


        # note 81 epoch : 56.527
        # self.bneck = nn.Sequential(
        #     DACB2(3, 24, 96, 24, nn.SiLU, False, 2),  # H/4(64)
        #     PCB(in_channels=24, atrous_rates=[2, 3, 5]),
        #     DACB2(3, 24, 144, 64, nn.SiLU, False, 2),  # H/8(32)
        #     MSDCB(64, 224),
        #     DACB(3, 64, 224, 64, nn.SiLU, True, 1),
        #     PCB(in_channels=64, atrous_rates=[2, 4, 6]),
        #     MSDCB(64, 224),
        #     DACB(3, 64, 224, 64, nn.SiLU, True, 1),
        #     DACB2(3, 64, 448, 112, nn.SiLU, True, 2),    # H/16(16)
        #     MSDCB(112, 448),
        #     DACB(3, 112, 448, 112, nn.SiLU, True, 1),
        #     DACB(3, 112, 448, 148, nn.SiLU, True, 1),
        #     DACB(3, 148, 672, 148, nn.SiLU, True, 1),
        #     DACB(3, 148, 960, 160, nn.SiLU, True, 1),
        # )


        # note 44 epoch : 50.044
        # self.bneck = nn.Sequential(
        #     DACB2(3, 32, 64, 32, nn.SiLU, False, 2),  # H/4(64)
        #     PCB(in_channels=32, atrous_rates=[2, 3, 5]),
        #     DACB2(3, 32, 128, 80, nn.SiLU, False, 2),  # H/8(32)
        #     MSDCB(80, 320),
        #     DACB(3, 80, 320, 80, nn.SiLU, True, 1),
        #     PCB(in_channels=80, atrous_rates=[2, 4, 6]),
        #     MSDCB(80, 320),
        #     DACB(3, 80, 320, 80, nn.SiLU, True, 1),
        #     DACB2(3, 80, 320, 112, nn.SiLU, True, 2),    # H/16(16)
        #     MSDCB(112, 448),
        #     DACB(3, 112, 448, 112, nn.SiLU, True, 1),
        #     DACB(3, 112, 448, 112, nn.SiLU, True, 1),
        #     DACB(3, 112, 672, 160, nn.SiLU, True, 1),
        # )


        self.conv2 = nn.Conv2d(176, 960, kernel_size=1, stride=1, padding=0, bias=False)
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
        # out = self.bneck1(out)
        # out = self.bneck2(out)
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
            if isinstance(module, MSDCB) or isinstance(module, DACB) or isinstance(module, SPADB):
                module.reparameterize()
        return new_model


def rdm_3(num_classes=1000):
    return LDRMNet(num_classes=num_classes)


if __name__ == '__main__':
    net = LDRMNet(num_classes=1000)
    net_m = net.reparameterize_model()
    print(net_m)