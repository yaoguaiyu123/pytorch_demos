
import torch
import torch.nn as nn
from typing import Callable, Optional
import numpy as np

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, stride=stride)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x:(B,C,H,W)
        max_result, _ = torch.max(x, dim=1, keepdim=True)  # 通过最大池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 最大值和对应的索引.
        avg_result = torch.mean(x, dim=1, keepdim=True)  # 通过平均池化压缩全局通道信息:(B,C,H,W)-->(B,1,H,W); 返回通道维度上的: 平均值
        result = torch.cat([max_result, avg_result], 1)  # 在通道上拼接两个矩阵:(B,2,H,W)
        output = self.conv(result)  # 然后重新降维为1维:(B,1,H,W)
        output = self.sigmoid(output)  # 通过sigmoid获得权重:(B,1,H,W)
        return output

# note 膨胀卷积 + 深度可分离卷积
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        padding = (kernel_size - 1) // 2 * dilation
        modules = [
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False, groups=in_channels),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)




# note Conv + Bn + Relu
class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups ,norm_cfg=None, act_cfg=None):
        super(ConvBnAct, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups,bias=False)
        self.bn = nn.BatchNorm2d(out_channels) if norm_cfg is None else norm_cfg
        self.act = nn.ReLU(inplace=True) if act_cfg is None else act_cfg
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x





# note 膨胀卷积需要满足 rates 缓慢增长的要求，不能就会出现稀疏的空洞，这里使用条形卷积改善这一点
class MFEBlockPlus(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(MFEBlockPlus, self).__init__()

        if atrous_rates is None:
            atrous_rates = [2, 4, 8]

        out_channels = in_channels
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = InceptionDWConv2d(in_channels)
        self.layer2 = ASPPConv(in_channels, out_channels, rate1)
        self.layer3 = ASPPConv(in_channels, out_channels, rate3)
        self.layer4 = ASPPConv(in_channels, out_channels, rate2)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
            #nn.Dropout(0.5))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()
        self.SE1 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1))
        self.SE2 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1))
        self.SE3 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1))
        self.SE4 = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(3, 1), padding=(1, 0), dilation=(1, 1))
    def forward(self, x):

        y0 = self.layer1(x)  # [B C H W]
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)

        y0_weight = self.SE1(self.gap(y0).permute(0,2,1,3)).permute(0,2,1,3)
        y1_weight = self.SE1(self.gap(y1).permute(0,2,1,3)).permute(0,2,1,3)
        y2_weight = self.SE1(self.gap(y2).permute(0,2,1,3)).permute(0,2,1,3)
        y3_weight = self.SE1(self.gap(y3).permute(0,2,1,3)).permute(0,2,1,3)

        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)  # [B,C,4,1]
        weight = self.softmax(self.Sigmoid(weight))  # sigmoid激活然后使用softmax得到归一化的权重

        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)

        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.conv1x1(x_att+x)  # 连接残差之后 1x1 卷积融合



# mod 两次 1x1 卷积
#  我在这边先使用 1x1 卷积升维的效果不好
class PCBlock(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(PCBlock, self).__init__()
        if atrous_rates is None:
            atrous_rates = [2, 4, 8]

        mid_channels = int(in_channels * 1)

        # note 多尺度卷积 (每个卷积都是深度可分离卷积)
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Sequential(
            InceptionDWConv2dPro(in_channels),
            nn.BatchNorm2d(in_channels),)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(inplace=True),)
        self.layer2 = ASPPConv(in_channels, in_channels, rate1, kernel_size=3)
        self.layer3 = ASPPConv(in_channels, in_channels, rate2, kernel_size=3)
        self.layer4 = ASPPConv(in_channels, in_channels, rate3, kernel_size=3)

        # note 3D卷积多尺度融合
        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), groups=in_channels, bias=False),
            nn.BatchNorm3d(in_channels),
            nn.SiLU(inplace=True),
        )

        # note 通道注意力
        self.se = SeBlock(mid_channels)

        # note 3d池化(平均池化才是融合，最大池化感觉是一个选择)
        self.pool_3d = nn.MaxPool3d(kernel_size=(5, 1, 1))

        # note 1x1通道卷积
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x):
        skip = x

        y0 = self.conv(self.layer1(x))  # [B C H W]
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)

        xm = torch.unsqueeze(x, -3) # [B C H W] -> [B C 1 H W]
        ym0 = torch.unsqueeze(y0, -3)
        ym1 = torch.unsqueeze(y1, -3)
        ym2 = torch.unsqueeze(y2, -3)
        ym3 = torch.unsqueeze(y3, -3)
        combine = torch.cat([xm, ym1, ym2, ym3, ym0], dim=2)  # [B C 5 H W]
        conv_3d = self.conv3d(combine)
        # conv_3d = self.conv3d_1x1(conv_3d)  # 1x1 卷积升维度

        out = self.pool_3d(conv_3d)  # [B C 1 H W]
        out = torch.squeeze(out, 2)  # [B C H W]

        out = self.se(out)

        out = self.conv1x1(out)  # 1x1 卷积降维

        return self.act(out + skip)  # 残差连接



# note 膨胀大核卷积
class MulDilatedConv(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        if kernel_size == 13:
            self.kernel_sizes = [5, 7, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [5, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [5, 5, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [5, 3, 3]
            self.dilates = [1, 2, 3]
        else:
            raise ValueError('Kernel Size Error(Only 7 9 11 13)')

        self.dilated_convs = nn.ModuleList()

        for k, r in zip(self.kernel_sizes, self.dilates):
            dil_conv = nn.Conv2d(channels, channels, kernel_size=k, stride=1,
                                 padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels, bias=False)
            self.dilated_convs.append(dil_conv)

    def forward(self, x):
        out = torch.zeros_like(x)
        for conv in self.dilated_convs:
            out += conv(x)

        return out

# note 膨胀大核卷积(以3x3卷积为核)
class MulDilatedConv_m(nn.Module):
    def __init__(self, channels, kernel_size):
        super().__init__()

        if kernel_size == 13:
            self.kernel_sizes = [3, 5, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 11:
            self.kernel_sizes = [3, 3, 3, 3, 3]
            self.dilates = [1, 2, 3, 4, 5]
        elif kernel_size == 9:
            self.kernel_sizes = [3, 3, 3, 3]
            self.dilates = [1, 2, 3, 4]
        elif kernel_size == 7:
            self.kernel_sizes = [3, 3, 3]
            self.dilates = [1, 2, 3]
        else:
            raise ValueError('Kernel Size Error(Only 7 9 11 13)')

        self.dilated_convs = nn.ModuleList()

        for k, r in zip(self.kernel_sizes, self.dilates):
            dil_conv = nn.Conv2d(channels, channels, kernel_size=k, stride=1,
                                 padding=(r * (k - 1) + 1) // 2, dilation=r, groups=channels, bias=False)
            self.dilated_convs.append(dil_conv)

    def forward(self, x):
        out = torch.zeros_like(x)
        for conv in self.dilated_convs:
            out += conv(x)

        return out


# note 膨胀大核卷积的 Block
class MulDilatedBlock(nn.Module):
    def __init__(self, in_channels, kernel_size=13, expend_size = None):
        super().__init__()
        expand_size = expend_size or int(in_channels * 2.4)

        self.bigconv = MulDilatedConv(in_channels, kernel_size)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act1 = nn.Hardswish()

        self.conv1 = nn.Conv2d(in_channels,expand_size,1, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = nn.Hardswish()

        self.se = SeBlock(expand_size)

        self.conv2 = nn.Conv2d(expand_size,in_channels,1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act3 = nn.Hardswish()


    def forward(self, x):
        r = x
        x = self.act1(self.bn1(self.bigconv(x)))
        x = self.act2(self.bn2(self.conv1(x)))
        x = self.se(x)
        x = self.bn3(self.conv2(x))
        x = self.act3(x + r)  # 残差连接 + 激活函数
        return x



class MulDilatedBlock1(nn.Module):
    def __init__(self, in_channels, kernel_size=13, expend_size = None):
        super().__init__()
        expand_size = expend_size or int(in_channels * 2.4)
        self.conv1 = nn.Conv2d(in_channels,expand_size,1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.Hardswish()


        self.bigconv = MulDilatedConv(expand_size, kernel_size)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = nn.Hardswish()


        self.se = SeBlock(expand_size)

        self.conv2 = nn.Conv2d(expand_size,in_channels,1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act3 = nn.Hardswish()


    def forward(self, x):
        r = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.bigconv(x)))
        x = self.se(x)
        x = self.bn3(self.conv2(x))
        x = self.act3(x + r)  # 残差连接 + 激活函数
        return x


class MulDilatedBlock2(nn.Module):
    def __init__(self, in_channels, kernel_size=13, expend_size = None):
        super().__init__()
        expand_size = expend_size or int(in_channels * 2.4)
        self.conv1 = nn.Conv2d(in_channels,expand_size,1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.Hardswish()


        self.bigconv = MulDilatedConv_m(expand_size, kernel_size)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = nn.Hardswish()


        self.se = SeBlock(expand_size)

        self.conv2 = nn.Conv2d(expand_size,in_channels,1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act3 = nn.Hardswish()


    def forward(self, x):
        r = x
        x = self.act1(self.bn1(self.conv1(x)))
        x = self.act2(self.bn2(self.bigconv(x)))
        x = self.se(x)
        x = self.bn3(self.conv2(x))
        x = self.act3(x + r)  # 残差连接 + 激活函数
        return x


# begin -----------------------------重参数化的Block-----------------------------
class MDCBlock(nn.Module):
    def __init__(self, in_channels, expand_size=None):
        super(MDCBlock, self).__init__()
        expand_size = expand_size or int(in_channels * 2.4)
        self.expand_size = expand_size
        self.conv1 = nn.Conv2d(in_channels, expand_size, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.SiLU(inplace=True)

        self.mid_conv1 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=7, padding=3, dilation=1,
            groups=expand_size, bias=False
        )
        self.mid_conv2 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=3, padding=1, dilation=1,
            groups=expand_size, bias=False
        )
        self.mid_conv3 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=3, padding=2, dilation=2,
            groups=expand_size, bias=False
        )
        self.mid_conv4 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=3, padding=3, dilation=3,
            groups=expand_size, bias=False
        )

        self.mid_bn1 = nn.BatchNorm2d(expand_size)
        self.mid_bn2 = nn.BatchNorm2d(expand_size)
        self.mid_bn3 = nn.BatchNorm2d(expand_size)
        self.mid_bn4 = nn.BatchNorm2d(expand_size)
        self.act2 = nn.SiLU(inplace=True)

        self.se = SeBlock(expand_size)

        self.conv2 = nn.Conv2d(expand_size, in_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.act3 = nn.SiLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.conv1(x)))

        x1 = self.mid_bn1(self.mid_conv1(x))
        x2 = self.mid_bn2(self.mid_conv2(x))
        x3 = self.mid_bn3(self.mid_conv3(x))
        x4 = self.mid_bn4(self.mid_conv4(x))

        x = x1 + x2 + x3 + x4 + x
        x = self.act2(x)
        x = self.se(x)
        x = self.bn3(self.conv2(x))
        x = self.act3(x + identity)
        return x

    def reparameterize(self):
        with torch.no_grad():
            # 重参数化中间的卷积和对应的 BN 层
            mid_convs = [self.mid_conv1, self.mid_conv2, self.mid_conv3, self.mid_conv4]
            mid_bns = [self.mid_bn1, self.mid_bn2, self.mid_bn3, self.mid_bn4]
            expanded_weights = []
            expanded_biases = []

            for conv, bn in zip(mid_convs, mid_bns):
                # 融合每个分支上的卷积和 BN 参数
                weight, bias = self._fuse_conv_bn(conv, bn)
                expanded_weight = self._expand_weight(weight, conv.kernel_size[0], conv.dilation[0])
                expanded_weights.append(expanded_weight)
                expanded_biases.append(bias)

            # 恒等映射
            identity_weight = torch.zeros_like(expanded_weights[0])
            center = 3
            identity_weight[:, 0, center, center] = 1.0
            identity_bias = torch.zeros_like(expanded_biases[0])

            # 合并参数
            merged_weight = sum(expanded_weights) + identity_weight
            merged_bias = sum(expanded_biases) + identity_bias

            # 创建新的融合后的卷积层
            self.reparam_conv = nn.Conv2d(
                in_channels=self.expand_size,
                out_channels=self.expand_size,
                kernel_size=7,
                padding=3,
                groups=self.expand_size,
                bias=True
            )
            self.reparam_conv.weight.data.copy_(merged_weight)
            self.reparam_conv.bias.data.copy_(merged_bias)

            # 删除原有的中间卷积层和 BN 层
            del self.mid_conv1, self.mid_conv2, self.mid_conv3, self.mid_conv4
            del self.mid_bn1, self.mid_bn2, self.mid_bn3, self.mid_bn4

            # 重参数化 conv1 + bn1 和 conv2 + bn3
            self.reparam_conv1 = self._fuse_conv_bn_layer(self.conv1, self.bn1)
            self.reparam_conv2 = self._fuse_conv_bn_layer(self.conv2, self.bn3)

            # 删除原始的 conv1、bn1 和 conv2、bn3
            del self.conv1, self.bn1
            del self.conv2, self.bn3

            # 修改 forward 方法
            self.forward = self.forward_reparam

    # note 重参数化后的 forward
    def forward_reparam(self, x):
        identity = x
        x = self.act1(self.reparam_conv1(x))
        x = self.reparam_conv(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.reparam_conv2(x)
        x = self.act3(x + identity)
        return x

    def _fuse_conv_bn(self, conv, bn):
        conv_weight = conv.weight.clone()
        conv_bias = torch.zeros(conv_weight.size(0), device=conv_weight.device) if conv.bias is None else conv.bias.clone()

        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_running_mean = bn.running_mean
        bn_running_var = bn.running_var
        bn_eps = bn.eps

        std = (bn_running_var + bn_eps).sqrt()
        t = bn_weight / std

        fused_weight = conv_weight * t.reshape(-1, 1, 1, 1)
        fused_bias = bn_bias + (conv_bias - bn_running_mean) * t

        return fused_weight, fused_bias

    def _fuse_conv_bn_layer(self, conv, bn):
        weight, bias = self._fuse_conv_bn(conv, bn)
        reparam_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True
        )
        reparam_conv.weight.data.copy_(weight)
        reparam_conv.bias.data.copy_(bias)
        return reparam_conv

    def _expand_weight(self, weight, kernel_size, dilation):
        groups = weight.size(0)
        expanded_weight = torch.zeros((groups, 1, 7, 7), device=weight.device, dtype=weight.dtype)
        center = 3  # 7x7 卷积核的中心索引

        if dilation == 1 and kernel_size == 3:
            start = center - 1
            expanded_weight[:, :, start:start + 3, start:start + 3] = weight
        elif dilation == 1 and kernel_size == 7:
            expanded_weight[:, :, :, :] = weight
        else:
            effective_kernel_size = (kernel_size - 1) * dilation + 1
            start = center - (effective_kernel_size // 2)
            indices = range(kernel_size)
            for i in indices:
                for j in indices:
                    idx = start + i * dilation
                    idy = start + j * dilation
                    if 0 <= idx < 7 and 0 <= idy < 7:
                        expanded_weight[:, 0, idx, idy] = weight[:, 0, i, j]
        return expanded_weight






# note 下采样模块
class ADown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = nn.Conv2d(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = nn.Conv2d(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 1, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)

        return torch.cat((x1, x2), 1)


# note 相对于MFEBlock01将条形卷积换为大核卷积
class MFEBlockPro1(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(MFEBlockPro1, self).__init__()

        if atrous_rates is None:
            atrous_rates = [2, 4, 8]

        out_channels = in_channels
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Conv2d(in_channels,in_channels, 7,bias=False, padding=3, groups=in_channels)
        self.conv = nn.Conv2d(in_channels,in_channels,1,bias=False)
        self.layer2 = ASPPConv(in_channels, in_channels, rate1, kernel_size=3)
        self.layer3 = ASPPConv(in_channels, in_channels, rate3, kernel_size=5)
        self.layer4 = ASPPConv(in_channels, in_channels, rate2, kernel_size=5)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim = 2)
        self.Sigmoid = nn.Sigmoid()
        self.attn1 = SeAttn(in_channels)
        self.attn2 = SeAttn(in_channels)
        self.attn3 = SeAttn(in_channels)
        self.attn4 = SeAttn(in_channels)
    def forward(self, x):

        y0 = self.conv(self.layer1(x))  # [B C H W]
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)
        y3 = self.layer4(y2+x)

        y0_weight = self.attn1(y0)
        y1_weight = self.attn2(y1)
        y2_weight = self.attn3(y2)
        y3_weight = self.attn4(y3)

        weight = torch.cat([y0_weight,y1_weight,y2_weight,y3_weight],2)  # [B,C,4,1]
        weight = self.softmax(self.Sigmoid(weight))  # sigmoid激活然后使用softmax得到归一化的权重

        y0_weight = torch.unsqueeze(weight[:,:,0],2)
        y1_weight = torch.unsqueeze(weight[:,:,1],2)
        y2_weight = torch.unsqueeze(weight[:,:,2],2)
        y3_weight = torch.unsqueeze(weight[:,:,3],2)

        x_att = y0_weight*y0+y1_weight*y1+y2_weight*y2+y3_weight*y3
        return self.conv1x1(x_att+x)  # 连接残差之后 1x1 卷积融合


# note 下采样模块
class ADown1(nn.Module):
    def __init__(self, in_channels, out_channels):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        expend_channels = in_channels * 4
        self.conv1 = nn.Conv2d(in_channels, expend_channels, 1, 1,0, bias= False)
        self.cv1 = nn.Conv2d(expend_channels // 2, expend_channels // 2, 5, 2, 2, groups=expend_channels // 2)
        self.cv2 = nn.Conv2d(expend_channels // 2, expend_channels // 2, 3, 1, 1, groups=expend_channels // 2)
        self.conv2 = nn.Conv2d(expend_channels, out_channels, 1, 1,0, bias= False)
        self.bn1 = nn.BatchNorm2d(expend_channels)
        self.bn2 = nn.BatchNorm2d(expend_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.Hardswish()
        self.act2 = nn.Hardswish()
        self.act3 = nn.Hardswish()
        self.convd = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, groups=in_channels, stride=2, padding=2,
                      bias=False),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_channels)
        )
    def forward(self, x):
        r = x
        x = self.act1(self.bn1(self.conv1(x)))  # 1x1 卷积升维
        x = torch.nn.functional.avg_pool2d(x, 3, 1, 1, False, True)
        x1, x2 = x.chunk(2, 1)  # 沿通道平均分成两块
        x1 = self.cv1(x1)  # 5x5 卷积下采样
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)  # 最大池化下采样
        x2 = self.cv2(x2)  # 3x3 卷积
        out = torch.cat((x1, x2), dim=1)
        out = self.act2(self.bn2(out))
        out = self.bn3(self.conv2(out))  # 1x1 卷积降维
        r = self.convd(r)
        out = out + r
        return self.act3(out)


# note 进一步改进的block
class MFEBlockPro2(nn.Module):
    def __init__(self, in_channels, out_channels=None, atrous_rates=None):
        super(MFEBlockPro2, self).__init__()

        if atrous_rates is None:
            atrous_rates = [2, 2, 3, 4]

        out_channels = out_channels or in_channels
        rate1, rate2, rate3, rate4 = tuple(atrous_rates)
        # mod 改进是从条形卷积变为正常卷积
        self.layer1 = nn.Conv2d(in_channels,in_channels,9,1, padding=4,groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv_m = nn.Conv2d(in_channels,in_channels,1,bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.act2 = nn.Hardswish()

        self.layer2 = ASPPConv(in_channels, in_channels, rate1, kernel_size=3)
        self.layer3 = ASPPConv(in_channels, in_channels, rate2, kernel_size=3)
        self.layer4 = ASPPConv(in_channels, in_channels, rate3, kernel_size=5)
        self.layer5 = ASPPConv(in_channels, in_channels, rate4, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(in_channels)
        self.bn4 = nn.BatchNorm2d(in_channels)
        self.bn5 = nn.BatchNorm2d(in_channels)
        self.act3 = nn.Hardswish()

        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
        self.attn = SeBlock(in_channels)
    def forward(self, x):
        y0 = self.bn1(self.layer1(x))   # 空间可分离卷积
        y0 = self.act2(self.bn2(self.conv_m(y0)))  # 1x1 卷积
        y1 = self.bn3(self.layer2(y0))
        y2 = self.bn4(self.layer3(y0))
        y3 = self.layer4(y0 + y1)
        y4 = self.layer5(y0 + y2)
        yy = y0 + y1 + y2 + y3 + y4
        yy = self.act3(self.bn5(yy))
        # sp 添加注意力机制
        yy = self.attn(yy)
        out = self.conv1x1(yy + x)
        return out


# note 进一步改进的(受到shufflenet启发)
class MFEBlockPro3(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(MFEBlockPro3, self).__init__()

        if atrous_rates is None:
            atrous_rates = [2, 4, 8]
        gc = int(in_channels * 0.4)
        gc1 = in_channels - gc
        self.split_index = [gc ,in_channels - gc]

        out_channels = gc1
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.layer1 = nn.Conv2d(gc1, gc1, 7,1, groups=gc1, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(gc1)
        self.conv = nn.Conv2d(gc1,gc1,1,bias=False)
        self.bn2 = nn.BatchNorm2d(gc1)
        self.act2 = nn.Hardswish()
        self.layer2 = ASPPConv(gc1, gc1, rate1, kernel_size=3)
        self.layer3 = ASPPConv(gc1, gc1, rate2, kernel_size=5)
        self.layer4 = ASPPConv(gc1, gc1, rate3, kernel_size=5)
        self.bn3 = nn.BatchNorm2d(gc1)
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(gc1, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),)
        self.attn = SeBlock(gc1)

    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1]
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        # print("形状1: ",index.shape)
        index = index.view(-1)
        # print("形状2: ",index.shape)

        index = index.to(features.device)

        features = features[:, index]
        return features


    def forward(self, x):
        x1, x2 = torch.split(x,self.split_index, dim=1)

        y0 = self.act2(self.bn2(self.conv(self.bn1(self.layer1(x2)))))  # [B C H W]
        y1 = self.layer2(y0)
        y2 = self.layer3(y0)
        y3 = self.layer4(y0)
        yy = y0 + y1 + y2 + y3
        yy =  self.bn3(yy)
        yy = self.attn(yy)
        out = self.conv1x1(yy)
        # print(out.shape, x2.shape)
        out = out + x2
        out = torch.cat([x1, out], dim=1)
        return self._channel_shuffle(out)



# note 继续改进的 MFEBlock
class MFEBlockPro4(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(MFEBlockPro4, self).__init__()

        if atrous_rates is None:
            atrous_rates = [2, 4]

        rate1, rate2 = tuple(atrous_rates)
        self.layer1 = InceptionDWConv2dPro2(in_channels)
        self.layer2 = ASPPConv(in_channels, in_channels, rate1, kernel_size=3)
        self.layer3 = ASPPConv(in_channels, in_channels, rate2, kernel_size=5)

        self.conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.Hardswish()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.softmax = nn.Softmax(dim=2)
        self.Sigmoid = nn.Sigmoid()
        self.attn1 = ECAAttention(5)
        self.attn2 = ECAAttention(5)
        self.attn3 = ECAAttention(5)
    def forward(self, x):

        y0 = self.layer1(x) # 多尺度条形卷积
        y1 = self.layer2(y0+x)
        y2 = self.layer3(y1+x)

        y0_weight = self.attn1(y0)
        y1_weight = self.attn2(y1)
        y2_weight = self.attn3(y2)

        weight = torch.cat([y0_weight, y1_weight, y2_weight], 2)
        weight = self.softmax(self.Sigmoid(weight))

        y0_weight = torch.unsqueeze(weight[:, :, 0], 2)
        y1_weight = torch.unsqueeze(weight[:, :, 1], 2)
        y2_weight = torch.unsqueeze(weight[:, :, 2], 2)

        out = y0_weight * y0 + y1_weight * y1 + y2_weight * y2
        out = self.bn(self.conv(out))
        return self.act(out + x)  # 残差连接





# note 多维度深度可分离卷积(3x3卷积 + 1x11卷积 + 11x1卷积)
class InceptionDWConv2d(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.25):
        super().__init__()

        gc = int(in_channels * branch_ratio)
        self.dwconv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2, groups=gc)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc)
        self.split_indexes = [in_channels - 3 * gc, gc, gc, gc]

    def forward(self, x):
        x_id, x_hw, x_w, x_h = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((x_id, self.dwconv_hw(x_hw), self.dwconv_w(x_w), self.dwconv_h(x_h)), dim=1)


class InceptionDWConv2dPro(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()

        gc2 = int(in_channels * 0.7)
        gc3 = in_channels - gc2
        self.dwconv_hw = nn.Conv2d(gc2, gc2, square_kernel_size, padding=square_kernel_size // 2, groups=gc2)
        self.dwconv_w = nn.Conv2d(gc3, gc3, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=gc3)
        self.dwconv_h = nn.Conv2d(gc3, gc3, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=gc3)
        self.split_indexes = [gc2, gc3]

    def forward(self, x):
        x2, x3 = torch.split(x, self.split_indexes, dim=1)
        return torch.cat((self.dwconv_hw(x2), self.dwconv_h(self.dwconv_w(x3))),dim=1)



class InceptionDWConv2dPro2(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()

        self.dwconv_3x3 = nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size // 2, groups=in_channels)
        self.dwconv_w = nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=in_channels)
        self.dwconv_h = nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=in_channels)

    def forward(self, x):
        x1 = self.dwconv_3x3(x)
        x2 = self.dwconv_h(self.dwconv_w(x))
        return x1 + x2


# note 空间可分离卷积 + 1x1卷积融合
class InceptionDWConv2dPro1(nn.Module):

    def __init__(self, in_channels, band_kernel_size=11):
        super().__init__()
        self.dwconv_w = nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=in_channels, bias=False)
        self.dwconv_h = nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.act = nn.Hardswish()
        self.conv1x1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)
    def forward(self, x):
        x = self.dwconv_w(self.dwconv_h(x))
        x = self.bn1(x)
        x = self.conv1x1(x)
        x = self.act(self.bn2(x))
        return x



# 空间可分离的卷积
class SSDWConv2d(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv_w = nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size // 2),
                                  groups=in_channels)
        self.dwconv_h = nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size // 2, 0),
                                  groups=in_channels)

    def forward(self, x):
        return self.dwconv_h(self.dwconv_w(x))



class DoubleConv1x1(nn.Module):
    def __init__(
            self, in_channels, hidden_channels=None, out_channels=None, act_layer=nn.ReLU,
            norm_layer=None, bias=False, drop=0.):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels  # 默认为 in_features * 4

        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
        self.norm = norm_layer(hidden_channels) if norm_layer else nn.Identity()
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.fc1(x)  # 1x1 卷积 升维
        x = self.norm(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)  # 1x1卷积 降维
        return x


class InceptionNeXtBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels = None,
            mlp_ratio=4,
            ls_init_value=1e-6,
            branch_ratio=0.25
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.inceptionDWConv2d = InceptionDWConv2d(in_channels=in_channels,branch_ratio=branch_ratio)
        self.norm = nn.BatchNorm2d(in_channels)
        self.mlp = DoubleConv1x1(in_channels, int(mlp_ratio * in_channels),out_channels=out_channels, act_layer=nn.GELU)
        self.gamma = nn.Parameter(ls_init_value * torch.ones(out_channels)) if ls_init_value else None

    def forward(self, x):
        residual = x
        x = self.inceptionDWConv2d(x)  # 多维深度可分离卷积
        x = self.norm(x)
        x = self.mlp(x)    # 2次1x1卷积
        # sp 可学习参数调整
        if self.gamma is not None:
            x = x.mul(self.gamma.reshape(1, -1, 1, 1))  # reshape: [C] -> [1 C 1 1]

        if x.shape == residual.shape:
            x = x + residual
        return x


# note SE注意力
class SeAttn(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeAttn, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return self.se(x)



# note SE注意力模块
class SeBlock(nn.Module):
    def __init__(self, in_size, reduction=4):
        super(SeBlock, self).__init__()
        expand_size = max(in_size // reduction, 8)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False),
            nn.BatchNorm2d(expand_size),
            nn.ReLU(inplace=True),
            nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False),
            nn.Hardsigmoid()
        )

    def forward(self, x):
        return x * self.se(x)



# note 重参数化 Block
class Block(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block, self).__init__()
        self.stride = stride
        self.expand_size = expand_size

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # 3x3 卷积
        self.conv2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=expand_size,  # 深度卷积
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(expand_size)

        # 1x3 卷积
        self.conv2_1 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=(kernel_size, 1),
            stride=stride,
            padding=(kernel_size // 2, 0),
            groups=expand_size,  # 深度卷积
            bias=False
        )
        self.bn2_1 = nn.BatchNorm2d(expand_size)

        # 3x1 卷积
        self.conv2_2 = nn.Conv2d(
            expand_size,
            expand_size,
            kernel_size=(1, kernel_size),
            stride=stride,
            padding=(0, kernel_size // 2),
            groups=expand_size,  # 深度卷积
            bias=False
        )
        self.bn2_2 = nn.BatchNorm2d(expand_size)

        self.act2 = act(inplace=True)
        self.se = SeBlock(expand_size) if se else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=in_size,
                    kernel_size=3,
                    groups=in_size,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_size,
                    out_channels=out_size,
                    kernel_size=3,
                    groups=in_size,
                    stride=2,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))
        x1 = self.bn2(self.conv2(out))
        x2 = self.bn2_1(self.conv2_1(out))
        x3 = self.bn2_2(self.conv2_2(out))
        out = self.act2(x1 + x2 + x3)
        out = self.se(out)
        out = self.bn3(self.conv3(out))

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

    # 合并卷积和 BN
    def _fuse_conv_bn(self, conv, bn):
        conv_weight = conv.weight.clone()
        if conv.bias is not None:
            conv_bias = conv.bias.clone()
        else:
            conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device)

        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_running_mean = bn.running_mean
        bn_running_var = bn.running_var
        bn_eps = bn.eps

        std = (bn_running_var + bn_eps).sqrt()
        t = bn_weight / std

        fused_weight = conv_weight * t.reshape(-1, 1, 1, 1)
        fused_bias = bn_bias + (conv_bias - bn_running_mean) * t

        return fused_weight, fused_bias

    # 用于 conv + bn 的重参数化
    def _fuse_conv_bn_layer(self, conv, bn):
        weight, bias = self._fuse_conv_bn(conv, bn)
        reparam_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True
        )
        reparam_conv.weight.data.copy_(weight)
        reparam_conv.bias.data.copy_(bias)
        return reparam_conv

    def reparameterize(self):
        with torch.no_grad():
            # 重参数化 conv1 + bn1 和 conv3 + bn3
            self.rep_conv1 = self._fuse_conv_bn_layer(self.conv1, self.bn1)
            self.rep_conv3 = self._fuse_conv_bn_layer(self.conv3, self.bn3)

            # 对于 conv2 的三个分支，需要融合成一个卷积
            # 1. 融合每个分支的 conv 和 bn
            weight_3x3, bias_3x3 = self._fuse_conv_bn(self.conv2, self.bn2)
            weight_1x3, bias_1x3 = self._fuse_conv_bn(self.conv2_1, self.bn2_1)
            weight_3x1, bias_3x1 = self._fuse_conv_bn(self.conv2_2, self.bn2_2)

            # 2. 将 1x3 和 3x1 的卷积核扩展为 3x3，并适当填充
            # 创建与 3x3 卷积相同形状的空权重
            weight_1x3_expanded = torch.zeros_like(weight_3x3)
            weight_1x3_expanded[:, :, :, 1] = weight_1x3[:, :, :, 0]

            weight_3x1_expanded = torch.zeros_like(weight_3x3)
            weight_3x1_expanded[:, :, 1, :] = weight_3x1[:, :, 0, :]

            # 3. 将所有权重和偏置相加
            merged_weight = weight_3x3 + weight_1x3_expanded + weight_3x1_expanded
            merged_bias = bias_3x3 + bias_1x3 + bias_3x1

            # 4. 创建新的融合后的卷积层
            self.rep_conv2 = nn.Conv2d(
                self.expand_size,
                self.expand_size,
                kernel_size=3,
                stride=self.conv2.stride,
                padding=self.conv2.padding,
                groups=self.conv2.groups,
                bias=True
            )
            self.rep_conv2.weight.data.copy_(merged_weight)
            self.rep_conv2.bias.data.copy_(merged_bias)

            # 删除原始的卷积和 BN 层
            del self.conv1, self.bn1
            del self.conv2, self.bn2
            del self.conv2_1, self.bn2_1
            del self.conv2_2, self.bn2_2
            del self.conv3, self.bn3

            # 修改 forward 方法
            self.forward = self.forward_reparam

    def forward_reparam(self, x):
        skip = x
        out = self.act1(self.rep_conv1(x))
        out = self.act2(self.rep_conv2(out))
        out = self.se(out)
        out = self.rep_conv3(out)

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)



class ElementScale(nn.Module):
    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale


# note 空间注意力下采样模块
class SPADownBlock(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, attention, stride):
        super(SPADownBlock, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.spa = SpatialAttention(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)  # sp 深度可分离卷积

        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        # note SE注意力机制
        self.attn = SeBlock(expand_size) if attention else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip_method = None
        if stride == 2 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))  # sp  1x1 卷积  升维
        spatten = self.spa(out)  # 空间注意力

        out = self.act2(self.bn2(self.conv2(out)))  # sp  3x3 或 5x5 卷积

        out = self.attn(out)  # 通道注意力
        out = spatten * out   # 空间注意力
        out = self.bn3(self.conv3(out))  # sp 1x1 卷积 降维

        if self.skip_method is not None:
            skip = self.skip_method(skip)

        return self.act3(out + skip)

    # 合并卷积和 BN
    def _fuse_conv_bn(self, conv, bn):
        conv_weight = conv.weight.clone()
        if conv.bias is not None:
            conv_bias = conv.bias.clone()
        else:
            conv_bias = torch.zeros(conv.weight.size(0), device=conv.weight.device)

        bn_weight = bn.weight
        bn_bias = bn.bias
        bn_running_mean = bn.running_mean
        bn_running_var = bn.running_var
        bn_eps = bn.eps

        std = (bn_running_var + bn_eps).sqrt()
        t = bn_weight / std

        fused_weight = conv_weight * t.reshape(-1, 1, 1, 1)
        fused_bias = bn_bias + (conv_bias - bn_running_mean) * t

        return fused_weight, fused_bias

    # 用于 conv + bn 的重参数化
    def _fuse_conv_bn_layer(self, conv, bn):
        weight, bias = self._fuse_conv_bn(conv, bn)
        reparam_conv = nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True
        )
        reparam_conv.weight.data.copy_(weight)
        reparam_conv.bias.data.copy_(bias)
        return reparam_conv

    def reparameterize(self):
        with torch.no_grad():
            # 重参数化每个卷积和 BN 层
            self.rep_conv1 = self._fuse_conv_bn_layer(self.conv1, self.bn1)
            self.rep_conv2 = self._fuse_conv_bn_layer(self.conv2, self.bn2)
            self.rep_conv3 = self._fuse_conv_bn_layer(self.conv3, self.bn3)

            # 删除原始的卷积和 BN 层
            del self.conv1, self.bn1
            del self.conv2, self.bn2
            del self.conv3, self.bn3

            # 修改 forward 方法
            self.forward = self.forward_reparam

    def forward_reparam(self, x):
        skip = x

        out = self.act1(self.rep_conv1(x))
        spatten = self.spa(out)  # 空间注意力

        out = self.act2(self.rep_conv2(out))

        out = self.attn(out)  # 通道注意力
        out = spatten * out   # 空间注意力
        out = self.rep_conv3(out)

        if self.skip_method is not None:
            skip = self.skip_method(skip)

        return self.act3(out + skip)


# note 部分卷积
class Block02(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, attention, stride, ratio=0.5):
        super(Block02, self).__init__()
        self.ratio = ratio
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        size1 = int(expand_size * ratio)
        size2 = int(expand_size - size1)
        self.split_indexes = [size1, size2]
        self.conv3x3 = nn.Conv2d(size2, size2, kernel_size=kernel_size, stride=stride,
                               padding=3 // 2, groups=size2, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)


        # note ECA注意力机制
        self.attn = ECAAttention(5) if attention else nn.Identity()
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip_method = None
        if stride == 1 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )


    def forward(self, x):
        skip = x
        out = self.act1(self.bn1(self.conv1(x)))  # sp  1x1 卷积  升维
        # 多尺度部分卷积
        x1,x2 = torch.split(out, self.split_indexes, dim=1)
        x2 = self.conv3x3(x2)
        out = torch.cat([x1,x2], dim=1)
        out = self.act2(self.bn2(out))

        out = self.attn(out)
        out = self.bn3(self.conv3(out))  # sp 1x1 卷积 降维

        if self.skip_method is not None:
            skip = self.skip_method(skip)

        return self.act3(out + skip)   # 残差链接


# note 用于下采样的block，灵感来源于 mobilenetV4
class DownBlock(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, attention, stride):
        super(DownBlock, self).__init__()
        self.stride = stride

        self.conv0 = nn.Conv2d(in_size, in_size, kernel_size=kernel_size, stride=1,
                               padding=kernel_size // 2, groups=in_size, bias=False)
        self.bn0 = nn.BatchNorm2d(in_size)

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv3x3 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)


        # note ECA注意力机制
        self.attn = ECAAttention(5) if attention else nn.Identity()
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip_method = None
        if stride == 1 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )


    def forward(self, x):
        skip = x
        out = self.bn0(self.conv0(x))
        out = self.act1(self.bn1(self.conv1(out)))
        out = self.conv3x3(out)
        out = self.act2(self.bn2(out))

        out = self.attn(out)
        out = self.bn3(self.conv3(out))

        if self.skip_method is not None:
            skip = self.skip_method(skip)

        return self.act3(out + skip)   # 残差连接


# note 结合了多尺度膨胀卷积+ mobileNetV4的思想
class DownBlock1(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, attention, stride):
        super(DownBlock1, self).__init__()
        self.stride = stride
        self.conv0 = MulDilatedConv(in_size,7)  # mod 修改 9->7
        self.bn0 = nn.BatchNorm2d(in_size)

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)
        self.spattn = SpatialAttention(stride=2)  # 空间注意力(伴有下采样)

        self.conv5x5 = nn.Conv2d(expand_size, expand_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=expand_size, bias=False)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)


        # 通道注意力
        self.attn = SeAttn(in_size=expand_size) if attention else nn.Identity()
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip_method = nn.Sequential(
            nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(in_size),
            nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
            nn.BatchNorm2d(out_size)
        )

    def forward(self, x):
        skip = x
        out = self.bn0(self.conv0(x))  # 5x5初次卷积
        out = self.act1(self.bn1(self.conv1(out)))  # 1x1 卷积
        spattn = self.spattn(out)

        out = self.conv5x5(out)  # 5x5 下采样
        out = self.act2(self.bn2(out))

        # mod 有机会还可以魔改一下( 空间注意力 + 多尺度)
        out = out * self.attn(out)  # 通道注意力
        out = out * spattn # 空间注意力
        out = self.bn3(self.conv3(out))  # 1x1 卷积降维

        if self.skip_method is not None:
            skip = self.skip_method(skip)

        return self.act3(out + skip)   # 残差连接


# note 卷积 + BN + Act
class ConvBNAct(nn.Module):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU

        self.conv = nn.Conv2d(in_channels=in_planes,
                              out_channels=out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)

        self.bn = norm_layer(out_planes)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result


#  note 大核卷积的Block
class Block03(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act,attention, stride):
        super(Block03, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        self.conv2 = InceptionDWConv2dPro(expand_size, kernel_size, 11)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        # note SE注意力
        self.attn = SeBlock(expand_size) if attention else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip_method = None
        if stride == 1 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip_method = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )


    def forward(self, x):
        skip = x

        out = self.act1(self.bn1(self.conv1(x)))  # sp  1x1 卷积  升维
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.attn(out)
        out = self.bn3(self.conv3(out))  # sp 1x1 卷积 降维

        if self.skip_method is not None:
            skip = self.skip_method(skip)

        return self.act3(out + skip)


# note 多分支卷积 + shuffle通道的Block
class Block04(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act,attention, stride, shuffle_rates=0.3):
        super(Block04, self).__init__()
        self.stride = stride
        # sp 第一次划分
        gc1 = int(in_size * (1-shuffle_rates))
        expand_size = int(expand_size * shuffle_rates)
        self.split_index1 = [gc1 ,in_size-gc1]

        self.conv1 = nn.Conv2d(in_size-gc1, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # sp 第二次划分
        gc2 = int(expand_size // 2)
        self.split_index2 = [gc2, expand_size - gc2]
        self.conv2_1 = nn.Conv2d(gc2, gc2, kernel_size, 1, padding=kernel_size // 2, groups=gc2)
        self.conv2_2 = SSDWConv2d(expand_size - gc2, kernel_size, 11)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        # note SE注意力
        self.attn = SeBlock(expand_size) if attention else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, in_size-gc1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_size-gc1)
        self.act3 = act(inplace=True)

    # note 打乱通道
    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1]
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        index = index.view(-1)

        index = index.to(features.device)

        features = features[:, index]
        return features

    def forward(self, x):
        x1, x2 =torch.split(x, self.split_index1, dim=1)
        skip = x2

        x2 = self.act1(self.bn1(self.conv1(x2)))  # sp  1x1 卷积  升维

        x2_1,x2_2 = torch.split(x2 , self.split_index2, dim=1)
        x2_1 = self.conv2_1(x2_1)
        x2_2 = self.conv2_2(x2_2)
        x2 = torch.cat([x2_1,x2_2], dim=1)
        x2 = self.act2(self.bn2(x2))
        x2 = self.attn(x2)

        x2 = self.bn3(self.conv3(x2))  # sp 1x1 卷积 降维
        x2 = x2 + skip
        self.act3(x2)
        output = torch.cat([x1, x2], dim=1)

        return self._channel_shuffle(output)


# note 受到 shuffleNet 启发的改进
class Block05(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act,attention, stride, shuffle_rates=0.5):
        super(Block05, self).__init__()
        self.stride = stride
        # sp 第一次划分
        gc1 = int(in_size * (1-shuffle_rates))
        expand_size = int(expand_size * shuffle_rates)
        self.split_index1 = [gc1 ,in_size-gc1]

        self.conv1 = nn.Conv2d(in_size-gc1, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # sp 第二次划分
        gc2 = int(expand_size // 2)
        self.split_index2 = [gc2, expand_size - gc2]
        # 3x3 卷积
        self.conv2_1 = nn.Conv2d(gc2, gc2, kernel_size, 1, padding=kernel_size // 2, groups=gc2)
        kernel_size = kernel_size + 2
        # 5x5 扩张卷积
        self.conv2_2 = nn.Conv2d(expand_size - gc2, expand_size - gc2, kernel_size=kernel_size, stride=1,
                                 padding=(kernel_size // 2) * 2, dilation=2, groups=expand_size - gc2)
        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        # note SE注意力
        self.attn = SeBlock(expand_size) if attention else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, in_size-gc1, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_size-gc1)
        self.act3 = act(inplace=True)

    # note 打乱通道
    #  g是分组数
    def _channel_shuffle(self, features, g=2):
        channels = features.size()[1]
        index = torch.from_numpy(np.asarray([i for i in range(channels)]))
        index = index.view(-1, g).t().contiguous()
        # print("形状1: ",index.shape)
        index = index.view(-1)
        # print("形状2: ",index.shape)

        index = index.to(features.device)

        features = features[:, index]
        return features

    def forward(self, x):
        x1, x2 =torch.split(x, self.split_index1, dim=1)
        skip = x2

        x2 = self.act1(self.bn1(self.conv1(x2)))  # sp  1x1 卷积  升维

        x2_1,x2_2 = torch.split(x2 , self.split_index2, dim=1)
        x2_1 = self.conv2_1(x2_1)
        x2_2 = self.conv2_2(x2_2)
        x2 = torch.cat([x2_1,x2_2], dim=1)
        x2 = self.act2(self.bn2(x2))
        x2 = self.attn(x2)

        x2 = self.bn3(self.conv3(x2))  # sp 1x1 卷积 降维
        x2 = x2 + skip
        self.act3(x2)
        output = torch.cat([x1, x2], dim=1)

        return self._channel_shuffle(output)  # note 打乱通道


# note 受到moganet启发的改进, 一个深度卷积之后两个膨胀卷积
class Block06(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act,attention, stride):
        super(Block06, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # sp 划分
        gc1 = int(expand_size * 0.5)
        gc2 = (expand_size - gc1) // 2
        gc3 = expand_size - gc1 - gc2
        self.split_index = [gc1, gc2, gc3]
        # 3x3 深度卷积
        self.conv2 = nn.Conv2d(expand_size, expand_size, kernel_size=3, stride=1, padding=1, groups=expand_size)
        # 3x3 扩张卷积
        self.conv2_1 = nn.Conv2d(gc2, gc2, kernel_size=3, stride=1, padding=2, groups=gc2, dilation=2)
        # 5x5 扩张卷积
        self.conv2_2 = nn.Conv2d(gc3, gc3, kernel_size=5, stride=1, padding=4,
                                 dilation=2, groups=gc3)

        self.bn2 = nn.BatchNorm2d(expand_size)
        self.act2 = act(inplace=True)

        # note SE注意力
        self.attn = SeBlock(expand_size) if attention else nn.Identity()

        self.conv3 = nn.Conv2d(expand_size, in_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(in_size)
        self.act3 = act(inplace=True)

    def forward(self, x):
        skip = x
        x = self.act1(self.bn1(self.conv1(x)))  # sp  1x1 卷积  升维
        x = self.conv2(x)
        x1, x2, x3 = torch.split(x , self.split_index, dim=1)  # 通道划分
        x2 = self.conv2_1(x2)
        x3 = self.conv2_2(x3)
        catx = torch.cat([x1, x2, x3], dim=1)
        x = self.act2(self.bn2(catx))
        x = self.attn(x)  # sp 注意力机制

        x = self.bn3(self.conv3(x))  # sp 1x1 卷积 降维
        x = x + skip
        x = self.act3(x)
        return x


# note 类似ConvNext的现代化的Block
class Block07(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(Block07, self).__init__()
        self.stride = stride

        self.conv1 = nn.Conv2d(in_size, in_size, kernel_size=kernel_size, stride=stride,
                               padding=kernel_size // 2, groups=in_size, bias=False)
        self.norm = nn.BatchNorm2d(in_size)

        self.pwconv1 = nn.Linear(in_size, expand_size)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()

        self.pwconv2 = nn.Linear(expand_size, out_size)

        self.se = SeBlock(out_size) if se else nn.Identity()

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=in_size, kernel_size=3, groups=in_size, stride=2, padding=1,
                          bias=False),
                nn.BatchNorm2d(in_size),
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=True),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2 and in_size == out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels=in_size, out_channels=out_size, kernel_size=3, groups=in_size, stride=2,
                          padding=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

    def forward(self, x):
        skip = x
        x = self.norm(self.conv1(x))
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.act((self.pwconv1(x)))
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        out = self.se(x)

        if self.skip is not None:
            skip = self.skip(skip)

        return out + skip



# CutD
class Cut(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_fusion = nn.Conv2d(in_channels * 4, out_channels, kernel_size=1, stride=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x0 = x[:, :, 0::2, 0::2]  # x = [B, C, H/2, W/2]
        x1 = x[:, :, 1::2, 0::2]
        x2 = x[:, :, 0::2, 1::2]
        x3 = x[:, :, 1::2, 1::2]
        x = torch.cat([x0, x1, x2, x3], dim=1)  # x = [B, 4*C, H/2, W/2]
        x = self.conv_fusion(x)     # x = [B, out_channels, H/2, W/2]
        x = self.batch_norm(x)
        return x


# 高频下采样模块
class DRFD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cut_c = Cut(in_channels=in_channels, out_channels=out_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.conv_x = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1, groups=out_channels)
        self.act_x = nn.GELU()
        self.batch_norm_x = nn.BatchNorm2d(out_channels)
        self.batch_norm_m = nn.BatchNorm2d(out_channels)
        self.max_m = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fusion = nn.Conv2d(3 * out_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):       # input: x = [B, C, H, W]
        c = x                   # c = [B, C, H, W]
        x = self.conv(x)        # x = [B, C, H, W] --> [B, 2C, H, W]
        m = x                   # m = [B, 2C, H, W]

        # CutD
        c = self.cut_c(c)       # c = [B, C, H, W] --> [B, 2C, H/2, W/2]

        # ConvD
        x = self.conv_x(x)      # x = [B, 2C, H, W] --> [B, 2C, H/2, W/2]
        x = self.act_x(x)
        x = self.batch_norm_x(x)

        # MaxD
        m = self.max_m(m)       # m = [B, 2C, H/2, W/2]
        m = self.batch_norm_m(m)

        # Concat + conv
        x = torch.cat([c, x, m], dim=1)  # x = [B, 6C, H/2, W/2]
        x = self.fusion(x)      # x = [B, 6C, H/2, W/2] --> [B, 2C, H/2, W/2]

        return x                # x = [B, 2C, H/2, W/2]



# Begin 测试
def main():
    block = MDCBlock(in_channels=80)
    block.eval()  # 切换到评估模式
    input_data = torch.randn(128, 80, 56, 56)
    output_before = block(input_data)
    block.reparameterize()
    output_after = block(input_data)
    difference = (output_before - output_after).abs().max()
    print(f"输出差异: {difference.item()}")

if __name__ == "__main__":
    main()
