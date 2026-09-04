import torch
import torch.nn as nn

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


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation, kernel_size=3):
        padding = (kernel_size - 1) // 2 * dilation

        modules = [
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                      padding=padding, dilation=dilation, bias=False, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class PCB(nn.Module):
    def __init__(self, in_channels, atrous_rates=None):
        super(PCB, self).__init__()
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
            nn.Conv3d(in_channels, in_channels, kernel_size=(5, 1, 1), padding=(0, 0, 0), groups=in_channels, bias=False),
        )

        # note 通道注意力
        self.se = SeBlock(mid_channels)

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

        out = torch.squeeze(conv_3d, 2)  # [B C H W]
        out = self.se(out)
        out = self.conv1x1(out)  # 1x1 卷积降维
        return self.act(out + skip)  # 残差连接 + 激活函数




# note 重参数化的7x7 Block
class MSDCB(nn.Module):
    def __init__(self, in_channels, expand_size=None, stride=1, out_channels=None):
        super(MSDCB, self).__init__()
        expand_size = expand_size or int(in_channels * 2.4)
        self.expand_size = expand_size
        if out_channels is None:
            out_channels = in_channels

        self.stride = stride
        if stride == 2:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=3,
                    groups=in_channels,
                    stride=stride,
                    padding=1,
                    bias=False
                ),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        # 1x1 卷积，用于通道扩张
        self.conv1 = nn.Conv2d(in_channels, expand_size, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = nn.SiLU(inplace=True)

        # 四个不同尺度的深度卷积
        self.mid_conv1 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=7, padding=3, dilation=1,
            groups=expand_size, bias=False, stride=stride
        )
        self.mid_conv2 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=3, padding=1, dilation=1,
            groups=expand_size, bias=False, stride=stride
        )
        self.mid_conv3 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=3, padding=2, dilation=2,
            groups=expand_size, bias=False, stride=stride
        )
        self.mid_conv4 = nn.Conv2d(
            expand_size, expand_size,
            kernel_size=3, padding=3, dilation=3,
            groups=expand_size, bias=False, stride=stride
        )

        # 对应的批归一化层
        self.mid_bn1 = nn.BatchNorm2d(expand_size)
        self.mid_bn2 = nn.BatchNorm2d(expand_size)
        self.mid_bn3 = nn.BatchNorm2d(expand_size)
        self.mid_bn4 = nn.BatchNorm2d(expand_size)

        # 用于通道加权融合的 Conv3D
        self.conv3d = nn.Conv3d(
            expand_size, expand_size,
            kernel_size=(4, 1, 1), padding=(0, 0, 0),
            groups=expand_size, bias=False
        )
        self.act2 = nn.SiLU(inplace=True)
        self.se = SeBlock(expand_size)

        # 1x1 卷积，用于通道压缩
        self.conv2 = nn.Conv2d(expand_size, out_channels, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.SiLU(inplace=True)

    # note forward方法
    def forward(self, x):
        identity = x
        x = self.act1(self.bn1(self.conv1(x)))

        # 分别通过四个不同尺度的深度卷积
        x1 = self.mid_bn1(self.mid_conv1(x))
        x2 = self.mid_bn2(self.mid_conv2(x))
        x3 = self.mid_bn3(self.mid_conv3(x))
        x4 = self.mid_bn4(self.mid_conv4(x))

        xm1 = torch.unsqueeze(x1, -3)  # [B, C, 1, H, W]
        xm2 = torch.unsqueeze(x2, -3)
        xm3 = torch.unsqueeze(x3, -3)
        xm4 = torch.unsqueeze(x4, -3)

        combine = torch.cat([xm1, xm2, xm3, xm4], dim=2)  # [B, C, 4, H, W]

        conv_3d = self.conv3d(combine)  # [B, C, 1, H, W]
        out = torch.squeeze(conv_3d, 2)  # [B, C, H, W]

        x = self.act2(out)
        x = self.se(x)
        x = self.bn3(self.conv2(x))
        x = self.act3(x + self.skip(identity))
        return x

    def reparameterize(self):
        with torch.no_grad():
            # 重参数化中间的卷积和对应的 BN 层
            mid_convs = [self.mid_conv1, self.mid_conv2, self.mid_conv3, self.mid_conv4]
            mid_bns = [self.mid_bn1, self.mid_bn2, self.mid_bn3, self.mid_bn4]
            expanded_weights = []
            expanded_biases = []

            # 对每个分支进行卷积和 BN 的融合，并扩展卷积核到统一大小
            for conv, bn in zip(mid_convs, mid_bns):
                # 融合卷积和 BN
                weight, bias = self._fuse_conv_bn(conv, bn)
                # 扩展卷积核到 7x7
                expanded_weight = self._expand_weight(weight, conv.kernel_size[0], conv.dilation[0])
                expanded_weights.append(expanded_weight)
                expanded_biases.append(bias)

            # 从 conv3d 获取每个分支的通道权重
            weight_3d = self.conv3d.weight.data.clone()  # [C, 1, 4, 1, 1]
            weight_3d = weight_3d.squeeze(-1).squeeze(-1)  # [C, 1, 4]
            weight_3d = weight_3d.squeeze(1)  # [C, 4]

            # 对每个分支的权重和偏置进行通道加权
            for i in range(4):
                expanded_weights[i] = expanded_weights[i] * weight_3d[:, i].reshape(-1, 1, 1, 1)
                expanded_biases[i] = expanded_biases[i] * weight_3d[:, i]

            # 将所有分支的权重和偏置相加
            merged_weight = sum(expanded_weights)
            merged_bias = sum(expanded_biases)

            # 创建新的融合后的卷积层
            self.reparam_conv = nn.Conv2d(
                in_channels=self.expand_size,
                out_channels=self.expand_size,
                kernel_size=7,
                padding=3,
                groups=self.expand_size,
                bias=True,
                stride=self.stride
            )
            self.reparam_conv.weight.data.copy_(merged_weight)
            self.reparam_conv.bias.data.copy_(merged_bias)

            del self.mid_conv1, self.mid_conv2, self.mid_conv3, self.mid_conv4
            del self.mid_bn1, self.mid_bn2, self.mid_bn3, self.mid_bn4
            del self.conv3d

            # 重参数化 conv1 + bn1 和 conv2 + bn3
            self.reparam_conv1 = self._fuse_conv_bn_layer(self.conv1, self.bn1)
            self.reparam_conv2 = self._fuse_conv_bn_layer(self.conv2, self.bn3)

            del self.conv1, self.bn1
            del self.conv2, self.bn3

            # 修改 forward 方法
            self.forward = self.forward_reparam

    # note 重参数化后的 forward 方法
    def forward_reparam(self, x):
        identity = x
        x = self.act1(self.reparam_conv1(x))
        x = self.reparam_conv(x)
        x = self.act2(x)
        x = self.se(x)
        x = self.reparam_conv2(x)
        iden = self.skip(identity)
        x = self.act3(x + iden)
        return x

    # 融合卷积和 BN 层的函数
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

    # 将卷积和 BN 层融合为一个卷积层
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

    # 将卷积核扩展到 7x7 大小
    def _expand_weight(self, weight, kernel_size, dilation):
        expanded_weight = torch.zeros((self.expand_size, 1, 7, 7), device=weight.device, dtype=weight.dtype)
        center = 3  # 7x7 卷积核的中心索引

        if kernel_size == 7 and dilation == 1:
            # 若卷积核为 7x7，无需扩展
            expanded_weight[:, :, :, :] = weight
        elif kernel_size == 3:
            # 计算有效卷积核尺寸
            effective_kernel_size = (kernel_size - 1) * dilation + 1
            start = center - (effective_kernel_size // 2)
            indices = range(kernel_size)
            for i in indices:
                for j in indices:
                    idx = start + i * dilation
                    idy = start + j * dilation
                    if 0 <= idx < 7 and 0 <= idy < 7:
                        expanded_weight[:, 0, idx, idy] = weight[:, 0, i, j]
        else:
            raise ValueError("不支持的卷积核大小和膨胀率")
        return expanded_weight





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
class DACB(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride, rep=True):
        super(DACB, self).__init__()
        self.stride = stride
        self.expand_size = expand_size
        self.rep = rep


        # 1x1卷积
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # 3x3深度可分离卷积
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

        # 1x3深度可分离卷积
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

        # 3x1深度可分离卷积
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

        # 用于通道加权的3D卷积
        self.conv3d = nn.Conv3d(
            expand_size,
            expand_size,
            kernel_size=(3, 1, 1),
            padding=(0, 0, 0),
            groups=expand_size,
            bias=False
        )

        self.act2 = act(inplace=True)
        self.se = SeBlock(expand_size) if se else nn.Identity()

        # 最后一层1x1卷积用于调整输出通道数
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2:
            if in_size != out_size:
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
            else:
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

        # 将3个分支的结果添加一个维度并合并
        xm1 = torch.unsqueeze(x1, -3)  # [B, C, 1, H, W]
        xm2 = torch.unsqueeze(x2, -3)
        xm3 = torch.unsqueeze(x3, -3)
        combine = torch.cat([xm1, xm2, xm3], dim=2)  # [B, C, 3, H, W]

        # 用3D卷积加权
        conv_3d = self.conv3d(combine)  # [B, C, 1, H, W]
        out = torch.squeeze(conv_3d, 2)  # [B, C, H, W]

        out = self.act2(out)
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

    # 将卷积和BN融合的方法
    def _fuse_conv_bn(self, conv, bn):
        # 提取卷积和BN参数
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

        # BN归一化标准差
        std = (bn_running_var + bn_eps).sqrt()
        t = bn_weight / std

        # 融合卷积权重和偏置
        fused_weight = conv_weight * t.reshape(-1, 1, 1, 1)
        fused_bias = bn_bias + (conv_bias - bn_running_mean) * t

        return fused_weight, fused_bias

    # 将卷积+BN层融合为单卷积层
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
            # 融合conv1 + bn1 和 conv3 + bn3
            self.rep_conv1 = self._fuse_conv_bn_layer(self.conv1, self.bn1)
            self.rep_conv3 = self._fuse_conv_bn_layer(self.conv3, self.bn3)

            # 分别融合3个分支的卷积
            weight_3x3, bias_3x3 = self._fuse_conv_bn(self.conv2, self.bn2)
            weight_1x3, bias_1x3 = self._fuse_conv_bn(self.conv2_1, self.bn2_1)
            weight_3x1, bias_3x1 = self._fuse_conv_bn(self.conv2_2, self.bn2_2)

            # 扩展1x3和3x1卷积核到3x3大小
            weight_1x3_expanded = torch.zeros_like(weight_3x3)
            weight_1x3_expanded[:, :, :, 1] = weight_1x3[:, :, :, 0]

            weight_3x1_expanded = torch.zeros_like(weight_3x3)
            weight_3x1_expanded[:, :, 1, :] = weight_3x1[:, :, 0, :]

            # 获取3D卷积权重
            weight_3d = self.conv3d.weight.data.clone()  # [C, 1, 3, 1, 1]
            weight_3d = weight_3d.squeeze(-1).squeeze(-1)  # [C, 1, 3]
            weight_3d = weight_3d.squeeze(1)  # [C, 3]

            # 按通道加权分支权重
            weight_3x3 = weight_3x3 * weight_3d[:, 0].reshape(-1, 1, 1, 1)
            weight_1x3_expanded = weight_1x3_expanded * weight_3d[:, 1].reshape(-1, 1, 1, 1)
            weight_3x1_expanded = weight_3x1_expanded * weight_3d[:, 2].reshape(-1, 1, 1, 1)

            # 调整偏置
            bias_3x3 = bias_3x3 * weight_3d[:, 0]
            bias_1x3 = bias_1x3 * weight_3d[:, 1]
            bias_3x1 = bias_3x1 * weight_3d[:, 2]

            # 合并所有分支的权重和偏置
            merged_weight = weight_3x3 + weight_1x3_expanded + weight_3x1_expanded
            merged_bias = bias_3x3 + bias_1x3 + bias_3x1

            # 创建新的融合卷积层
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

            # 删除原始卷积和BN层
            del self.conv1, self.bn1
            del self.conv2, self.bn2
            del self.conv2_1, self.bn2_1
            del self.conv2_2, self.bn2_2
            del self.conv3d
            del self.conv3, self.bn3

            # 修改forward方法
            self.forward = self.forward_reparam

    def forward_reparam(self, x):
        # 重参数化后的前向传播
        skip = x
        out = self.act1(self.rep_conv1(x))
        out = self.act2(self.rep_conv2(out))
        out = self.se(out)
        out = self.rep_conv3(out)

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)




import torch
import torch.nn as nn

class DACB2(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride):
        super(DACB2, self).__init__()
        self.stride = stride
        self.expand_size = expand_size

        # 1x1卷积
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # 3x3深度可分离卷积
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

        self.act2 = act(inplace=True)
        self.se = SeBlock(expand_size) if se else nn.Identity()

        # 最后一层1x1卷积用于调整输出通道数
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        # Skip connection
        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2:
            if in_size != out_size:
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
            else:
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
        out = self.act1(self.bn1(self.conv1(x)))  # 1x1卷积
        out = self.act2(self.bn2(self.conv2(out)))  # 3x3深度可分离卷积
        out = self.se(out)  # SE模块（如果启用）
        out = self.bn3(self.conv3(out))  # 1x1卷积调整通道数

        if self.skip is not None:
            skip = self.skip(skip)  # Skip connection
        return self.act3(out + skip)  # 残差连接



class DACB1(nn.Module):
    def __init__(self, kernel_size, in_size, expand_size, out_size, act, se, stride, rep=True):
        super(DACB1, self).__init__()
        self.stride = stride
        self.expand_size = expand_size
        self.rep = rep

        # 1x1卷积
        self.conv1 = nn.Conv2d(in_size, expand_size, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(expand_size)
        self.act1 = act(inplace=True)

        # 3x3深度可分离卷积
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

        # 1x3深度可分离卷积
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

        # 3x1深度可分离卷积
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

        # 最后一层1x1卷积用于调整输出通道数
        self.conv3 = nn.Conv2d(expand_size, out_size, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_size)
        self.act3 = act(inplace=True)

        self.skip = None
        if stride == 1 and in_size != out_size:
            self.skip = nn.Sequential(
                nn.Conv2d(in_size, out_size, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_size)
            )

        if stride == 2:
            if in_size != out_size:
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
            else:
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

        # 将3个分支的结果相加
        out = x1 + x2 + x3

        out = self.act2(out)
        out = self.se(out)
        out = self.bn3(self.conv3(out))
        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)

    # 将卷积和BN融合的方法
    def _fuse_conv_bn(self, conv, bn):
        # 提取卷积和BN参数
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

        # BN归一化标准差
        std = (bn_running_var + bn_eps).sqrt()
        t = bn_weight / std

        # 融合卷积权重和偏置
        fused_weight = conv_weight * t.reshape(-1, 1, 1, 1)
        fused_bias = bn_bias + (conv_bias - bn_running_mean) * t

        return fused_weight, fused_bias

    # 将卷积+BN层融合为单卷积层
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
            # 融合conv1 + bn1 和 conv3 + bn3
            self.rep_conv1 = self._fuse_conv_bn_layer(self.conv1, self.bn1)
            self.rep_conv3 = self._fuse_conv_bn_layer(self.conv3, self.bn3)

            # 分别融合3个分支的卷积
            weight_3x3, bias_3x3 = self._fuse_conv_bn(self.conv2, self.bn2)
            weight_1x3, bias_1x3 = self._fuse_conv_bn(self.conv2_1, self.bn2_1)
            weight_3x1, bias_3x1 = self._fuse_conv_bn(self.conv2_2, self.bn2_2)

            # 扩展1x3和3x1卷积核到3x3大小
            weight_1x3_expanded = torch.zeros_like(weight_3x3)
            weight_1x3_expanded[:, :, :, 1] = weight_1x3[:, :, :, 0]

            weight_3x1_expanded = torch.zeros_like(weight_3x3)
            weight_3x1_expanded[:, :, 1, :] = weight_3x1[:, :, 0, :]

            # 合并所有分支的权重和偏置
            merged_weight = weight_3x3 + weight_1x3_expanded + weight_3x1_expanded
            merged_bias = bias_3x3 + bias_1x3 + bias_3x1

            # 创建新的融合卷积层
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

            # 删除原始卷积和BN层
            del self.conv1, self.bn1
            del self.conv2, self.bn2
            del self.conv2_1, self.bn2_1
            del self.conv2_2, self.bn2_2
            del self.conv3, self.bn3

            # 修改forward方法
            self.forward = self.forward_reparam

    def forward_reparam(self, x):
        # 重参数化后的前向传播
        skip = x
        out = self.act1(self.rep_conv1(x))
        out = self.act2(self.rep_conv2(out))
        out = self.se(out)
        out = self.rep_conv3(out)

        if self.skip is not None:
            skip = self.skip(skip)
        return self.act3(out + skip)





# note 空间注意力下采样模块
class SPADB(nn.Module):

    def __init__(self, kernel_size, in_size, expand_size, out_size, act, attention, stride):
        super(SPADB, self).__init__()
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



# Begin 测试
def main():
    block = DACB1(3, 80, 224, 80, nn.SiLU, True, 1)
    # block = MSDCB(32, 64)
    block.eval()
    input_data = torch.randn(32, 80, 80, 80)
    output_before = block(input_data)
    block.reparameterize()
    output_after = block(input_data)
    difference = (output_before - output_after).abs().max()
    print(f"输出差异: {difference.item()}")


if __name__ == "__main__":
    main()
