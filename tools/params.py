# begin 计算模型 参数量, FLOPs
import torch
from ptflops import get_model_complexity_info
import timm
from torchvision.models import (shufflenet_v2_x2_0, efficientnet_b1, regnet_x_800mf,
                                regnet_y_800mf, regnet_x_400mf, regnet_y_400mf, regnet_x_1_6gf,
                                resnet34, resnet50, densenet121, resnet18, shufflenet_v2_x1_5, shufflenet_v2_x1_0,
                                mobilenet_v3_small, mobilenet_v3_large)
from timm import create_model, list_models

from my_model import rdm_1, rdm_2, rdm_3
import os
from torchvision import models

def main():
    models = {
        "rdm_1": rdm_1(num_classes=8),
        "rdm_2": rdm_2(num_classes=8),
        "rdm_3": rdm_3(num_classes=8),
        "mobilenet_v3_small": mobilenet_v3_small(num_classes=8),
        "mobilenet_v3_large": mobilenet_v3_large(num_classes=8),
        "shufflenet_v2_x2_0": shufflenet_v2_x2_0(num_classes=8),
        "shufflenet_v2_x1_0": shufflenet_v2_x1_0(num_classes=8),
        "shufflenet_v2_x1_5": shufflenet_v2_x1_5(num_classes=8),
        "tf_efficientnetv2_b0": create_model('tf_efficientnetv2_b0', pretrained=False, num_classes=8),
        "tf_efficientnetv2_b1": create_model('tf_efficientnetv2_b1', pretrained=False, num_classes=8),
        "tf_efficientnetv2_b2": create_model('tf_efficientnetv2_b2', pretrained=False, num_classes=8),
        "efficientnet_b1": efficientnet_b1(num_classes=8),
        "regnet_x_400mf": regnet_x_400mf(num_classes=8),
        "regnet_x_800mf": regnet_x_800mf(num_classes=8),
        "regnet_y_400mf": regnet_y_400mf(num_classes=8),
        "regnet_y_800mf": regnet_y_800mf(num_classes=8),
        "regnet_x_1_6gf": regnet_x_1_6gf(num_classes=8),
        "resnet18": resnet18(num_classes=8),
        "resnet34": resnet34(num_classes=8),
        "resnet50": resnet50(num_classes=8),
        "densenet121": densenet121(num_classes=8),
        "mobilevitv2_150": create_model('mobilevitv2_150', pretrained=False, num_classes=8),
        "mobilevitv2_125": create_model('mobilevitv2_125', pretrained=False, num_classes=8),
        "mobilevitv2_100": create_model('mobilevitv2_100', pretrained=False, num_classes=8),
        "mobilevitv2_075": create_model('mobilevitv2_075', pretrained=False, num_classes=8),
        "mobilevitv2_050": create_model('mobilevitv2_050', pretrained=False, num_classes=8),
        "mobilevit_xs": create_model('mobilevit_xs', pretrained=False, num_classes=8),
        "mobilevit_s": create_model('mobilevit_s', pretrained=False, num_classes=8),
        "mobilenetv4_conv_medium": create_model('mobilenetv4_conv_medium', pretrained=False, num_classes=8),
        "mobilenetv4_conv_small": create_model('mobilenetv4_conv_small', pretrained=False, num_classes=8),
    }

    input = torch.randn(1, 3, 224, 224)

    def to_megabytes(value: str) -> float:
        """Convert FLOPs or params strings to millions (M)."""
        value = value.strip()
        if "G" in value:  # Convert Giga (GMac) to Mega (MMac)
            return float(value.replace("GMac", "").replace(" G", "").strip()) * 1000
        elif "M" in value:  # Remove M or MMac to handle millions
            return float(value.replace("MMac", "").replace(" M", "").strip())
        else:
            return float(value)

    for name, model in models.items():
        print(f"\033[93m{name}\033[0m")
        re_model = None
        if hasattr(model, 'reparameterize_model'):
            re_model = model.reparameterize_model()

        flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=False)
        flops_m = to_megabytes(flops)
        params_m = to_megabytes(params)
        print(f'FLOPs (M): {flops_m:.2f}')
        print(f'Params (M): {params_m:.2f}')
        print('-' * 50)

        if re_model is not None:
            flops, params = get_model_complexity_info(re_model, (3, 224, 224), as_strings=True,
                                                      print_per_layer_stat=False)
            flops_m = to_megabytes(flops)
            params_m = to_megabytes(params)
            print(f"    \033[93mre_model\033[0m")
            print(f'    FLOPs (M): {flops_m:.2f}')
            print(f'    Params (M): {params_m:.2f}')
            print('-' * 50)


import requests

def download_weights():
    """
    下载 torchvision 模型权重到默认路径，并打印 timm 模型的预训练权重 URL。
    """
    # note 下载 torchvision 模型的权重
    print("\033[1;32m开始下载 torchvision 模型的权重...\033[0m")
    torchvision_models = {
        # "efficientnetV1_b1": models.efficientnet_b1(weights=models.EfficientNet_B1_Weights.IMAGENET1K_V1),
        # "regnet_x_800mf": models.regnet_x_800mf(weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V1),
        # "regnet_y_800mf": models.regnet_y_800mf(weights=models.RegNet_Y_800MF_Weights.IMAGENET1K_V1),
        # "regnet_y_1_6gf": models.regnet_y_1_6gf(weights=models.RegNet_Y_1_6GF_Weights.IMAGENET1K_V1),
        # "mobilenet_v3_small": models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights),
    }


    for model_name, model in torchvision_models.items():
        print(f"\033[1;32m{model_name} 的权重已下载到默认缓存目录。\033[0m")

    # note 打印 timm 模型的预训练权重 URL
    save_dir = "./download_weights"
    os.makedirs(save_dir, exist_ok=True)

    # 打印开始信息
    print("\033[1;32m开始处理 timm 模型的权重加载...\033[0m")

    # 定义模型字典
    timm_models = {
        # "mobilenetv4_conv_medium": "mobilenetv4_conv_medium",
        "mobilenetv4_conv_small": "mobilenetv4_conv_small",
        # "tf_efficientnetv2_b2": "tf_efficientnetv2_b2",
    }

    # 遍历模型，加载权重
    for model_name, timm_model in timm_models.items():
        try:
            print(f"\033[1;34m正在加载 {model_name} 模型...\033[0m")

            model = create_model(timm_model, pretrained=True)
            print(model)
            model.eval()
            input_tensor = torch.randn(1, 3, 224, 224)

            with torch.no_grad():
                output = model(input_tensor)

            print(f"\033[1;34m输入张量形状: {input_tensor.shape}\033[0m")
            print(f"\033[1;34m输出张量形状: {output.shape}\033[0m")
            print(f"\033[1;34m输出结果 (部分): {output[0][:5]}\033[0m")  # 打印前 5 个值

        except Exception as e:
            print(f"\033[1;31m处理 {model_name} 时发生错误：{e}\033[0m")

    # 打印完成信息
    print("\033[1;32m所有模型处理完成！\033[0m")

if __name__ == '__main__':
   main()
   # print(timm.list_models('*mobilenetv4*'))
   # download_weights()
