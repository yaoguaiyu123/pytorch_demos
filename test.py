import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from utils import CustomCosineScheduler, optimizerChoice, choose_dataset, choose_model, weightFrozen,CnnValidator
import numpy as np
from utils.util import choose_val_model
import time

# 验证softmax的输出
def func01():
    logits = torch.tensor([1.0966e-01, 2.9803e-01, 1.2494e+00, -3.5193e-02, -1.1853e+00, 1.4485e+00, -1.2355e-01, -1.7152e+00])
    softmax_output = F.softmax(logits, dim=0)
    print(logits)
    print(softmax_output)


import pandas as pd

def validate_model():
    # begin 数据集选择
    from colorama import Fore, Style

    # 用户选择数据集
    print(f"{Fore.CYAN}请选择数据集:{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}[3]{Style.RESET_ALL} ISIC2019 (8分类)")
    print(f"{Fore.YELLOW}[4]{Style.RESET_ALL} ISIC2018 (7分类)")

    data_choice = input(f"{Fore.GREEN}输入你的选择 (3 或 4): {Style.RESET_ALL}").strip()

    num_classes = None
    labels = None
    if data_choice == '3':
        print(f"{Fore.BLUE}数据集: ISIC2019 (8分类){Style.RESET_ALL}")
        num_classes = 8
        labels = [0, 1, 2, 3, 4, 5, 6, 7]
    elif data_choice == '4':
        print(f"{Fore.BLUE}数据集: ISIC2018 (7分类){Style.RESET_ALL}")
        num_classes = 7
        labels = [0, 1, 2, 3, 4, 5, 6]
    else:
        print(f"{Fore.RED}无效输入，请输入 3 或 4!{Style.RESET_ALL}")
        exit(1)

    # begin 模型选择
    model_options = {
        '003': ('regnet_y_800mf', 'regnet_y_800mf', './weights/regnet_y_800mf.pth'),
        '006': ('mobilevitv2_100', 'mobilevitv2_100', './weights/mobilevitv2_100.pth'),
        '007': ('mobilevitv2_075', 'mobilevitv2_075', './weights/mobilevitv2_075.pth'),
        '008': ('mobilevitv2_050', 'mobilevitv2_050', './weights/mobilevitv2_050.pth'),
        '009': ('efficientnetV2_b0', 'efficientnetV2_b0', './weights/efficientnetV2_b0.pth'),
        '010': ('efficientnetV2_b1', 'efficientnetV2_b1', './weights/efficientnetV2_b1.pth'),
        '011': ('efficientnetV2_b2', 'efficientnetV2_b2', './weights/efficientnetV2_b2.pth'),
        '012': ('mobilenetv4_conv_small', 'mobilenetv4_conv_small', './weights/mobilenetv4_conv_small.pth'),
        '3': ('rdm_3', 'rdm_3(num_classes=200)', './weights/rdm_3.pth'),
        '4': ('mobileNetV3_large_m1', 'mobileNetV3_large_m1(num_classes=8)', './weights/mobilenet_large_m1.pth'),
    }
    print("请选择模型: ")
    for index, (key, (name, _, _)) in enumerate(model_options.items()):
        print(f"{key}: {name}", end=" " * (32 - len(name)))
        if (index + 1) % 4 == 0:
            print()
    print()

    model_choice = input("输入模型对应的数字选择模型: ")

    # note 加载数据集
    train_loader, val_loader = choose_dataset(data_choice, 1)

    # note 初始化模型
    model, pretrained_path = choose_val_model(model_choice, model_options, data_choice, weight_path="./weights/rdm_3_ISIC2019.pth")
    # model, pretrained_path = choose_val_model(model_choice, model_options, data_choice, weight_path=None)

    # note 相关变量初始化
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("神经网络将在 ", device, " 进行验证")
    model = model.to(device)

    start_time = time.time()

    # begin 使用 CnnValidator 进行验证
    tester = CnnValidator(device, num_classes=num_classes, model=model, test_loader=val_loader)
    tester.print_throughput()

    # note 打印信息
    print('\n##################  测试信息打印（原始模型）  ##################')
    print('验证准确率 (Accuracy): {:.3f}%'.format(tester.TestAcc * 100))
    print('验证损失 (Loss): {:.4f}'.format(tester.TestLoss))
    print('验证精确率 (Precision, Macro): {:.4f}'.format(np.mean(tester.Precision)))
    print('验证召回率 (Recall, Macro): {:.4f}'.format(np.mean(tester.Recall)))
    print('验证 F1 分数 (F1 Score, Macro): {:.4f}'.format(np.mean(tester.F1Score)))
    print('验证特异性 (Specificity, Macro): {:.4f}'.format(np.mean(tester.Specificity)))
    print('验证 AUC: {:.4f}'.format(tester.AUC))
    print('耗时: {:.2f} 秒'.format(time.time() - start_time))

    if hasattr(model, 'reparameterize_model'):
        print('\n##################  测试信息打印（重参数化模型）  ##################')
        print('验证准确率 (Accuracy): {:.3f}%'.format(tester.re_TestAcc * 100))
        print('验证损失 (Loss): {:.4f}'.format(tester.re_TestLoss))
        print('验证精确率 (Precision, Macro): {:.4f}'.format(np.mean(tester.re_Precision)))
        print('验证召回率 (Recall, Macro): {:.4f}'.format(np.mean(tester.re_Recall)))
        print('验证 F1 分数 (F1 Score, Macro): {:.4f}'.format(np.mean(tester.re_F1Score)))
        print('验证特异性 (Specificity, Macro): {:.4f}'.format(np.mean(tester.re_Specificity)))
        print('验证 AUC: {:.4f}'.format(tester.re_AUC))
        print('吞吐量提升: {:.2f}%'.format(
            (tester.re_throughput - tester.orig_throughput) / tester.orig_throughput * 100))

    # 绘制 ROC 曲线
    tester.plot_roc_curve(num_classes)



if __name__ == '__main__':
    validate_model()



