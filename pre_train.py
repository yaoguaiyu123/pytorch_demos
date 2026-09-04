# begin
#  1. 在 Stable ImageNet-1K 上的的预训练，数据集来源于
#  https://www.kaggle.com/datasets/vitaliykinakh/stable-imagenet1k
#  2. 在 ImageNet100 上的预训练，数据集来源于
#  https://www.kaggle.com/datasets/ambityga/imagenet100
import shutup
shutup.please()
import datetime
import torch
import time
from tensorboardX import SummaryWriter
from utils import CustomCosineScheduler, optimizerChoice, choose_pretrain_dataset, choose_model_to_pretrain, weightFrozen, \
    CnnTrainer, CnnTrainerProWithOptionalEMA, CnnTrainerProConVal
import numpy as np
import multiprocessing as mp
import math
from torch.optim.lr_scheduler import LinearLR, MultiStepLR, SequentialLR, CosineAnnealingLR
# begin tensorboard --logdir=../important_files/show --port=5550

# note 权重衰减函数
def cosine_weight_decay(epoch, num_epochs, initial_wd, final_wd):
    cosine_decay = 0.5 * (1 + math.cos(math.pi * epoch / num_epochs))
    weight_decay = final_wd + (initial_wd - final_wd) * cosine_decay
    return weight_decay


if __name__ == '__main__':
    mp.set_start_method('spawn')

    print("请选择数据集：")
    print("1. ImageNet100")
    print("2. ImageNet1000")
    data_choice = input("请选择(1 ~ 2): ")

    model_options = {
        '001': ('mobileNetV3_large_model', 'models.mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.IMAGENET1K_V1)', './weights/mobileNetV3_large_models_预训练.pth'),
        '002': ('mobileNetV3_large_model', 'models.mobilenet_v3_large(weights=None)', './weights/mobileNetV3_large_models_预训练.pth'),
        '1': ('mobileNetv3_large', 'mobileNetV3_large(num_classes=200)', './weights/mobilenetv3_large_预训练.pth'),
        '2': ('mobilenetv4_small', 'mobilenetv4_small(num_classes=200)', './weights/mobilenetv4_small_预训练.pth'),
        '3': ('net_t2', 'net_t2(num_classes=200)', './weights/net_t2_预训练.pth'),
        '4': ('net_t1', 'net_t1(num_classes=200)', './weights/net_t1_预训练.pth'),
    }
    print("请选择模型: ")
    for index, (key, (name, _, _)) in enumerate(model_options.items()):
        print(f"{key}: {name}", end=" " * (32 - len(name)))
        if (index + 1) % 4 == 0:
            print()
    print()

    model_choice = input("输入模型对应的数字选择模型: ")

    # note 初始化数据集
    train_loader, val_loader = choose_pretrain_dataset(data_choice)

    # note 初始化模型
    model, pretrained_path = choose_model_to_pretrain(data_choice, model_choice, model_options)

    # note 相关参数初始化
    indicatorType = 'accuracy'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\033[33m神经网络将在 {device} 进行训练\033[0m")
    model = model.to(device)
    initial_lr = 1e-1    #  begin 初始学习率

    # note 权重衰减  如果模型过拟合，考虑增加 weight_decay  |  如果模型欠拟合，可以适当减小 weight_decay
    initial_wd = 1e-5
    final_wd = 5e-6
    # 优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=initial_lr, momentum=0.9, weight_decay=initial_wd)
    num_epochs = 250
    # 学习率预热参数
    lr_warmup_epochs = 10
    lr_warmup_method = 'linear'
    lr_warmup_decay = 0.1   # 预热时的初始学习率从 initial_lr * lr_warmup_decay -> lr_warmup_decay

    # note 阶梯下降学习率调度器
    warmup_scheduler = LinearLR(optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs)
    multi_step_scheduler = MultiStepLR(optimizer, milestones=[100, 150, 180, 210], gamma=0.1)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, multi_step_scheduler],
        milestones=[lr_warmup_epochs]
    )

    # note 余弦退火学习率调度器
    # warmup_scheduler = LinearLR(optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs)
    # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - lr_warmup_epochs, eta_min=final_wd)
    # scheduler = SequentialLR(
    #     optimizer,
    #     schedulers=[warmup_scheduler, cosine_scheduler],
    #     milestones=[lr_warmup_epochs]
    # )

    writer = SummaryWriter(logdir="./runs")
    best_val_acc = 0.0
    best_train_acc = 0.0
    best_epoch = 0
    interval = 1  # note 进行验证的间隔
    # begin ---------------Begin Train---------------
    for epoch in range(num_epochs):
        start_time = time.time()

        # 更新权重衰减参数
        current_wd = cosine_weight_decay(epoch, num_epochs, initial_wd, final_wd)
        for param_group in optimizer.param_groups:
            param_group['weight_decay'] = current_wd

        # 训练阶段
        trainer = CnnTrainerProConVal(
            device=device,
            num_classes=1000,
            optim=optimizer,
            model=model,
            train_dataloader=train_loader,
            test_dataloader=val_loader,
            rep=False,
            use_mixup_cutmix=True
        )
        trainer.train_one_epoch()  # 执行一个完整的训练过程

        # begin 打印前2轮 + 每 interval 轮打印一次验证信息
        if (epoch + 1) % interval == 0 or epoch <= 1:
            trainer.validate(epoch)
            # 计算平均指标
            with torch.no_grad():
                AvgRecall = np.mean(trainer.Recall)
                AvgPrecision = np.mean(trainer.Precision)
                AvgF1Score = np.mean(trainer.F1Score)
                AvgSpecificity = np.mean(trainer.Specificity)

                # 更新最佳验证指标
                indicator = trainer.TestLoss if indicatorType == 'loss' else trainer.TestAcc
                if (indicator < best_val_acc and indicatorType == 'loss') or \
                        (indicator > best_val_acc and indicatorType != 'loss'):
                    best_val_acc = indicator
                    best_epoch = epoch + 1
                    # note 保存模型
                    if pretrained_path is not None:
                        torch.save(model.state_dict(), pretrained_path)

            # 打印验证信息
            print(
                f"\033[96m\nEpoch: [{epoch + 1}/{num_epochs}] \t Best Val Acc: {best_val_acc * 100:.3f}   Best Train Acc: {best_train_acc * 100:.3f} "
                f"Train Acc: {trainer.TrainAcc * 100:.3f}  Val Acc: {trainer.TestAcc * 100:.3f}"
                f"\nEpochs since best: [{epoch + 1 - best_epoch}] \t Train Loss: {trainer.TrainLoss:.5f} \t Val Loss: {trainer.TestLoss:.5f}"
                f"\nRecall: {AvgRecall * 100:.3f}  Precision: {AvgPrecision * 100:.3f}  F1 Score: {AvgF1Score:.4f}  Specificity: {AvgSpecificity:.4f}"
                f"\nTime Elapsed: {time.time() - start_time:.2f} sec \t Learning Rate: {optimizer.param_groups[0]['lr']:.6f}\033[0m")

            # 写入 ./record.md 文件
            with open('./record.md', 'a') as f:
                f.write(f"## Epoch {epoch + 1}\n")
                f.write(f"- **Time**: {datetime.datetime.now()}\n")
                f.write(f"- **Train Acc**: {trainer.TrainAcc * 100:.3f}%\n")
                f.write(f"- **Val Acc**: {trainer.TestAcc * 100:.3f}%\n")
                f.write(f"- **Train Loss**: {trainer.TrainLoss:.5f}\n")
                f.write(f"- **Val Loss**: {trainer.TestLoss:.5f}\n")
                f.write(f"- **Best Val Acc**: {best_val_acc * 100:.3f}%\n")
                f.write(f"- **Metrics**:\n")
                f.write(f"  - Recall: {AvgRecall * 100:.3f}%\n")
                f.write(f"  - Precision: {AvgPrecision * 100:.3f}%\n")
                f.write(f"  - F1 Score: {AvgF1Score:.4f}\n")
                f.write(f"  - Specificity: {AvgSpecificity:.4f}\n\n")

        # begin 训练信息每轮都打印
        print(
            f"\nEpoch: [{epoch + 1}/{num_epochs}] \t Train Acc: {trainer.TrainAcc * 100:.3f} \t Train Loss: {trainer.TrainLoss:.5f}")
        print(
            f"Time Elapsed: {time.time() - start_time:.2f} sec \t Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        writer.add_scalar('train/loss', trainer.TrainLoss, epoch + 1)
        writer.add_scalar('train/accuracy', trainer.TrainAcc, epoch + 1)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)

        # note 更新学习率
        scheduler.step()

    # begin 结束的时候再保存一次模型权重
    if pretrained_path is not None:
        torch.save(model.state_dict(), pretrained_path)

    writer.close()  # 关闭 SummaryWriter



