def main():
    # begin 前置项
    import shutup
    shutup.please()
    import random
    import torch
    import time
    from tensorboardX import SummaryWriter
    from utils import (CustomCosineScheduler, optimizerChoice, choose_dataset, choose_model, weightFrozen,
                       CnnTrainerProWithOptionalEMA, CustomLearningRateScheduler, ThreeStageScheduler)
    import numpy as np
    # tensorboard --logdir=./runs/show --port=4565
    from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, StepLR

    def load_loss():
        loss_mode = 0
        loss_options = {
            0: "交叉熵损失函数",
            1: "焦点损失函数",
            2: "焦点损失 + 类平衡损失函数",
            3: "SeeSaw损失函数"
        }
        try:
            loss_mode = int(input("请选择损失函数模式 (0-3): "))
            if loss_mode in loss_options:
                print(f"\033[93m使用了{loss_options[loss_mode]}\033[0m")
            else:
                print("输入无效，请输入0-3之间的数字")
        except ValueError:
            print("请输入有效的数字")
        return loss_mode

    def choose_pretrained_model():
        load_pretrained = input("是否加载本地模型的预训练模型？(y/n): ").strip().lower()

        if load_pretrained == 'y':
            print("\033[93m成功加载预训练模型\033[0m")
            return "./weights/net_t2_预训练_imageNet1000.pth"
        else:
            print("\033[93m不加载预训练模型\033[0m")
            return None

    # begin 用户交互部分
    print("请选择数据集：")
    print("1. PAD2020")
    print("2. ImageNet1k")
    print("3. ISIC2019")
    print("4. ISIC2018")
    print("5. ISIC2017")
    data_choice = input("请选择(1 ~ 5): ")
    num_classes = None
    if data_choice == '3':
        num_classes = 8
    elif data_choice == '4':
        num_classes = 7
    elif data_choice == '1':
        num_classes = 6

    model_options = {
        '001': ('efficientnetV1_b1', 'efficientnetV1_b1', './weights/efficientnetV1_b1.pth'),
        '002': ('regnet_x_800mf', 'regnet_x_800mf', './weights/regnet_x_800mf.pth'),
        '003': ('regnet_y_800mf', 'regnet_y_800mf', './weights/regnet_y_800mf.pth'),
        '004': ('regnet_y_1_6gf', 'regnet_y_1_6gf', './weights/regnet_y_1_6gf.pth'),
        '005': ('mobilenetv4_conv_medium', 'mobilenetv4_conv_medium', './weights/mobilenetv4_conv_medium.pth'),
        '006': ('mobilevitv2_100', 'mobilevitv2_100', './weights/mobilevitv2_100.pth'),
        '007': ('mobilevitv2_075', 'mobilevitv2_075', './weights/mobilevitv2_075.pth'),
        '008': ('mobilevitv2_050', 'mobilevitv2_050', './weights/mobilevitv2_050.pth'),
        '009': ('efficientnetV2_b0', 'efficientnetV2_b0', './weights/efficientnetV2_b0.pth'),
        '010': ('efficientnetV2_b1', 'efficientnetV2_b1', './weights/efficientnetV2_b1.pth'),
        '011': ('efficientnetV2_b2', 'efficientnetV2_b2', './weights/efficientnetV2_b2.pth'),
        '012': ('mobilenetv4_conv_small', 'mobilenetv4_conv_small', './weights/mobilenetv4_conv_small.pth'),
        '2': ('mobilenetv4_mid', 'mobilenetv4_mid(num_classes=200)', './weights/mobilenetv4_mid.pth'),
        '3': ('rdm_2', 'rdm_2(num_classes=200)', './weights/rdm_2.pth'),
        '4': ('rdm_3', 'rdm_3(num_classes=200)', './weights/rdm_3.pth'),
        '5': ('efficientnetv2_b0', 'efficientnetv2_b0(num_classes=200)', './weights/efficientnetv2_b0.pth'),
    }
    print("请选择模型: ")
    for index, (key, (name, _, _)) in enumerate(model_options.items()):
        print(f"{key}: {name}", end=" " * (32 - len(name)))
        if (index + 1) % 4 == 0:
            print()
    print()

    model_choice = input("输入模型对应的数字选择模型: ")

    pretrained_path = choose_pretrained_model()
    print("预训练权重路径: ",pretrained_path)

    # begin 训练部分
    for split in range(1, 2):
        # note 初始化数据集
        train_loader, val_loader = choose_dataset(data_choice, split)

        # note 初始化模型
        model, model_weights_path = choose_model(data_choice, model_choice,
                                                 model_options, pretrained_path=pretrained_path)

        # note 相关变量初始化
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\033[33m神经网络将在 {device} 进行训练\033[0m")
        model = model.to(device)
        model = weightFrozen(model, True)   # begin 冻结特征提取的权重
        # begin 初始学习率
        # initial_lr = 2e-4
        # num_epochs = 450
        # mod initial_lr = 2e-3
        initial_lr = 1e-3
        num_epochs = 152
        final_wd = 1e-5
        lr_warmup_epochs = 5

        # optimizer = optimizerChoice(model.parameters(), lr=initial_lr, Choice="adamw",
        #                             betas=(0.9, 0.999), weight_decay=1e-2)
        # note 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=initial_lr,weight_decay=1e-2,betas=(0.9 ,0.999))

        unfreeze_epoch = 1  # mod
        milestones = 30
        use_ema = False   # sp 是否启用 EMA
        use_rep = True   # sp 是否进行重参数化
        use_mixup_cutmix = True   # sp 是否进行mixup的数据增强
        lr_warmup_decay = 0.1  # 预热时的初始学习率从 initial_lr * lr_warmup_decay -> lr_warmup_decay
        # note 自定义的余弦退火学习率调度器
        # scheduler = MyCosineScheduler(optimizer, Milestones=milestones, MaxEpochs=num_epochs, MinLrRate=initial_lr)

        # note 官方提供的余弦退火学习率调度器
        # warmup_scheduler = LinearLR(optimizer, start_factor=lr_warmup_decay, total_iters=lr_warmup_epochs)
        # cosine_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - lr_warmup_epochs, eta_min=final_wd)
        # scheduler = SequentialLR(
        #     optimizer,
        #     schedulers=[warmup_scheduler, cosine_scheduler],
        #     milestones=[lr_warmup_epochs]
        # )

        # note 更好的学习率调度器
        # scheduler = CustomLearningRateScheduler(optimizer, initial_lr=initial_lr)
        scheduler = ThreeStageScheduler(optimizer, initial_lr=initial_lr)


        writer = SummaryWriter(logdir="./runs")
        best_val_acc = 0.0
        best_val_acc_ema = 0.0
        best_epoch = 0

        # begin ---------------Begin Train---------------
        for epoch in range(num_epochs):
            start_time = time.time()
            scheduler.step(epoch)

            # note 判断是否解冻模型
            if unfreeze_epoch == epoch + 1:
                model = weightFrozen(model, freezeClassifier=False)

            trainer = CnnTrainerProWithOptionalEMA(device, num_classes=num_classes, optim=optimizer,
                                                   model=model, train_dataloader=train_loader, rep=use_rep, test_dataloader=val_loader,
                                                   use_mixup_cutmix=use_mixup_cutmix, use_ema=use_ema)
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # note 混淆矩阵与模型的保存`
            with torch.no_grad():
                AvgRecall = np.mean(trainer.Recall)
                AvgPrecision = np.mean(trainer.Precision)
                AvgF1Score = np.mean(trainer.F1Score)
                AvgSpecificity = np.mean(trainer.Specificity)
                if (trainer.TestAcc > best_val_acc):
                    best_val_acc = trainer.TestAcc
                    best_epoch = epoch + 1
                    torch.save(model.state_dict(), model_weights_path)
                    trainer.save_model_confusion_matrix(trainer.confusion_matrix, "./model_confusion_matrix.xlsx")


                if use_ema and trainer.ema_TestAcc > best_val_acc_ema:
                    best_val_acc_ema = trainer.ema_TestAcc
                    trainer.save_ema_confusion_matrix(trainer.ema_confusion_matrix, "./ema_confusion_matrix.xlsx")

            # note 打印信息与记录信息
            if use_ema:
                print('\n' + '=' * 80)
                print('Split: [%d/5] \t Epoch: [%d/%d]' % (split, epoch + 1, num_epochs))
                print('目前最好的验证准确率: %.3f %% (EMA: %.3f %%) \t 训练准确率: %.3f %% \t 验证准确率: %.3f %% (EMA: %.3f %%)'
                      % (best_val_acc * 100, best_val_acc_ema * 100,
                         trainer.TrainAcc * 100, trainer.TestAcc * 100,
                         trainer.ema_TestAcc * 100))
                print('距离最佳轮次: [%d] \t 训练损失: %.5f \t 验证损失: %.5f (EMA: %.5f)'
                      % (epoch + 1 - best_epoch, trainer.TrainLoss, trainer.TestLoss, trainer.ema_TestLoss))
                print('召回率: %.3f %% \t 精确率: %.3f %% \t F1得分: %.4f \t 特异性: %.4f'
                      % (AvgRecall * 100, AvgPrecision * 100, AvgF1Score, AvgSpecificity))
                print('EMA召回率: %.3f %% \t EMA精确率: %.3f %% \t EMA F1得分: %.4f \t EMA特异性: %.4f'
                      % (np.mean(trainer.ema_Recall) * 100, np.mean(trainer.ema_Precision) * 100,
                         np.mean(trainer.ema_F1Score), np.mean(trainer.ema_Specificity)))

                # 在这里添加 AUC 的打印
                print('AUC: %.4f \t EMA AUC: %.4f' % (trainer.AUC, trainer.emaAUC))

                print('耗时: %.2f 秒 \t 学习率: %.6f' % (time.time() - start_time, optimizer.param_groups[0]['lr']))
                print('=' * 80)

                # 在 TensorBoardX 中记录每个 epoch 的验证集和 EMA 模型的性能指标
                writer.add_scalar('train/loss', trainer.TrainLoss, epoch + 1)
                writer.add_scalar('train/accuracy', trainer.TrainAcc, epoch + 1)
                writer.add_scalar('val/loss', trainer.TestLoss, epoch + 1)
                writer.add_scalar('val/accuracy', trainer.TestAcc, epoch + 1)
                writer.add_scalar('val/ema_loss', trainer.ema_TestLoss, epoch + 1)
                writer.add_scalar('val/ema_accuracy', trainer.ema_TestAcc, epoch + 1)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
                writer.add_scalar('best_val_acc', best_val_acc, epoch + 1)

                # 记录 AUC 到 TensorBoardX
                writer.add_scalar('val/auc', trainer.AUC, epoch + 1)
                writer.add_scalar('val/ema_auc', trainer.emaAUC, epoch + 1)

            else:
                print('\n' + '=' * 80)
                print('Split: [%d/5] \t Epoch: [%d/%d] \t 最佳验证准确率: %.3f %% \t 训练准确率: %.3f %% \t 验证准确率: %.3f %%'
                      % (
                      split, epoch + 1, num_epochs, best_val_acc * 100, trainer.TrainAcc * 100, trainer.TestAcc * 100))
                print('距离最佳轮次: [%d] \t 训练损失: %.5f \t 验证损失: %.5f'
                      % (epoch + 1 - best_epoch, trainer.TrainLoss, trainer.TestLoss))
                print('召回率: %.3f %% \t 精确率: %.3f %% \t F1得分: %.4f \t 特异性: %.4f'
                      % (AvgRecall * 100, AvgPrecision * 100, AvgF1Score, AvgSpecificity))

                # 在这里添加 AUC 的打印
                print('AUC: %.4f' % trainer.AUC)

                print('耗时: %.2f 秒 \t 学习率: %.6f' % (time.time() - start_time, optimizer.param_groups[0]['lr']))
                print('=' * 80)

                # 在 TensorBoardX 中记录
                writer.add_scalar('train/loss', trainer.TrainLoss, epoch + 1)
                writer.add_scalar('train/accuracy', trainer.TrainAcc, epoch + 1)
                writer.add_scalar('val/loss', trainer.TestLoss, epoch + 1)
                writer.add_scalar('val/accuracy', trainer.TestAcc, epoch + 1)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)
                writer.add_scalar('best_val_acc', best_val_acc, epoch + 1)

                # 记录 AUC 到 TensorBoardX
                writer.add_scalar('val/auc', trainer.AUC, epoch + 1)

            # note 更新学习率
            # scheduler.step()

        writer.close()  # 关闭 SummaryWriter



# note 设置随机数种子
def seed_everything(seed=8080):
    print(f"\033[93m设置随机数种子{seed}\033[0m")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    import torch
    import numpy as np
    import random
    # seed_everything(3407)
    seed_everything(1111)
    main()











