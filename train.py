import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
import os

# 导入项目中的模块
from model.SentimentClassifier import SentimentClassifier
from utils import (
    load_test,
    load_train,                                                          
    set_seed,
    train_model,
    evaluate_model,
    plot_training_history,
    model_complexity,
)
import argparse
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="bert-base-chinese", help="预训练模型名称")
    p.add_argument("--model_type", type=str, default="bert", help="模型类型")
    p.add_argument("--epochs", type=int, default=3, help="训练轮数")
    p.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
    p.add_argument("--batch_size", type=int, default=32, help="批大小")
    p.add_argument("--max_length", type=int, default=128, help="最大序列长度")
    p.add_argument("--save_directory", type=str, default="checkpoints", help="模型保存目录")
    p.add_argument("--visualize_filters", action="store_true", help="是否可视化卷积核")
    p.add_argument("--visualize_predictions", action="store_true", help="是否可视化预测结果")

    return p.parse_args()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 从参数中获取值
    model_name = args.model_name
    model_type = args.model_type
    epochs = args.epochs
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    max_length = args.max_length
    save_directory = args.save_directory
    visualize_filters = args.visualize_filters
    visualize_predictions = args.visualize_predictions
    
    # 设置随机种子
    set_seed()
    # 检查是否有可用的GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    train_loader, valid_loader = load_train(batch_size=batch_size, num_workers=2, max_length=max_length,split_ratio=0.1)
    test_loader = load_test(batch_size=batch_size, num_workers=2, max_length=max_length)
    
    # 假设情感分类有3个类别：积极、中性、消极
    classes = ["negative", "neural", "positive"]
    
    # 初始化模型
    model = SentimentClassifier(model_name=model_name, num_classes=len(classes))

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 可以添加学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 确保checkpoints目录存在
    os.makedirs(save_directory, exist_ok=True)

    # 训练模型
    trained_model, history = train_model(
        model, train_loader, valid_loader, criterion, optimizer, scheduler,
        num_epochs=epochs, device=device, save_dir=save_directory
    )

    # 绘制训练历史
    plot_training_history(history, title=f"{model_name} Training History")

    # 在测试集上评估模型
    print("\n在测试集上评估模型:")
    test_loss, test_acc = evaluate_model(trained_model, test_loader, criterion, device, classes)

    print(f"{model_name} 最终测试准确率: {test_acc:.4f}")

    print(f"\n{model_name}的训练和评估已完成！")

if __name__ == "__main__":
    main()

