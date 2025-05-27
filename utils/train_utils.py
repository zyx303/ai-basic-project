import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import time
import os

def train_model(model, train_loader, valid_loader, criterion, optimizer, scheduler=None,
                num_epochs=10, device=None, save_dir='./checkpoints',progress_callback=None, 
                metrics_callback=None, save_name='model'):
    """
    训练模型并记录性能指标

    参数:
        model: 要训练的模型
        train_loader, valid_loader: 训练和验证数据加载器
        criterion: 损失函数
        optimizer: 优化器
        scheduler: 学习率调度器（可选）
        num_epochs: 训练轮数
        device: 使用的设备
        save_dir: 模型保存目录
        progress_callback: 进度回调函数
        metrics_callback: 指标回调函数，每个epoch结束时调用
        save_name: 保存的模型名称

    返回:
        history: 包含训练历史的字典
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    start_time = time.time()
    model = model.to(device)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []
    }

    best_val_acc = 0.0

    # 确保保存目录存在
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start = time.time()
        print(f"Epoch {epoch+1}/{num_epochs}")

        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 修改：支持有无进度回调的情况
        total_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            # 从字典中获取各个组件
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)
            
            # 清除之前的梯度
            optimizer.zero_grad()
            # 前向传播
            if token_type_ids is not None:
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                
            # 计算损失
            loss = criterion(outputs, labels)
            
            # 反向传播和优化
            loss.backward()
            optimizer.step()
            
            # 更新统计信息
            train_loss += loss.item() * input_ids.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            # 调用进度回调（如果提供）
            if progress_callback:
                progress_callback(epoch, batch_idx + 1, total_batches)

        # 计算训练指标
        train_loss = train_loss / len(train_loader.sampler)
        train_acc = train_correct / train_total

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in valid_loader:  # 修改: 直接获取批次数据
                # 从字典中获取各个组件
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
                labels = batch['labels'].to(device)
                
                # 前向传播
                if token_type_ids is not None:
                    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                # 计算损失
                loss = criterion(outputs, labels)
                
                # 统计
                val_loss += loss.item() * input_ids.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        # 计算验证指标
        val_loss = val_loss / len(valid_loader.sampler)
        val_acc = val_correct / val_total

        # 更新学习率
        if scheduler:
            scheduler.step()

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 记录每个epoch的时间
        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        history['epoch_times'].append(epoch_time)
        
        # 如果提供了指标回调函数，调用它
        if metrics_callback:
            metrics = {
                'loss': train_loss,
                'accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'epoch': epoch + 1
            }
            metrics_callback(metrics)

        # 如果是最佳模型，保存权重
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{save_dir}/{save_name}_best.pth")
            print(f"模型已保存到 {save_dir}/{save_name}_best.pth")

        print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
        print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
        print(f"本轮用时: {epoch_time:.2f}s")
        print("-" * 50)

    # 计算总训练时间
    total_time = time.time() - start_time
    print(f"总训练时间: {total_time:.2f}s")

    return model, history

def plot_training_history(history, title="Training History"):
    """
    绘制训练历史曲线

    参数:
        history: 包含训练历史的字典
        title: 图表标题
    """
    plt.figure(figsize=(12, 5))

    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')  # 英文标签
    plt.plot(history['val_loss'], label='Validation Loss')  # 英文标签
    plt.xlabel('Epochs')  # 英文标签
    plt.ylabel('Loss')  # 英文标签
    plt.title('Loss Curves')  # 英文标题
    plt.legend()

    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')  # 英文标签
    plt.plot(history['val_acc'], label='Validation Accuracy')  # 英文标签
    plt.xlabel('Epochs')  # 英文标签
    plt.ylabel('Accuracy')  # 英文标签
    plt.title('Accuracy Curves')  # 英文标题
    plt.legend()

    plt.suptitle(title)  # 英文总标题
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.show()


def model_complexity(model, input_size=(3, 32, 32), batch_size=128, device=None):
    """
    计算模型参数量和推理时间

    参数:
        model: 要评估的模型
        input_size: 输入尺寸
        batch_size: 批量大小
        device: 使用的设备

    返回:
        num_params: 参数量
        inference_time: 每批次推理时间
    """
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # 计算参数量
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 创建随机输入
    dummy_input = torch.randn(batch_size, *input_size).to(device)

    # 预热
    with torch.no_grad():
        for _ in range(10):
            _ = model(dummy_input)

    # 计时
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):
            _ = model(dummy_input)
    end_time = time.time()

    inference_time = (end_time - start_time) / 100

    print(f"参数量: {num_params:,}")
    print(f"每批次({batch_size}个样本)推理时间: {inference_time*1000:.2f}ms")

    return num_params, inference_time
