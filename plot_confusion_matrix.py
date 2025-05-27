import warnings
warnings.filterwarnings("ignore")

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

# 添加当前目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# 导入项目模块
try:
    from model.SentimentClassifier import SentimentClassifier
    from utils.data_loader import load_test, set_seed
except ImportError:
    try:
        from .model.SentimentClassifier import SentimentClassifier
        from .utils.data_loader import load_test, set_seed
    except ImportError:
        model_path = os.path.join(current_dir, 'model')
        utils_path = os.path.join(current_dir, 'utils')
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        
        from SentimentClassifier import SentimentClassifier
        from data_loader import load_test, set_seed

# 设置matplotlib中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

class ConfusionMatrixPlotter:
    """混淆矩阵绘制器"""
    
    def __init__(self, class_names=None):
        self.class_names = class_names or ["消极", "中性", "积极"]
        set_seed()
    
    def evaluate_model_predictions(self, model, test_loader, device=None):
        """
        使用模型对测试集进行预测并获取结果
        
        Args:
            model: 训练好的模型
            test_loader: 测试数据加载器
            device: 计算设备
            
        Returns:
            tuple: (真实标签列表, 预测标签列表, 预测概率列表)
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model.eval()
        model.to(device)
        
        all_labels = []
        all_preds = []
        all_probs = []
        
        print("正在进行模型预测...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if batch_idx % 10 == 0:
                    print(f"处理批次 {batch_idx}/{len(test_loader)}")
                
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids, attention_mask, token_type_ids)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        return all_labels, all_preds, all_probs
    
    def plot_confusion_matrix(self, y_true, y_pred, title="混淆矩阵", 
                            normalize=False, figsize=(8, 6), save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            title: 图表标题
            normalize: 是否标准化
            figsize: 图像大小
            save_path: 保存路径
        """
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
            title += " (标准化)"
        else:
            fmt = 'd'
        
        # 创建图像
        plt.figure(figsize=figsize)
        
        # 使用seaborn绘制热力图
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': '预测数量' if not normalize else '比例'})
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('预测标签', fontsize=12)
        plt.ylabel('真实标签', fontsize=12)
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存到: {save_path}")
        
        plt.show()
        
        return cm
    
    def plot_detailed_analysis(self, y_true, y_pred, save_dir=None):
        """
        绘制详细的分析图表，包括混淆矩阵和分类报告
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_dir: 保存目录
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 1. 绘制原始数量混淆矩阵
        save_path1 = os.path.join(save_dir, "confusion_matrix_counts.png") if save_dir else None
        cm_counts = self.plot_confusion_matrix(
            y_true, y_pred, 
            title="混淆矩阵 - 预测数量",
            normalize=False,
            save_path=save_path1
        )
        
        # 2. 绘制标准化混淆矩阵
        save_path2 = os.path.join(save_dir, "confusion_matrix_normalized.png") if save_dir else None
        cm_norm = self.plot_confusion_matrix(
            y_true, y_pred,
            title="混淆矩阵 - 标准化比例", 
            normalize=True,
            save_path=save_path2
        )
        
        # 3. 打印详细的分类报告
        print("\n" + "="*60)
        print("分类报告")
        print("="*60)
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        print(report)
        
        # 4. 计算并打印关键指标
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        print("\n" + "="*60)
        print("关键指标汇总")
        print("="*60)
        print(f"准确率 (Accuracy): {accuracy:.4f}")
        print(f"宏平均精确率 (Macro Precision): {precision_macro:.4f}")
        print(f"宏平均召回率 (Macro Recall): {recall_macro:.4f}")
        print(f"宏平均F1分数 (Macro F1): {f1_macro:.4f}")
        
        # 5. 按类别显示详细指标
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        print("\n各类别详细指标:")
        print("-" * 60)
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:>6}: 精确率={precision_per_class[i]:.4f}, "
                  f"召回率={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
        # 6. 保存分类报告到文件
        if save_dir:
            report_path = os.path.join(save_dir, "classification_report.txt")
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("分类报告\n")
                f.write("="*60 + "\n")
                f.write(report + "\n")
                f.write("\n关键指标汇总\n")
                f.write("="*60 + "\n")
                f.write(f"准确率 (Accuracy): {accuracy:.4f}\n")
                f.write(f"宏平均精确率 (Macro Precision): {precision_macro:.4f}\n")
                f.write(f"宏平均召回率 (Macro Recall): {recall_macro:.4f}\n")
                f.write(f"宏平均F1分数 (Macro F1): {f1_macro:.4f}\n")
                f.write("\n各类别详细指标:\n")
                f.write("-" * 60 + "\n")
                for i, class_name in enumerate(self.class_names):
                    f.write(f"{class_name:>6}: 精确率={precision_per_class[i]:.4f}, "
                           f"召回率={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}\n")
            print(f"\n分类报告已保存到: {report_path}")
        
        return cm_counts, cm_norm

def load_model_and_evaluate(model_path, test_data_path=None, batch_size=32, max_length=128):
    """
    加载模型并评估，绘制混淆矩阵
    
    Args:
        model_path: 模型文件路径
        test_data_path: 测试数据路径
        batch_size: 批次大小
        max_length: 最大序列长度
    """
    print(f"正在加载模型: {model_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    try:
        # 加载模型
        model = SentimentClassifier.load_from_checkpoint(model_path, device=device)
        print("模型加载成功!")
        
        # 加载测试数据
        print("正在加载测试数据...")
        num_workers = 0 if hasattr(sys, 'frozen') else 4
        test_loader = load_test(
            batch_size=batch_size,
            num_workers=num_workers,
            max_length=max_length,
            custom_test_path=test_data_path
        )
        print(f"测试数据加载成功! 共 {len(test_loader.dataset)} 个样本")
        
        # 创建混淆矩阵绘制器
        plotter = ConfusionMatrixPlotter()
        
        # 进行预测
        y_true, y_pred, y_probs = plotter.evaluate_model_predictions(model, test_loader, device)
        
        # 创建保存目录
        save_dir = "confusion_matrix_results"
        
        # 绘制详细分析
        plotter.plot_detailed_analysis(y_true, y_pred, save_dir)
        
        return y_true, y_pred, y_probs
        
    except Exception as e:
        print(f"评估过程出错: {str(e)}")
        return None, None, None

def main():
    """主函数，提供命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description='绘制模型评估混淆矩阵')
    parser.add_argument('--model_path', type=str, required=True,
                       help='模型文件路径 (.pth)')
    parser.add_argument('--test_data', type=str, default="./data/test.csv",
                       help='测试数据文件路径 (可选，默认使用内置测试集)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小 (默认: 32)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='最大序列长度 (默认: 128)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_path):
        print(f"错误: 模型文件不存在: {args.model_path}")
        return
    
    # 执行评估和绘制
    load_model_and_evaluate(
        model_path=args.model_path,
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        max_length=args.max_length
    )

if __name__ == "__main__":
    # 如果直接运行此文件，可以在这里设置默认参数进行测试
    if len(sys.argv) == 1:
        # 示例用法，你可以修改这些路径
        model_path = "checkpoints/sentiment_model.pth"  # 修改为你的模型路径
        test_data_path = None  # 使用默认测试集，或指定你的测试数据路径
        
        if os.path.exists(model_path):
            print("使用默认参数进行评估...")
            load_model_and_evaluate(model_path, test_data_path)
        else:
            print("请提供模型路径参数，或修改代码中的默认路径")
            print("用法示例:")
            print("python plot_confusion_matrix.py --model_path checkpoints/sentiment_model.pth")
            print("python plot_confusion_matrix.py --model_path checkpoints/sentiment_model.pth --test_data data/test.csv")
    else:
        main()
