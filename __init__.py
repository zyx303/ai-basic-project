"""
AI情感分类项目

这是一个基于BERT的中文情感分析系统，支持训练、评估和预测功能。
"""

__version__ = "1.0.0"
__author__ = "AI Lab"

# 导入主要模块 - 使用异常处理来兼容不同的导入方式
try:
    # 尝试相对导入（开发环境）
    from .model.SentimentClassifier import SentimentClassifier
    from .utils.data_loader import load_train, load_test, set_seed
    from .utils.train_utils import train_model, plot_training_history
except ImportError:
    # 回退到绝对导入（打包环境）
    from model.SentimentClassifier import SentimentClassifier
    from utils.data_loader import load_train, load_test, set_seed
    from utils.train_utils import train_model, plot_training_history

__all__ = [
    'SentimentClassifier',
    'load_train',
    'load_test', 
    'set_seed',
    'train_model',
    'plot_training_history'
]
