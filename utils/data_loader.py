import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, Dataset, random_split
import pandas as pd
import os
from transformers import AutoTokenizer

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        """情感分类数据集"""
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros(self.max_length, dtype=torch.long)).flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }
    
def get_dataset_info(train_path=None, test_path=None, max_length=128):
    """
    获取数据集的统计信息
    
    参数:
        train_path: 训练数据路径，如果为None则使用默认数据
        test_path: 测试数据路径，如果为None则使用默认数据
        max_length: 文本序列的最大长度
        
    返回:
        info: 包含数据集统计信息的字典
    """
    # 加载训练和测试数据
    train_texts, train_labels = load_sentiment_data('train', custom_path=train_path)
    test_texts, test_labels = load_sentiment_data('test', custom_path=test_path)
    
    # 合并标签以计算类别数和分布
    all_labels = train_labels + test_labels
    unique_labels = sorted(set(all_labels))
    num_classes = len(unique_labels)
    
    # 计算类别分布
    class_distribution = {}
    label_names = ["积极", "中性", "消极"]  # 假设标签0,1,2对应积极,中性,消极
    for label in unique_labels:
        train_count = train_labels.count(label)
        test_count = test_labels.count(label)
        label_name = label_names[label] if label < len(label_names) else f"类别{label}"
        class_distribution[label_name] = {
            '训练': train_count,
            '测试': test_count,
            '总计': train_count + test_count
        }
    
    # 计算平均文本长度
    all_texts = train_texts + test_texts
    avg_length = sum(len(text) for text in all_texts) / len(all_texts) if all_texts else 0
    
    # 示例数据
    # examples = {
    #     '训练': train_texts[:3] if train_texts else [],
    #     '测试': test_texts[:3] if test_texts else []
    # }
    examples = {
        'train': [],
        'test': []
    }
    for i in range(3):
        examples['train'].append(train_texts[i] if i < len(train_texts) else "无数据")
        examples['test'].append(test_texts[i] if i < len(test_texts) else "无数据")
    
    # 整理信息
    info = {
        'train_count': len(train_texts),
        'test_count': len(test_texts),
        'total_count': len(train_texts) + len(test_texts),
        'num_classes': num_classes,
        'class_distribution': class_distribution,
        'avg_length': avg_length,
        'max_length': max_length,
        'examples': examples
    }
    
    return info

# 设置随机种子，确保实验可重复性
def set_seed(seed=42):
    """
    设置随机种子，确保实验可重复性

    参数:
        seed: 随机种子
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_train(batch_size=128, num_workers=2, split_ratio=0.1, max_length=128, custom_train_path=None):
    """
    加载训练数据并分割为训练集和验证集
    
    参数:
        batch_size: 批处理大小
        num_workers: 数据加载的工作线程数
        split_ratio: 验证集所占比例
        max_length: 文本序列的最大长度
        custom_train_path: 自定义训练数据路径
    
    返回:
        train_loader: 训练数据加载器
        valid_loader: 验证数据加载器
    """
    # 加载原始数据
    texts, labels = load_sentiment_data('train', custom_path=custom_train_path)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 创建数据集
    dataset = SentimentDataset(texts, labels, tokenizer, max_length)
    
    # 计算训练集和验证集的大小
    dataset_size = len(dataset)
    valid_size = int(split_ratio * dataset_size)
    train_size = dataset_size - valid_size
    
    # 随机分割数据集
    train_dataset, valid_dataset = random_split(
        dataset, 
        [train_size, valid_size],
        generator=torch.Generator()
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True
    )
    
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(valid_dataset)}")
    
    return train_loader, valid_loader

def load_test(batch_size=128, num_workers=2, max_length=128, custom_test_path=None):
    """
    加载测试数据
    
    参数:
        batch_size: 批处理大小
        num_workers: 数据加载的工作线程数
        max_length: 文本序列的最大长度
        custom_test_path: 自定义测试数据路径
    
    返回:
        test_loader: 测试数据加载器
    """
    # 加载原始数据
    texts, labels = load_sentiment_data('test', custom_path=custom_test_path)
    
    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    
    # 创建数据集
    test_dataset = SentimentDataset(texts, labels, tokenizer, max_length)
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    print(f"测试集大小: {len(test_dataset)}")
    
    return test_loader

def load_sentiment_data(data_type='train', custom_path=None):
    """
    加载情感分析数据
    
    参数:
        data_type: 'train' 或 'test'，指定要加载的数据类型
        custom_path: 自定义数据文件路径，如果提供则忽略data_type
    
    返回:
        texts: 文本列表
        labels: 标签列表
    """
    try:
        # 尝试加载自定义文件或默认CSV文件
        if custom_path and os.path.exists(custom_path):
            file_path = custom_path
        else:
            data_dir = os.path.join('data')
            if data_type == 'train':
                file_path = os.path.join(data_dir, 'train.csv')
            else:
                file_path = os.path.join(data_dir, 'test.csv')
        
        # 根据文件扩展名加载数据
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            texts = df['text'].tolist()
            labels = df['label'].tolist()
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
            texts = df['text'].tolist() if 'text' in df.columns else df.iloc[:, 0].tolist()
            labels = df['label'].tolist() if 'label' in df.columns else df.iloc[:, 1].tolist()
        elif file_path.endswith('.txt'):
            texts = []
            labels = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        texts.append(parts[0])
                        try:
                            labels.append(int(parts[1]))
                        except ValueError:
                            labels.append(0)  # 默认标签
                    elif len(parts) == 1:
                        texts.append(parts[0])
                        labels.append(0)  # 默认标签
        else:
            raise ValueError(f"不支持的文件格式: {file_path}")
        
    except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
        print(f"警告: 加载数据出错 ({e})，使用示例数据代替")
        # 使用与原来相同的示例数据
        if data_type == 'train':
            texts = [
                "这部电影真的太棒了，我非常喜欢！",
                "这家餐厅的服务态度很差，食物也不好吃。",
                "这本书还可以，但有些地方写得不够好。",
                "今天天气不错，心情很好。",
                "这次考试我考砸了，感觉很沮丧。",
                "我对这个产品非常满意，质量超出预期。",
                "昨天的会议很无聊，浪费了我好几个小时。",
                "这个手机的电池续航能力一般。",
                "老师的讲解非常清晰，让我轻松理解了难点。",
                "这家酒店环境很差，而且价格还贵。"
            ]
            labels = [0, 2, 1, 0, 2, 0, 2, 1, 0, 2]  # 0: 积极, 1: 中性, 2: 消极
        else:
            texts = [
                "这个APP的用户体验设计很棒！",
                "这个地方真是太无聊了，什么都没有。",
                "这部剧情节发展一般，不过演员表演不错。",
                "我很喜欢这个城市的文化氛围。",
                "这家公司的客服态度实在太差了。"
            ]
            labels = [0, 2, 1, 0, 2]  # 0: 积极, 1: 中性, 2: 消极
    
    return texts, labels

def visualize_data_distribution(labels, title="数据分布"):
    """
    可视化数据分布
    
    参数:
        labels: 标签列表
        title: 图表标题
    """
    unique_labels = np.unique(labels)
    counts = [labels.count(label) for label in unique_labels]
    
    plt.figure(figsize=(8, 6))
    plt.bar(['积极', '中性', '消极'], counts)
    plt.title(title)
    plt.xlabel('情感类别')
    plt.ylabel('样本数量')
    plt.savefig(f'{title}.png')
    plt.close()

def validate_data_format(file_path):
    """
    验证数据文件格式是否正确
    
    参数:
        file_path: 数据文件路径
        
    返回:
        is_valid: 是否有效
        message: 错误信息，有效则为空
    """
    if not os.path.exists(file_path):
        return False, "文件不存在"
    
    try:
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
            if 'text' not in df.columns or 'label' not in df.columns:
                return False, "CSV文件需要包含'text'和'label'列"
        elif file_path.endswith('.tsv'):
            df = pd.read_csv(file_path, sep='\t')
            if len(df.columns) < 2:
                return False, "TSV文件需要至少包含2列(文本和标签)"
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                if '\t' not in first_line:
                    return False, "TXT文件每行应为'文本\\t标签'格式"
        else:
            return False, "不支持的文件格式(支持.csv, .tsv, .txt)"
            
        return True, ""
    except Exception as e:
        return False, f"验证文件格式出错: {str(e)}"
