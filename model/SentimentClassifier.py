import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, BertConfig
from torch.optim import AdamW
import numpy as np
from sklearn.metrics import accuracy_score
import os

class SentimentClassifier(nn.Module):
    def __init__(self, model_name="bert-base-chinese", num_classes=3, dropout_rate=0.1, hidden_dim=768, local_model=False):
        """
        初始化情感分类模型
        Args:
            model_name: 预训练模型名称或路径
            num_classes: 类别数量(3,positive,negative,neutral)
            dropout_rate: Dropout比率
            hidden_dim: MLP隐藏层维度
            local_model: 是否从本地加载模型(不下载预训练权重)
        """
        super(SentimentClassifier, self).__init__()
        self.model_name = model_name
        self.num_classes = num_classes
        
        if not local_model:
            # 加载预训练模型
            try:
                self.embed = AutoModel.from_pretrained(model_name)
                # 加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(f"加载预训练模型出错: {str(e)}")
                # 使用配置创建模型，但不加载预训练权重
                config = BertConfig.from_pretrained(model_name)
                self.embed = AutoModel.from_config(config)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        else:
            # 使用BERT配置创建模型，但不下载预训练权重
            config = BertConfig.from_pretrained(model_name, local_files_only=True)
            self.embed = AutoModel.from_config(config)
            # 尝试本地加载tokenizer或使用默认配置
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            except:
                print("无法从本地加载tokenizer，将使用默认配置")
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", local_files_only=False)
        
        # 分类器部分
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Sequential(
            nn.Linear(self.embed.config.hidden_size, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        """
        前向传播函数
        """
        outputs = self.embed(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # 取[CLS]对应的隐藏状态作为句子表示
        pooled_output = outputs.pooler_output
        
        # 添加dropout并分类
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def prepare_input(self, texts):
        """准备模型输入数据"""
        return self.tokenizer(
            texts, 
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512
        )
    
    def predict(self, texts, device="cpu"):
        """对输入文本进行情感分类预测"""
        self.eval()
        self.to(device)
        
        inputs = self.prepare_input(texts)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids", None)
            )
            
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, device="cpu"):
        """
        从检查点加载模型
        Args:
            checkpoint_path: 保存的模型路径
            device: 使用的设备
        Returns:
            加载好的模型
        """
        try:
            # 首先加载状态字典(不加载到模型)
            state_dict = torch.load(checkpoint_path, map_location=device)
            
            # 从状态字典中推断参数
            # classifier.0.weight的形状是[hidden_dim, embed_dim]
            # classifier.3.weight的形状是[num_classes, hidden_dim]
            if 'classifier.0.bias' in state_dict:
                hidden_dim = state_dict['classifier.0.bias'].size(0)
            else:
                # 备用方案：从权重矩阵推断
                hidden_dim = state_dict['classifier.0.weight'].size(0)
            
            # 推断类别数
            if 'classifier.3.bias' in state_dict:
                num_classes = state_dict['classifier.3.bias'].size(0)
            else:
                num_classes = state_dict['classifier.3.weight'].size(0)
            
            # 推断dropout率（从权重名称推断，默认值0.1）
            dropout_rate = 0.1
            
            # 使用推断出的参数创建模型
            model = cls(
                local_model=True, 
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                dropout_rate=dropout_rate
            )
            
            # 加载状态字典
            model.load_state_dict(state_dict)
            
            # 将模型放到指定设备
            model.to(device)
            model.eval()
            
            return model
        except Exception as e:
            print(f"加载检查点出错: {str(e)}")
            raise e





# def sentiment_example():
#     """情感分类示例"""
#     # 示例数据
#     texts = [
#         "这部电影真的太棒了，我非常喜欢！",
#         "这家餐厅的服务态度很差，食物也不好吃。",
#         "这本书还可以，但有些地方写得不够好。",
#         "今天天气不错，心情很好。",
#         "这次考试我考砸了，感觉很沮丧。"
#     ]
#     labels = [0, 2, 1, 0, 2]  # 0: 积极, 1: 中性, 2: 消极
    
#     # 创建模型
#     model = SentimentClassifier(model_name="bert-base-chinese", num_classes=3)
    
#     # 创建数据集和数据加载器
#     dataset = SentimentDataset(texts, labels, model.tokenizer)
#     dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
#     # 训练模型
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     history = model.train_model(dataloader, epochs=1, device=device)
    
#     # 进行预测
#     new_texts = [
#         "真是太开心了，我得了第一名！",
#         "这个产品的质量一般般。",
#         "服务太差了，我再也不会来这家店了。"
#     ]
#     predictions, probabilities = model.predict(new_texts, device=device)
    
#     # 将预测结果映射到情感标签
#     sentiment_labels = ["积极", "中性", "消极"]
#     for i, text in enumerate(new_texts):
#         pred_label = sentiment_labels[predictions[i]]
#         print(f"文本: {text}")
#         print(f"预测情感: {pred_label}")
#         print(f"各类别概率: {probabilities[i]}")
#         print("-" * 50)
    
#     return model


# if __name__ == "__main__":
#     model = sentiment_example()
