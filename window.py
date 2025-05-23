import warnings 
warnings.filterwarnings("ignore")
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, 
                            QTabWidget, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QProgressBar, QTextEdit, QSplitter, QTableWidget,
                            QTableWidgetItem, QSlider,QMessageBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入情感分类相关模块
import torch
import torch.nn as nn
import torch.optim as optim
from model.SentimentClassifier import SentimentClassifier
from utils.data_loader import (load_test, load_train, set_seed, get_dataset_info, 
                               validate_data_format, visualize_data_distribution)
from utils.train_utils import train_model

class DataLoadingThread(QThread):
    """用于后台加载数据集的线程类"""
    update_progress = pyqtSignal(int)
    loading_complete = pyqtSignal(object, object, object, str)
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    def run(self):
        try:
            # 获取参数
            batch_size = self.params.get('batch_size', 64)
            max_length = self.params.get('max_length', 128)
            train_split = self.params.get('train_split', 0.9)
            train_path = self.params.get('train_path')
            test_path = self.params.get('test_path')
            
            # 模拟加载进度
            self.update_progress.emit(10)
            
            # 加载训练和验证集
            self.update_progress.emit(20)
            train_loader, valid_loader = load_train(
                batch_size=batch_size,
                num_workers=4,
                split_ratio=1.0-train_split,
                max_length=max_length,
                custom_train_path=train_path
            )
            
            self.update_progress.emit(60)
            
            # 加载测试集
            test_loader = load_test(
                batch_size=batch_size,
                num_workers=4,
                max_length=max_length,
                custom_test_path=test_path
            )
            
            self.update_progress.emit(90)
            
            # 获取数据集信息
            info_str = f"数据集已成功加载: {len(train_loader.dataset)} 训练样本,\n {len(valid_loader.dataset)} 验证样本,\n {len(test_loader.dataset)} 测试样本"
            
            self.update_progress.emit(100)
            
            # 发送加载完成信号
            self.loading_complete.emit(train_loader, valid_loader, test_loader, info_str)
            
        except Exception as e:
            self.loading_complete.emit(None, None, None, f"加载数据集时出错: {str(e)}")

class ModelLoadingThread(QThread):
    """用于后台加载模型的线程类"""
    update_progress = pyqtSignal(int)
    loading_complete = pyqtSignal(object, str)
    
    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        
    def run(self):
        try:
            # 模拟进度更新
            self.update_progress.emit(10)
            
            # 加载模型 - 使用新的类方法
            self.update_progress.emit(30)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.update_progress.emit(50)
            model = SentimentClassifier.load_from_checkpoint(self.model_path, device=device)
            
            self.update_progress.emit(80)
            
            # 发送加载完成信号
            self.update_progress.emit(100)
            self.loading_complete.emit(model, "模型加载成功")
            
        except Exception as e:
            self.loading_complete.emit(None, f"加载模型出错: {str(e)}")

class TrainingThread(QThread):
    """用于后台训练模型的线程类"""
    update_progress = pyqtSignal(int)
    update_metrics = pyqtSignal(dict)
    training_complete = pyqtSignal(object)  # 传递训练后的模型
    
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.train_loader = None
        self.valid_loader = None
        self.model = None
        self.history = None
        
    def run(self):
        try:
            # 设置随机种子
            set_seed()
            
            # 参数准备
            batch_size = self.params.get('batch_size', 32)
            learning_rate = self.params.get('learning_rate', 2e-5)
            epochs = self.params.get('epochs', 3)
            model_name = self.params.get('model_name', 'bert-base-chinese')
            max_length = self.params.get('max_length', 128)
            save_directory = self.params.get('save_directory', 'checkpoints')
            save_name = self.params.get('save_name', 'sentiment_model')
            device = self.params.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')    
            hidden_dim = self.params.get('hidden_dim', 768)  # 新增隐藏层维度参数
            dropout_rate = self.params.get('dropout_rate', 0.1)  # 新增dropout比率参数

            optimizer_name = self.params.get('optimizer', 'Adam')
            loss_function = self.params.get('loss_function', 'CrossEntropyLoss')

            
            # 确保保存目录存在
            os.makedirs(save_directory, exist_ok=True)
            
            # 检查设备
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            classes = ["negative", "neural", "positive"]
            
            # 初始化模型，传递新的超参数
            model = SentimentClassifier(
                model_name=model_name, 
                num_classes=len(classes),
                dropout_rate=dropout_rate,
                hidden_dim=hidden_dim
            )
            
            # 定义损失函数和优化器
            # criterion = nn.CrossEntropyLoss()
            # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            if loss_function == 'CrossEntropyLoss':
                criterion = nn.CrossEntropyLoss()
            elif loss_function == 'BCEWithLogitsLoss':
                criterion = nn.BCEWithLogitsLoss()
            else:
                raise ValueError("Unsupported loss function")
            
            if optimizer_name == 'Adam':
                optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'AdamW':
                optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'SGD':
                optimizer = optim.SGD(model.parameters(), lr=learning_rate)
            elif optimizer_name == 'RMSprop':
                optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
            else:
                raise ValueError("Unsupported optimizer")


            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # 自定义进度回调函数
            def progress_callback(epoch, batch, total_batches):
                progress = int(epoch / epochs * 100 + (batch / total_batches) * (100 / epochs))
                self.update_progress.emit(progress)
            
            # 增加指标回调函数
            def metrics_callback(metrics):
                self.update_metrics.emit(metrics)
            
            # 训练模型
            trained_model, history = train_model(
                model, self.train_loader, self.valid_loader, criterion, optimizer, scheduler,
                num_epochs=epochs, device=device, save_dir=save_directory,save_name=save_name,
                progress_callback=progress_callback, metrics_callback=metrics_callback
            )
            
            # 保存结果
            self.model = trained_model
            self.history = history
            
            # 发出训练完成信号
            self.training_complete.emit(trained_model)
            
        except Exception as e:
            print(f"训练出错: {str(e)}")
            self.training_complete.emit(None)

class EvaluationThread(QThread):
    """用于后台评估模型的线程类"""
    update_progress = pyqtSignal(int)
    evaluation_complete = pyqtSignal(dict)  # 传递所有评估指标
    
    def __init__(self, model, params, test_loader=None):
        super().__init__()
        self.model = model
        self.params = params
        self.test_loader = test_loader
        
    def run(self):
        try:
            # 进度指示
            self.update_progress.emit(10)
            
            # 参数准备
            device = self.params.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            max_length = self.params.get('max_length', 128)
            test_path = self.params.get('test_path')
            batch_size = self.params.get('batch_size', 32)
            
            # 如果没有提供测试加载器，则创建一个
            if not self.test_loader:
                self.update_progress.emit(20)
                self.test_loader = load_test(
                    batch_size=batch_size, 
                    num_workers=4,
                    max_length=max_length,
                    custom_test_path=test_path
                )
            
            self.update_progress.emit(30)
            
            # 定义评估标准
            criterion = nn.CrossEntropyLoss()
            classes = ["negative", "neural", "positive"]
            
            # 评估模型
            self.update_progress.emit(40)
            from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
            
            self.model.eval()
            self.model.to(device)
            
            all_preds = []
            all_labels = []
            total_loss = 0.0
            total_samples = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(self.test_loader):
                    # 更新进度
                    progress = 40 + int(50 * batch_idx / len(self.test_loader))
                    self.update_progress.emit(progress)
                    
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
                    labels = batch['labels'].to(device)
                    
                    outputs = self.model(input_ids, attention_mask, token_type_ids)
                    loss = criterion(outputs, labels)
                    
                    # 累计损失
                    total_loss += loss.item() * labels.size(0)
                    total_samples += labels.size(0)
                    
                    # 获取预测结果
                    _, predicted = torch.max(outputs, 1)
                    
                    # 收集预测和真实标签用于计算指标
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # 计算所有指标
            avg_loss = total_loss / total_samples
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='macro')
            recall = recall_score(all_labels, all_preds, average='macro')
            f1 = f1_score(all_labels, all_preds, average='macro')
            
            # 汇总结果
            metrics = {
                'test_loss': avg_loss,
                'test_acc': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1
            }
            
            self.update_progress.emit(100)
            
            # 发送完成信号
            self.evaluation_complete.emit(metrics)
            
        except Exception as e:
            print(f"评估出错: {str(e)}")
            self.evaluation_complete.emit({})

class PredictionThread(QThread):
    """用于后台进行情感预测的线程类"""
    update_progress = pyqtSignal(int)
    prediction_complete = pyqtSignal(dict)  # 传递预测结果
    
    def __init__(self, model, text, max_length=128):
        super().__init__()
        self.model = model
        self.text = text
        self.max_length = max_length
        
    def run(self):
        try:
            # 进度指示 - 开始
            self.update_progress.emit(10)
            
            # 设置设备
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # 加载tokenizer
            self.update_progress.emit(30)
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            
            # 处理文本
            self.update_progress.emit(50)
            inputs = tokenizer(
                self.text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # print(inputs)
            
            # 将输入移至与模型相同的设备
            self.update_progress.emit(60)
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            token_type_ids = inputs.get('token_type_ids', torch.zeros_like(input_ids)).to(device)
            
            # 将模型设置为评估模式
            self.model.eval()
            self.model.to(device)
            
            # 使用模型进行预测
            self.update_progress.emit(80)
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask, token_type_ids)
                
            # 获取预测结果
            sentiment_labels = ["消极", "中性", "积极"]
            probabilities = torch.nn.functional.softmax(outputs, dim=1)

            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
            
            # 整理结果
            result = {
                'prediction': prediction,
                'label': sentiment_labels[prediction],
                'confidence': confidence,
                'probabilities': probabilities[0].cpu().numpy().tolist()
            }
            
            self.update_progress.emit(100)
            
            # 发送完成信号
            self.prediction_complete.emit(result)
            
        except Exception as e:
            print(f"预测出错: {str(e)}")
            self.prediction_complete.emit({})

class MetricPlotCanvas(FigureCanvas):
    """用于显示训练指标的画布"""
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(1, 2, figsize=(width, height), dpi=dpi)
        super(MetricPlotCanvas, self).__init__(self.fig)
        
        self.loss_data = {'train': [], 'val': []}
        self.acc_data = {'train': [], 'val': []}
        self.epochs = []
        
        self.setup_plots()
        
    def setup_plots(self):
        self.axes[0].set_title('损失')
        self.axes[0].set_xlabel('Epoch')
        self.axes[0].set_ylabel('Loss')
        self.axes[0].grid(True)
        
        self.axes[1].set_title('准确率')
        self.axes[1].set_xlabel('Epoch')
        self.axes[1].set_ylabel('Accuracy')
        self.axes[1].grid(True)
        
        self.fig.tight_layout()
    
    def update_plots(self, metrics):
        epoch = metrics.get('epoch', 0)
        
        if epoch not in self.epochs:
            self.epochs.append(epoch)
            self.loss_data['train'].append(metrics.get('loss', 0))
            self.loss_data['val'].append(metrics.get('val_loss', 0))
            self.acc_data['train'].append(metrics.get('accuracy', 0))
            self.acc_data['val'].append(metrics.get('val_accuracy', 0))
            
            # 清除并重绘图表
            self.axes[0].clear()
            self.axes[1].clear()
            
            self.axes[0].plot(self.epochs, self.loss_data['train'], 'bo-', label='训练')
            self.axes[0].plot(self.epochs, self.loss_data['val'], 'ro-', label='验证')
            self.axes[0].legend()
            self.axes[0].set_title('损失')
            self.axes[0].grid(True)
            
            self.axes[1].plot(self.epochs, self.acc_data['train'], 'bo-', label='训练')
            self.axes[1].plot(self.epochs, self.acc_data['val'], 'ro-', label='验证')
            self.axes[1].legend()
            self.axes[1].set_title('准确率')
            self.axes[1].grid(True)
            
            self.fig.canvas.draw()

class AIModelGUI(QMainWindow):
    """主应用窗口"""
    def __init__(self):
        super().__init__()
        self.init_ui()
        self.training_thread = None
        self.current_metrics = {}
        self.trained_model = None
        
        # 设置随机种子
        set_seed()
    
    def init_ui(self):
        self.setWindowTitle('情感分类模型训练工具')
        self.setGeometry(100, 100, 1200, 1400)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡窗口部件
        tabs = QTabWidget()
        
        # 创建各选项卡
        dataset_tab = self.create_dataset_tab()
        train_tab = self.create_training_tab()
        eval_tab = self.create_evaluation_tab()
        
        # 添加选项卡到选项卡窗口部件
        tabs.addTab(dataset_tab, "数据集")
        tabs.addTab(train_tab, "训练")
        tabs.addTab(eval_tab, "评估")
        
        # 将选项卡窗口部件添加到主布局
        main_layout.addWidget(tabs)
    
    def create_dataset_tab(self):
        """创建数据集选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 数据集选择组
        dataset_group = QGroupBox("数据集选择")
        dataset_layout = QVBoxLayout(dataset_group)
        
        # 训练集选择
        train_layout = QHBoxLayout()
        train_layout.addWidget(QLabel("训练集文件:"))
        self.train_path = QLineEdit()
        train_layout.addWidget(self.train_path)
        browse_train = QPushButton("浏览...")
        browse_train.clicked.connect(self.browse_train_path)
        train_layout.addWidget(browse_train)
        
        # 测试集选择
        test_layout = QHBoxLayout()
        test_layout.addWidget(QLabel("测试集文件:"))
        self.test_path = QLineEdit()
        test_layout.addWidget(self.test_path)
        browse_test = QPushButton("浏览...")
        browse_test.clicked.connect(self.browse_test_path)
        test_layout.addWidget(browse_test)
        
        dataset_layout.addLayout(train_layout)
        dataset_layout.addLayout(test_layout)
        
        # 数据集预处理选项
        preprocess_group = QGroupBox("数据预处理")
        preprocess_layout = QFormLayout(preprocess_group)
        
        self.max_length = QSpinBox()
        self.max_length.setRange(16, 512)
        self.max_length.setValue(128)
        self.max_length.setSingleStep(16)
        preprocess_layout.addRow("最大序列长度:", self.max_length)
        
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.95)
        self.train_split.setSingleStep(0.05)
        self.train_split.setValue(0.9)  # 90%训练，10%验证
        self.train_split.setDecimals(2)
        preprocess_layout.addRow("训练集比例:", self.train_split)
        
        # 数据加载按钮
        button_layout = QHBoxLayout()
        load_button = QPushButton("加载数据")
        load_button.clicked.connect(self.load_dataset)
        self.preview_button = QPushButton("预览数据")
        self.preview_button.clicked.connect(self.preview_dataset)
        # self.preview_button.setEnabled(False)  # 初始禁用，直到数据加载完成
        button_layout.addWidget(load_button)
        button_layout.addWidget(self.preview_button)
        
        # 数据集信息显示
        self.dataset_info = QTextEdit()
        self.dataset_info.setReadOnly(True)
        self.dataset_info.setFixedHeight(200)

        # 数据加载进度条
        loading_progress_layout = QHBoxLayout()
        loading_progress_layout.addWidget(QLabel("加载进度:"))
        self.loading_progress_bar = QProgressBar()
        self.loading_progress_bar.setValue(0)
        loading_progress_layout.addWidget(self.loading_progress_bar)

        
        # 添加组到布局
        layout.addWidget(dataset_group)
        layout.addWidget(preprocess_group)
        layout.addLayout(button_layout)
        layout.addLayout(loading_progress_layout)  # 新增的进度条
        layout.addWidget(QLabel("数据集信息:"))
        layout.addWidget(self.dataset_info)
        layout.addStretch()
        
        return tab
    
    def create_training_tab(self):
        """创建训练选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 超参数设置
        hyperparams_group = QGroupBox("超参数设置")
        hyperparams_layout = QFormLayout(hyperparams_group)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 256)
        self.batch_size.setValue(64)
        self.batch_size.setSingleStep(8)
        hyperparams_layout.addRow("Batch Size:", self.batch_size)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(3)
        hyperparams_layout.addRow("Epochs:", self.epochs)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0000001, 0.1)
        self.learning_rate.setDecimals(7)
        self.learning_rate.setSingleStep(0.00001)
        self.learning_rate.setValue(0.00002)
        hyperparams_layout.addRow("学习率:", self.learning_rate)
        
        self.optimizer = QComboBox()
        self.optimizer.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        hyperparams_layout.addRow("优化器:", self.optimizer)
        
        self.loss_function = QComboBox()
        self.loss_function.addItems(["CrossEntropyLoss", "BCEWithLogitsLoss"])
        hyperparams_layout.addRow("损失函数:", self.loss_function)
        
        # 模型参数设置
        model_params_group = QGroupBox("模型参数设置")
        model_params_layout = QFormLayout(model_params_group)
        self.hidden_dim = QSpinBox()
        self.hidden_dim.setRange(1, 1024)
        self.hidden_dim.setValue(768)
        self.hidden_dim.setSingleStep(16)
        model_params_layout.addRow("隐藏层维度:", self.hidden_dim)
        self.dropout_rate = QDoubleSpinBox()
        self.dropout_rate.setRange(0.0, 1.0)
        self.dropout_rate.setValue(0.1)
        self.dropout_rate.setDecimals(2)
        self.dropout_rate.setSingleStep(0.01)
        model_params_layout.addRow("Dropout比率:", self.dropout_rate)
        self.model_name = QComboBox()
        self.model_name.addItems(["bert-base-chinese", "hfl/chinese-bert-wwm", "hfl/chinese-roberta-wwm-ext", "nghuyong/ernie-3.0-base-zh"])
        model_params_layout.addRow("预训练模型:", self.model_name)


        # 训练曲线可视化
        plot_group = QGroupBox("训练过程可视化")
        plot_layout = QVBoxLayout(plot_group)
        
        # 创建绘图画布
        self.metrics_canvas = MetricPlotCanvas(width=8, height=4, dpi=100)
        plot_layout.addWidget(self.metrics_canvas)
        
        # 训练控制
        training_group = QGroupBox("训练控制")
        training_layout = QVBoxLayout(training_group)
        
        # 训练按钮组
        button_layout = QHBoxLayout()
        self.start_button = QPushButton("开始训练")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("停止")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop_training)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        # 保存路径设置
        save_layout = QHBoxLayout()
        save_layout.addWidget(QLabel("保存路径:"))
        self.save_path = QLineEdit("checkpoints")
        save_layout.addWidget(self.save_path)
        
        browse_save = QPushButton("浏览...")
        browse_save.clicked.connect(self.browse_save_path)
        save_layout.addWidget(browse_save)

        # 保存的名称
        save_name_layout = QHBoxLayout()
        save_name_layout.addWidget(QLabel("保存名称:"))
        self.save_name = QLineEdit("sentiment_model")
        save_name_layout.addWidget(self.save_name)
        
        # 进度条
        progress_layout = QHBoxLayout()
        progress_layout.addWidget(QLabel("训练进度:"))
        self.progress_bar = QProgressBar()
        progress_layout.addWidget(self.progress_bar)
        
        # 日志输出
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        
        # 添加到训练控制组
        training_layout.addLayout(button_layout)
        training_layout.addLayout(save_layout)
        training_layout.addLayout(save_name_layout)
        training_layout.addLayout(progress_layout)
        training_layout.addWidget(QLabel("训练日志:"))
        training_layout.addWidget(self.log_output)
        
        # 添加组到布局
        layout.addWidget(hyperparams_group)
        layout.addWidget(model_params_group)
        layout.addWidget(plot_group)  # 添加训练曲线可视化
        layout.addWidget(training_group)
        
        return tab
    
    def create_evaluation_tab(self):
        """创建评估选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 输入框和按钮
        input_group = QGroupBox("模型输入")
        input_layout = QVBoxLayout(input_group)
        
        # 添加文本输入区域
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("请在此输入要分析的文本...")
        self.text_input.setMinimumHeight(100)
        input_layout.addWidget(QLabel("输入文本:"))
        input_layout.addWidget(self.text_input)
        
        # 添加预测按钮
        predict_button = QPushButton("进行情感预测")
        predict_button.clicked.connect(self.predict_sentiment)
        predict_button.setEnabled(False)
        self.predict_button = predict_button  # 保存为类成员以便后续启用/禁用
        input_layout.addWidget(predict_button)
        
        # 添加预测进度条
        predict_progress_layout = QHBoxLayout()
        predict_progress_layout.addWidget(QLabel("预测进度:"))
        self.predict_progress_bar = QProgressBar()
        self.predict_progress_bar.setValue(0)
        predict_progress_layout.addWidget(self.predict_progress_bar)
        input_layout.addLayout(predict_progress_layout)
        
        # 添加预测结果显示区域
        input_layout.addWidget(QLabel("预测结果:"))
        self.prediction_result = QTextEdit()
        self.prediction_result.setReadOnly(True)
        self.prediction_result.setMaximumHeight(80)
        input_layout.addWidget(self.prediction_result)
        
        # 评估控制组
        eval_control_group = QGroupBox("评估控制")
        eval_control_layout = QHBoxLayout(eval_control_group)
        
        self.eval_button = QPushButton("评估模型")
        self.eval_button.clicked.connect(self.evaluate_model)
        self.eval_button.setEnabled(False)  # 初始禁用，直到有训练好的模型
        
        self.load_model_button = QPushButton("加载模型")
        self.load_model_button.clicked.connect(self.load_model)
        
        eval_control_layout.addWidget(self.eval_button)
        eval_control_layout.addWidget(self.load_model_button)
        
        # 添加模型加载进度条
        model_loading_progress_layout = QHBoxLayout()
        model_loading_progress_layout.addWidget(QLabel("模型加载进度:"))
        self.model_loading_progress_bar = QProgressBar()
        self.model_loading_progress_bar.setValue(0)
        model_loading_progress_layout.addWidget(self.model_loading_progress_bar)
        
        # 添加评估进度条
        eval_progress_layout = QHBoxLayout()
        eval_progress_layout.addWidget(QLabel("评估进度:"))
        self.eval_progress_bar = QProgressBar()
        self.eval_progress_bar.setValue(0)
        eval_progress_layout.addWidget(self.eval_progress_bar)
        
        # 创建指标表格
        metrics_group = QGroupBox("评估指标")
        metrics_layout = QVBoxLayout(metrics_group)
        
        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics_table.setRowCount(5)
        
        # 预设一些指标行
        metrics = ["准确率", "损失", "精确率", "召回率", "F1分数"]
        for i, metric in enumerate(metrics):
            self.metrics_table.setItem(i, 0, QTableWidgetItem(metric))
            self.metrics_table.setItem(i, 1, QTableWidgetItem("N/A"))
        
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        metrics_layout.addWidget(self.metrics_table)
        
        # 添加组件到布局
        layout.addWidget(eval_control_group)
        layout.addLayout(model_loading_progress_layout)
        layout.addLayout(eval_progress_layout)  # 添加评估进度条
        layout.addWidget(metrics_group)
        layout.addWidget(input_group)
        
        return tab
    
    
    def create_spin_box(self, min_val, max_val, default, step=1):
        """创建数值选择框"""
        spin_box = QSpinBox()
        spin_box.setRange(min_val, max_val)
        spin_box.setValue(default)
        spin_box.setSingleStep(step)
        return spin_box
    
    def create_double_spin_box(self, min_val, max_val, default, step=0.1, decimals=2):
        """创建浮点数选择框"""
        spin_box = QDoubleSpinBox()
        spin_box.setRange(min_val, max_val)
        spin_box.setValue(default)
        spin_box.setSingleStep(step)
        spin_box.setDecimals(decimals)
        return spin_box
    
    def create_combo_box(self, items):
        """创建下拉选择框"""
        combo_box = QComboBox()
        combo_box.addItems(items)
        return combo_box
    
    def create_check_box(self, checked=False):
        """创建复选框"""
        check_box = QCheckBox()
        check_box.setChecked(checked)
        return check_box
    
    def browse_dataset(self):
        """打开文件对话框选择数据集"""
        directory = QFileDialog.getExistingDirectory(self, "选择数据集文件夹")
        if directory:
            self.dataset_path.setText(directory)
    
    def browse_save_path(self):
        """打开文件对话框选择保存路径"""
        directory = QFileDialog.getExistingDirectory(self, "选择保存路径")
        if directory:
            self.save_path.setText(directory)
    
    def browse_train_path(self):
        """选择训练数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择训练数据文件", "", "数据文件 (*.csv *.tsv *.txt);;CSV文件 (*.csv);;TSV文件 (*.tsv);;文本文件 (*.txt)"
        )
        if file_path:
            self.train_path.setText(file_path)

    def browse_test_path(self):
        """选择测试数据文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择测试数据文件", "", "数据文件 (*.csv *.tsv *.txt);;CSV文件 (*.csv);;TSV文件 (*.tsv);;文本文件 (*.txt)"
        )
        if file_path:
            self.test_path.setText(file_path)

    def update_loading_progress(self, value):
        """更新数据加载进度条"""
        self.loading_progress_bar.setValue(value)

    def loading_complete(self, train_loader, valid_loader, test_loader, info):
        """数据加载完成的处理"""
        if train_loader and valid_loader and test_loader:
            self.train_loader = train_loader
            self.valid_loader = valid_loader
            self.test_loader = test_loader
            self.dataset_ready = True
            
            # 清除并显示结果信息
            self.dataset_info.clear()
            self.dataset_info.append(info)
            
            # 尝试生成并显示分布图
            try:
                from utils.data_loader import load_sentiment_data
                train_path = self.train_path.text() if self.train_path.text() else None
                test_path = self.test_path.text() if self.test_path.text() else None
                
                train_texts, train_labels = load_sentiment_data('train', train_path)
                test_texts, test_labels = load_sentiment_data('test', test_path)
                
                visualize_data_distribution(train_labels, "训练集分布")
                visualize_data_distribution(test_labels, "测试集分布")
                self.preview_button.setEnabled(True)  # 启用预览按钮
            except Exception as e:
                self.dataset_info.append(f"无法生成分布图: {str(e)}")
        else:
            self.dataset_info.append(info)
            self.dataset_ready = False


    def load_dataset(self):
        """加载数据集并应用预处理参数"""
        self.dataset_info.append("正在加载数据集...")
        self.loading_progress_bar.setValue(0)
        
        try:
            # 获取用户选择的文件路径
            train_path = self.train_path.text() if self.train_path.text() else None
            test_path = self.test_path.text() if self.test_path.text() else None
            
            # 验证文件格式
            if train_path:
                is_valid, message = validate_data_format(train_path)
                if not is_valid:
                    self.dataset_info.append(f"训练集文件格式错误: {message}")
                    return
                    
            if test_path:
                is_valid, message = validate_data_format(test_path)
                if not is_valid:
                    self.dataset_info.append(f"测试集文件格式错误: {message}")
                    return
            
            # 获取用户设置的参数
            max_length = self.max_length.value()
            train_split = self.train_split.value()
            batch_size = self.batch_size.value()    
            
            # 保存到类变量中，以便在训练时使用
            self.dataset_params = {
                'max_length': max_length,
                'train_split': train_split,
                'train_path': train_path,
                'test_path': test_path,
                'batch_size': batch_size
            }
            
            self.dataset_info.append("正在后台加载和处理数据集，请稍候...")
            
            # 创建并启动数据加载线程
            self.loading_thread = DataLoadingThread(self.dataset_params)
            self.loading_thread.update_progress.connect(self.update_loading_progress)
            self.loading_thread.loading_complete.connect(self.loading_complete)
            self.loading_thread.start()
            
        except Exception as e:
            self.dataset_info.append(f"数据集准备过程出错: {str(e)}")
            self.dataset_ready = False

    def preview_dataset(self):
        """预览数据集内容和分布"""
        from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTabWidget, QLabel
        from PyQt5.QtGui import QPixmap
        import os

        train_path = self.train_path.text() if self.train_path.text() else None
        test_path = self.test_path.text() if self.test_path.text() else None
        
        try:
            # 创建一个对话框窗口
            preview_dialog = QDialog(self)
            preview_dialog.setWindowFlags(preview_dialog.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            preview_dialog.setWindowTitle("数据集预览")
            preview_dialog.resize(1920, 1080)
            layout = QVBoxLayout(preview_dialog)
            
            # 创建选项卡
            tabs = QTabWidget()
            
            # 示例数据选项卡
            examples_tab = QWidget()
            examples_layout = QVBoxLayout(examples_tab)
            
            # 获取数据集信息
            dataset_info = get_dataset_info(train_path, test_path)
            examples = dataset_info['examples']
            
            label = ["消极", "中性", "积极"]
            # 训练集示例
            examples_layout.addWidget(QLabel("<h3>训练集示例</h3>"))
            for i, text in enumerate(examples['训练']):
                examples_layout.addWidget(QLabel(f"<b>示例 {i+1}:</b> 文本：{text['text']}      标签：{label[text['label']]}"))
            
            # 测试集示例
            examples_layout.addWidget(QLabel("<h3>测试集示例</h3>"))
            for i, text in enumerate(examples['测试']):
                examples_layout.addWidget(QLabel(f"<b>示例 {i+1}:</b> 文本：{text['text']}      标签：{label[text['label']]}"))
            
            examples_layout.addStretch()
            
            # 分布图选项卡
            distribution_tab = QWidget()
            distribution_layout = QHBoxLayout(distribution_tab)
            
            # 加载并显示分布图
            train_dist_image = QLabel()
            test_dist_image = QLabel()
            
            # 检查图片文件是否存在
            train_dist_path = os.path.join(os.getcwd(), "训练集分布.png")
            test_dist_path = os.path.join(os.getcwd(), "测试集分布.png")
            
            # 在 preview_dataset 方法中，找到加载图片的部分并修改
            if os.path.exists(train_dist_path):
                pixmap = QPixmap(train_dist_path)
                # 不用缩放到QLabel的当前大小，而是设置固定大小
                train_dist_image.setMinimumSize(1200, 900)  # 设置最小尺寸
                # 使用QLabel的大小而不是原始QLabel大小来缩放
                train_dist_image.setPixmap(pixmap.scaled(1200, 900, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                train_dist_image.setAlignment(Qt.AlignCenter)
            else:
                train_dist_image.setText("训练集分布图不可用")
                train_dist_image.setAlignment(Qt.AlignCenter)

            if os.path.exists(test_dist_path):
                pixmap = QPixmap(test_dist_path)
                # 同样设置固定大小
                test_dist_image.setMinimumSize(1200, 900)  # 设置最小尺寸
                test_dist_image.setPixmap(pixmap.scaled(1200, 900, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                test_dist_image.setAlignment(Qt.AlignCenter)
            else:
                test_dist_image.setText("测试集分布图不可用")
                test_dist_image.setAlignment(Qt.AlignCenter)
            
            distribution_layout.addWidget(train_dist_image)
            distribution_layout.addWidget(test_dist_image)
            
            # 添加选项卡
            tabs.addTab(examples_tab, "示例数据")
            tabs.addTab(distribution_tab, "数据分布")
            
            layout.addWidget(tabs)
            
            # 显示对话框
            preview_dialog.setLayout(layout)
            preview_dialog.exec_()
            
            self.log_message("数据集预览已打开")
            
        except Exception as e:
            self.log_message(f"预览数据集出错: {str(e)}")

    def start_training(self):
        """开始训练过程"""
        self.log_message("准备开始训练...")
        
        if not hasattr(self, 'train_loader') or not hasattr(self, 'valid_loader'):
            self.log_message("请先加载数据集!")
            return
        
        # 重置训练曲线图表
        if hasattr(self, 'metrics_canvas'):
            self.metrics_canvas.loss_data = {'train': [], 'val': []}
            self.metrics_canvas.acc_data = {'train': [], 'val': []}
            self.metrics_canvas.epochs = []
    
        # 收集训练参数
        training_params = {
            'batch_size': self.batch_size.value(),
            'epochs': self.epochs.value(),
            'learning_rate': self.learning_rate.value(),
            'optimizer': self.optimizer.currentText(),
            'loss_function': self.loss_function.currentText(),
            'model_name': self.model_name.currentText(),
            'max_length': self.max_length.value(),
            'save_directory': self.save_path.text(),
            'save_name': self.save_name.text(),
            'dropout_rate': self.dropout_rate.value(),
            'hidden_dim': self.hidden_dim.value(),
            'device': 'cuda:0' if torch.cuda.is_available() else 'cpu'
        }
        
        self.log_message(f"训练参数: {training_params}")
        
        # 创建并启动训练线程
        self.training_thread = TrainingThread(training_params)
        # 传递已加载的数据集给训练线程
        self.training_thread.train_loader = self.train_loader
        self.training_thread.valid_loader = self.valid_loader
        
        self.training_thread.update_progress.connect(self.update_progress)
        self.training_thread.update_metrics.connect(self.update_metrics)
        self.training_thread.training_complete.connect(self.training_complete)
        self.training_thread.start()
        
        # 更新按钮状态
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        
        self.log_message("训练已启动...")
    
    def stop_training(self):
        """停止训练过程"""
        if self.training_thread and self.training_thread.isRunning():
            self.training_thread.terminate()
            self.log_message("训练已停止")
            
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
    
    def update_progress(self, value):
        """更新训练进度条"""
        self.progress_bar.setValue(value)
    
    def update_metrics(self, metrics):
        """更新训练指标"""
        self.current_metrics = metrics
        
        # 更新指标表格
        self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics.get('accuracy', 0):.4f}"))
        self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{metrics.get('loss', 0):.4f}"))
        
        # 更新日志
        self.log_message(f"Epoch {metrics.get('epoch')}: loss={metrics.get('loss', 0):.4f}, "
                        f"acc={metrics.get('accuracy', 0):.4f}, val_loss={metrics.get('val_loss', 0):.4f}, "
                        f"val_acc={metrics.get('val_accuracy', 0):.4f}")
        
        # 更新图表
        self.metrics_canvas.update_plots(metrics)

    def show_warning_dialog(self,text):
        reply = QMessageBox.warning(
            self,
            "警告",
            text,
            QMessageBox.Ok   # 按钮选项
        )

    def training_complete(self, model):
        """训练完成时的处理"""
        if model:
            self.log_message("训练完成！")
            self.trained_model = model
            self.eval_button.setEnabled(True)  # 启用评估按钮
            self.predict_button.setEnabled(True)  # 启用预测按钮
        else:
            self.log_message("训练失败！")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
    
    def evaluate_model(self):
        """评估模型"""
        if not self.trained_model:
            # self.log_message("没有可评估的模型！")
            self.show_warning_dialog("没有可评估的模型！")
            return
            
        # self.log_message("开始评估模型...")
        self.eval_progress_bar.setValue(0)
        self.eval_button.setEnabled(False)  # 禁用按钮，防止重复点击
        
        try:
            # 设置参数
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            test_path = self.test_path.text() if hasattr(self, 'test_path') and self.test_path.text() else None
            
            # 准备参数
            eval_params = {
                'device': device,
                'max_length': self.max_length.value(),
                'test_path': test_path,
                'batch_size': self.batch_size.value()
            }
            
            # 使用已加载的测试集
            test_loader = self.test_loader if hasattr(self, 'test_loader') else None

            if test_loader is None:
                # self.log_message("没有可用的测试集！")
                self.show_warning_dialog("没有可用的测试集！")
                raise ValueError("没有可用的测试集！")
            
            # 创建并启动评估线程
            self.eval_thread = EvaluationThread(self.trained_model, eval_params, test_loader)
            self.eval_thread.update_progress.connect(self.update_eval_progress)
            self.eval_thread.evaluation_complete.connect(self.evaluation_complete)
            self.eval_thread.start()
            
        except Exception as e:
            # self.log_message(f"评估过程出错: {str(e)}")
            self.eval_button.setEnabled(True)  # 重新启用评估按钮

    def update_eval_progress(self, value):
        """更新评估进度条"""
        self.eval_progress_bar.setValue(value)

    def evaluation_complete(self, metrics):
        """评估完成的处理"""
        self.eval_button.setEnabled(True)  # 重新启用评估按钮
        
        if not metrics:
            self.log_message("评估失败！")
            return
        
        # 更新UI显示所有指标
        self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{metrics.get('test_acc', 0):.4f}"))
        self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{metrics.get('test_loss', 0):.4f}"))
        self.metrics_table.setItem(2, 1, QTableWidgetItem(f"{metrics.get('test_precision', 0):.4f}"))
        self.metrics_table.setItem(3, 1, QTableWidgetItem(f"{metrics.get('test_recall', 0):.4f}"))
        self.metrics_table.setItem(4, 1, QTableWidgetItem(f"{metrics.get('test_f1', 0):.4f}"))
        
        self.log_message(f"评估完成 - 准确率: {metrics.get('test_acc', 0):.4f}, "
                        f"损失: {metrics.get('test_loss', 0):.4f}, "
                        f"精确率: {metrics.get('test_precision', 0):.4f}, "
                        f"召回率: {metrics.get('test_recall', 0):.4f}, "
                        f"F1分数: {metrics.get('test_f1', 0):.4f}")
    
    def load_model(self):
        """加载预训练模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "checkpoints", 
            "PyTorch模型 (*.pth *.pt)"
        )
        
        if file_path:
            self.log_message(f"开始加载模型: {file_path}")
            self.model_loading_progress_bar.setValue(0)
            
            # 创建并启动模型加载线程
            self.model_loading_thread = ModelLoadingThread(file_path)
            self.model_loading_thread.update_progress.connect(self.update_model_loading_progress)
            self.model_loading_thread.loading_complete.connect(self.model_loading_complete)
            self.model_loading_thread.start()
    
    def update_model_loading_progress(self, value):
        """更新模型加载进度条"""
        self.model_loading_progress_bar.setValue(value)
        
    def model_loading_complete(self, model, message):
        """模型加载完成的处理"""
        self.log_message(message)
        
        if model:
            self.trained_model = model
            self.eval_button.setEnabled(True)  # 启用评估按钮
            self.predict_button.setEnabled(True)  # 启用预测按钮
        else:
            self.eval_button.setEnabled(False)  # 禁用评估按钮
            self.predict_button.setEnabled(False)  # 禁用预测按钮
    
    def predict_sentiment(self):
        """使用加载的模型对输入文本进行情感预测"""
        if not self.trained_model or not self.text_input.toPlainText().strip():
            self.prediction_result.setText("请先加载模型并输入文本")
            return
        
        # 获取输入文本
        text = self.text_input.toPlainText().strip()
        self.log_message(f"正在预测文本: {text}")
        self.predict_progress_bar.setValue(0)
        self.predict_button.setEnabled(False)  # 预测时禁用按钮
        
        try:
            # 创建并启动预测线程
            self.prediction_thread = PredictionThread(
                self.trained_model, 
                text, 
                max_length=self.max_length.value()
            )
            self.prediction_thread.update_progress.connect(self.update_predict_progress)
            self.prediction_thread.prediction_complete.connect(self.prediction_complete)
            self.prediction_thread.start()
            
        except Exception as e:
            self.prediction_result.setText(f"预测出错: {str(e)}")
            self.log_message(f"预测过程出错: {str(e)}")
            self.predict_button.setEnabled(True)  # 出错时重新启用按钮

    def update_predict_progress(self, value):
        """更新预测进度条"""
        self.predict_progress_bar.setValue(value)

    def prediction_complete(self, result):
        """预测完成的处理"""
        self.predict_button.setEnabled(True)  # 重新启用预测按钮
        
        if not result:
            self.prediction_result.setText("预测失败！")
            return
        
        # 显示预测结果
        sentiment_label = result.get('label', '未知')
        confidence = result.get('confidence', 0)
        
        result_text = f"<p style='font-size:14pt;'><b>预测情感:</b> {sentiment_label}</p>"
        result_text += f"<p>置信度: {confidence:.4f}</p>"
        self.prediction_result.setHtml(result_text)
        
        self.log_message(f"预测完成: 情感={sentiment_label}, 置信度={confidence:.4f}")

    def log_message(self, message):
        """将消息添加到日志输出框"""
        self.log_output.append(message)
        # 滚动到最新消息
        self.log_output.verticalScrollBar().setValue(
            self.log_output.verticalScrollBar().maximum()
        )



if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = AIModelGUI()
    ex.show()
    sys.exit(app.exec_())
