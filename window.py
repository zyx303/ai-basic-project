import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QLineEdit, QComboBox, QPushButton, QFileDialog, 
                            QTabWidget, QGroupBox, QFormLayout, QSpinBox, QDoubleSpinBox, 
                            QCheckBox, QProgressBar, QTextEdit, QSplitter, QTableWidget,
                            QTableWidgetItem, QSlider)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

# 导入情感分类相关模块
import torch
import torch.nn as nn
import torch.optim as optim
from model.SentimentClassifier import SentimentClassifier
from utils.data_loader import (load_test, load_train, set_seed, get_dataset_info, 
                               validate_data_format, visualize_data_distribution)
from utils.train_utils import train_model, evaluate_model

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
            device = self.params.get('device', 'cuda:0' if torch.cuda.is_available() else 'cpu')    
            
            # 确保保存目录存在
            os.makedirs(save_directory, exist_ok=True)
            
            # 检查设备
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            classes = ["negative", "neural", "positive"]
            
            # 初始化模型
            model = SentimentClassifier(model_name=model_name, num_classes=len(classes))
            
            # 定义损失函数和优化器
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # 自定义进度回调函数
            def progress_callback(epoch, batch, total_batches):
                progress = int(epoch / epochs * 100 + (batch / total_batches) * (100 / epochs))
                self.update_progress.emit(progress)
            
            # 训练模型
            trained_model, history = train_model(
                model, self.train_loader, self.valid_loader, criterion, optimizer, scheduler,
                num_epochs=epochs, device=device, save_dir=save_directory,
                progress_callback=progress_callback
            )
            
            # 保存结果
            self.model = trained_model
            self.history = history
            
            # 每个epoch更新一次指标
            for epoch in range(len(history['train_loss'])):
                metrics = {
                    'loss': history['train_loss'][epoch],
                    'accuracy': history['train_acc'][epoch],
                    'val_loss': history['val_loss'][epoch],
                    'val_accuracy': history['val_acc'][epoch],
                    'epoch': epoch + 1
                }
                self.update_metrics.emit(metrics)
            
            # 发出训练完成信号
            self.training_complete.emit(trained_model)
            
        except Exception as e:
            print(f"训练出错: {str(e)}")
            self.training_complete.emit(None)

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
            
            self.axes[0].plot(self.epochs, self.loss_data['train'], 'b-', label='训练')
            self.axes[0].plot(self.epochs, self.loss_data['val'], 'r-', label='验证')
            self.axes[0].legend()
            self.axes[0].set_title('损失')
            self.axes[0].grid(True)
            
            self.axes[1].plot(self.epochs, self.acc_data['train'], 'b-', label='训练')
            self.axes[1].plot(self.epochs, self.acc_data['val'], 'r-', label='验证')
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
        self.setGeometry(100, 100, 1200, 800)
        
        # 创建中央窗口部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局
        main_layout = QVBoxLayout(central_widget)
        
        # 创建选项卡窗口部件
        tabs = QTabWidget()
        
        # 创建各选项卡
        dataset_tab = self.create_dataset_tab()
        model_tab = self.create_model_tab()
        train_tab = self.create_training_tab()
        eval_tab = self.create_evaluation_tab()
        
        # 添加选项卡到选项卡窗口部件
        tabs.addTab(dataset_tab, "数据集")
        tabs.addTab(model_tab, "模型参数")
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
        
        self.normalize_check = QCheckBox("标准化")
        self.normalize_check.setChecked(True)
        preprocess_layout.addRow("", self.normalize_check)
        
        self.max_length = QSpinBox()
        self.max_length.setRange(16, 512)
        self.max_length.setValue(128)
        self.max_length.setSingleStep(16)
        preprocess_layout.addRow("最大序列长度:", self.max_length)
        
        self.train_split = QDoubleSpinBox()
        self.train_split.setRange(0.1, 0.9)
        self.train_split.setSingleStep(0.05)
        self.train_split.setValue(0.9)  # 90%训练，10%验证
        self.train_split.setDecimals(2)
        preprocess_layout.addRow("训练集比例:", self.train_split)
        
        # 数据加载按钮
        button_layout = QHBoxLayout()
        load_button = QPushButton("加载数据")
        load_button.clicked.connect(self.load_dataset)
        preview_button = QPushButton("预览数据")
        preview_button.clicked.connect(self.preview_dataset)
        button_layout.addWidget(load_button)
        button_layout.addWidget(preview_button)
        
        # 数据集信息显示
        self.dataset_info = QTextEdit()
        self.dataset_info.setReadOnly(True)
        self.dataset_info.setFixedHeight(200)
        
        # 添加组到布局
        layout.addWidget(dataset_group)
        layout.addWidget(preprocess_group)
        layout.addLayout(button_layout)
        layout.addWidget(QLabel("数据集信息:"))
        layout.addWidget(self.dataset_info)
        layout.addStretch()
        
        return tab
    
    def create_model_tab(self):
        """创建模型参数选项卡"""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # 左侧 - 模型类型选择
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        model_group = QGroupBox("模型类型")
        model_layout = QFormLayout(model_group)
        
        self.model_name = QComboBox()
        self.model_name.addItems(["bert-base-chinese", "chinese-bert-wwm", "chinese-roberta-wwm"])
        model_layout.addRow("预训练模型:", self.model_name)
        
        left_layout.addWidget(model_group)
        
        # 预训练模型选项
        pretrain_group = QGroupBox("模型设置")
        pretrain_layout = QFormLayout(pretrain_group)
        
        self.num_classes = QSpinBox()
        self.num_classes.setRange(2, 10)
        self.num_classes.setValue(3)  # 默认3个类别：积极、中性、消极
        pretrain_layout.addRow("类别数:", self.num_classes)
        
        left_layout.addWidget(pretrain_group)
        left_layout.addStretch()
        
        # 右侧 - 模型参数设置
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        self.model_params_group = QGroupBox("模型参数")
        self.model_params_layout = QFormLayout(self.model_params_group)
        
        # 添加BERT模型特有参数
        self.model_params_layout.addRow("隐藏层维度:", self.create_spin_box(64, 1024, 768, 64))
        self.model_params_layout.addRow("注意力头数:", self.create_spin_box(1, 16, 12))
        self.model_params_layout.addRow("Dropout率:", self.create_double_spin_box(0, 0.9, 0.1, 0.05))
        
        right_layout.addWidget(self.model_params_group)
        right_layout.addStretch()
        
        # 添加面板到布局
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setSizes([300, 700])
        
        layout.addWidget(splitter)
        
        return tab
    
    def create_training_tab(self):
        """创建训练选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 超参数设置
        hyperparams_group = QGroupBox("超参数设置")
        hyperparams_layout = QFormLayout(hyperparams_group)
        
        self.batch_size = QSpinBox()
        self.batch_size.setRange(1, 128)
        self.batch_size.setValue(32)
        self.batch_size.setSingleStep(8)
        hyperparams_layout.addRow("Batch Size:", self.batch_size)
        
        self.epochs = QSpinBox()
        self.epochs.setRange(1, 100)
        self.epochs.setValue(3)
        hyperparams_layout.addRow("Epochs:", self.epochs)
        
        self.learning_rate = QDoubleSpinBox()
        self.learning_rate.setRange(0.0000001, 0.1)
        self.learning_rate.setValue(0.00002)
        self.learning_rate.setDecimals(7)
        self.learning_rate.setSingleStep(0.00001)
        hyperparams_layout.addRow("学习率:", self.learning_rate)
        
        self.optimizer = QComboBox()
        self.optimizer.addItems(["Adam", "AdamW", "SGD", "RMSprop"])
        hyperparams_layout.addRow("优化器:", self.optimizer)
        
        self.loss_function = QComboBox()
        self.loss_function.addItems(["CrossEntropyLoss", "BCEWithLogitsLoss"])
        hyperparams_layout.addRow("损失函数:", self.loss_function)
        
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
        training_layout.addLayout(progress_layout)
        training_layout.addWidget(QLabel("训练日志:"))
        training_layout.addWidget(self.log_output)
        
        # 添加组到布局
        layout.addWidget(hyperparams_group)
        layout.addWidget(training_group)
        
        return tab
    
    def create_evaluation_tab(self):
        """创建评估选项卡"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # 创建图表用于显示训练指标
        self.metrics_canvas = MetricPlotCanvas(width=10, height=4)
        
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
        layout.addWidget(self.metrics_canvas)
        layout.addWidget(eval_control_group)
        layout.addWidget(metrics_group)
        
        return tab
    
    def update_model_params(self, index):
        """根据选择的模型类型更新参数设置"""
        # 清除现有的参数控件
        while self.model_params_layout.rowCount() > 0:
            self.model_params_layout.removeRow(0)
        
        model_type = self.model_name.currentText()
        
        if model_type == "CNN":
            self.model_params_layout.addRow("卷积层数:", self.create_spin_box(1, 10, 3))
            self.model_params_layout.addRow("初始卷积核数:", self.create_spin_box(8, 128, 32, 8))
            self.model_params_layout.addRow("卷积核大小:", self.create_spin_box(2, 7, 3, 2))
            self.model_params_layout.addRow("池化类型:", self.create_combo_box(["MaxPool", "AvgPool"]))
            self.model_params_layout.addRow("Dropout率:", self.create_double_spin_box(0, 0.9, 0.5, 0.05))
        elif model_type in ["RNN", "LSTM", "GRU"]:
            self.model_params_layout.addRow("隐藏层大小:", self.create_spin_box(8, 512, 128, 8))
            self.model_params_layout.addRow("层数:", self.create_spin_box(1, 5, 2))
            self.model_params_layout.addRow("序列长度:", self.create_spin_box(5, 100, 20))
            self.model_params_layout.addRow("双向:", self.create_check_box(False))
            self.model_params_layout.addRow("Dropout率:", self.create_double_spin_box(0, 0.9, 0.3, 0.05))
        elif model_type == "Transformer":
            self.model_params_layout.addRow("注意力头数:", self.create_spin_box(1, 16, 8))
            self.model_params_layout.addRow("编码器层数:", self.create_spin_box(1, 12, 6))
            self.model_params_layout.addRow("前馈网络维度:", self.create_spin_box(128, 2048, 512, 128))
            self.model_params_layout.addRow("Dropout率:", self.create_double_spin_box(0, 0.9, 0.1, 0.05))
        elif model_type == "MLP":
            self.model_params_layout.addRow("隐藏层数:", self.create_spin_box(1, 10, 2))
            self.model_params_layout.addRow("隐藏层大小:", self.create_spin_box(8, 1024, 256, 8))
            self.model_params_layout.addRow("激活函数:", self.create_combo_box(["ReLU", "Sigmoid", "Tanh", "LeakyReLU"]))
            self.model_params_layout.addRow("Dropout率:", self.create_double_spin_box(0, 0.9, 0.2, 0.05))
    
    def toggle_pretrained(self, state):
        """启用/禁用预训练模型选择"""
        self.pretrained_model.setEnabled(state == Qt.Checked)
    
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

    def load_dataset(self):
        """加载数据集并应用预处理参数"""
        self.dataset_info.append("正在加载数据集...")
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
            
            # 保存到类变量中，以便在训练时使用
            self.dataset_params = {
                'max_length': max_length,
                'train_split': train_split,
                'normalize': self.normalize_check.isChecked(),
                'train_path': train_path,
                'test_path': test_path
            }
            
            self.dataset_info.append("正在加载和处理数据集，请稍候...")
            
            # 实际加载数据集，包括训练集、验证集和测试集
            try:
                # 加载训练和验证集
                self.train_loader, self.valid_loader = load_train(
                    batch_size=self.batch_size.value(),
                    num_workers=2,
                    split_ratio=1.0-train_split,  # 因为train_split是训练集占比
                    max_length=max_length,
                    custom_train_path=train_path
                )
                
                # 加载测试集
                self.test_loader = load_test(
                    batch_size=self.batch_size.value(),
                    num_workers=2,
                    max_length=max_length,
                    custom_test_path=test_path
                )
                
                self.dataset_info.append(f"数据集加载完成，训练集和验证集已准备就绪")
            
                # 获取和显示数据集信息
                dataset_info = get_dataset_info(train_path, test_path, max_length)
                
                # 显示数据集统计信息
                self.dataset_info.clear()
                self.dataset_info.append(f"数据集已成功加载: {len(self.train_loader.dataset)} 训练样本,\n {len(self.valid_loader.dataset)} 验证样本,\n {len(self.test_loader.dataset)} 测试样本")
                
                self.dataset_ready = True
                
                # 尝试生成并显示分布图
                try:
                    from utils.data_loader import load_sentiment_data
                    train_texts, train_labels = load_sentiment_data('train', train_path)
                    test_texts, test_labels = load_sentiment_data('test', test_path)
                    
                    visualize_data_distribution(train_labels, "训练集分布")
                    visualize_data_distribution(test_labels, "测试集分布")
                    
                    self.dataset_info.append("已生成数据分布图")
                except Exception as e:
                    self.dataset_info.append(f"无法生成分布图: {str(e)}")
            
            except Exception as e:
                self.dataset_info.append(f"加载数据集时出错: {str(e)}")
                self.dataset_ready = False
                return
            
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
            preview_dialog.setWindowTitle("数据集预览")
            preview_dialog.resize(800, 600)
            layout = QVBoxLayout(preview_dialog)
            
            # 创建选项卡
            tabs = QTabWidget()
            
            # 示例数据选项卡
            examples_tab = QWidget()
            examples_layout = QVBoxLayout(examples_tab)
            
            # 获取数据集信息
            dataset_info = get_dataset_info(train_path, test_path)
            examples = dataset_info['examples']
            
            # 训练集示例
            examples_layout.addWidget(QLabel("<h3>训练集示例</h3>"))
            for i, text in enumerate(examples['训练'][:3]):
                examples_layout.addWidget(QLabel(f"<b>示例 {i+1}:</b> {text}"))
            
            # 测试集示例
            examples_layout.addWidget(QLabel("<h3>测试集示例</h3>"))
            for i, text in enumerate(examples['测试'][:3]):
                examples_layout.addWidget(QLabel(f"<b>示例 {i+1}:</b> {text}"))
            
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
            
            if os.path.exists(train_dist_path):
                pixmap = QPixmap(train_dist_path)
                train_dist_image.setPixmap(pixmap.scaled(350, 300))
                train_dist_image.setAlignment(Qt.AlignCenter)
            else:
                train_dist_image.setText("训练集分布图不可用")
                train_dist_image.setAlignment(Qt.AlignCenter)
            
            if os.path.exists(test_dist_path):
                pixmap = QPixmap(test_dist_path)
                test_dist_image.setPixmap(pixmap.scaled(350, 300))
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
    
    def training_complete(self, model):
        """训练完成时的处理"""
        if model:
            self.log_message("训练完成！")
            self.trained_model = model
            self.eval_button.setEnabled(True)  # 启用评估按钮
        else:
            self.log_message("训练失败！")
        
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.progress_bar.setValue(100)
    
    def evaluate_model(self):
        """评估模型"""
        if not self.trained_model:
            self.log_message("没有可评估的模型！")
            return
            
        self.log_message("开始评估模型...")
        
        try:
            # 检查设备
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
            
            # 获取自定义测试集路径
            test_path = self.test_path.text() if hasattr(self, 'test_path') and self.test_path.text() else None
            
            # 加载测试数据
            test_loader = load_test(
                batch_size=self.batch_size.value(), 
                num_workers=2,
                max_length=self.max_length.value(),
                custom_test_path=test_path
            )
            
            # 假设情感分类有3个类别
            classes = ["negative", "neural", "positive"]
            criterion = nn.CrossEntropyLoss()
            
            # 评估模型
            test_loss, test_acc = evaluate_model(
                self.trained_model, test_loader, criterion, device, classes
            )
            
            # 更新UI
            self.metrics_table.setItem(0, 1, QTableWidgetItem(f"{test_acc:.4f}"))
            self.metrics_table.setItem(1, 1, QTableWidgetItem(f"{test_loss:.4f}"))
            
            self.log_message(f"评估完成 - 准确率: {test_acc:.4f}, 损失: {test_loss:.4f}")
        
        except Exception as e:
            self.log_message(f"评估过程出错: {str(e)}")
    
    def load_model(self):
        """加载预训练模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "checkpoints", 
            "PyTorch模型 (*.pth *.pt)"
        )
        
        if file_path:
            try:
                # 加载模型
                self.log_message(f"加载模型: {file_path}")
                
                self.trained_model = SentimentClassifier()
                self.trained_model.load_state_dict(torch.load(file_path))
                self.trained_model.eval()
                
                self.eval_button.setEnabled(True)
                self.log_message("模型加载成功，可以开始评估")
            except Exception as e:
                self.log_message(f"加载模型出错: {str(e)}")
                
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
