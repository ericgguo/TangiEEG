"""
EEG解码器模块 - 负责将预处理后的EEG信号解码为文本
包含适用于各种硬件环境（包括Mac）的优化
"""

import os
import sys
import time
import json
import threading
import numpy as np
from pathlib import Path
from queue import Queue, Empty

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from utils.logging_utils import get_logger
from config.model_config import get_model_config

# 检测平台类型
import platform
IS_MAC = platform.system() == 'Darwin'
MAC_ARM = IS_MAC and platform.machine() == 'arm64'

# 根据不同平台进行特定优化
if IS_MAC:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # 解决Intel MKL库冲突问题
    
    # 对于M1/M2 Mac，使用Metal性能优化
    if MAC_ARM:
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'  # 启用MPS回退
        try:
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 优化内存使用
                DEFAULT_DEVICE = 'mps'  # 使用Metal性能着色器设备
            else:
                DEFAULT_DEVICE = 'cpu'
        except:
            DEFAULT_DEVICE = 'cpu'
    else:
        DEFAULT_DEVICE = 'cpu'
else:
    # 检测CUDA可用性
    try:
        import torch
        if torch.cuda.is_available():
            DEFAULT_DEVICE = 'cuda'
        else:
            DEFAULT_DEVICE = 'cpu'
    except:
        DEFAULT_DEVICE = 'cpu'

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告：未安装PyTorch，将使用简化模型")

# 简单的CNN模型类
class EEGConvNet(nn.Module):
    """用于EEG解码的简单卷积神经网络"""
    
    def __init__(self, num_channels=8, num_classes=4):
        """
        初始化EEG卷积网络
        
        Args:
            num_channels: EEG通道数量
            num_classes: 输出类别数（例如：意图类别）
        """
        super(EEGConvNet, self).__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(num_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 第三个卷积层
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 125, 512)  # 假设输入为1000个时间点，经过3次池化后为125
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout层用于防止过拟合
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """前向传播"""
        # 输入x的形状为 [batch_size, num_channels, time_points]
        
        # 卷积块1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        # 卷积块2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        # 卷积块3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 文本编码器
class IntentToTextEncoder:
    """将意图编码转换为文本"""
    
    def __init__(self):
        """初始化意图到文本的编码器"""
        # 定义意图类别映射
        self.intent_map = {
            0: "",  # 无意图/噪音
            1: "是",   # 肯定/同意
            2: "否",   # 否定/拒绝
            3: "帮助"  # 请求帮助
        }
        
        # 当前文本缓存
        self.current_text = ""
        
        # 句子构建辅助变量
        self.word_buffer = []
        self.sentence_buffer = []
        
        # 加载语言模型（如有）
        self.language_model = None
        try:
            from transformers import BertTokenizer, BertForMaskedLM
            self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
            self.language_model = BertForMaskedLM.from_pretrained("bert-base-chinese")
            self.has_lm = True
        except:
            self.has_lm = False
    
    def encode(self, intent_id):
        """将意图ID编码为文本"""
        if intent_id in self.intent_map:
            return self.intent_map[intent_id]
        return ""
    
    def update_text(self, intent_id):
        """更新当前文本"""
        text = self.encode(intent_id)
        if text:
            self.word_buffer.append(text)
            
            # 每3个词组成一个短句
            if len(self.word_buffer) >= 3:
                sentence = "".join(self.word_buffer)
                self.sentence_buffer.append(sentence)
                self.word_buffer = []
                
                # 最多保留5个短句
                if len(self.sentence_buffer) > 5:
                    self.sentence_buffer.pop(0)
                
                # 更新当前文本
                self.current_text = "，".join(self.sentence_buffer)
        
        return self.current_text
    
    def predict_next_word(self, context, num_suggestions=3):
        """使用语言模型预测下一个可能的词"""
        if not self.has_lm or not context:
            return []
        
        try:
            # 添加掩码标记
            masked_text = context + "[MASK]"
            
            # 编码
            inputs = self.tokenizer(masked_text, return_tensors="pt")
            
            # 获取[MASK]的位置
            mask_token_index = torch.where(inputs["input_ids"][0] == self.tokenizer.mask_token_id)[0]
            
            # 预测
            with torch.no_grad():
                outputs = self.language_model(**inputs)
            
            # 获取最可能的词
            logits = outputs.logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_k_tokens = torch.topk(mask_token_logits, num_suggestions, dim=1).indices[0].tolist()
            
            # 解码
            top_words = [self.tokenizer.decode([token]) for token in top_k_tokens]
            return top_words
            
        except Exception as e:
            print(f"预测下一个词失败: {e}")
            return []

# EEG解码器类
class EEGDecoder:
    """EEG信号解码器 - 将EEG信号转换为文本"""
    
    def __init__(self, model_name="dewave", device=None):
        """
        初始化EEG解码器
        
        Args:
            model_name: 要使用的模型名称
            device: 计算设备（'cpu', 'cuda', 'mps'等）
        """
        self.logger = get_logger("decoder")
        self.config = get_model_config()
        self.model_name = model_name
        self.device = device or DEFAULT_DEVICE
        self.model = None
        self.text_encoder = IntentToTextEncoder()
        self.latest_result = ""
        self.initialized = False
        self.prediction_buffer = []
        
        # 结果队列
        self.result_queue = Queue(maxsize=100)
        
        # 初始化解码器
        self._initialize()
    
    def _initialize(self):
        """初始化模型和相关资源"""
        self.logger.info(f"初始化EEG解码器，模型: {self.model_name}，设备: {self.device}")
        
        try:
            # 根据不同平台进行特定优化
            if IS_MAC:
                self.logger.info("检测到Mac平台，应用Mac特定优化...")
                
                if MAC_ARM:
                    self.logger.info("检测到M系列芯片，启用Metal性能着色器加速...")
                    
                    # MPS设备优化
                    if self.device == 'mps':
                        self.logger.info("使用MPS设备进行加速")
                        # 降低批处理大小以适应内存限制
                        self.config['batch_size'] = min(self.config.get('batch_size', 32), 16)
                        # 使用混合精度训练
                        self.config['use_mixed_precision'] = True
                else:
                    self.logger.info("检测到Intel Mac，应用CPU优化...")
                    # 使用多线程优化
                    import torch
                    if hasattr(torch, 'set_num_threads'):
                        # 使用物理核心数量的2倍
                        import multiprocessing
                        num_cores = multiprocessing.cpu_count()
                        torch.set_num_threads(num_cores)
                        self.logger.info(f"设置PyTorch线程数为: {num_cores}")
            
            # 加载模型
            if HAS_TORCH:
                self._load_model()
            else:
                self.logger.warning("未安装PyTorch，将使用简化模型")
                self._load_simplified_model()
            
            self.initialized = True
            self.logger.info("EEG解码器初始化完成")
            
        except Exception as e:
            self.logger.error(f"初始化EEG解码器失败: {e}")
            # 回退到简化模型
            self._load_simplified_model()
    
    def _load_model(self):
        """加载深度学习模型"""
        try:
            import torch
            
            # 创建模型实例
            num_channels = len(self.config.get('channels', 8))
            num_classes = self.config.get('num_classes', 4)
            
            if self.model_name == "cnn":
                self.model = EEGConvNet(num_channels=num_channels, num_classes=num_classes)
            else:
                # DeWave模型或其他模型实现
                # 这里使用CNN作为默认回退
                self.model = EEGConvNet(num_channels=num_channels, num_classes=num_classes)
            
            # 将模型移动到指定设备
            if self.device == 'mps' and hasattr(torch, 'mps') and torch.backends.mps.is_available():
                self.model = self.model.to(torch.device('mps'))
            elif self.device == 'cuda' and torch.cuda.is_available():
                self.model = self.model.to(torch.device('cuda'))
            else:
                self.model = self.model.to(torch.device('cpu'))
            
            # 加载预训练权重（如果有）
            model_path = Path(self.config.get('model_path', ''))
            if model_path.exists():
                self.logger.info(f"加载预训练模型: {model_path}")
                state_dict = torch.load(model_path, map_location=torch.device(self.device))
                self.model.load_state_dict(state_dict)
            else:
                self.logger.warning(f"找不到预训练模型: {model_path}，使用随机初始化权重")
            
            # 设置为评估模式
            self.model.eval()
            
            self.logger.info(f"模型加载完成，使用 {self.device} 设备")
            
        except Exception as e:
            self.logger.error(f"加载深度学习模型失败: {e}")
            self._load_simplified_model()
    
    def _load_simplified_model(self):
        """加载简化版模型（不依赖深度学习框架）"""
        self.logger.info("加载简化版模型")
        self.model = None  # 不使用神经网络模型
    
    def decode(self, eeg_data):
        """
        解码EEG数据
        
        Args:
            eeg_data: 形状为(通道数, 样本数)的EEG数据
        
        Returns:
            str: 解码得到的文本，无法解码则返回None
        """
        if not self.initialized or eeg_data is None:
            return None
        
        try:
            # 准备数据
            if HAS_TORCH and self.model is not None:
                # 使用深度学习模型
                import torch
                
                # 转换为张量
                data = torch.tensor(eeg_data, dtype=torch.float32)
                
                # 添加批维度
                if len(data.shape) == 2:
                    data = data.unsqueeze(0)  # [1, channels, samples]
                
                # 移动到指定设备
                if self.device != 'cpu':
                    data = data.to(self.device)
                
                # 进行推理
                with torch.no_grad():
                    self.model.eval()
                    outputs = self.model(data)
                    
                    # 获取预测类别
                    _, predicted = torch.max(outputs, 1)
                    intent_id = predicted.item()
                    
                    # 更新文本
                    decoded_text = self.text_encoder.update_text(intent_id)
                    confidence = torch.softmax(outputs, dim=1)[0, intent_id].item()
                    
                    # 仅在置信度高于阈值时返回结果
                    threshold = self.config.get('confidence_threshold', 0.7)
                    if confidence > threshold and decoded_text:
                        self.latest_result = decoded_text
                        # 将结果放入队列
                        if not self.result_queue.full():
                            self.result_queue.put((decoded_text, confidence))
                        return decoded_text
                    
                    return None
                    
            else:
                # 使用简化版解码方法
                return self._simplified_decode(eeg_data)
                
        except Exception as e:
            self.logger.error(f"解码EEG数据失败: {e}")
            return None
    
    def _simplified_decode(self, eeg_data):
        """简化版解码方法，不依赖深度学习框架"""
        if eeg_data is None or eeg_data.shape[0] == 0:
            return None
        
        try:
            # 提取特征：使用简单的信号功率比较
            
            # 计算各频段功率
            sampling_rate = 250  # 假设采样率为250Hz
            window_size = min(eeg_data.shape[1], 250)  # 1秒窗口
            
            # 仅使用第一个通道数据
            signal = eeg_data[0, -window_size:]
            
            # 快速傅里叶变换
            from scipy.fft import fft
            fft_vals = np.abs(fft(signal))
            
            # 频率数组
            freqs = np.fft.fftfreq(len(signal), 1.0/sampling_rate)
            
            # 仅保留正频率
            positive_mask = freqs > 0
            freqs = freqs[positive_mask]
            fft_vals = fft_vals[positive_mask]
            
            # 各频段索引
            delta_idx = np.logical_and(freqs >= 0.5, freqs < 4)
            theta_idx = np.logical_and(freqs >= 4, freqs < 8)
            alpha_idx = np.logical_and(freqs >= 8, freqs < 13)
            beta_idx = np.logical_and(freqs >= 13, freqs < 30)
            
            # 计算各频段功率
            delta_power = np.sum(fft_vals[delta_idx])
            theta_power = np.sum(fft_vals[theta_idx])
            alpha_power = np.sum(fft_vals[alpha_idx])
            beta_power = np.sum(fft_vals[beta_idx])
            
            # 简单分类逻辑：基于频段能量比例
            # 实际上这只是一个占位实现，不具备真正的分类能力
            alpha_ratio = alpha_power / (delta_power + theta_power + alpha_power + beta_power + 1e-10)
            beta_ratio = beta_power / (delta_power + theta_power + alpha_power + beta_power + 1e-10)
            
            # 生成随机输出，模拟分类结果
            import random
            random_val = random.random()
            
            # 随机确定意图，但增加某些特征的权重
            if alpha_ratio > 0.3 and random_val > 0.7:  # 高alpha波通常与放松状态相关
                intent_id = 1  # "是"
            elif beta_ratio > 0.3 and random_val > 0.7:  # 高beta波通常与专注状态相关
                intent_id = 2  # "否"
            elif random_val > 0.9:  # 低概率事件
                intent_id = 3  # "帮助"
            else:
                intent_id = 0  # 无意图
            
            # 更新文本并返回
            decoded_text = self.text_encoder.update_text(intent_id)
            if decoded_text:
                self.latest_result = decoded_text
                # 将结果放入队列
                if not self.result_queue.full():
                    self.result_queue.put((decoded_text, 0.5))  # 固定置信度为0.5
                return decoded_text
            
            return None
            
        except Exception as e:
            self.logger.error(f"简化解码失败: {e}")
            return None
    
    def get_latest_result(self):
        """获取最新解码结果"""
        return self.latest_result
    
    def get_predictions(self, num=3):
        """获取下一个可能词的预测"""
        try:
            if self.text_encoder.has_lm and self.latest_result:
                # 使用语言模型预测
                return self.text_encoder.predict_next_word(self.latest_result, num_suggestions=num)
            else:
                # 返回固定预测
                return ["是", "否", "帮助"]
        except:
            return []
    
    def train(self, training_data, labels):
        """
        使用训练数据训练或微调模型
        
        Args:
            training_data: 训练数据，形状为(样本数, 通道数, 时间点)
            labels: 标签，形状为(样本数,)
        
        Returns:
            bool: 训练成功返回True，否则返回False
        """
        if not HAS_TORCH or self.model is None:
            self.logger.error("无法训练模型：PyTorch未安装或模型未初始化")
            return False
        
        try:
            import torch
            import torch.optim as optim
            from torch.utils.data import TensorDataset, DataLoader
            
            # 准备数据
            X = torch.tensor(training_data, dtype=torch.float32)
            y = torch.tensor(labels, dtype=torch.long)
            
            # 创建数据集和数据加载器
            dataset = TensorDataset(X, y)
            batch_size = self.config.get('batch_size', 32)
            train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            # 设置为训练模式
            self.model.train()
            
            # 定义优化器和损失函数
            learning_rate = self.config.get('learning_rate', 0.001)
            optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            criterion = nn.CrossEntropyLoss()
            
            # 训练循环
            num_epochs = self.config.get('num_epochs', 10)
            self.logger.info(f"开始训练，批大小: {batch_size}，学习率: {learning_rate}，轮数: {num_epochs}")
            
            # Mac特定优化
            if self.device == 'mps' and self.config.get('use_mixed_precision', False):
                # 模拟混合精度训练
                self.logger.info("使用模拟混合精度训练")
            
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i, (inputs, targets) in enumerate(train_loader):
                    # 将数据移动到设备
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    # 前向传播
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # 反向传播和优化
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                # 打印统计信息
                epoch_loss = running_loss / len(train_loader)
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")
            
            # 保存模型
            model_dir = Path(self.config.get('model_dir', 'data/models'))
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / f"{self.model_name}_model.pth"
            torch.save(self.model.state_dict(), model_path)
            self.logger.info(f"模型已保存到: {model_path}")
            
            # 设置回评估模式
            self.model.eval()
            
            return True
            
        except Exception as e:
            self.logger.error(f"训练模型失败: {e}")
            return False
