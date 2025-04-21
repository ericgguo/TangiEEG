"""
模型配置模块 - 包含解码模型的配置参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 模型基本配置
MODEL_CONFIG = {
    # 模型类型设置
    'model_type': 'dewave',        # 可选: 'cnn', 'rnn', 'hybrid', 'dewave'
    'model_path': str(PROJECT_ROOT / 'data' / 'models' / 'dewave_model.pth'),
    'model_dir': str(PROJECT_ROOT / 'data' / 'models'),
    
    # 输入设置
    'channels': [1, 2, 3, 4, 5, 6, 7, 8],  # 使用的通道索引列表
    'sampling_rate': 250,          # 采样率(Hz)
    'window_size': 1000,           # 时间窗口大小(样本数)
    'window_stride': 100,          # 滑动窗口步长(样本数)
    
    # 解码器参数
    'num_classes': 4,              # 分类类别数
    'confidence_threshold': 0.7,   # 置信度阈值
    'prediction_buffer_size': 5,   # 预测缓冲区大小
    
    # 深度学习训练参数
    'batch_size': 32,              # 批处理大小
    'learning_rate': 0.001,        # 学习率
    'num_epochs': 20,              # 训练轮数
    'use_mixed_precision': False,  # 是否使用混合精度训练
    
    # 特征提取设置
    'features': {
        'time_domain': True,       # 是否提取时域特征
        'freq_domain': True,       # 是否提取频域特征
        'wavelets': False,         # 是否使用小波变换
        'bands': {                 # 频段设置
            'delta': [0.5, 4.0],   # Delta波段范围(Hz)
            'theta': [4.0, 8.0],   # Theta波段范围(Hz)
            'alpha': [8.0, 13.0],  # Alpha波段范围(Hz)
            'beta': [13.0, 30.0],  # Beta波段范围(Hz)
            'gamma': [30.0, 50.0]  # Gamma波段范围(Hz)
        }
    },
    
    # 文本解码设置
    'text_decoder': {
        'use_language_model': True,      # 是否使用语言模型增强
        'language_model_weight': 0.5,    # 语言模型权重
        'intent_transition_smoothing': 0.7,  # 意图转换平滑因子
        'min_confidence_for_output': 0.6,    # 输出所需的最小置信度
    },
    
    # MAC平台特定优化
    'mac_optimization': {
        'use_mps': True,                 # 是否使用Metal性能着色器
        'reduced_precision': True,       # 是否使用降低精度（节省内存）
        'optimize_for_battery': False,   # 是否优化电池使用
        'thread_count': 0                # 线程数，0表示自动确定
    }
}

# DeWave特定配置
DEWAVE_CONFIG = {
    'encoder_layers': 6,           # 编码器层数
    'decoder_layers': 6,           # 解码器层数
    'hidden_dim': 768,             # 隐藏层维度
    'intermediate_dim': 3072,      # 中间层维度
    'num_attention_heads': 12,     # 注意力头数量
    'max_position_embeddings': 512,# 最大位置嵌入长度
    'vocab_size': 21128,           # 词汇表大小（中文）
}

# CNN特定配置
CNN_CONFIG = {
    'filter_sizes': [32, 64, 128, 256], # 卷积滤波器大小
    'kernel_sizes': [3, 3, 3, 3],       # 卷积核大小
    'pool_sizes': [2, 2, 2, 2],         # 池化大小
    'dropout_rate': 0.5,                # Dropout比例
}

# RNN特定配置
RNN_CONFIG = {
    'rnn_type': 'lstm',            # RNN类型: 'lstm', 'gru'
    'hidden_sizes': [128, 256],    # 隐藏层大小
    'bidirectional': True,         # 是否使用双向RNN
    'dropout_rate': 0.3,           # Dropout比例
}

# 获取模型配置
def get_model_config():
    """返回当前模型配置"""
    config = MODEL_CONFIG.copy()
    
    # 根据模型类型添加特定配置
    model_type = config['model_type']
    if model_type == 'dewave':
        config['dewave'] = DEWAVE_CONFIG
    elif model_type == 'cnn':
        config['cnn'] = CNN_CONFIG
    elif model_type == 'rnn':
        config['rnn'] = RNN_CONFIG
    elif model_type == 'hybrid':
        config['cnn'] = CNN_CONFIG
        config['rnn'] = RNN_CONFIG
    
    return config

# 更新模型配置
def update_model_config(new_config):
    """更新模型配置"""
    global MODEL_CONFIG
    
    # 递归更新嵌套字典
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    MODEL_CONFIG = update_dict(MODEL_CONFIG, new_config)
    return get_model_config()
