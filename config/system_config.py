"""
系统全局配置模块 - 包含TangiEEG系统的全局配置参数
"""

import os
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# 系统基本配置
SYSTEM_CONFIG = {
    # 基本路径配置
    'paths': {
        'project_root': str(PROJECT_ROOT),
        'data_dir': str(PROJECT_ROOT / 'data'),
        'raw_data_dir': str(PROJECT_ROOT / 'data' / 'raw'),
        'processed_data_dir': str(PROJECT_ROOT / 'data' / 'processed'),
        'models_dir': str(PROJECT_ROOT / 'data' / 'models'),
        'logs_dir': str(PROJECT_ROOT / 'logs'),
    },
    
    # 日志配置
    'logging': {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'file_enabled': True,
        'console_enabled': True,
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5
    },
    
    # 会话配置
    'session': {
        'auto_save': True,
        'save_interval': 60,  # 自动保存间隔(秒)
        'session_timeout': 3600,  # 会话超时时间(秒)
    },
    
    # 调试配置
    'debug': {
        'enabled': False,
        'simulate_device': False,
        'verbose_output': False
    },
    
    # 界面配置
    'ui': {
        'visualization_enabled': True,
        'update_interval': 100,  # ms
        'theme': 'dark',
        'window_size': (1280, 720)
    },
    
    # 多线程配置
    'performance': {
        'num_workers': os.cpu_count() or 4,
        'use_gpu': True,
        'buffer_size': 1024
    }
}

# 模式设置
OPERATIONAL_MODES = {
    'offline_analysis': {
        'description': '离线分析模式 - 分析已记录的数据文件',
        'requires_device': False,
        'data_recording': False
    },
    'online_recording': {
        'description': '在线记录模式 - 记录数据但不进行实时解码',
        'requires_device': True,
        'data_recording': True
    },
    'online_decoding': {
        'description': '在线解码模式 - 实时记录并解码数据',
        'requires_device': True,
        'data_recording': True
    },
    'simulation': {
        'description': '模拟模式 - 使用模拟数据进行测试',
        'requires_device': False,
        'data_recording': True
    }
}

# 获取当前系统配置
def get_system_config():
    """返回当前系统配置"""
    return SYSTEM_CONFIG.copy()

# 获取指定模式配置
def get_mode_config(mode_name):
    """根据模式名称返回对应的配置"""
    if mode_name in OPERATIONAL_MODES:
        return OPERATIONAL_MODES[mode_name].copy()
    raise ValueError(f"未知操作模式: {mode_name}")

# 更新系统配置
def update_system_config(new_config):
    """更新系统配置"""
    global SYSTEM_CONFIG
    
    # 递归更新嵌套字典
    def update_dict(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = update_dict(d[k], v)
            else:
                d[k] = v
        return d
    
    SYSTEM_CONFIG = update_dict(SYSTEM_CONFIG, new_config)
    return get_system_config()
