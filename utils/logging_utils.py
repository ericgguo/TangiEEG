"""
日志工具模块 - 提供日志记录和管理功能
"""

import logging
import os
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.system_config import SYSTEM_CONFIG

# 日志格式
DEFAULT_LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DEFAULT_LOG_LEVEL = logging.INFO
DEFAULT_LOG_DIR = os.path.join(Path(__file__).parent.parent.absolute(), 'logs')

def setup_logging(level=None, log_dir=None, log_to_file=True, log_to_console=True):
    """
    设置日志系统
    
    Args:
        level: 日志级别，默认为INFO
        log_dir: 日志目录，默认为项目根目录下的logs文件夹
        log_to_file: 是否将日志写入文件
        log_to_console: 是否将日志输出到控制台
    
    Returns:
        logger: 根日志记录器
    """
    # 使用配置中的值或默认值
    config = SYSTEM_CONFIG.get('logging', {})
    level = level or config.get('level', DEFAULT_LOG_LEVEL)
    
    # 如果级别是字符串，转换为对应的日志级别
    if isinstance(level, str):
        level = getattr(logging, level.upper(), DEFAULT_LOG_LEVEL)
    
    log_dir = log_dir or config.get('logs_dir', DEFAULT_LOG_DIR)
    log_format = config.get('format', DEFAULT_LOG_FORMAT)
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 配置根日志记录器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除现有处理器
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # 设置格式化器
    formatter = logging.Formatter(log_format)
    
    # 添加控制台处理器
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # 添加文件处理器
    if log_to_file:
        log_file = os.path.join(log_dir, f'tangieeg_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        max_size = config.get('max_file_size', 10 * 1024 * 1024)  # 默认10MB
        backup_count = config.get('backup_count', 5)
        
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_size, 
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 创建模块专用日志记录器
    logger = logging.getLogger('tangieeg')
    
    # 记录初始日志
    logger.info(f"日志系统初始化完成，日志级别：{logging.getLevelName(level)}")
    if log_to_file:
        logger.info(f"日志文件位置：{log_file}")
    
    return logger

def get_logger(name):
    """
    获取指定名称的日志记录器
    
    Args:
        name: 日志记录器名称，通常是模块名
    
    Returns:
        logger: 指定名称的日志记录器
    """
    return logging.getLogger(f'tangieeg.{name}')
