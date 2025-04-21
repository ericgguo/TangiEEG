#!/usr/bin/env python3
"""
TangiEEG 主程序入口
脑电转文本界面启动模块
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config.system_config import SYSTEM_CONFIG
from utils.logging_utils import setup_logging

# 导入相关模块
try:
    from acquisition.device_manager import DeviceManager
    from preprocessing.filters import FilterPipeline
    from decoding.decoder import EEGDecoder
    from generation.output_manager import OutputManager
    from visualization.signal_viewer import SignalViewer
except ImportError as e:
    print(f"导入模块失败: {e}")
    print("请确保已安装所有必要依赖: pip install -r requirements.txt")
    sys.exit(1)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TangiEEG - 脑电信号转文本系统')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据，无需真实设备')
    parser.add_argument('--config', type=str, help='指定配置文件路径')
    parser.add_argument('--record', action='store_true', help='记录会话数据')
    return parser.parse_args()

def main():
    """主程序入口函数"""
    args = parse_arguments()
    
    # 设置日志
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    logger = logging.getLogger('main')
    
    logger.info("正在启动 TangiEEG 系统...")
    
    try:
        # 初始化设备管理器
        logger.info("初始化设备连接...")
        device_manager = DeviceManager(simulate=args.simulate)
        
        # 初始化信号处理管道
        logger.info("初始化信号处理模块...")
        filter_pipeline = FilterPipeline()
        
        # 初始化解码器
        logger.info("加载解码模型...")
        decoder = EEGDecoder()
        
        # 初始化输出管理器
        logger.info("初始化文本输出模块...")
        output_manager = OutputManager()
        
        # 初始化可视化
        logger.info("初始化信号可视化...")
        signal_viewer = SignalViewer()
        
        # 连接设备
        logger.info("连接到 OpenBCI 设备...")
        if not device_manager.connect():
            logger.error("设备连接失败")
            return
        
        # 主处理循环
        logger.info("系统准备就绪，按 Ctrl+C 停止...")
        try:
            while True:
                # 获取EEG数据
                eeg_data = device_manager.get_latest_data()
                
                # 预处理数据
                filtered_data = filter_pipeline.process(eeg_data)
                
                # 解码
                decoded_result = decoder.decode(filtered_data)
                
                # 更新可视化
                signal_viewer.update(filtered_data)
                
                # 输出结果
                if decoded_result:
                    output_manager.display(decoded_result)
                
                time.sleep(0.1)  # 短暂休眠避免CPU占用过高
                
        except KeyboardInterrupt:
            logger.info("接收到退出信号")
        
        # 清理资源
        logger.info("关闭设备连接...")
        device_manager.disconnect()
        logger.info("系统已关闭")
        
    except Exception as e:
        logger.exception(f"系统运行时发生错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
