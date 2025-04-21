#!/usr/bin/env python3
"""
TangiEEG 主程序入口
脑电转文本界面启动模块 - 采用模块化架构设计
"""

import argparse
import logging
import os
import sys
import time
from enum import Enum
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from config.system_config import SYSTEM_CONFIG, get_mode_config
from utils.logging_utils import setup_logging, get_logger

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

# 操作模式枚举
class OperationMode(Enum):
    """系统可用的操作模式"""
    IDLE = "idle"                 # 空闲/配置模式
    DATA_ACQUISITION = "acquire"  # 仅数据采集模式
    SIGNAL_MONITOR = "monitor"    # 信号监测模式
    OFFLINE_ANALYSIS = "analyze"  # 离线分析模式
    ONLINE_DECODING = "decode"    # 在线解码模式
    SIMULATION = "simulate"       # 模拟数据模式
    
class TangiEEGSystem:
    """TangiEEG系统主类 - 负责协调各个子模块"""
    
    def __init__(self, args):
        """初始化系统"""
        self.args = args
        self.logger = None
        self.mode = OperationMode.IDLE
        self.ui_manager = None
        self.device_manager = None
        self.filter_pipeline = None
        self.decoder = None
        self.output_manager = None
        self.visualizer = None
        self.running = False
        self.initialized = False
        
        # 初始化日志系统
        self._setup_logging()
        
    def _setup_logging(self):
        """设置日志系统"""
        log_level = logging.DEBUG if self.args.debug else logging.INFO
        setup_logging(log_level)
        self.logger = get_logger("main")
    
    def initialize(self):
        """初始化系统组件"""
        if self.initialized:
            self.logger.warning("系统已经初始化，跳过重复初始化")
            return True
            
        self.logger.info("正在初始化TangiEEG系统...")
        
        try:
            # 导入相关模块 - 延迟导入以加快启动速度
            self.logger.info("加载系统模块...")
            
            # 1. 初始化UI管理器
            from visualization.ui_manager import UIManager
            self.ui_manager = UIManager(self)
            self.logger.info("用户界面管理器初始化完成")
            
            # 2. 初始化设备管理器（根据命令行选项决定是否使用模拟数据）
            from acquisition.device_manager import DeviceManager
            self.device_manager = DeviceManager(simulate=self.args.simulate)
            self.logger.info("设备管理器初始化完成")
            
            # 3. 初始化信号处理管道
            from preprocessing.filters import FilterPipeline
            self.filter_pipeline = FilterPipeline()
            self.logger.info("信号处理管道初始化完成")
            
            # 4. 初始化可视化模块
            from visualization.signal_viewer import SignalViewer
            self.visualizer = SignalViewer()
            self.logger.info("可视化模块初始化完成")
            
            # 5. 根据需要加载解码模型
            if self.args.decode or self.args.mode == 'decode':
                from decoding.decoder import EEGDecoder
                self.decoder = EEGDecoder()
                self.logger.info("EEG解码器初始化完成")
            
            # 6. 初始化输出管理器
            from generation.output_manager import OutputManager
            self.output_manager = OutputManager()
            self.logger.info("输出管理器初始化完成")
            
            # 设置初始操作模式
            self._set_operation_mode()
            
            self.initialized = True
            self.logger.info("TangiEEG系统初始化完成")
            return True
            
        except Exception as e:
            self.logger.exception(f"系统初始化失败: {e}")
            return False
    
    def _set_operation_mode(self):
        """根据命令行参数设置操作模式"""
        mode_str = self.args.mode.lower() if self.args.mode else 'idle'
        
        # 映射命令行参数到操作模式
        mode_mapping = {
            'idle': OperationMode.IDLE,
            'acquire': OperationMode.DATA_ACQUISITION,
            'monitor': OperationMode.SIGNAL_MONITOR,
            'analyze': OperationMode.OFFLINE_ANALYSIS,
            'decode': OperationMode.ONLINE_DECODING,
            'simulate': OperationMode.SIMULATION
        }
        
        if mode_str in mode_mapping:
            self.mode = mode_mapping[mode_str]
        else:
            self.logger.warning(f"未知操作模式: {mode_str}，使用默认的IDLE模式")
            self.mode = OperationMode.IDLE
        
        self.logger.info(f"系统操作模式设置为: {self.mode.value}")
        
        # 对于特定的模式，按需加载组件
        if self.mode == OperationMode.SIMULATION:
            self.args.simulate = True
            
        if self.mode == OperationMode.ONLINE_DECODING and not self.decoder:
            from decoding.decoder import EEGDecoder
            self.decoder = EEGDecoder()
            self.logger.info("按需加载: EEG解码器初始化完成")
    
    def change_mode(self, new_mode):
        """动态更改操作模式"""
        if isinstance(new_mode, str):
            try:
                new_mode = OperationMode(new_mode)
            except ValueError:
                self.logger.error(f"无效的操作模式: {new_mode}")
                return False
        
        old_mode = self.mode
        self.mode = new_mode
        self.logger.info(f"操作模式从 {old_mode.value} 更改为 {new_mode.value}")
        
        # 根据新模式动态加载/卸载组件
        if new_mode == OperationMode.ONLINE_DECODING and not self.decoder:
            from decoding.decoder import EEGDecoder
            self.decoder = EEGDecoder()
            self.logger.info("按需加载: EEG解码器初始化完成")
        
        return True
    
    def start(self):
        """启动系统运行"""
        if not self.initialized:
            if not self.initialize():
                self.logger.error("系统初始化失败，无法启动")
                return False
        
        self.logger.info(f"启动TangiEEG系统，模式: {self.mode.value}...")
        
        try:
            # 连接设备(如果需要)
            if self.mode in [OperationMode.DATA_ACQUISITION, 
                           OperationMode.SIGNAL_MONITOR, 
                           OperationMode.ONLINE_DECODING]:
                self.logger.info("连接到EEG设备...")
                if not self.device_manager.connect():
                    self.logger.error("设备连接失败")
                    return False
            
            # 启动UI
            if self.ui_manager:
                self.ui_manager.start()
            
            self.running = True
            
            # 启动主处理循环
            self._main_loop()
            
            return True
            
        except Exception as e:
            self.logger.exception(f"系统启动失败: {e}")
            self.stop()
            return False
    
    def stop(self):
        """停止系统运行"""
        self.logger.info("正在停止TangiEEG系统...")
        
        self.running = False
        
        # 断开设备连接
        if self.device_manager and self.device_manager.is_connected():
            self.logger.info("断开设备连接...")
            self.device_manager.disconnect()
        
        # 停止UI
        if self.ui_manager:
            self.ui_manager.stop()
        
        self.logger.info("TangiEEG系统已停止")
    
    def _main_loop(self):
        """系统主处理循环"""
        self.logger.info("进入主处理循环，按Ctrl+C停止...")
        
        try:
            while self.running:
                # 根据当前模式执行相应操作
                if self.mode == OperationMode.IDLE:
                    # 空闲模式，只更新UI
                    pass
                    
                elif self.mode == OperationMode.DATA_ACQUISITION:
                    # 数据采集模式
                    eeg_data = self.device_manager.get_latest_data()
                    if eeg_data is not None:
                        # 可选地可视化原始数据
                        if self.visualizer:
                            self.visualizer.update(eeg_data)
                
                elif self.mode == OperationMode.SIGNAL_MONITOR:
                    # 信号监测模式
                    eeg_data = self.device_manager.get_latest_data()
                    if eeg_data is not None:
                        # 预处理数据
                        filtered_data = self.filter_pipeline.process(eeg_data)
                        # 可视化处理后的数据
                        if self.visualizer:
                            self.visualizer.update(filtered_data)
                
                elif self.mode == OperationMode.ONLINE_DECODING:
                    # 在线解码模式
                    eeg_data = self.device_manager.get_latest_data()
                    if eeg_data is not None:
                        # 预处理数据
                        filtered_data = self.filter_pipeline.process(eeg_data)
                        # 解码
                        if self.decoder:
                            decoded_result = self.decoder.decode(filtered_data)
                            # 输出结果
                            if decoded_result and self.output_manager:
                                self.output_manager.display(decoded_result)
                        # 可视化
                        if self.visualizer:
                            self.visualizer.update(filtered_data)
                
                elif self.mode == OperationMode.OFFLINE_ANALYSIS:
                    # 离线分析模式 - 主要由UI驱动
                    pass
                
                elif self.mode == OperationMode.SIMULATION:
                    # 模拟数据模式
                    eeg_data = self.device_manager.get_simulated_data()
                    if eeg_data is not None:
                        # 预处理数据
                        filtered_data = self.filter_pipeline.process(eeg_data)
                        # 根据需要解码
                        if self.decoder:
                            decoded_result = self.decoder.decode(filtered_data)
                            # 输出结果
                            if decoded_result and self.output_manager:
                                self.output_manager.display(decoded_result)
                        # 可视化
                        if self.visualizer:
                            self.visualizer.update(filtered_data)
                
                # 更新UI
                if self.ui_manager:
                    self.ui_manager.update()
                
                # 短暂休眠以避免CPU占用过高
                time.sleep(0.01)
                
        except KeyboardInterrupt:
            self.logger.info("接收到用户中断信号")
        except Exception as e:
            self.logger.exception(f"主循环中发生错误: {e}")
        finally:
            self.stop()

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TangiEEG - 脑电信号转文本系统')
    
    # 基本选项
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--config', type=str, help='指定配置文件路径')
    
    # 操作模式
    parser.add_argument('--mode', type=str, choices=['idle', 'acquire', 'monitor', 'analyze', 'decode', 'simulate'], 
                        default='idle', help='指定操作模式')
    
    # 功能性选项
    parser.add_argument('--simulate', action='store_true', help='使用模拟数据，无需真实设备')
    parser.add_argument('--record', action='store_true', help='记录会话数据')
    parser.add_argument('--decode', action='store_true', help='启用实时解码')
    parser.add_argument('--no-ui', action='store_true', help='禁用图形界面，使用命令行模式')
    parser.add_argument('--input-file', type=str, help='离线分析模式下的输入文件')
    
    return parser.parse_args()

def main():
    """主程序入口"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 创建系统实例
    system = TangiEEGSystem(args)
    
    # 初始化并启动系统
    if system.initialize():
        success = system.start()
        return 0 if success else 1
    else:
        return 1

if __name__ == "__main__":
    sys.exit(main())
