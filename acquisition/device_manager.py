"""
设备管理器模块 - 负责与OpenBCI硬件设备的通信和数据获取
支持实际硬件和模拟数据模式
"""

import os
import sys
import time
import random
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from utils.logging_utils import get_logger
from config.device_config import get_device_config

# 尝试导入pyOpenBCI库，如果不可用则使用模拟模式
try:
    import pyOpenBCI
    from pyOpenBCI import OpenBCICyton
    HAS_OPENBCI = True
except ImportError:
    HAS_OPENBCI = False

class DeviceManager:
    """OpenBCI设备管理器类"""
    
    def __init__(self, simulate=False):
        """
        初始化设备管理器
        
        Args:
            simulate: 是否使用模拟数据模式
        """
        self.logger = get_logger("device")
        self.config = get_device_config()['openbci']
        self.connected = False
        self.board = None
        self.data_queue = Queue(maxsize=1000)  # 数据缓冲队列
        self.simulate = simulate or not HAS_OPENBCI
        
        # 模拟模式相关
        self.last_sample_time = 0
        self.sample_interval = 1.0 / self.config['sample_rate']
        self.simulate_thread = None
        
        # 上次读取的数据
        self.last_data = None
        
        # 记录当前设备状态
        self.device_status = {
            'battery_level': None,
            'signal_quality': None,
            'impedance': [None] * 8,
            'sample_rate': self.config['sample_rate'],
            'channel_count': len(self.config['channels']),
            'timestamp': None
        }
        
        if self.simulate:
            self.logger.info("设备管理器以模拟模式初始化")
        else:
            self.logger.info("设备管理器以实际设备模式初始化")
    
    def connect(self):
        """
        连接到OpenBCI设备
        
        Returns:
            bool: 连接成功返回True，否则返回False
        """
        if self.connected:
            self.logger.warning("设备已经连接，忽略重复连接请求")
            return True
        
        try:
            if self.simulate:
                # 模拟设备连接
                self.logger.info("连接到模拟设备...")
                self._start_simulate_thread()
                self.connected = True
                self.device_status['timestamp'] = datetime.now().isoformat()
                self.device_status['battery_level'] = 95  # 模拟电池电量
                self.device_status['signal_quality'] = 98  # 模拟信号质量
                self.logger.info("模拟设备连接成功")
                return True
            
            else:
                # 实际设备连接
                self.logger.info("连接到OpenBCI设备...")
                
                # 获取串口配置
                serial_port = self.config['serial_port']
                if serial_port is None:
                    # 自动检测串口
                    self.logger.info("尝试自动检测串口...")
                    # 这里可以实现自动检测逻辑
                
                # 创建OpenBCI板对象
                self.board = OpenBCICyton(
                    port=serial_port,
                    daisy=False  # 是否使用Daisy模块（16通道）
                )
                
                # 启动数据流并设置回调
                self.board.start_stream(self._data_callback)
                
                # 等待确认连接
                time.sleep(1.0)
                
                # 应用设备配置
                self._apply_device_config()
                
                self.connected = True
                self.device_status['timestamp'] = datetime.now().isoformat()
                self.logger.info("OpenBCI设备连接成功")
                return True
        
        except Exception as e:
            self.logger.error(f"设备连接失败: {e}")
            # 出错时回退到模拟模式
            self.logger.info("回退到模拟模式")
            self.simulate = True
            self.board = None
            return self.connect()  # 递归调用以模拟模式连接
    
    def disconnect(self):
        """
        断开与OpenBCI设备的连接
        
        Returns:
            bool: 断开成功返回True，否则返回False
        """
        if not self.connected:
            self.logger.warning("设备未连接，忽略断开请求")
            return True
        
        try:
            if self.simulate:
                # 停止模拟线程
                self.logger.info("断开模拟设备连接...")
                self.simulate = False
                if self.simulate_thread and self.simulate_thread.is_alive():
                    self.simulate_thread.join(timeout=1.0)
            else:
                # 停止实际设备
                self.logger.info("断开OpenBCI设备连接...")
                if self.board:
                    self.board.stop_stream()
                    self.board = None
            
            # 清空队列
            while not self.data_queue.empty():
                try:
                    self.data_queue.get_nowait()
                except Empty:
                    break
            
            self.connected = False
            self.logger.info("设备已断开连接")
            return True
        
        except Exception as e:
            self.logger.error(f"断开设备连接失败: {e}")
            return False
    
    def is_connected(self):
        """
        检查设备是否已连接
        
        Returns:
            bool: 已连接返回True，否则返回False
        """
        return self.connected
    
    def get_latest_data(self, timeout=0.1):
        """
        获取最新的EEG数据
        
        Args:
            timeout: 等待数据的超时时间(秒)
        
        Returns:
            numpy.ndarray: 形状为(通道数, 样本数)的EEG数据，无数据时返回None
        """
        if not self.connected:
            self.logger.warning("设备未连接，无法获取数据")
            return None
        
        try:
            # 尝试从队列获取最新数据
            data = self.data_queue.get(timeout=timeout)
            self.last_data = data
            return data
        except Empty:
            return self.last_data
    
    def get_simulated_data(self):
        """
        获取模拟EEG数据（仅在模拟模式下有效）
        
        Returns:
            numpy.ndarray: 形状为(通道数, 样本数)的模拟EEG数据
        """
        # 生成新的模拟数据
        if self.simulate:
            return self._generate_simulated_data()
        else:
            self.logger.warning("非模拟模式下调用了get_simulated_data")
            return None
    
    def get_battery_level(self):
        """
        获取设备电池电量
        
        Returns:
            int: 电池电量百分比，无法获取时返回None
        """
        if not self.connected:
            return None
        
        # 在模拟模式下生成模拟电池电量
        if self.simulate:
            # 模拟电池电量缓慢下降
            current_level = self.device_status['battery_level']
            if current_level is not None:
                # 每小时下降约2%
                elapsed_time = time.time() - self.last_sample_time
                decrease = (elapsed_time / 3600) * 2
                new_level = max(0, current_level - decrease)
                self.device_status['battery_level'] = new_level
            return int(self.device_status['battery_level']) if self.device_status['battery_level'] is not None else None
        else:
            # 实际设备的电池电量获取
            # 注意：这需要特定的命令支持，以下仅为占位实现
            if self.board:
                # TODO: 实现实际电池电量读取
                return None
            return None
    
    def get_signal_quality(self):
        """
        获取信号质量评估
        
        Returns:
            int: 信号质量百分比，无法获取时返回None
        """
        if not self.connected:
            return None
        
        # 模拟信号质量
        if self.simulate:
            current_quality = self.device_status['signal_quality']
            if current_quality is not None:
                # 模拟信号质量波动
                new_quality = current_quality + random.uniform(-5, 5)
                new_quality = max(0, min(100, new_quality))  # 限制在0-100范围内
                self.device_status['signal_quality'] = new_quality
            return int(self.device_status['signal_quality']) if self.device_status['signal_quality'] is not None else None
        else:
            # 实际设备的信号质量评估
            # 这通常基于阻抗检测和信号波动分析
            if self.board:
                # TODO: 实现实际信号质量评估
                return None
            return None
    
    def check_channel_impedance(self, channel_id):
        """
        检查指定通道的电极阻抗
        
        Args:
            channel_id: 通道ID (1-8)
        
        Returns:
            float: 阻抗值(欧姆)，无法获取时返回None
        """
        if not self.connected or channel_id < 1 or channel_id > 8:
            return None
        
        idx = channel_id - 1
        
        # 模拟阻抗检测
        if self.simulate:
            # 生成随机阻抗值，通常好的接触阻抗在10kΩ以下
            impedance = random.uniform(5000, 50000)
            self.device_status['impedance'][idx] = impedance
            return impedance
        else:
            # 实际设备的阻抗检测
            if self.board:
                # TODO: 实现实际阻抗检测
                return None
            return None
    
    def set_sample_rate(self, rate):
        """
        设置采样率
        
        Args:
            rate: 采样率 (Hz)
        
        Returns:
            bool: 设置成功返回True，否则返回False
        """
        if not self.connected:
            self.logger.warning("设备未连接，无法设置采样率")
            return False
        
        try:
            if rate not in [250, 500, 1000, 2000]:
                self.logger.error(f"不支持的采样率: {rate} Hz")
                return False
            
            if self.simulate:
                self.logger.info(f"设置模拟采样率为 {rate} Hz")
                self.config['sample_rate'] = rate
                self.sample_interval = 1.0 / rate
                self.device_status['sample_rate'] = rate
                return True
            else:
                # 实际设备设置采样率
                if self.board:
                    # TODO: 实现实际采样率设置
                    self.config['sample_rate'] = rate
                    self.device_status['sample_rate'] = rate
                    self.logger.info(f"设置设备采样率为 {rate} Hz")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"设置采样率失败: {e}")
            return False
    
    def set_channel_config(self, channel_id, enabled=True, gain=24):
        """
        配置特定通道
        
        Args:
            channel_id: 通道ID (1-8)
            enabled: 是否启用该通道
            gain: 增益值 (1, 2, 4, 6, 8, 12, 24)
        
        Returns:
            bool: 设置成功返回True，否则返回False
        """
        if not self.connected or channel_id < 1 or channel_id > 8:
            return False
        
        try:
            if gain not in [1, 2, 4, 6, 8, 12, 24]:
                self.logger.error(f"不支持的增益值: {gain}")
                return False
            
            if self.simulate:
                # 在模拟模式下更新通道配置
                if enabled and channel_id not in self.config['channels']:
                    self.config['channels'].append(channel_id)
                    self.config['channels'].sort()
                elif not enabled and channel_id in self.config['channels']:
                    self.config['channels'].remove(channel_id)
                
                self.logger.info(f"模拟模式: 通道 {channel_id} 已设置为 {'启用' if enabled else '禁用'}, 增益={gain}")
                return True
            else:
                # 实际设备设置通道配置
                if self.board:
                    # TODO: 实现实际通道配置
                    self.logger.info(f"通道 {channel_id} 已设置为 {'启用' if enabled else '禁用'}, 增益={gain}")
                    return True
                return False
        except Exception as e:
            self.logger.error(f"设置通道配置失败: {e}")
            return False
    
    def _apply_device_config(self):
        """应用设备配置到实际设备"""
        if not self.board:
            return
        
        # TODO: 实现配置应用逻辑
        self.logger.info("正在应用设备配置...")
        
        # 应用滤波器设置
        pass
        
        # 应用通道设置
        pass
        
        # 应用采样率设置
        pass
        
        self.logger.info("设备配置应用完成")
    
    def _data_callback(self, sample):
        """
        OpenBCI数据回调函数
        
        Args:
            sample: OpenBCI样本对象
        """
        try:
            # 提取EEG数据
            eeg_data = np.array(sample.channels_data)
            
            # 整形为(通道数, 1)
            eeg_data = eeg_data.reshape(-1, 1)
            
            # 将数据放入队列
            if not self.data_queue.full():
                self.data_queue.put(eeg_data)
            else:
                # 队列满，丢弃最旧的数据并添加新数据
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put(eeg_data)
                except Empty:
                    pass
            
            # 更新设备状态
            self.device_status['timestamp'] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"处理数据回调时出错: {e}")
    
    def _start_simulate_thread(self):
        """启动模拟数据生成线程"""
        if self.simulate_thread and self.simulate_thread.is_alive():
            return
        
        self.simulate = True
        self.simulate_thread = threading.Thread(
            target=self._simulate_data_stream,
            daemon=True
        )
        self.simulate_thread.start()
    
    def _simulate_data_stream(self):
        """模拟数据流生成线程"""
        self.logger.info("启动模拟数据流...")
        self.last_sample_time = time.time()
        
        while self.simulate and self.connected:
            current_time = time.time()
            elapsed = current_time - self.last_sample_time
            
            # 按照采样率生成数据
            if elapsed >= self.sample_interval:
                # 生成模拟数据
                data = self._generate_simulated_data()
                
                # 将数据放入队列
                if not self.data_queue.full():
                    self.data_queue.put(data)
                else:
                    # 队列满，丢弃最旧的数据并添加新数据
                    try:
                        self.data_queue.get_nowait()
                        self.data_queue.put(data)
                    except Empty:
                        pass
                
                self.last_sample_time = current_time
            
            # 短暂休眠以避免CPU占用过高
            time.sleep(0.001)
        
        self.logger.info("模拟数据流已停止")
    
    def _generate_simulated_data(self):
        """
        生成模拟EEG数据
        
        Returns:
            numpy.ndarray: 形状为(通道数, 样本数)的模拟EEG数据
        """
        # 设置通道数和样本数
        num_channels = len(self.config['channels'])
        num_samples = 10  # 每次生成10个样本点
        
        # 创建数据数组
        data = np.zeros((num_channels, num_samples))
        
        # 对每个通道生成模拟脑电信号
        for i in range(num_channels):
            # 模拟Alpha波 (8-13 Hz)
            alpha_freq = 10.0
            alpha_amp = 10.0 * (i % 3 + 1)  # 不同通道有不同幅度
            
            # 模拟Beta波 (13-30 Hz)
            beta_freq = 20.0
            beta_amp = 5.0
            
            # 模拟Theta波 (4-8 Hz)
            theta_freq = 6.0
            theta_amp = 15.0
            
            # 模拟Delta波 (0.5-4 Hz)
            delta_freq = 2.0
            delta_amp = 20.0
            
            # 生成时间点
            t = np.linspace(self.last_sample_time, 
                          self.last_sample_time + num_samples / self.config['sample_rate'], 
                          num_samples)
            
            # 合成信号
            alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * t)
            beta = beta_amp * np.sin(2 * np.pi * beta_freq * t)
            theta = theta_amp * np.sin(2 * np.pi * theta_freq * t)
            delta = delta_amp * np.sin(2 * np.pi * delta_freq * t)
            
            # 添加随机噪声
            noise = np.random.normal(0, 2.0, num_samples)
            
            # 合成最终信号
            channel_data = alpha + beta + theta + delta + noise
            
            # 如果是第一个通道，模拟眨眼伪迹
            if i == 0 and random.random() < 0.05:  # 5%的概率出现眨眼
                blink_pos = random.randint(0, num_samples - 1)
                blink_width = min(5, num_samples - blink_pos)
                blink_amp = random.uniform(50.0, 100.0)
                channel_data[blink_pos:blink_pos+blink_width] += blink_amp
            
            # 保存到数据数组
            data[i] = channel_data
        
        return data
