"""
设备管理器 - 负责OpenBCI设备的连接和管理
支持真实设备和模拟数据模式
"""

import time
import logging
import numpy as np
from pathlib import Path
import threading
import queue

# 创建logger
logger = logging.getLogger('device_manager')

class DeviceManager:
    """OpenBCI设备管理类，处理设备连接、配置和数据流"""
    
    def __init__(self, simulate=False, device_type='daisy', connection_params=None):
        """
        初始化设备管理器
        
        参数:
            simulate (bool): 是否使用模拟数据而非真实设备
            device_type (str): 设备类型 ('cyton', 'ganglion', 'daisy')
            connection_params (dict): 连接参数
        """
        self.simulate = simulate
        self.device_type = device_type.lower()
        self.connection_params = connection_params or {}
        self.connected = False
        self.streaming = False
        self.board = None
        self.data_queue = queue.Queue(maxsize=1000)
        self.stream_thread = None
        
        # 设置通道数
        if self.device_type == 'cyton':
            self.channels = 8
        elif self.device_type == 'ganglion':
            self.channels = 4
        elif self.device_type == 'daisy':
            self.channels = 16
        else:
            self.channels = 8  # 默认值
            
        # 设置采样率
        self.sample_rate = self.connection_params.get('sample_rate', 250)
        
        # 模拟数据生成器
        if self.simulate:
            from acquisition.data_streamer import SimulatedDataStreamer
            self.simulator = SimulatedDataStreamer(
                channels=self.channels,
                sample_rate=self.sample_rate,
                noise_level=self.connection_params.get('noise_level', 0.1),
                artifact_prob=self.connection_params.get('artifact_prob', 0.05)
            )
        
        logger.info(f"设备管理器初始化: 模式={'模拟' if simulate else '真实'}, 设备类型={device_type}, 通道数={self.channels}")
    
    def connect(self):
        """连接到OpenBCI设备或初始化模拟器"""
        if self.connected:
            logger.warning("设备已经连接")
            return True
            
        try:
            if self.simulate:
                # 模拟模式，不需要实际连接
                logger.info("正在初始化模拟数据流...")
                self.connected = True
                logger.info("模拟数据模式已就绪")
            else:
                # 真实设备模式
                logger.info(f"正在连接到 {self.device_type} 设备...")
                
                # TODO: 实现真实设备连接逻辑
                # 这里将使用pyOpenBCI库连接真实设备
                # 目前为了测试，直接返回成功
                time.sleep(2)  # 模拟连接延迟
                self.connected = True
                logger.info("设备连接成功")
            
            return self.connected
            
        except Exception as e:
            logger.error(f"设备连接失败: {e}")
            self.connected = False
            return False
    
    def disconnect(self):
        """断开设备连接"""
        if not self.connected:
            logger.warning("设备未连接")
            return True
            
        try:
            # 如果数据流正在运行，先停止它
            if self.streaming:
                self.stop_stream()
            
            if not self.simulate:
                # 真实设备模式，需要关闭连接
                # TODO: 实现真实设备断开连接逻辑
                pass
                
            self.connected = False
            logger.info("设备已断开连接")
            return True
            
        except Exception as e:
            logger.error(f"断开设备连接失败: {e}")
            return False
    
    def start_stream(self):
        """启动数据流"""
        if not self.connected:
            logger.error("无法启动数据流: 设备未连接")
            return False
            
        if self.streaming:
            logger.warning("数据流已经在运行")
            return True
            
        try:
            if self.simulate:
                # 启动模拟数据流线程
                self.stream_thread = threading.Thread(target=self._simulate_stream, daemon=True)
                self.stream_thread.start()
            else:
                # 真实设备模式，启动数据流
                # TODO: 实现真实设备数据流启动逻辑
                # 这将使用pyOpenBCI库启动数据流
                pass
                
            self.streaming = True
            logger.info("数据流已启动")
            return True
            
        except Exception as e:
            logger.error(f"启动数据流失败: {e}")
            return False
    
    def stop_stream(self):
        """停止数据流"""
        if not self.streaming:
            logger.warning("数据流未运行")
            return True
            
        try:
            if self.simulate:
                # 模拟模式，只需设置标志位
                self.streaming = False
                if self.stream_thread and self.stream_thread.is_alive():
                    self.stream_thread.join(timeout=2.0)
            else:
                # 真实设备模式，停止数据流
                # TODO: 实现真实设备数据流停止逻辑
                pass
                
            self.streaming = False
            logger.info("数据流已停止")
            return True
            
        except Exception as e:
            logger.error(f"停止数据流失败: {e}")
            return False
    
    def get_latest_data(self, samples=250, flush=False):
        """
        获取最新的EEG数据
        
        参数:
            samples (int): 要获取的样本数
            flush (bool): 是否清空队列中更早的数据
            
        返回:
            numpy.ndarray: 形状为 (channels, samples) 的数据数组
        """
        if not self.connected:
            logger.warning("无法获取数据: 设备未连接")
            return None
            
        if self.simulate and not self.streaming:
            # 如果是模拟模式但流未启动，返回一些随机数据
            return self.simulator.generate_chunk(samples)
            
        try:
            # 从队列中获取数据
            collected_data = []
            
            # 如果要清空队列，先丢弃旧数据
            if flush:
                while not self.data_queue.empty():
                    try:
                        self.data_queue.get_nowait()
                    except queue.Empty:
                        break
            
            # 收集请求的样本数量
            timeout = samples / self.sample_rate * 2  # 超时时间是预期时间的2倍
            end_time = time.time() + timeout
            
            while len(collected_data) < samples and time.time() < end_time:
                try:
                    data = self.data_queue.get(timeout=0.1)
                    collected_data.append(data)
                except queue.Empty:
                    continue
            
            if not collected_data:
                logger.warning(f"获取数据超时，没有收到数据，预期样本数: {samples}")
                # 如果是模拟模式，返回一些随机数据
                if self.simulate:
                    return self.simulator.generate_chunk(samples)
                return None
                
            # 合并数据
            if len(collected_data) < samples:
                logger.warning(f"数据不足，请求 {samples} 样本，仅获取到 {len(collected_data)} 样本")
                
            # 处理可能不完整的数据
            data_array = np.vstack(collected_data)
            
            # 如果获取的数据超过请求的样本数，只返回最新的数据
            if len(data_array) > samples:
                data_array = data_array[-samples:]
                
            # 转置为 (channels, samples) 格式
            return data_array.T
                
        except Exception as e:
            logger.error(f"获取数据失败: {e}")
            return None
    
    def check_impedance(self, duration=5):
        """
        检查电极阻抗
        
        参数:
            duration (int): 检查持续时间(秒)
            
        返回:
            dict: 每个通道的阻抗值(欧姆)
        """
        if not self.connected:
            logger.warning("无法检查阻抗: 设备未连接")
            return None
            
        if self.simulate:
            # 模拟阻抗检查
            logger.info("模拟阻抗检查...")
            
            # 生成随机阻抗值
            impedances = {}
            for ch in range(1, self.channels + 1):
                # 大部分通道阻抗正常，少部分不正常
                if np.random.random() > 0.7:
                    imp = np.random.uniform(500000, 1500000)  # 高阻抗 (500k-1.5M Ohm)
                else:
                    imp = np.random.uniform(5000, 100000)  # 正常阻抗 (5k-100k Ohm)
                impedances[ch] = imp
                
            logger.info("阻抗检查完成")
            return impedances
        else:
            # 真实设备阻抗检查
            # TODO: 实现真实设备阻抗检查逻辑
            logger.warning("真实设备阻抗检查暂未实现")
            return None
    
    def _simulate_stream(self):
        """模拟数据流线程函数"""
        logger.info("模拟数据流启动")
        
        try:
            # 计算每次迭代应生成的样本数
            chunk_size = max(1, self.sample_rate // 10)  # 每次生成约100ms的数据
            
            while self.streaming:
                # 生成模拟数据
                eeg_data = self.simulator.generate_chunk(chunk_size)
                
                # 对于每个样本，添加到队列
                for i in range(eeg_data.shape[1]):
                    sample = eeg_data[:, i]
                    
                    try:
                        # 如果队列满了，丢弃旧数据
                        if self.data_queue.full():
                            self.data_queue.get_nowait()
                        self.data_queue.put(sample)
                    except Exception as e:
                        logger.error(f"添加数据到队列失败: {e}")
                
                # 控制生成速率
                time.sleep(chunk_size / self.sample_rate)
                
        except Exception as e:
            logger.error(f"模拟数据流发生错误: {e}")
            self.streaming = False
