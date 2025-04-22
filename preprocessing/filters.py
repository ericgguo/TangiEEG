"""
滤波器模块 - 实现各种数字滤波器用于EEG信号预处理
"""

import logging
import numpy as np
from scipy import signal
import mne

# 创建logger
logger = logging.getLogger('filters')

class BaseFilter:
    """基础滤波器类"""
    
    def __init__(self, name="BaseFilter"):
        """初始化基础滤波器"""
        self.name = name
        logger.debug(f"初始化滤波器: {name}")
    
    def apply(self, data, sampling_rate):
        """
        应用滤波器
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 处理后的数据
        """
        # 基类不做任何处理，直接返回原始数据
        return data

class BandpassFilter(BaseFilter):
    """带通滤波器 - 保留特定频率范围内的信号"""
    
    def __init__(self, low_cutoff=1.0, high_cutoff=50.0, order=4, method='butterworth'):
        """
        初始化带通滤波器
        
        参数:
            low_cutoff (float): 低切频率(Hz)
            high_cutoff (float): 高切频率(Hz)
            order (int): 滤波器阶数
            method (str): 滤波器类型，'butterworth', 'cheby1', 'cheby2', 'elliptic'
        """
        super().__init__(name=f"{method.capitalize()}Bandpass({low_cutoff:.1f}-{high_cutoff:.1f}Hz)")
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order
        self.method = method
        
        # 确保方法有效
        valid_methods = ['butterworth', 'cheby1', 'cheby2', 'elliptic']
        if method not in valid_methods:
            logger.warning(f"无效的滤波器方法: {method}，使用butterworth代替")
            self.method = 'butterworth'
    
    def apply(self, data, sampling_rate):
        """应用带通滤波"""
        if data is None or data.size == 0:
            logger.error("无法应用滤波: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 确保截止频率有效
            nyquist = sampling_rate / 2
            if self.low_cutoff >= nyquist or self.high_cutoff >= nyquist:
                logger.error(f"截止频率无效: 低切={self.low_cutoff}Hz, 高切={self.high_cutoff}Hz, 奈奎斯特频率={nyquist}Hz")
                return data
                
            # 归一化截止频率
            low = self.low_cutoff / nyquist
            high = self.high_cutoff / nyquist
            
            # 设计滤波器
            if self.method == 'butterworth':
                b, a = signal.butter(self.order, [low, high], btype='band')
            elif self.method == 'cheby1':
                b, a = signal.cheby1(self.order, 0.5, [low, high], btype='band')
            elif self.method == 'cheby2':
                b, a = signal.cheby2(self.order, 30, [low, high], btype='band')
            elif self.method == 'elliptic':
                b, a = signal.ellip(self.order, 0.5, 30, [low, high], btype='band')
                
            # 应用滤波器
            filtered_data = np.zeros_like(data)
            for ch in range(channels):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
                
            logger.debug(f"应用带通滤波: {self.low_cutoff:.1f}-{self.high_cutoff:.1f}Hz")
            return filtered_data
            
        except Exception as e:
            logger.error(f"带通滤波失败: {e}")
            return data

class NotchFilter(BaseFilter):
    """陷波滤波器 - 去除电源线噪声"""
    
    def __init__(self, freq=50.0, quality_factor=30.0):
        """
        初始化陷波滤波器
        
        参数:
            freq (float): 要去除的频率(Hz)，通常为50Hz(欧洲)或60Hz(美国)
            quality_factor (float): 品质因数，控制陷波带宽
        """
        super().__init__(name=f"Notch({freq:.1f}Hz)")
        self.freq = freq
        self.quality_factor = quality_factor
    
    def apply(self, data, sampling_rate):
        """应用陷波滤波"""
        if data is None or data.size == 0:
            logger.error("无法应用滤波: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 确保频率有效
            nyquist = sampling_rate / 2
            if self.freq >= nyquist:
                logger.error(f"陷波频率无效: {self.freq}Hz, 奈奎斯特频率={nyquist}Hz")
                return data
                
            # 归一化频率
            freq = self.freq / nyquist
            
            # 设计陷波滤波器
            b, a = signal.iirnotch(freq, self.quality_factor)
                
            # 应用滤波器
            filtered_data = np.zeros_like(data)
            for ch in range(channels):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
                
            logger.debug(f"应用陷波滤波: {self.freq:.1f}Hz")
            return filtered_data
            
        except Exception as e:
            logger.error(f"陷波滤波失败: {e}")
            return data

class HighpassFilter(BaseFilter):
    """高通滤波器 - 去除低频漂移"""
    
    def __init__(self, cutoff=0.5, order=4, method='butterworth'):
        """
        初始化高通滤波器
        
        参数:
            cutoff (float): 截止频率(Hz)
            order (int): 滤波器阶数
            method (str): 滤波器类型，'butterworth', 'cheby1', 'cheby2', 'elliptic'
        """
        super().__init__(name=f"{method.capitalize()}Highpass({cutoff:.1f}Hz)")
        self.cutoff = cutoff
        self.order = order
        self.method = method
        
        # 确保方法有效
        valid_methods = ['butterworth', 'cheby1', 'cheby2', 'elliptic']
        if method not in valid_methods:
            logger.warning(f"无效的滤波器方法: {method}，使用butterworth代替")
            self.method = 'butterworth'
    
    def apply(self, data, sampling_rate):
        """应用高通滤波"""
        if data is None or data.size == 0:
            logger.error("无法应用滤波: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 确保截止频率有效
            nyquist = sampling_rate / 2
            if self.cutoff >= nyquist:
                logger.error(f"截止频率无效: {self.cutoff}Hz, 奈奎斯特频率={nyquist}Hz")
                return data
                
            # 归一化截止频率
            freq = self.cutoff / nyquist
            
            # 设计滤波器
            if self.method == 'butterworth':
                b, a = signal.butter(self.order, freq, btype='high')
            elif self.method == 'cheby1':
                b, a = signal.cheby1(self.order, 0.5, freq, btype='high')
            elif self.method == 'cheby2':
                b, a = signal.cheby2(self.order, 30, freq, btype='high')
            elif self.method == 'elliptic':
                b, a = signal.ellip(self.order, 0.5, 30, freq, btype='high')
                
            # 应用滤波器
            filtered_data = np.zeros_like(data)
            for ch in range(channels):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
                
            logger.debug(f"应用高通滤波: {self.cutoff:.1f}Hz")
            return filtered_data
            
        except Exception as e:
            logger.error(f"高通滤波失败: {e}")
            return data

class LowpassFilter(BaseFilter):
    """低通滤波器 - 平滑信号并去除高频噪声"""
    
    def __init__(self, cutoff=50.0, order=4, method='butterworth'):
        """
        初始化低通滤波器
        
        参数:
            cutoff (float): 截止频率(Hz)
            order (int): 滤波器阶数
            method (str): 滤波器类型，'butterworth', 'cheby1', 'cheby2', 'elliptic'
        """
        super().__init__(name=f"{method.capitalize()}Lowpass({cutoff:.1f}Hz)")
        self.cutoff = cutoff
        self.order = order
        self.method = method
        
        # 确保方法有效
        valid_methods = ['butterworth', 'cheby1', 'cheby2', 'elliptic']
        if method not in valid_methods:
            logger.warning(f"无效的滤波器方法: {method}，使用butterworth代替")
            self.method = 'butterworth'
    
    def apply(self, data, sampling_rate):
        """应用低通滤波"""
        if data is None or data.size == 0:
            logger.error("无法应用滤波: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 确保截止频率有效
            nyquist = sampling_rate / 2
            if self.cutoff >= nyquist:
                logger.error(f"截止频率无效: {self.cutoff}Hz, 奈奎斯特频率={nyquist}Hz")
                return data
                
            # 归一化截止频率
            freq = self.cutoff / nyquist
            
            # 设计滤波器
            if self.method == 'butterworth':
                b, a = signal.butter(self.order, freq, btype='low')
            elif self.method == 'cheby1':
                b, a = signal.cheby1(self.order, 0.5, freq, btype='low')
            elif self.method == 'cheby2':
                b, a = signal.cheby2(self.order, 30, freq, btype='low')
            elif self.method == 'elliptic':
                b, a = signal.ellip(self.order, 0.5, 30, freq, btype='low')
                
            # 应用滤波器
            filtered_data = np.zeros_like(data)
            for ch in range(channels):
                filtered_data[ch] = signal.filtfilt(b, a, data[ch])
                
            logger.debug(f"应用低通滤波: {self.cutoff:.1f}Hz")
            return filtered_data
            
        except Exception as e:
            logger.error(f"低通滤波失败: {e}")
            return data

class SpatialFilter(BaseFilter):
    """空间滤波器 - 如CAR(Common Average Reference)、拉普拉斯滤波等"""
    
    def __init__(self, method='car', neighbor_indices=None):
        """
        初始化空间滤波器
        
        参数:
            method (str): 空间滤波方法，'car'(共均值参考), 'laplacian'(拉普拉斯), 'bipolar'(双极)
            neighbor_indices (dict): 每个通道的邻居通道索引，用于拉普拉斯滤波
        """
        super().__init__(name=f"Spatial({method})")
        self.method = method
        self.neighbor_indices = neighbor_indices
        
        # 确保方法有效
        valid_methods = ['car', 'laplacian', 'bipolar']
        if method not in valid_methods:
            logger.warning(f"无效的空间滤波方法: {method}，使用car代替")
            self.method = 'car'
    
    def apply(self, data, sampling_rate):
        """应用空间滤波"""
        if data is None or data.size == 0:
            logger.error("无法应用滤波: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 根据方法应用不同的空间滤波
            if self.method == 'car':
                # 共均值参考 - 从每个通道减去所有通道的平均值
                filtered_data = data - np.mean(data, axis=0, keepdims=True)
                
            elif self.method == 'laplacian':
                # 拉普拉斯滤波 - 从每个通道减去其邻居通道的平均值
                if self.neighbor_indices is None:
                    logger.error("拉普拉斯滤波需要提供邻居通道索引")
                    return data
                    
                filtered_data = np.zeros_like(data)
                for ch in range(channels):
                    if ch in self.neighbor_indices and self.neighbor_indices[ch]:
                        # 计算邻居通道的平均值
                        neighbors = self.neighbor_indices[ch]
                        neighbor_avg = np.mean(data[neighbors, :], axis=0)
                        # 从当前通道减去邻居通道的平均值
                        filtered_data[ch] = data[ch] - neighbor_avg
                    else:
                        # 没有邻居通道信息，不做处理
                        filtered_data[ch] = data[ch]
                        
            elif self.method == 'bipolar':
                # 双极滤波 - 相邻通道间的差值
                # 注意：这会减少通道数，这里我们保持通道数不变，每个通道与后一通道的差
                filtered_data = np.zeros_like(data)
                for ch in range(channels - 1):
                    filtered_data[ch] = data[ch] - data[ch + 1]
                # 最后一个通道保持不变
                filtered_data[-1] = data[-1]
                
            logger.debug(f"应用空间滤波: {self.method}")
            return filtered_data
            
        except Exception as e:
            logger.error(f"空间滤波失败: {e}")
            return data

class FilterPipeline:
    """滤波器管道 - 将多个滤波器串联应用"""
    
    def __init__(self):
        """初始化滤波器管道"""
        self.filters = []
        self.sampling_rate = None
        logger.debug("初始化滤波器管道")
    
    def add_filter(self, filter_obj):
        """
        添加滤波器到管道
        
        参数:
            filter_obj (BaseFilter): 滤波器对象
        """
        if not isinstance(filter_obj, BaseFilter):
            logger.error(f"无效的滤波器类型: {type(filter_obj)}")
            return
            
        self.filters.append(filter_obj)
        logger.info(f"添加滤波器到管道: {filter_obj.name}")
    
    def set_sampling_rate(self, rate):
        """
        设置采样率
        
        参数:
            rate (int): 采样率(Hz)
        """
        self.sampling_rate = rate
        logger.debug(f"设置采样率: {rate}Hz")
    
    def process(self, data, sampling_rate=None):
        """
        对数据应用整个滤波器管道
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)，如果为None则使用之前设置的值
            
        返回:
            numpy.ndarray: 处理后的数据
        """
        if data is None or data.size == 0:
            logger.error("无法处理数据: 数据为空")
            return data
            
        # 使用提供的采样率或之前设置的采样率
        rate = sampling_rate if sampling_rate is not None else self.sampling_rate
        if rate is None:
            logger.error("未设置采样率")
            return data
            
        # 依次应用每个滤波器
        processed_data = data.copy()
        for filter_obj in self.filters:
            processed_data = filter_obj.apply(processed_data, rate)
            
        logger.debug(f"应用了 {len(self.filters)} 个滤波器")
        return processed_data
    
    def clear(self):
        """清空滤波器管道"""
        self.filters = []
        logger.debug("清空滤波器管道")
    
    def get_default_pipeline(self, sampling_rate):
        """
        获取默认滤波器管道
        
        参数:
            sampling_rate (int): 采样率(Hz)
            
        返回:
            FilterPipeline: 配置好的滤波器管道
        """
        self.set_sampling_rate(sampling_rate)
        
        # 添加标准EEG预处理滤波器
        # 1. 高通滤波去除基线漂移
        self.add_filter(HighpassFilter(cutoff=0.5))
        
        # 2. 低通滤波去除高频噪声
        self.add_filter(LowpassFilter(cutoff=50.0))
        
        # 3. 陷波滤波去除电源线噪声
        self.add_filter(NotchFilter(freq=50.0))  # 使用50Hz或60Hz取决于地区
        
        # 4. 空间滤波增强空间分辨率
        self.add_filter(SpatialFilter(method='car'))
        
        return self
    
    def get_raw_signal_pipeline(self, sampling_rate):
        """
        获取最小处理的滤波器管道，保留更多原始信号特征
        
        参数:
            sampling_rate (int): 采样率(Hz)
            
        返回:
            FilterPipeline: 配置好的滤波器管道
        """
        self.set_sampling_rate(sampling_rate)
        
        # 仅添加必要的预处理滤波器
        # 1. 高通滤波去除直流分量
        self.add_filter(HighpassFilter(cutoff=0.1))
        
        # 2. 陷波滤波去除电源线噪声
        self.add_filter(NotchFilter(freq=50.0))  # 使用50Hz或60Hz取决于地区
        
        return self
    
    def get_beta_focused_pipeline(self, sampling_rate):
        """
        专注于beta频段(13-30Hz)的滤波器管道
        
        参数:
            sampling_rate (int): 采样率(Hz)
            
        返回:
            FilterPipeline: 配置好的滤波器管道
        """
        self.set_sampling_rate(sampling_rate)
        
        # 添加预处理滤波器
        # 1. 带通滤波保留beta频段
        self.add_filter(BandpassFilter(low_cutoff=13.0, high_cutoff=30.0))
        
        # 2. 陷波滤波去除电源线噪声
        self.add_filter(NotchFilter(freq=50.0))  # 使用50Hz或60Hz取决于地区
        
        # 3. 空间滤波增强空间分辨率
        self.add_filter(SpatialFilter(method='car'))
        
        return self
    
    def get_alpha_focused_pipeline(self, sampling_rate):
        """
        专注于alpha频段(8-13Hz)的滤波器管道
        
        参数:
            sampling_rate (int): 采样率(Hz)
            
        返回:
            FilterPipeline: 配置好的滤波器管道
        """
        self.set_sampling_rate(sampling_rate)
        
        # 添加预处理滤波器
        # 1. 带通滤波保留alpha频段
        self.add_filter(BandpassFilter(low_cutoff=8.0, high_cutoff=13.0))
        
        # 2. 空间滤波增强空间分辨率
        self.add_filter(SpatialFilter(method='car'))
        
        return self
