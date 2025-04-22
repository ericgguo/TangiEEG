"""
数据分段模块 - 将连续数据分割为适合解码的时间窗口
"""

import numpy as np
import logging
from scipy import signal
import mne

# 创建logger
logger = logging.getLogger('data_segmentation')

class DataSegmenter:
    """数据分段基类"""
    
    def __init__(self, window_size=1.0, overlap=0.5, padding='reflect'):
        """
        初始化数据分段器
        
        参数:
            window_size (float): 窗口大小(秒)
            overlap (float): 重叠比例 (0.0-1.0)
            padding (str): 边界处理方式，'reflect', 'constant', 'nearest', 'mirror', 或 'wrap'
        """
        self.window_size = window_size
        self.overlap = max(0.0, min(0.95, overlap))  # 限制在0-0.95之间
        self.padding = padding
        
        logger.info(f"数据分段器初始化: 窗口大小={window_size}秒, 重叠={overlap}")
    
    def segment(self, data, sampling_rate):
        """
        将数据分段
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 形状为 (segments, channels, window_samples) 的分段数据
            numpy.ndarray: 每个分段的起始时间(秒)
        """
        if data is None or data.size == 0:
            logger.warning("数据为空，无法分段")
            return np.array([]), np.array([])
        
        try:
            # 计算窗口点数和步长
            window_samples = int(self.window_size * sampling_rate)
            step_samples = int(window_samples * (1 - self.overlap))
            
            # 获取数据维度
            if len(data.shape) == 1:
                # 单通道数据，添加通道维度
                data = data.reshape(1, -1)
            
            channels, total_samples = data.shape
            
            # 计算分段数量
            n_segments = max(1, (total_samples - window_samples) // step_samples + 1)
            
            # 如果数据太短，使用填充
            if total_samples < window_samples:
                logger.warning(f"数据长度({total_samples}样本)小于窗口大小({window_samples}样本)，使用填充")
                pad_width = window_samples - total_samples
                padded_data = np.pad(data, ((0, 0), (0, pad_width)), mode=self.padding)
                segments = padded_data.reshape(1, channels, window_samples)
                segment_times = np.array([0.0])
                return segments, segment_times
            
            # 创建结果数组
            segments = np.zeros((n_segments, channels, window_samples))
            segment_times = np.zeros(n_segments)
            
            # 分段
            for i in range(n_segments):
                start = i * step_samples
                end = start + window_samples
                
                # 检查是否需要填充
                if end <= total_samples:
                    segments[i] = data[:, start:end]
                else:
                    # 需要填充
                    pad_width = end - total_samples
                    temp_data = np.pad(data[:, start:], ((0, 0), (0, pad_width)), mode=self.padding)
                    segments[i] = temp_data
                
                # 计算每个分段的时间
                segment_times[i] = start / sampling_rate
            
            return segments, segment_times
            
        except Exception as e:
            logger.error(f"数据分段失败: {e}")
            return np.array([]), np.array([])
    
    def set_parameters(self, window_size=None, overlap=None, padding=None):
        """
        设置分段参数
        
        参数:
            window_size (float): 窗口大小(秒)
            overlap (float): 重叠比例 (0.0-1.0)
            padding (str): 边界处理方式
        """
        if window_size is not None:
            self.window_size = window_size
            
        if overlap is not None:
            self.overlap = max(0.0, min(0.95, overlap))
            
        if padding is not None:
            self.padding = padding
            
        logger.info(f"数据分段参数已更新: 窗口大小={self.window_size}秒, 重叠={self.overlap}, 填充方式='{self.padding}'")


class EventBasedSegmenter(DataSegmenter):
    """基于事件的数据分段器"""
    
    def __init__(self, pre_event=0.2, post_event=0.8, baseline_correction=True):
        """
        初始化基于事件的数据分段器
        
        参数:
            pre_event (float): 事件前时间(秒)
            post_event (float): 事件后时间(秒)
            baseline_correction (bool): 是否进行基线校正
        """
        super().__init__(window_size=pre_event+post_event, overlap=0.0)
        
        self.pre_event = pre_event
        self.post_event = post_event
        self.baseline_correction = baseline_correction
        
        logger.info(f"基于事件的数据分段器初始化: 事件前={pre_event}秒, 事件后={post_event}秒, 基线校正={baseline_correction}")
    
    def segment_by_events(self, data, sampling_rate, events, event_ids=None):
        """
        根据事件分段数据
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据
            sampling_rate (int): 采样率(Hz)
            events (numpy.ndarray): 形状为 (n_events, 2) 的事件数组，每行包含 [样本索引, 事件类型]
            event_ids (list): 事件ID列表，如果为None则使用所有事件
            
        返回:
            numpy.ndarray: 形状为 (n_events, channels, window_samples) 的分段数据
            numpy.ndarray: 事件类型数组
            numpy.ndarray: 事件发生时间(秒)
        """
        if data is None or data.size == 0:
            logger.warning("数据为空，无法分段")
            return np.array([]), np.array([]), np.array([])
            
        if events is None or events.size == 0:
            logger.warning("事件列表为空，无法分段")
            return np.array([]), np.array([]), np.array([])
        
        try:
            # 检查事件格式
            if len(events.shape) == 1:
                # 只有样本索引，假设所有事件类型相同
                events = np.column_stack((events, np.ones(len(events))))
            
            # 过滤事件类型
            if event_ids is not None:
                mask = np.isin(events[:, 1], event_ids)
                filtered_events = events[mask]
                if len(filtered_events) == 0:
                    logger.warning(f"没有找到指定的事件类型: {event_ids}")
                    return np.array([]), np.array([]), np.array([])
                events = filtered_events
            
            # 计算窗口点数
            pre_samples = int(self.pre_event * sampling_rate)
            post_samples = int(self.post_event * sampling_rate)
            window_samples = pre_samples + post_samples
            
            # 获取数据维度
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            channels, total_samples = data.shape
            
            # 过滤超出范围的事件
            valid_indices = []
            for i, (sample_idx, _) in enumerate(events):
                if 0 <= sample_idx < total_samples and sample_idx - pre_samples >= 0 and sample_idx + post_samples <= total_samples:
                    valid_indices.append(i)
            
            if not valid_indices:
                logger.warning("所有事件都超出数据范围")
                return np.array([]), np.array([]), np.array([])
            
            valid_events = events[valid_indices]
            n_events = len(valid_events)
            
            # 创建结果数组
            segments = np.zeros((n_events, channels, window_samples))
            event_types = np.zeros(n_events)
            event_times = np.zeros(n_events)
            
            # 分段
            for i, (sample_idx, event_type) in enumerate(valid_events):
                start = int(sample_idx - pre_samples)
                end = int(sample_idx + post_samples)
                
                # 提取分段
                segment = data[:, start:end]
                
                # 基线校正
                if self.baseline_correction:
                    # 使用事件前的数据作为基线
                    baseline = np.mean(segment[:, :pre_samples], axis=1, keepdims=True)
                    segment = segment - baseline
                
                segments[i] = segment
                event_types[i] = event_type
                event_times[i] = sample_idx / sampling_rate
            
            return segments, event_types, event_times
            
        except Exception as e:
            logger.error(f"基于事件的分段失败: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def set_parameters(self, pre_event=None, post_event=None, baseline_correction=None):
        """
        设置分段参数
        
        参数:
            pre_event (float): 事件前时间(秒)
            post_event (float): 事件后时间(秒)
            baseline_correction (bool): 是否进行基线校正
        """
        if pre_event is not None:
            self.pre_event = pre_event
            
        if post_event is not None:
            self.post_event = post_event
            
        if baseline_correction is not None:
            self.baseline_correction = baseline_correction
        
        # 更新窗口大小
        self.window_size = self.pre_event + self.post_event
        
        logger.info(f"基于事件的分段参数已更新: 事件前={self.pre_event}秒, 事件后={self.post_event}秒, 基线校正={self.baseline_correction}")


class AdaptiveSegmenter(DataSegmenter):
    """自适应数据分段器 - 根据信号特性动态调整分段"""
    
    def __init__(self, min_window_size=0.5, max_window_size=2.0, 
                 threshold=0.7, step_size=0.1, metric='variance'):
        """
        初始化自适应数据分段器
        
        参数:
            min_window_size (float): 最小窗口大小(秒)
            max_window_size (float): 最大窗口大小(秒)
            threshold (float): 变化检测阈值 (0.0-1.0)
            step_size (float): 窗口增长步长(秒)
            metric (str): 用于检测变化的指标，'variance', 'energy', 'zero_crossings'
        """
        super().__init__(window_size=min_window_size, overlap=0.5)
        
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.threshold = threshold
        self.step_size = step_size
        self.metric = metric
        
        logger.info(f"自适应数据分段器初始化: 窗口范围={min_window_size}-{max_window_size}秒, 阈值={threshold}, 指标='{metric}'")
    
    def segment_adaptive(self, data, sampling_rate):
        """
        自适应地将数据分段
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 形状为 (segments, channels, max_window_samples) 的分段数据
            numpy.ndarray: 每个分段的实际窗口大小(秒)
            numpy.ndarray: 每个分段的起始时间(秒)
        """
        if data is None or data.size == 0:
            logger.warning("数据为空，无法分段")
            return np.array([]), np.array([]), np.array([])
        
        try:
            # 获取数据维度
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            
            channels, total_samples = data.shape
            
            # 计算最小和最大窗口点数
            min_samples = int(self.min_window_size * sampling_rate)
            max_samples = int(self.max_window_size * sampling_rate)
            step_samples = int(self.step_size * sampling_rate)
            
            # 初始化结果列表
            segments = []
            window_sizes = []
            segment_times = []
            
            # 遍历数据
            start_idx = 0
            while start_idx < total_samples - min_samples:
                # 初始窗口大小为最小值
                current_size = min_samples
                end_idx = start_idx + current_size
                
                # 如果剩余的样本数不足最小窗口大小，结束处理
                if end_idx > total_samples:
                    break
                
                # 获取初始分段
                current_segment = data[:, start_idx:end_idx]
                
                # 计算初始指标值
                last_metric_value = self._compute_metric(current_segment)
                
                # 动态增长窗口
                growing = True
                while growing and current_size < max_samples and end_idx + step_samples <= total_samples:
                    # 增加窗口大小
                    next_size = current_size + step_samples
                    next_end = start_idx + next_size
                    
                    # 获取新分段
                    next_segment = data[:, start_idx:next_end]
                    
                    # 计算新指标值
                    new_metric_value = self._compute_metric(next_segment)
                    
                    # 检查指标值的变化
                    change_ratio = abs(new_metric_value - last_metric_value) / (last_metric_value + 1e-10)
                    
                    if change_ratio > self.threshold:
                        # 变化超过阈值，停止增长
                        growing = False
                    else:
                        # 更新当前大小和指标值
                        current_size = next_size
                        end_idx = next_end
                        last_metric_value = new_metric_value
                
                # 添加找到的分段
                segment = data[:, start_idx:end_idx]
                
                # 填充到最大长度
                if current_size < max_samples:
                    padded_segment = np.zeros((channels, max_samples))
                    padded_segment[:, :current_size] = segment
                    segment = padded_segment
                
                segments.append(segment)
                window_sizes.append(current_size / sampling_rate)
                segment_times.append(start_idx / sampling_rate)
                
                # 更新开始索引 (使用重叠)
                step = int(current_size * (1 - self.overlap))
                start_idx += max(1, step)  # 确保至少前进1个样本
            
            if not segments:
                logger.warning("未能创建任何分段")
                return np.array([]), np.array([]), np.array([])
            
            # 转换为数组
            segments_array = np.array(segments)
            window_sizes_array = np.array(window_sizes)
            segment_times_array = np.array(segment_times)
            
            return segments_array, window_sizes_array, segment_times_array
            
        except Exception as e:
            logger.error(f"自适应分段失败: {e}")
            return np.array([]), np.array([]), np.array([])
    
    def _compute_metric(self, data):
        """
        计算分段的特征指标
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据分段
            
        返回:
            float: 指标值
        """
        # 对所有通道求平均
        avg_data = np.mean(data, axis=0)
        
        if self.metric == 'variance':
            # 使用方差作为指标
            return np.var(avg_data)
            
        elif self.metric == 'energy':
            # 使用能量作为指标
            return np.sum(avg_data**2) / len(avg_data)
            
        elif self.metric == 'zero_crossings':
            # 使用过零率作为指标
            return np.sum(np.diff(np.signbit(avg_data))) / len(avg_data)
            
        else:
            # 默认使用方差
            return np.var(avg_data)
    
    def set_parameters(self, min_window_size=None, max_window_size=None, 
                      threshold=None, step_size=None, metric=None):
        """
        设置分段参数
        
        参数:
            min_window_size (float): 最小窗口大小(秒)
            max_window_size (float): 最大窗口大小(秒)
            threshold (float): 变化检测阈值 (0.0-1.0)
            step_size (float): 窗口增长步长(秒)
            metric (str): 用于检测变化的指标
        """
        if min_window_size is not None:
            self.min_window_size = min_window_size
            self.window_size = min_window_size
            
        if max_window_size is not None:
            self.max_window_size = max_window_size
            
        if threshold is not None:
            self.threshold = threshold
            
        if step_size is not None:
            self.step_size = step_size
            
        if metric is not None:
            self.metric = metric
        
        logger.info(f"自适应分段参数已更新: 窗口范围={self.min_window_size}-{self.max_window_size}秒, 阈值={self.threshold}, 指标='{self.metric}'")


def create_epochs_from_data(data, sampling_rate, event_times=None, event_ids=None, 
                           tmin=-0.2, tmax=0.8, baseline=(None, 0)):
    """
    从原始数据创建MNE Epochs对象
    
    参数:
        data (numpy.ndarray): 形状为 (channels, samples) 的数据
        sampling_rate (int): 采样率(Hz)
        event_times (list/array): 事件时间点列表(秒)，如果为None则尝试自动检测
        event_ids (dict): 事件ID字典，格式为{名称: ID}
        tmin (float): 事件前时间(秒)
        tmax (float): 事件后时间(秒)
        baseline (tuple): 基线校正范围，格式为(开始时间, 结束时间)，单位为秒
        
    返回:
        mne.Epochs: MNE Epochs对象
    """
    try:
        # 检查数据
        if data is None or data.size == 0:
            logger.warning("数据为空，无法创建Epochs")
            return None
        
        # 创建MNE Info对象
        n_channels = data.shape[0] if len(data.shape) > 1 else 1
        
        # 创建通道名称
        ch_names = [f'CH{i+1}' for i in range(n_channels)]
        
        # 所有通道类型设为EEG
        ch_types = ['eeg'] * n_channels
        
        # 创建Info对象
        info = mne.create_info(ch_names=ch_names, sfreq=sampling_rate, ch_types=ch_types)
        
        # 创建原始数据对象
        raw = mne.io.RawArray(data, info)
        
        # 创建事件数组
        if event_times is not None:
            # 将时间点转换为样本索引
            event_samples = (np.array(event_times) * sampling_rate).astype(int)
            
            # 创建事件数组，默认事件ID为1
            default_id = 1
            if event_ids is None:
                events_array = np.column_stack((event_samples, 
                                             np.zeros(len(event_samples), dtype=int),
                                             np.ones(len(event_samples), dtype=int) * default_id))
            else:
                # 如果提供了事件ID映射，使用第一个ID作为默认值
                default_id = list(event_ids.values())[0] if event_ids else 1
                events_array = np.column_stack((event_samples, 
                                             np.zeros(len(event_samples), dtype=int),
                                             np.ones(len(event_samples), dtype=int) * default_id))
        else:
            # 尝试自动检测事件
            logger.info("未提供事件时间点，尝试自动检测事件")
            
            # 计算所有通道的平均值，用于事件检测
            avg_data = np.mean(data, axis=0) if len(data.shape) > 1 else data
            
            # 使用简单阈值方法检测事件
            # 这里假设事件对应信号的局部最大值
            threshold = np.mean(avg_data) + 2 * np.std(avg_data)  # 均值+2倍标准差
            
            # 寻找超过阈值的点
            above_threshold = avg_data > threshold
            
            # 找到阈值交叉点 (上升沿)
            crosses = np.where(np.diff(above_threshold.astype(int)) > 0)[0] + 1
            
            if len(crosses) == 0:
                logger.warning("未能自动检测到任何事件")
                # 使用均匀分布的事件作为备选
                crosses = np.linspace(0, len(avg_data) - 1, 10).astype(int)
            
            # 创建事件数组
            default_id = list(event_ids.values())[0] if event_ids else 1
            events_array = np.column_stack((crosses, 
                                         np.zeros(len(crosses), dtype=int),
                                         np.ones(len(crosses), dtype=int) * default_id))
        
        # 创建Epochs对象
        epochs = mne.Epochs(raw, events_array, event_id=event_ids, tmin=tmin, tmax=tmax,
                         baseline=baseline, preload=True)
        
        return epochs
        
    except Exception as e:
        logger.error(f"创建MNE Epochs对象失败: {e}")
        return None
