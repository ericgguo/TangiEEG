"""
数据流处理模块 - 处理实时数据流，包括数据缓存、时间戳同步和数据格式转换
包含模拟数据生成功能，用于测试和开发
"""

import time
import logging
import numpy as np
from scipy import signal

# 创建logger
logger = logging.getLogger('data_streamer')

class DataStreamer:
    """数据流处理基类"""
    
    def __init__(self, channels=16, sample_rate=250):
        """
        初始化数据流处理器
        
        参数:
            channels (int): 通道数量
            sample_rate (int): 采样率(Hz)
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.buffer = []
        self.timestamp_offset = time.time()
        
        logger.info(f"数据流处理器初始化: 通道数={channels}, 采样率={sample_rate}Hz")
    
    def add_data(self, data, timestamp=None):
        """
        添加数据到缓冲区
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据
            timestamp (float): 数据采集时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.buffer.append((data, timestamp))
        
        # 保持缓冲区大小合理 (最多保存10秒的数据)
        max_buffer_size = 10 * self.sample_rate
        if len(self.buffer) > max_buffer_size:
            self.buffer = self.buffer[-max_buffer_size:]
    
    def get_data(self, duration=1.0, end_time=None):
        """
        获取指定时间段的数据
        
        参数:
            duration (float): 时间段长度(秒)
            end_time (float): 时间段结束时间，如果为None则使用最新数据时间戳
            
        返回:
            tuple: (数据数组, 开始时间戳, 结束时间戳)
        """
        if not self.buffer:
            logger.warning("缓冲区为空，无法获取数据")
            return None, None, None
        
        if end_time is None:
            end_time = self.buffer[-1][1]
        
        start_time = end_time - duration
        
        # 查找时间范围内的数据
        data_points = []
        for data, timestamp in self.buffer:
            if start_time <= timestamp <= end_time:
                data_points.append((data, timestamp))
        
        if not data_points:
            logger.warning(f"在指定时间范围内未找到数据: {start_time:.3f} - {end_time:.3f}")
            return None, None, None
        
        # 合并数据
        all_data = np.hstack([data for data, _ in data_points])
        actual_start_time = data_points[0][1]
        actual_end_time = data_points[-1][1]
        
        return all_data, actual_start_time, actual_end_time
    
    def clear_buffer(self):
        """清空缓冲区"""
        self.buffer = []
        logger.info("缓冲区已清空")


class SimulatedDataStreamer:
    """模拟EEG数据生成器"""
    
    def __init__(self, channels=16, sample_rate=250, noise_level=0.1, artifact_prob=0.05):
        """
        初始化模拟数据生成器
        
        参数:
            channels (int): 通道数量
            sample_rate (int): 采样率(Hz)
            noise_level (float): 噪声水平 (0.0-1.0)
            artifact_prob (float): 伪迹发生概率 (0.0-1.0)
        """
        self.channels = channels
        self.sample_rate = sample_rate
        self.noise_level = noise_level
        self.artifact_prob = artifact_prob
        
        # 频率配置 (Hz)
        self.freq_bands = {
            'delta': (0.5, 4),    # 深度睡眠
            'theta': (4, 8),      # 睡眠、冥想
            'alpha': (8, 13),     # 放松、闭眼
            'beta': (13, 30),     # 活跃思考、专注
            'gamma': (30, 100)    # 认知处理、学习
        }
        
        # 为每个通道设置随机相位差，使生成的数据更自然
        self.channel_phases = np.random.uniform(0, 2*np.pi, size=channels)
        
        # 设置频段振幅 - 默认以alpha为主导，模拟放松状态
        self.band_amplitudes = {
            'delta': 2.0,
            'theta': 1.0,
            'alpha': 5.0,  # alpha波振幅大，模拟闭眼放松
            'beta': 1.5,
            'gamma': 0.5
        }
        
        # 伪迹类型及其生成函数的映射
        self.artifact_generators = {
            'blink': self._generate_blink,
            'muscle': self._generate_muscle_artifact,
            'movement': self._generate_movement_artifact,
            'electrode_pop': self._generate_electrode_pop
        }
        
        # 伪迹概率分布 - 眨眼最常见
        self.artifact_weights = {
            'blink': 0.6,          # 眨眼最常见
            'muscle': 0.2,         # 肌肉伪迹
            'movement': 0.15,      # 移动伪迹
            'electrode_pop': 0.05  # 电极弹出（少见）
        }
        
        # 默认在前额叶区域的通道索引 (Fp1, Fp2, F7, F3, Fz, F4, F8)
        # 假设通道顺序遵循10-20系统
        self.frontal_channels = [0, 1, 2, 3, 4, 5, 6] if channels >= 8 else [0, 1]
        
        # 右脑区域的通道索引
        self.right_channels = [1, 5, 6, 10, 13, 15] if channels >= 16 else [1, 3] if channels >= 4 else [1]
        
        # 左脑区域的通道索引
        self.left_channels = [0, 2, 3, 8, 12, 14] if channels >= 16 else [0, 2] if channels >= 4 else [0]
        
        # 随时间变化的状态
        self.time_elapsed = 0
        self.current_state = 'relaxed'  # 初始状态：放松
        self.state_duration = np.random.uniform(5, 15)  # 当前状态持续5-15秒
        
        logger.info(f"模拟数据生成器初始化: 通道数={channels}, 采样率={sample_rate}Hz, 噪声水平={noise_level}, 伪迹概率={artifact_prob}")
    
    def generate_chunk(self, samples):
        """
        生成一段模拟的EEG数据
        
        参数:
            samples (int): 样本数量
            
        返回:
            numpy.ndarray: 形状为 (channels, samples) 的数据数组
        """
        # 创建时间向量
        t = np.linspace(self.time_elapsed, 
                         self.time_elapsed + samples/self.sample_rate, 
                         samples, endpoint=False)
        self.time_elapsed += samples/self.sample_rate
        
        # 更新状态
        if self.time_elapsed > self.state_duration:
            self._update_state()
            self.time_elapsed = 0
            self.state_duration = np.random.uniform(5, 15)  # 新状态持续5-15秒
        
        # 初始化数据数组
        data = np.zeros((self.channels, samples))
        
        # 生成底层脑电节律
        for band, (low_freq, high_freq) in self.freq_bands.items():
            # 在频段内选择几个代表频率
            n_freqs = 3
            freqs = np.random.uniform(low_freq, high_freq, n_freqs)
            
            for freq in freqs:
                # 根据当前状态调整振幅
                amp_scale = self._get_state_amplitude_scale(band)
                amplitude = self.band_amplitudes[band] * amp_scale
                
                # 为每个通道生成基于该频率的信号，加入随机相位差
                for ch in range(self.channels):
                    # 根据通道位置调整振幅（前额、左右半球）
                    ch_amp = amplitude * self._get_channel_amplitude_scale(ch, band)
                    
                    # 生成信号
                    phase = self.channel_phases[ch]
                    signal_component = ch_amp * np.sin(2 * np.pi * freq * t + phase)
                    data[ch, :] += signal_component
        
        # 添加背景噪声
        noise = np.random.normal(0, self.noise_level, size=data.shape)
        data += noise
        
        # 添加随机伪迹
        data = self._add_artifacts(data, t)
        
        return data
    
    def set_cognitive_state(self, state, duration=None):
        """
        设置认知状态以模拟不同的脑电活动
        
        参数:
            state (str): 认知状态 ('relaxed', 'focused', 'drowsy', 'imagining_motor', 'imagining_speech')
            duration (float): 持续时间(秒)，如果为None则随机生成
        """
        self.current_state = state
        if duration is not None:
            self.state_duration = duration
        else:
            self.state_duration = np.random.uniform(5, 15)
        
        logger.info(f"认知状态已更改为: {state}, 持续时间: {self.state_duration:.1f}秒")
    
    def set_artifact_level(self, prob):
        """
        设置伪迹水平
        
        参数:
            prob (float): 伪迹发生概率 (0.0-1.0)
        """
        self.artifact_prob = max(0.0, min(1.0, prob))
        logger.info(f"伪迹水平已设置为: {self.artifact_prob:.2f}")
    
    def set_noise_level(self, level):
        """
        设置噪声水平
        
        参数:
            level (float): 噪声水平 (0.0-1.0)
        """
        self.noise_level = max(0.0, min(1.0, level))
        logger.info(f"噪声水平已设置为: {self.noise_level:.2f}")
    
    def _update_state(self):
        """随机更新认知状态"""
        states = ['relaxed', 'focused', 'drowsy', 'imagining_motor', 'imagining_speech']
        # 倾向于保持当前状态，有20%的概率切换
        if np.random.random() < 0.2:
            new_state = np.random.choice([s for s in states if s != self.current_state])
            self.current_state = new_state
            logger.debug(f"认知状态自动更改为: {new_state}")
    
    def _get_state_amplitude_scale(self, band):
        """根据当前认知状态获取频段振幅缩放因子"""
        # 不同认知状态下脑电节律的强度不同
        state_scales = {
            'relaxed': {
                'delta': 0.5, 'theta': 0.7, 'alpha': 2.0, 'beta': 0.5, 'gamma': 0.3
            },
            'focused': {
                'delta': 0.3, 'theta': 0.5, 'alpha': 0.7, 'beta': 2.0, 'gamma': 1.5
            },
            'drowsy': {
                'delta': 2.0, 'theta': 1.5, 'alpha': 0.5, 'beta': 0.3, 'gamma': 0.2
            },
            'imagining_motor': {
                'delta': 0.5, 'theta': 0.8, 'alpha': 0.6, 'beta': 1.8, 'gamma': 1.2
            },
            'imagining_speech': {
                'delta': 0.5, 'theta': 0.9, 'alpha': 0.7, 'beta': 1.5, 'gamma': 1.8
            }
        }
        
        return state_scales[self.current_state].get(band, 1.0)
    
    def _get_channel_amplitude_scale(self, channel_idx, band):
        """根据通道位置和频段获取振幅缩放因子"""
        scale = 1.0
        
        # 在不同的认知状态下，不同脑区的活动不同
        if self.current_state == 'relaxed':
            # 放松状态下，后枕区alpha波增强
            if channel_idx in [14, 15, 16] and band == 'alpha':  # 假设这些是后枕区通道
                scale *= 1.5
                
        elif self.current_state == 'focused':
            # 专注状态下，前额叶beta波增强
            if channel_idx in self.frontal_channels and band == 'beta':
                scale *= 1.5
                
        elif self.current_state == 'drowsy':
            # 昏昏欲睡状态下，全脑delta和theta波增强
            if band in ['delta', 'theta']:
                scale *= 1.3
                
        elif self.current_state == 'imagining_motor':
            # 运动想象状态下，运动皮层区域mu节律（alpha频段）抑制
            motor_channels = [8, 9, 10] if self.channels >= 16 else [2, 3] if self.channels >= 4 else [0, 1]
            if channel_idx in motor_channels and band == 'alpha':
                scale *= 0.5
                
        elif self.current_state == 'imagining_speech':
            # 语言想象状态下，语言相关区域的beta和gamma活动增强
            speech_channels = [2, 3, 8, 12] if self.channels >= 16 else [0, 2] if self.channels >= 4 else [0]
            if channel_idx in speech_channels and band in ['beta', 'gamma']:
                scale *= 1.4
        
        return scale
    
    def _add_artifacts(self, data, time_vector):
        """
        向数据添加随机伪迹
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            time_vector (numpy.ndarray): 时间向量
            
        返回:
            numpy.ndarray: 添加了伪迹的数据数组
        """
        # 每100ms检查一次是否添加伪迹
        samples = data.shape[1]
        check_interval = int(0.1 * self.sample_rate)  # 100ms
        
        for i in range(0, samples, check_interval):
            # 判断是否添加伪迹
            if np.random.random() < self.artifact_prob:
                # 选择伪迹类型
                artifact_type = np.random.choice(
                    list(self.artifact_generators.keys()),
                    p=list(self.artifact_weights.values())
                )
                
                # 确定伪迹的窗口大小（不超过剩余样本数）
                max_window = min(samples - i, int(1.0 * self.sample_rate))  # 最大1秒
                window_size = np.random.randint(int(0.1 * self.sample_rate), max_window)
                
                # 生成并添加伪迹
                artifact_data = self.artifact_generators[artifact_type](window_size)
                
                # 将伪迹添加到相应通道的相应时间点
                for ch in range(self.channels):
                    if artifact_data[ch] is not None:
                        end_idx = min(i + window_size, samples)
                        actual_window = end_idx - i
                        data[ch, i:end_idx] += artifact_data[ch][:actual_window]
        
        return data
    
    def _generate_blink(self, window_size):
        """
        生成眨眼伪迹
        
        参数:
            window_size (int): 时间窗口大小（样本数）
            
        返回:
            list: 每个通道的伪迹数据，如果该通道没有伪迹则为None
        """
        # 眨眼主要影响前额叶通道
        artifact_data = [None] * self.channels
        
        # 创建高斯形状的眨眼
        t = np.linspace(-3, 3, window_size)
        blink_shape = 50 * np.exp(-t**2)  # 振幅为50uV的高斯形状
        
        # 仅在前额叶通道添加眨眼伪迹，振幅逐渐降低
        for idx, ch in enumerate(self.frontal_channels):
            if idx < len(self.frontal_channels):
                # 振幅随着远离前额逐渐降低
                scale = 1.0 - idx * 0.15
                if scale > 0:
                    artifact_data[ch] = blink_shape * scale
        
        return artifact_data
    
    def _generate_muscle_artifact(self, window_size):
        """
        生成肌肉伪迹（高频噪声突发）
        
        参数:
            window_size (int): 时间窗口大小（样本数）
            
        返回:
            list: 每个通道的伪迹数据，如果该通道没有伪迹则为None
        """
        artifact_data = [None] * self.channels
        
        # 确定受影响的通道（随机选择1-3个通道）
        n_affected = np.random.randint(1, 4)
        affected_channels = np.random.choice(range(self.channels), size=n_affected, replace=False)
        
        # 生成高频噪声
        noise = np.random.normal(0, 15, size=window_size)  # 较大振幅的高斯噪声
        
        # 对噪声进行高通滤波，保留高频成分
        b, a = signal.butter(4, 30/(self.sample_rate/2), 'high')
        high_freq_noise = signal.filtfilt(b, a, noise)
        
        # 用窗函数包络噪声，使开始和结束平滑
        window = signal.windows.hann(window_size)
        windowed_noise = high_freq_noise * window
        
        # 添加到受影响的通道
        for ch in affected_channels:
            # 随机调整每个通道的振幅
            scale = np.random.uniform(0.7, 1.3)
            artifact_data[ch] = windowed_noise * scale
        
        return artifact_data
    
    def _generate_movement_artifact(self, window_size):
        """
        生成运动伪迹（低频漂移）
        
        参数:
            window_size (int): 时间窗口大小（样本数）
            
        返回:
            list: 每个通道的伪迹数据，如果该通道没有伪迹则为None
        """
        artifact_data = [None] * self.channels
        
        # 生成低频漂移的柔和曲线
        t = np.linspace(0, 2*np.pi, window_size)
        drift = 30 * np.sin(t/2) + 10 * np.sin(t/6)  # 混合的低频正弦波
        
        # 所有通道都受影响，但程度不同
        for ch in range(self.channels):
            # 随机调整每个通道的漂移幅度和相位
            amplitude = np.random.uniform(0.5, 1.5)
            phase = np.random.uniform(0, 2*np.pi)
            shifted_drift = amplitude * drift * np.sin(t + phase)
            
            artifact_data[ch] = shifted_drift
        
        return artifact_data
    
    def _generate_electrode_pop(self, window_size):
        """
        生成电极弹出伪迹（突然的大幅值变化）
        
        参数:
            window_size (int): 时间窗口大小（样本数）
            
        返回:
            list: 每个通道的伪迹数据，如果该通道没有伪迹则为None
        """
        artifact_data = [None] * self.channels
        
        # 随机选择一个通道发生电极弹出
        affected_channel = np.random.randint(0, self.channels)
        
        # 创建突然的大幅跳变
        pop = np.zeros(window_size)
        pop_start = np.random.randint(0, window_size // 3)
        pop_end = np.random.randint(window_size // 3 * 2, window_size)
        
        # 突然跳变到一个大值，然后保持
        pop[pop_start:pop_end] = np.random.uniform(50, 100) * np.random.choice([-1, 1])
        
        # 添加到受影响的通道
        artifact_data[affected_channel] = pop
        
        return artifact_data
