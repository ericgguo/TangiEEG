"""
特征提取模块 - 提取时域、频域和时频域特征
"""

import logging
import numpy as np
from scipy import signal, stats
import mne
from sklearn.decomposition import PCA

# 创建logger
logger = logging.getLogger('feature_extraction')

class FeatureExtractor:
    """特征提取基类"""
    
    def __init__(self, name="BaseExtractor"):
        """初始化特征提取器"""
        self.name = name
        logger.debug(f"初始化特征提取器: {name}")
    
    def extract(self, data, sampling_rate):
        """
        从数据中提取特征
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 特征数组
        """
        # 基类不提取任何特征，返回空数组
        return np.array([])
    
    def get_feature_names(self):
        """
        获取特征名称列表
        
        返回:
            list: 特征名称列表
        """
        # 基类返回空列表
        return []

class TimeDomainFeatures(FeatureExtractor):
    """时域特征提取器 - 提取信号在时域的统计特性"""
    
    def __init__(self, features=None, window_size=None, overlap=0.5):
        """
        初始化时域特征提取器
        
        参数:
            features (list): 要提取的特征列表，如果为None则提取所有特征
            window_size (float): 窗口大小(秒)，如果为None则使用整个信号
            overlap (float): 窗口重叠比例(0.0-1.0)
        """
        super().__init__(name="TimeDomainFeatures")
        
        # 可用特征列表
        self.available_features = [
            'mean', 'variance', 'std', 'max', 'min', 'peak_to_peak',
            'rms', 'median', 'skewness', 'kurtosis', 'zero_crossings',
            'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'
        ]
        
        # 设置要提取的特征
        if features is None:
            self.features = self.available_features.copy()
        else:
            self.features = [f for f in features if f in self.available_features]
            if not self.features:
                logger.warning("未指定有效特征，将使用所有可用特征")
                self.features = self.available_features.copy()
        
        self.window_size = window_size
        self.overlap = max(0.0, min(1.0, overlap))
        
        logger.debug(f"初始化时域特征提取器: {len(self.features)}个特征")
    
    def extract(self, data, sampling_rate):
        """提取时域特征"""
        if data is None or data.size == 0:
            logger.error("无法提取特征: 数据为空")
            return np.array([])
        
        try:
            channels, samples = data.shape
            
            # 如果未指定窗口大小，使用整个信号
            if self.window_size is None:
                # 对整个信号提取特征
                features = self._extract_from_segment(data)
                
                # 将所有特征值平铺成一维数组
                return features.flatten()
            else:
                # 将信号分段，并对每段提取特征
                window_samples = int(self.window_size * sampling_rate)
                
                # 确保窗口大小不超过样本数
                if window_samples >= samples:
                    features = self._extract_from_segment(data)
                    return features.flatten()
                
                # 计算步长
                step_samples = int(window_samples * (1 - self.overlap))
                
                # 创建分段
                segments = []
                for start in range(0, samples - window_samples + 1, step_samples):
                    end = start + window_samples
                    segment = data[:, start:end]
                    segments.append(segment)
                
                # 对每个分段提取特征
                segment_features = []
                for segment in segments:
                    features = self._extract_from_segment(segment)
                    segment_features.append(features)
                
                # 计算所有分段特征的平均值
                if segment_features:
                    avg_features = np.mean(segment_features, axis=0)
                    return avg_features.flatten()
                else:
                    return np.array([])
        
        except Exception as e:
            logger.error(f"特征提取失败: {e}")
            return np.array([])
    
    def _extract_from_segment(self, segment):
        """
        从单个数据段中提取特征
        
        参数:
            segment (numpy.ndarray): 形状为 (channels, samples) 的数据段
            
        返回:
            numpy.ndarray: 形状为 (channels, n_features) 的特征数组
        """
        channels, samples = segment.shape
        
        # 创建特征数组
        feature_array = np.zeros((channels, len(self.features)))
        
        for ch in range(channels):
            # 获取当前通道数据
            ch_data = segment[ch]
            
            # 提取特征
            feature_idx = 0
            
            for feature in self.features:
                if feature == 'mean':
                    feature_array[ch, feature_idx] = np.mean(ch_data)
                    
                elif feature == 'variance':
                    feature_array[ch, feature_idx] = np.var(ch_data)
                    
                elif feature == 'std':
                    feature_array[ch, feature_idx] = np.std(ch_data)
                    
                elif feature == 'max':
                    feature_array[ch, feature_idx] = np.max(ch_data)
                    
                elif feature == 'min':
                    feature_array[ch, feature_idx] = np.min(ch_data)
                    
                elif feature == 'peak_to_peak':
                    feature_array[ch, feature_idx] = np.max(ch_data) - np.min(ch_data)
                    
                elif feature == 'rms':
                    # 均方根值
                    feature_array[ch, feature_idx] = np.sqrt(np.mean(np.square(ch_data)))
                    
                elif feature == 'median':
                    feature_array[ch, feature_idx] = np.median(ch_data)
                    
                elif feature == 'skewness':
                    # 偏度
                    feature_array[ch, feature_idx] = stats.skew(ch_data)
                    
                elif feature == 'kurtosis':
                    # 峭度
                    feature_array[ch, feature_idx] = stats.kurtosis(ch_data)
                    
                elif feature == 'zero_crossings':
                    # 过零点数量
                    zero_crossings = np.where(np.diff(np.signbit(ch_data)))[0]
                    feature_array[ch, feature_idx] = len(zero_crossings) / samples
                    
                elif feature == 'hjorth_activity':
                    # Hjorth活动度 - 信号方差
                    feature_array[ch, feature_idx] = np.var(ch_data)
                    
                elif feature == 'hjorth_mobility':
                    # Hjorth移动性 - 一阶差分信号的标准差与原信号标准差的比值
                    diff1 = np.diff(ch_data)
                    var_diff1 = np.var(diff1)
                    var_ch = np.var(ch_data)
                    if var_ch > 0:
                        feature_array[ch, feature_idx] = np.sqrt(var_diff1 / var_ch)
                    else:
                        feature_array[ch, feature_idx] = 0
                    
                elif feature == 'hjorth_complexity':
                    # Hjorth复杂度 - 二阶差分信号的移动性与一阶差分信号移动性的比值
                    diff1 = np.diff(ch_data)
                    diff2 = np.diff(diff1)
                    var_diff1 = np.var(diff1)
                    var_diff2 = np.var(diff2)
                    if var_diff1 > 0:
                        feature_array[ch, feature_idx] = np.sqrt(var_diff2 / var_diff1)
                    else:
                        feature_array[ch, feature_idx] = 0
                
                feature_idx += 1
        
        return feature_array
    
    def get_feature_names(self):
        """获取特征名称列表"""
        feature_names = []
        
        # 假设有N个通道
        for ch_idx in range(1):  # 只创建一个通道的名称，提取时会自动为每个通道重复
            for feature in self.features:
                feature_names.append(f"{feature}_ch{ch_idx}")
        
        return feature_names

class FrequencyDomainFeatures(FeatureExtractor):
    """频域特征提取器 - 提取信号的频域特性"""
    
    def __init__(self, features=None, window_size=None, overlap=0.5, nperseg=256, 
                 bands=None, relative_power=True):
        """
        初始化频域特征提取器
        
        参数:
            features (list): 要提取的特征列表，如果为None则提取所有特征
            window_size (float): 窗口大小(秒)，如果为None则使用整个信号
            overlap (float): 窗口重叠比例(0.0-1.0)
            nperseg (int): FFT分段长度
            bands (dict): 频带定义字典，格式为{名称: (最低频率, 最高频率)}
            relative_power (bool): 是否计算相对功率（相对于总功率的比例）
        """
        super().__init__(name="FrequencyDomainFeatures")
        
        # 默认脑电频带
        self.default_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # 设置频带
        self.bands = bands if bands is not None else self.default_bands
        
        # 可用特征列表
        self.available_features = [
            'band_power',          # 各频带功率
            'spectral_edge',       # 频谱边缘频率
            'spectral_entropy',    # 频谱熵
            'spectral_centroid',   # 频谱质心
            'spectral_variance',   # 频谱方差
            'spectral_skewness',   # 频谱偏度
            'spectral_kurtosis',   # 频谱峭度
            'dominant_frequency',  # 主频
            'median_frequency',    # 中值频率
            'peak_frequency',      # 峰值频率
            'band_ratios'          # 频带比例(如alpha/beta)
        ]
        
        # 设置要提取的特征
        if features is None:
            self.features = self.available_features.copy()
        else:
            self.features = [f for f in features if f in self.available_features]
            if not self.features:
                logger.warning("未指定有效特征，将使用所有可用特征")
                self.features = self.available_features.copy()
        
        self.window_size = window_size
        self.overlap = max(0.0, min(1.0, overlap))
        self.nperseg = nperseg
        self.relative_power = relative_power
        
        logger.debug(f"初始化频域特征提取器: {len(self.features)}个特征")
    
    def extract(self, data, sampling_rate):
        """提取频域特征"""
        if data is None or data.size == 0:
            logger.error("无法提取特征: 数据为空")
            return np.array([])
        
        try:
            channels, samples = data.shape
            
            # 如果未指定窗口大小，使用整个信号
            if self.window_size is None:
                # 对整个信号提取特征
                features = self._extract_from_segment(data, sampling_rate)
                
                # 将所有特征值平铺成一维数组
                return features.flatten()
            else:
                # 将信号分段，并对每段提取特征
                window_samples = int(self.window_size * sampling_rate)
                
                # 确保窗口大小不超过样本数
                if window_samples >= samples:
                    features = self._extract_from_segment(data, sampling_rate)
                    return features.flatten()
                
                # 计算步长
                step_samples = int(window_samples * (1 - self.overlap))
                
                # 创建分段
                segments = []
                for start in range(0, samples - window_samples + 1, step_samples):
                    end = start + window_samples
                    segment = data[:, start:end]
                    segments.append(segment)
                
                # 对每个分段提取特征
                segment_features = []
                for segment in segments:
                    features = self._extract_from_segment(segment, sampling_rate)
                    segment_features.append(features)
                
                # 计算所有分段特征的平均值
                if segment_features:
                    avg_features = np.mean(segment_features, axis=0)
                    return avg_features.flatten()
                else:
                    return np.array([])
        
        except Exception as e:
            logger.error(f"频域特征提取失败: {e}")
            return np.array([])
    
    def _extract_from_segment(self, segment, sampling_rate):
        """
        从单个数据段中提取频域特征
        
        参数:
            segment (numpy.ndarray): 形状为 (channels, samples) 的数据段
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 形状为 (channels, n_features) 的特征数组
        """
        channels, samples = segment.shape
        
        # 确定特征维度
        n_basic_features = len(self.features)
        n_band_features = len(self.bands) if 'band_power' in self.features else 0
        n_ratio_features = len(list(self._get_band_ratios().keys())) if 'band_ratios' in self.features else 0
        n_features = n_basic_features + n_band_features - 1 + n_ratio_features  # -1是因为band_power代表多个特征但只占一个位置
        
        # 创建特征数组
        feature_array = np.zeros((channels, n_features))
        
        for ch in range(channels):
            # 获取当前通道数据
            ch_data = segment[ch]
            
            # 计算PSD (Power Spectral Density)
            freqs, psd = signal.welch(ch_data, fs=sampling_rate, nperseg=min(self.nperseg, len(ch_data)),
                                     scaling='density')
            
            # 计算总功率
            total_power = np.sum(psd)
            
            # 提取特征
            feature_idx = 0
            
            # 为每个特征类型计算值
            for feature in self.features:
                if feature == 'band_power':
                    # 计算每个频带的功率
                    for band_name, (low_freq, high_freq) in self.bands.items():
                        # 找到对应频带的索引
                        band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                        
                        # 计算频带功率
                        band_power = np.sum(psd[band_idx])
                        
                        # 如果需要计算相对功率，除以总功率
                        if self.relative_power and total_power > 0:
                            band_power /= total_power
                            
                        feature_array[ch, feature_idx] = band_power
                        feature_idx += 1
                        
                elif feature == 'spectral_edge':
                    # 频谱边缘频率 (95% 的功率位于此频率以下)
                    # 计算累积功率分布
                    cumulative_power = np.cumsum(psd) / total_power if total_power > 0 else np.zeros_like(psd)
                    
                    # 找到95%功率对应的频率
                    edge_idx = np.argmax(cumulative_power >= 0.95) if np.any(cumulative_power >= 0.95) else -1
                    edge_freq = freqs[edge_idx] if edge_idx != -1 else 0
                    
                    feature_array[ch, feature_idx] = edge_freq
                    feature_idx += 1
                
                elif feature == 'spectral_entropy':
                    # 频谱熵 - 测量频谱的复杂度/随机性
                    # 归一化PSD使其和为1
                    norm_psd = psd / total_power if total_power > 0 else np.zeros_like(psd)
                    
                    # 计算信息熵 (-sum(p*log(p)))
                    entropy = 0
                    for p in norm_psd:
                        if p > 0:
                            entropy -= p * np.log2(p)
                            
                    feature_array[ch, feature_idx] = entropy
                    feature_idx += 1
                
                elif feature == 'spectral_centroid':
                    # 频谱质心 - 代表信号的"频谱中心"
                    if total_power > 0:
                        centroid = np.sum(freqs * psd) / total_power
                    else:
                        centroid = 0
                        
                    feature_array[ch, feature_idx] = centroid
                    feature_idx += 1
                
                elif feature == 'spectral_variance':
                    # 频谱方差 - 频谱分布的分散程度
                    if total_power > 0:
                        centroid = np.sum(freqs * psd) / total_power
                        variance = np.sum(((freqs - centroid) ** 2) * psd) / total_power
                    else:
                        variance = 0
                        
                    feature_array[ch, feature_idx] = variance
                    feature_idx += 1
                
                elif feature == 'spectral_skewness':
                    # 频谱偏度 - 频谱分布的不对称性
                    if total_power > 0:
                        centroid = np.sum(freqs * psd) / total_power
                        variance = np.sum(((freqs - centroid) ** 2) * psd) / total_power
                        
                        if variance > 0:
                            skewness = np.sum(((freqs - centroid) ** 3) * psd) / (total_power * (variance ** 1.5))
                        else:
                            skewness = 0
                    else:
                        skewness = 0
                        
                    feature_array[ch, feature_idx] = skewness
                    feature_idx += 1
                
                elif feature == 'spectral_kurtosis':
                    # 频谱峭度 - 频谱分布的"尖锐度"
                    if total_power > 0:
                        centroid = np.sum(freqs * psd) / total_power
                        variance = np.sum(((freqs - centroid) ** 2) * psd) / total_power
                        
                        if variance > 0:
                            kurtosis = np.sum(((freqs - centroid) ** 4) * psd) / (total_power * (variance ** 2)) - 3
                        else:
                            kurtosis = 0
                    else:
                        kurtosis = 0
                        
                    feature_array[ch, feature_idx] = kurtosis
                    feature_idx += 1
                
                elif feature == 'dominant_frequency':
                    # 主频 - PSD最大值对应的频率
                    max_idx = np.argmax(psd)
                    dominant_freq = freqs[max_idx]
                    
                    feature_array[ch, feature_idx] = dominant_freq
                    feature_idx += 1
                
                elif feature == 'median_frequency':
                    # 中值频率 - 功率的中值点
                    if total_power > 0:
                        cumulative_power = np.cumsum(psd) / total_power
                        median_idx = np.argmax(cumulative_power >= 0.5) if np.any(cumulative_power >= 0.5) else 0
                        median_freq = freqs[median_idx]
                    else:
                        median_freq = 0
                        
                    feature_array[ch, feature_idx] = median_freq
                    feature_idx += 1
                
                elif feature == 'peak_frequency':
                    # 峰值频率 - 每个频带中的峰值频率
                    for band_name, (low_freq, high_freq) in self.bands.items():
                        # 找到对应频带的索引
                        band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                        
                        if np.any(band_idx):
                            band_freqs = freqs[band_idx]
                            band_psd = psd[band_idx]
                            
                            # 找到功率最大值的频率
                            max_idx = np.argmax(band_psd)
                            peak_freq = band_freqs[max_idx]
                        else:
                            peak_freq = 0
                            
                        feature_array[ch, feature_idx] = peak_freq
                        feature_idx += 1
                        
                elif feature == 'band_ratios':
                    # 计算频带比例
                    band_powers = {}
                    
                    # 首先计算每个频带的功率
                    for band_name, (low_freq, high_freq) in self.bands.items():
                        band_idx = np.logical_and(freqs >= low_freq, freqs <= high_freq)
                        band_powers[band_name] = np.sum(psd[band_idx])
                    
                    # 计算常用的频带比例
                    ratios = self._get_band_ratios()
                    for ratio_name, (num_band, denom_band) in ratios.items():
                        if denom_band in band_powers and band_powers[denom_band] > 0:
                            ratio = band_powers.get(num_band, 0) / band_powers[denom_band]
                        else:
                            ratio = 0
                            
                        feature_array[ch, feature_idx] = ratio
                        feature_idx += 1
        
        return feature_array
    
    def _get_band_ratios(self):
        """定义要计算的频带比例"""
        ratios = {
            'alpha_theta': ('alpha', 'theta'),
            'alpha_beta': ('alpha', 'beta'),
            'theta_beta': ('theta', 'beta'),
            'delta_theta': ('delta', 'theta'),
            'delta_alpha': ('delta', 'alpha'),
            'delta_beta': ('delta', 'beta'),
            'theta_alpha': ('theta', 'alpha'),
            'beta_alpha': ('beta', 'alpha'),
            'gamma_beta': ('gamma', 'beta')
        }
        return ratios
    
    def get_feature_names(self):
        """获取特征名称列表"""
        feature_names = []
        
        for ch_idx in range(1):  # 只创建一个通道的名称，提取时会自动为每个通道重复
            for feature in self.features:
                if feature == 'band_power':
                    # 为每个频带添加名称
                    for band_name in self.bands.keys():
                        feature_names.append(f"{band_name}_power_ch{ch_idx}")
                        
                elif feature == 'band_ratios':
                    # 为每个频带比例添加名称
                    for ratio_name in self._get_band_ratios().keys():
                        feature_names.append(f"{ratio_name}_ratio_ch{ch_idx}")
                
                elif feature == 'peak_frequency':
                    # 为每个频带添加峰值频率名称
                    for band_name in self.bands.keys():
                        feature_names.append(f"{band_name}_peak_freq_ch{ch_idx}")
                
                else:
                    # 其他普通特征
                    feature_names.append(f"{feature}_ch{ch_idx}")
        
        return feature_names

class TimeFrequencyFeatures(FeatureExtractor):
    """时频域特征提取器 - 提取信号的时频特性"""
    
    def __init__(self, features=None, window_size=None, overlap=0.5, wavelet='morlet', 
                 freqs=None, n_cycles=7.0, bands=None):
        """
        初始化时频域特征提取器
        
        参数:
            features (list): 要提取的特征列表，如果为None则提取所有特征
            window_size (float): 窗口大小(秒)，如果为None则使用整个信号
            overlap (float): 窗口重叠比例(0.0-1.0)
            wavelet (str): 小波类型，'morlet'或'cmor'
            freqs (list/array): 要分析的频率列表，如果为None则使用默认频率
            n_cycles (float): 每个频率的周期数，可以是单个值或数组
            bands (dict): 频带定义字典，格式为{名称: (最低频率, 最高频率)}
        """
        super().__init__(name="TimeFrequencyFeatures")
        
        # 默认脑电频带
        self.default_bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
        
        # 设置频带
        self.bands = bands if bands is not None else self.default_bands
        
        # 默认分析频率范围
        if freqs is None:
            self.freqs = np.logspace(np.log10(1), np.log10(100), 30)
        else:
            self.freqs = np.array(freqs)
        
        # 可用特征列表
        self.available_features = [
            'band_power',            # 各频带功率随时间变化
            'power_variance',        # 功率时间方差
            'power_skew',            # 功率时间偏度
            'power_kurtosis',        # 功率时间峭度
            'band_power_ratio',      # 频带功率比例随时间变化
            'phase_coherence',       # 相位一致性
            'power_entropy',         # 功率熵
            'instantaneous_freq',    # 瞬时频率
            'spectral_edge_variation' # 频谱边缘频率变化
        ]
        
        # 设置要提取的特征
        if features is None:
            self.features = self.available_features.copy()
        else:
            self.features = [f for f in features if f in self.available_features]
            if not self.features:
                logger.warning("未指定有效特征，将使用所有可用特征")
                self.features = self.available_features.copy()
        
        self.window_size = window_size
        self.overlap = max(0.0, min(1.0, overlap))
        self.wavelet = wavelet
        self.n_cycles = n_cycles
        
        logger.debug(f"初始化时频域特征提取器: {len(self.features)}个特征")
    
    def extract(self, data, sampling_rate):
        """提取时频域特征"""
        if data is None or data.size == 0:
            logger.error("无法提取特征: 数据为空")
            return np.array([])
        
        try:
            channels, samples = data.shape
            
            # 如果未指定窗口大小，使用整个信号
            if self.window_size is None:
                # 对整个信号提取特征
                features = self._extract_from_segment(data, sampling_rate)
                
                # 将所有特征值平铺成一维数组
                return features.flatten()
            else:
                # 将信号分段，并对每段提取特征
                window_samples = int(self.window_size * sampling_rate)
                
                # 确保窗口大小不超过样本数
                if window_samples >= samples:
                    features = self._extract_from_segment(data, sampling_rate)
                    return features.flatten()
                
                # 计算步长
                step_samples = int(window_samples * (1 - self.overlap))
                
                # 创建分段
                segments = []
                for start in range(0, samples - window_samples + 1, step_samples):
                    end = start + window_samples
                    segment = data[:, start:end]
                    segments.append(segment)
                
                # 对每个分段提取特征
                segment_features = []
                for segment in segments:
                    features = self._extract_from_segment(segment, sampling_rate)
                    segment_features.append(features)
                
                # 计算所有分段特征的平均值
                if segment_features:
                    avg_features = np.mean(segment_features, axis=0)
                    return avg_features.flatten()
                else:
                    return np.array([])
        
        except Exception as e:
            logger.error(f"时频域特征提取失败: {e}")
            return np.array([])
    
    def _extract_from_segment(self, segment, sampling_rate):
        """
        从单个数据段中提取时频域特征
        
        参数:
            segment (numpy.ndarray): 形状为 (channels, samples) 的数据段
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 形状为 (channels, n_features) 的特征数组
        """
        channels, samples = segment.shape
        
        # 估计所需特征维度
        n_features = 0
        for feature in self.features:
            if feature == 'band_power':
                n_features += len(self.bands)
            elif feature == 'band_power_ratio':
                n_features += len(self._get_band_ratios())
            elif feature == 'phase_coherence':
                n_features += (channels * (channels - 1)) // 2  # 通道间两两组合
            else:
                n_features += 1
        
        # 创建特征数组
        feature_array = np.zeros((channels, n_features))
        
        try:
            # 将数据创建为MNE对象(如果为原始数据)
            # 对于每个通道，计算时频分解
            for ch in range(channels):
                # 提取当前通道数据
                ch_data = segment[ch]
                
                # 使用MNE进行小波分解
                # 注意：这里使用简单的方法计算。在实际应用中，可能需要更高级的预处理和设置
                if self.wavelet == 'morlet':
                    # 使用Morlet小波变换计算时频能量
                    tfr = self._compute_morlet_tfr(ch_data, sampling_rate)
                else:
                    # 使用短时傅里叶变换替代
                    tfr = self._compute_stft_tfr(ch_data, sampling_rate)
                
                # 计算各频带功率
                band_powers = {}
                for band_name, (low_freq, high_freq) in self.bands.items():
                    # 找到对应频带的索引
                    band_idx = np.logical_and(self.freqs >= low_freq, self.freqs <= high_freq)
                    
                    # 计算该频带的时频能量
                    if np.any(band_idx):
                        band_tfr = tfr[band_idx, :]
                        band_powers[band_name] = np.mean(band_tfr, axis=0)
                    else:
                        band_powers[band_name] = np.zeros(tfr.shape[1])
                
                # 提取特征
                feature_idx = 0
                
                for feature in self.features:
                    if feature == 'band_power':
                        # 各频带平均功率
                        for band_name in self.bands:
                            band_power = np.mean(band_powers[band_name])
                            feature_array[ch, feature_idx] = band_power
                            feature_idx += 1
                    
                    elif feature == 'power_variance':
                        # 功率时间方差 - 各频带功率的时间变化方差
                        total_var = 0
                        for band_name in self.bands:
                            band_var = np.var(band_powers[band_name])
                            total_var += band_var
                        
                        feature_array[ch, feature_idx] = total_var / len(self.bands)
                        feature_idx += 1
                    
                    elif feature == 'power_skew':
                        # 功率时间偏度 - 各频带功率的时间变化偏度
                        total_skew = 0
                        for band_name in self.bands:
                            band_skew = stats.skew(band_powers[band_name]) if len(band_powers[band_name]) > 2 else 0
                            total_skew += band_skew
                        
                        feature_array[ch, feature_idx] = total_skew / len(self.bands)
                        feature_idx += 1
                    
                    elif feature == 'power_kurtosis':
                        # 功率时间峭度 - 各频带功率的时间变化峭度
                        total_kurt = 0
                        for band_name in self.bands:
                            band_kurt = stats.kurtosis(band_powers[band_name]) if len(band_powers[band_name]) > 3 else 0
                            total_kurt += band_kurt
                        
                        feature_array[ch, feature_idx] = total_kurt / len(self.bands)
                        feature_idx += 1
                    
                    elif feature == 'band_power_ratio':
                        # 频带功率比例
                        ratios = self._get_band_ratios()
                        for ratio_name, (num_band, denom_band) in ratios.items():
                            # 计算每个比例的平均值
                            num_power = np.mean(band_powers[num_band])
                            denom_power = np.mean(band_powers[denom_band])
                            
                            ratio = num_power / denom_power if denom_power > 0 else 0
                            feature_array[ch, feature_idx] = ratio
                            feature_idx += 1
                    
                    elif feature == 'power_entropy':
                        # 功率熵 - 时频分布的熵
                        # 归一化TFR
                        tfr_norm = tfr / np.sum(tfr) if np.sum(tfr) > 0 else tfr
                        
                        # 计算熵 (-sum(p*log(p)))
                        entropy = 0
                        for p in tfr_norm.flatten():
                            if p > 0:
                                entropy -= p * np.log2(p)
                        
                        feature_array[ch, feature_idx] = entropy
                        feature_idx += 1
                    
                    elif feature == 'instantaneous_freq':
                        # 瞬时频率 - 根据相位导数估计
                        # 简化计算：使用最大功率对应的频率
                        avg_inst_freq = 0
                        for t in range(tfr.shape[1]):
                            max_freq_idx = np.argmax(tfr[:, t])
                            if max_freq_idx < len(self.freqs):
                                avg_inst_freq += self.freqs[max_freq_idx]
                        
                        avg_inst_freq /= tfr.shape[1]
                        feature_array[ch, feature_idx] = avg_inst_freq
                        feature_idx += 1
                    
                    elif feature == 'spectral_edge_variation':
                        # 频谱边缘频率变化 - 95%功率点的标准差
                        edge_freqs = []
                        for t in range(tfr.shape[1]):
                            col_sum = np.sum(tfr[:, t])
                            if col_sum > 0:
                                # 计算累积功率
                                cum_power = np.cumsum(tfr[:, t]) / col_sum
                                # 找到95%功率对应的频率
                                edge_idx = np.argmax(cum_power >= 0.95) if np.any(cum_power >= 0.95) else 0
                                edge_freqs.append(self.freqs[edge_idx])
                        
                        if edge_freqs:
                            feature_array[ch, feature_idx] = np.std(edge_freqs)
                        else:
                            feature_array[ch, feature_idx] = 0
                        feature_idx += 1
                    
                    elif feature == 'phase_coherence':
                        # 相位一致性 - 不同通道间的相位同步性
                        # 在单通道处理中跳过，在最后处理
                        pass
            
            # 如果需要计算通道间相位一致性
            if 'phase_coherence' in self.features and channels > 1:
                # 根据时频分解计算相位一致性
                # 这里简化为计算通道间的相关性，作为相位一致性的近似
                feature_idx = self.features.index('phase_coherence')
                for ch1 in range(channels):
                    for ch2 in range(ch1 + 1, channels):
                        # 计算两个通道的相关性
                        ch1_data = segment[ch1]
                        ch2_data = segment[ch2]
                        coherence = np.corrcoef(ch1_data, ch2_data)[0, 1]
                        
                        # 保存相位一致性指标
                        feature_array[ch1, feature_idx] = coherence
                        feature_array[ch2, feature_idx] = coherence
                        feature_idx += 1
            
            return feature_array
            
        except Exception as e:
            logger.error(f"时频特征提取失败: {e}")
            return np.zeros((channels, n_features))
    
    def _compute_morlet_tfr(self, data, sampling_rate):
        """使用Morlet小波计算时频表示"""
        try:
            # 生成MNE Info对象
            info = mne.create_info(['CH'], sampling_rate, ch_types=['eeg'])
            
            # 创建Epochs对象
            epochs = mne.EpochsArray(data.reshape(1, 1, -1), info)
            
            # 计算TFR
            power = mne.time_frequency.tfr_morlet(
                epochs, freqs=self.freqs, n_cycles=self.n_cycles, 
                return_itc=False, average=False
            )
            
            # 提取功率值
            tfr = power.data[0, 0, :, :]  # (frequencies, times)
            
            return tfr
            
        except Exception as e:
            logger.error(f"Morlet变换失败: {e}")
            # 如果MNE处理失败，尝试使用scipy实现的简化版本
            return self._compute_stft_tfr(data, sampling_rate)
    
    def _compute_stft_tfr(self, data, sampling_rate):
        """使用短时傅里叶变换计算时频表示"""
        try:
            # 计算STFT
            f, t, Zxx = signal.stft(data, fs=sampling_rate, nperseg=min(256, len(data)))
            
            # 计算功率谱
            tfr = np.abs(Zxx)**2
            
            # 如果需要，将频率重采样为self.freqs
            if len(f) != len(self.freqs):
                from scipy.interpolate import interp2d
                # 创建插值函数
                f_interp = interp2d(t, f, tfr)
                # 重采样到目标频率
                tfr = f_interp(t, self.freqs)
            
            return tfr
            
        except Exception as e:
            logger.error(f"STFT变换失败: {e}")
            # 返回空数组
            return np.zeros((len(self.freqs), len(data)))
    
    def _get_band_ratios(self):
        """定义要计算的频带比例"""
        ratios = {
            'alpha_theta': ('alpha', 'theta'),
            'alpha_beta': ('alpha', 'beta'),
            'theta_beta': ('theta', 'beta'),
            'delta_theta': ('delta', 'theta')
        }
        return ratios
    
    def get_feature_names(self):
        """获取特征名称列表"""
        feature_names = []
        
        for ch_idx in range(1):  # 只创建一个通道的名称，提取时会自动为每个通道重复
            for feature in self.features:
                if feature == 'band_power':
                    # 为每个频带添加名称
                    for band_name in self.bands.keys():
                        feature_names.append(f"tf_{band_name}_power_ch{ch_idx}")
                
                elif feature == 'band_power_ratio':
                    # 为每个频带比例添加名称
                    for ratio_name in self._get_band_ratios().keys():
                        feature_names.append(f"tf_{ratio_name}_ratio_ch{ch_idx}")
                
                elif feature == 'phase_coherence':
                    # 为每对通道添加相位一致性名称
                    feature_names.append(f"phase_coherence_ch{ch_idx}")
                
                else:
                    # 其他普通特征
                    feature_names.append(f"tf_{feature}_ch{ch_idx}")
        
        return feature_names

class FeatureExtractorPipeline:
    """特征提取器流水线 - 组合多个特征提取器"""
    
    def __init__(self):
        """初始化特征提取器流水线"""
        self.extractors = []
        logger.debug("初始化特征提取器流水线")
    
    def add_extractor(self, extractor):
        """
        添加特征提取器到流水线
        
        参数:
            extractor (FeatureExtractor): 特征提取器实例
        """
        if not isinstance(extractor, FeatureExtractor):
            logger.error(f"无法添加非特征提取器对象: {type(extractor)}")
            return
            
        self.extractors.append(extractor)
        logger.debug(f"添加特征提取器: {extractor.name}")
    
    def extract(self, data, sampling_rate):
        """
        从数据中提取所有特征
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 提取的特征数组
            list: 特征名称列表
        """
        if not self.extractors:
            logger.warning("特征提取器流水线为空")
            return np.array([]), []
            
        all_features = []
        all_feature_names = []
        
        for extractor in self.extractors:
            try:
                # 提取特征
                features = extractor.extract(data, sampling_rate)
                
                if features.size > 0:
                    all_features.append(features)
                    
                    # 获取特征名称
                    feature_names = extractor.get_feature_names()
                    
                    # 为多通道数据重复特征名称
                    if len(feature_names) > 0 and data.shape[0] > 1:
                        expanded_names = []
                        base_names = feature_names
                        
                        for ch in range(data.shape[0]):
                            for name in base_names:
                                # 替换通道索引
                                if 'ch0' in name:
                                    expanded_names.append(name.replace('ch0', f'ch{ch}'))
                                else:
                                    expanded_names.append(f"{name}_ch{ch}")
                        
                        all_feature_names.extend(expanded_names)
                    else:
                        all_feature_names.extend(feature_names)
                
            except Exception as e:
                logger.error(f"使用提取器 {extractor.name} 提取特征时失败: {e}")
        
        if all_features:
            # 合并所有特征
            combined_features = np.hstack(all_features) if len(all_features) > 1 else all_features[0]
            return combined_features, all_feature_names
        else:
            return np.array([]), all_feature_names
    
    def clear(self):
        """清空流水线"""
        self.extractors = []
        logger.debug("清空特征提取器流水线")
    
    def get_default_pipeline(self):
        """
        创建默认特征提取流水线
        
        返回:
            FeatureExtractorPipeline: 配置好的特征提取流水线
        """
        self.clear()
        
        # 添加时域特征提取器
        time_domain = TimeDomainFeatures(
            features=['mean', 'std', 'skewness', 'kurtosis', 'hjorth_activity', 'hjorth_mobility'],
            window_size=1.0,
            overlap=0.5
        )
        self.add_extractor(time_domain)
        
        # 添加频域特征提取器
        freq_domain = FrequencyDomainFeatures(
            features=['band_power', 'dominant_frequency', 'spectral_entropy'],
            window_size=1.0,
            overlap=0.5,
            relative_power=True
        )
        self.add_extractor(freq_domain)
        
        # 添加简化的时频域特征提取器
        tf_domain = TimeFrequencyFeatures(
            features=['band_power', 'power_entropy'],
            window_size=1.0,
            overlap=0.5
        )
        self.add_extractor(tf_domain)
        
        logger.info("创建了默认特征提取流水线")
        return self
