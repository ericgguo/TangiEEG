"""
信号增强模块 - 实现空间滤波、信号平滑等增强信号质量的方法
"""

import logging
import numpy as np
from scipy import signal, interpolate
import mne
from sklearn.preprocessing import StandardScaler, RobustScaler

# 创建logger
logger = logging.getLogger('signal_enhancer')

class SignalEnhancer:
    """信号增强基类"""
    
    def __init__(self, name="BaseEnhancer"):
        """初始化信号增强器"""
        self.name = name
        logger.debug(f"初始化信号增强器: {name}")
    
    def apply(self, data, sampling_rate):
        """
        应用信号增强
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 增强后的数据
        """
        # 基类不做任何处理，直接返回原始数据
        return data.copy()

class AmplitudeNormalizer(SignalEnhancer):
    """幅度归一化 - 标准化信号幅度"""
    
    def __init__(self, method='standard', per_channel=True):
        """
        初始化幅度归一化增强器
        
        参数:
            method (str): 归一化方法，'standard'(均值为0，标准差为1), 'robust'(中位数为0，IQR为1), 'minmax'(0-1之间)
            per_channel (bool): 是否对每个通道分别归一化
        """
        super().__init__(name=f"AmplitudeNormalizer({method})")
        self.method = method
        self.per_channel = per_channel
        
        # 确保方法有效
        valid_methods = ['standard', 'robust', 'minmax']
        if method not in valid_methods:
            logger.warning(f"无效的归一化方法: {method}，使用standard代替")
            self.method = 'standard'
    
    def apply(self, data, sampling_rate):
        """应用幅度归一化"""
        if data is None or data.size == 0:
            logger.error("无法增强信号: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            normalized_data = np.zeros_like(data)
            
            if self.per_channel:
                # 对每个通道分别归一化
                for ch in range(channels):
                    normalized_data[ch] = self._normalize_array(data[ch], self.method)
            else:
                # 对整个数据集归一化
                normalized_data = self._normalize_array(data, self.method)
                
            logger.debug(f"应用幅度归一化: {self.method}, 每通道={self.per_channel}")
            return normalized_data
            
        except Exception as e:
            logger.error(f"幅度归一化失败: {e}")
            return data
    
    def _normalize_array(self, arr, method):
        """
        对数组应用归一化
        
        参数:
            arr (numpy.ndarray): 要归一化的数组
            method (str): 归一化方法
            
        返回:
            numpy.ndarray: 归一化后的数组
        """
        if method == 'standard':
            # 均值为0，标准差为1
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0:
                return np.zeros_like(arr)
            return (arr - mean) / std
            
        elif method == 'robust':
            # 中位数为0，四分位范围为1
            median = np.median(arr)
            q75, q25 = np.percentile(arr, [75, 25])
            iqr = q75 - q25
            if iqr == 0:
                return np.zeros_like(arr)
            return (arr - median) / iqr
            
        elif method == 'minmax':
            # 将值缩放到0-1之间
            min_val = np.min(arr)
            max_val = np.max(arr)
            if min_val == max_val:
                return np.zeros_like(arr)
            return (arr - min_val) / (max_val - min_val)
            
        else:
            # 默认不处理
            return arr

class SpatialEnhancer(SignalEnhancer):
    """空间增强器 - 应用空间滤波增强特定区域的信号"""
    
    def __init__(self, method='csp', n_components=4, classes=None):
        """
        初始化空间增强器
        
        参数:
            method (str): 空间增强方法，'csp'(共同空间模式), 'ssd'(频谱空间分解), 'xdawn'(事件相关增强)
            n_components (int): 要保留的组件数量
            classes (array): 用于监督方法(如CSP)的类标签
        """
        super().__init__(name=f"SpatialEnhancer({method})")
        self.method = method
        self.n_components = n_components
        self.classes = classes
        self.model = None
        
        # 确保方法有效
        valid_methods = ['csp', 'ssd', 'xdawn']
        if method not in valid_methods:
            logger.warning(f"无效的空间增强方法: {method}，使用csp代替")
            self.method = 'csp'
    
    def fit(self, data, sampling_rate, classes=None):
        """
        根据数据拟合空间增强器
        
        参数:
            data (numpy.ndarray): 形状为 (trials, channels, samples) 的多试次数据
            sampling_rate (int): 采样率(Hz)
            classes (array): 类标签，如果为None则使用构造函数中的classes
            
        返回:
            self: 返回自身实例
        """
        if data is None or data.size == 0:
            logger.error("无法拟合空间增强器: 数据为空")
            return self
            
        try:
            if classes is not None:
                self.classes = classes
            
            # 如果没有提供类标签，且方法需要类标签
            if self.classes is None and self.method == 'csp':
                logger.error("CSP方法需要提供类标签")
                return self
                
            # 根据方法创建并拟合模型
            if self.method == 'csp':
                # 共同空间模式(Common Spatial Patterns)
                try:
                    from mne.decoding import CSP
                    self.model = CSP(n_components=self.n_components)
                    # CSP需要二分类标签和epochs格式数据
                    self.model.fit(data, self.classes)
                    logger.debug("CSP空间增强器拟合完成")
                except ImportError:
                    logger.error("未安装MNE库，无法使用CSP方法")
                    
            elif self.method == 'ssd':
                # 频谱空间分解(Spatio-Spectral Decomposition)
                try:
                    # 尝试导入MNE的SSD实现
                    from mne.preprocessing import SSD
                    # SSD需要指定信号频率和噪声频率
                    # 这里假设信号频率为8-13Hz(alpha)，噪声频率为其余部分
                    freq_signal = (8.0, 13.0)
                    freq_noise = [(1.0, 7.0), (14.0, 45.0)]
                    ssd = SSD(n_components=self.n_components, info=None,
                               reg='oas', rank='full',
                               sort_by_spectral_ratio=True)
                    # 创建MNE raw对象
                    # 注意：这里需要更多的配置，完整实现需要在实际项目中根据MNE文档调整
                    self.model = ssd
                    logger.debug("SSD空间增强器设置完成")
                except ImportError:
                    logger.error("未安装MNE库，无法使用SSD方法")
                    
            elif self.method == 'xdawn':
                # Xdawn空间滤波器
                try:
                    from mne.preprocessing import Xdawn
                    # Xdawn需要epochs格式数据和事件ID
                    # 这里简化处理，具体实现需要根据MNE文档调整
                    self.model = Xdawn(n_components=self.n_components)
                    logger.debug("Xdawn空间增强器设置完成")
                except ImportError:
                    logger.error("未安装MNE库，无法使用Xdawn方法")
            
            return self
            
        except Exception as e:
            logger.error(f"空间增强器拟合失败: {e}")
            self.model = None
            return self
    
    def apply(self, data, sampling_rate):
        """应用空间增强"""
        if data is None or data.size == 0:
            logger.error("无法增强信号: 数据为空")
            return data
            
        if self.model is None:
            logger.warning("空间增强器未拟合，无法应用")
            return data
            
        try:
            # 根据方法应用空间增强
            if self.method == 'csp':
                # CSP变换
                enhanced_data = self.model.transform(data)
                logger.debug(f"应用CSP空间增强: 保留{self.n_components}个组件")
                return enhanced_data
                
            elif self.method == 'ssd':
                # SSD变换
                # 注意：这里需要更详细的实现，取决于MNE的SSD API
                logger.debug(f"应用SSD空间增强: 保留{self.n_components}个组件")
                return data  # 暂时返回原始数据
                
            elif self.method == 'xdawn':
                # Xdawn变换
                # 注意：这里需要更详细的实现，取决于MNE的Xdawn API
                logger.debug(f"应用Xdawn空间增强: 保留{self.n_components}个组件")
                return data  # 暂时返回原始数据
                
            else:
                return data
                
        except Exception as e:
            logger.error(f"空间增强应用失败: {e}")
            return data

class FrequencyBandEnhancer(SignalEnhancer):
    """频段增强器 - 增强特定频段的信号"""
    
    def __init__(self, band_name='alpha', gain=2.0, low_freq=None, high_freq=None):
        """
        初始化频段增强器
        
        参数:
            band_name (str): 频段名称，'delta', 'theta', 'alpha', 'beta', 'gamma'或'custom'
            gain (float): 增益因子，用于放大指定频段
            low_freq (float): 如果band_name='custom'，自定义频段的低频(Hz)
            high_freq (float): 如果band_name='custom'，自定义频段的高频(Hz)
        """
        super().__init__(name=f"FrequencyBandEnhancer({band_name}, gain={gain})")
        self.band_name = band_name
        self.gain = gain
        self.low_freq = low_freq
        self.high_freq = high_freq
        
        # 预定义的频段范围
        self.freq_bands = {
            'delta': (0.5, 4.0),
            'theta': (4.0, 8.0),
            'alpha': (8.0, 13.0),
            'beta': (13.0, 30.0),
            'gamma': (30.0, 100.0)
        }
        
        # 如果是自定义频段，确保提供了频率范围
        if band_name == 'custom':
            if low_freq is None or high_freq is None:
                logger.warning("自定义频段需要提供低频和高频，将使用alpha频段替代")
                self.band_name = 'alpha'
                self.low_freq, self.high_freq = self.freq_bands['alpha']
            else:
                self.low_freq = low_freq
                self.high_freq = high_freq
        else:
            # 使用预定义频段
            if band_name in self.freq_bands:
                self.low_freq, self.high_freq = self.freq_bands[band_name]
            else:
                logger.warning(f"未知的频段名称: {band_name}，将使用alpha频段替代")
                self.band_name = 'alpha'
                self.low_freq, self.high_freq = self.freq_bands['alpha']
    
    def apply(self, data, sampling_rate):
        """应用频段增强"""
        if data is None or data.size == 0:
            logger.error("无法增强信号: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 确保频率有效
            nyquist = sampling_rate / 2
            if self.low_freq >= nyquist or self.high_freq >= nyquist:
                logger.error(f"频段范围无效: {self.low_freq}-{self.high_freq}Hz, 奈奎斯特频率={nyquist}Hz")
                return data
                
            # 使用FFT进行频域处理
            enhanced_data = np.zeros_like(data)
            
            for ch in range(channels):
                # 对每个通道应用FFT
                fft_data = np.fft.rfft(data[ch])
                fft_freqs = np.fft.rfftfreq(samples, d=1.0/sampling_rate)
                
                # 创建频率增益掩码
                gain_mask = np.ones(len(fft_freqs))
                
                # 对指定频段应用增益
                mask = (fft_freqs >= self.low_freq) & (fft_freqs <= self.high_freq)
                gain_mask[mask] = self.gain
                
                # 应用增益
                fft_data_enhanced = fft_data * gain_mask
                
                # 逆FFT回时域
                enhanced_data[ch] = np.fft.irfft(fft_data_enhanced, n=samples)
                
            logger.debug(f"应用频段增强: {self.band_name}({self.low_freq}-{self.high_freq}Hz), 增益={self.gain}")
            return enhanced_data
            
        except Exception as e:
            logger.error(f"频段增强失败: {e}")
            return data

class SignalSmoother(SignalEnhancer):
    """信号平滑器 - 减少信号噪声"""
    
    def __init__(self, method='moving_average', window_size=5):
        """
        初始化信号平滑器
        
        参数:
            method (str): 平滑方法，'moving_average', 'savgol', 'gaussian'
            window_size (int): 滑动窗口大小
        """
        super().__init__(name=f"SignalSmoother({method}, window={window_size})")
        self.method = method
        self.window_size = window_size
        
        # 确保方法有效
        valid_methods = ['moving_average', 'savgol', 'gaussian']
        if method not in valid_methods:
            logger.warning(f"无效的平滑方法: {method}，使用moving_average代替")
            self.method = 'moving_average'
        
        # 确保窗口大小有效
        if window_size < 3:
            logger.warning(f"窗口大小太小: {window_size}，设置为3")
            self.window_size = 3
        
        # 确保窗口大小为奇数（某些滤波器需要）
        if window_size % 2 == 0:
            self.window_size += 1
    
    def apply(self, data, sampling_rate):
        """应用信号平滑"""
        if data is None or data.size == 0:
            logger.error("无法增强信号: 数据为空")
            return data
            
        try:
            channels, samples = data.shape
            
            # 确保窗口大小不超过样本数
            window_size = min(self.window_size, samples)
            
            # 如果窗口大小太小，直接返回原始数据
            if window_size < 3:
                logger.warning(f"窗口大小 {window_size} 太小，无法应用平滑")
                return data
                
            smoothed_data = np.zeros_like(data)
            
            for ch in range(channels):
                # 根据方法应用不同的平滑
                if self.method == 'moving_average':
                    # 简单移动平均
                    window = np.ones(window_size) / window_size
                    smoothed_data[ch] = np.convolve(data[ch], window, mode='same')
                    
                elif self.method == 'savgol':
                    # Savitzky-Golay滤波
                    # 多项式阶数，通常为2-5之间，不超过窗口大小
                    poly_order = min(3, window_size - 1)
                    smoothed_data[ch] = signal.savgol_filter(data[ch], window_size, poly_order)
                    
                elif self.method == 'gaussian':
                    # 高斯滤波
                    sigma = window_size / 5.0  # 标准差，控制高斯曲线宽度
                    smoothed_data[ch] = signal.gaussian_filter1d(data[ch], sigma)
                    
            logger.debug(f"应用信号平滑: {self.method}, 窗口大小={window_size}")
            return smoothed_data
            
        except Exception as e:
            logger.error(f"信号平滑失败: {e}")
            return data

class SignalEnhancementPipeline:
    """信号增强管道，组合多种方法"""
    
    def __init__(self):
        """初始化信号增强管道"""
        self.enhancers = []
        logger.debug("初始化信号增强管道")
    
    def add_enhancer(self, enhancer):
        """
        添加信号增强器到管道
        
        参数:
            enhancer (SignalEnhancer): 信号增强器实例
        """
        if not isinstance(enhancer, SignalEnhancer):
            logger.error(f"无效的信号增强器类型: {type(enhancer)}")
            return
        
        self.enhancers.append(enhancer)
        logger.info(f"添加信号增强器到管道: {enhancer.name}")
    
    def process(self, data, sampling_rate):
        """
        对数据应用整个信号增强管道
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 处理后的数据
        """
        if data is None or data.size == 0:
            logger.error("无法处理数据: 数据为空")
            return data
        
        # 依次应用每个信号增强器
        enhanced_data = data.copy()
        for enhancer in self.enhancers:
            enhanced_data = enhancer.apply(enhanced_data, sampling_rate)
        
        logger.debug(f"应用了 {len(self.enhancers)} 个信号增强器")
        return enhanced_data
    
    def clear(self):
        """清空信号增强管道"""
        self.enhancers = []
        logger.debug("清空信号增强管道")
    
    def get_default_pipeline(self):
        """
        获取默认信号增强管道
        
        返回:
            SignalEnhancementPipeline: 配置好的信号增强管道
        """
        # 添加常用的信号增强器
        # 1. 幅度归一化 - 标准化信号幅度
        self.add_enhancer(AmplitudeNormalizer(method='robust', per_channel=True))
        
        # 2. 信号平滑 - 去除高频噪声
        self.add_enhancer(SignalSmoother(method='savgol', window_size=7))
        
        # 3. 频段增强 - 增强alpha频段(8-13Hz)，通常与注意力相关
        self.add_enhancer(FrequencyBandEnhancer(band_name='alpha', gain=1.5))
        
        return self
