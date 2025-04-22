"""
伪迹去除模块 - 实现眨眼、肌电等伪迹的自动检测与去除
"""

import logging
import numpy as np
from scipy import signal, stats
import mne
from sklearn.decomposition import FastICA

# 创建logger
logger = logging.getLogger('artifact_removal')

class ArtifactRemover:
    """伪迹检测与去除基类"""
    
    def __init__(self, name="BaseArtifactRemover"):
        """初始化伪迹去除器"""
        self.name = name
        logger.debug(f"初始化伪迹去除器: {name}")
    
    def detect(self, data, sampling_rate):
        """
        检测数据中的伪迹
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            list: 伪迹时间点的列表，每个元素为 (开始样本索引, 结束样本索引, 类型)
        """
        # 基类不做任何检测，返回空列表
        return []
    
    def remove(self, data, sampling_rate):
        """
        从数据中去除伪迹
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 去除伪迹后的数据
        """
        # 基类不做任何去除，直接返回原始数据
        return data.copy()

class ThresholdArtifactRemover(ArtifactRemover):
    """基于阈值的伪迹去除 - 简单但有效的方法处理极端值"""
    
    def __init__(self, threshold=5.0, window_size=0.2, replace_method='linear'):
        """
        初始化基于阈值的伪迹去除器
        
        参数:
            threshold (float): 检测阈值(标准差的倍数)
            window_size (float): 伪迹窗口大小(秒)
            replace_method (str): 替换方法，'linear', 'mean', 'median', 'zero'
        """
        super().__init__(name=f"ThresholdArtifactRemover(threshold={threshold})")
        self.threshold = threshold
        self.window_size = window_size
        self.replace_method = replace_method
        
        # 确保替换方法有效
        valid_methods = ['linear', 'mean', 'median', 'zero']
        if replace_method not in valid_methods:
            logger.warning(f"无效的替换方法: {replace_method}，使用linear代替")
            self.replace_method = 'linear'
    
    def detect(self, data, sampling_rate):
        """检测超过阈值的伪迹"""
        if data is None or data.size == 0:
            logger.error("无法检测伪迹: 数据为空")
            return []
            
        try:
            channels, samples = data.shape
            artifacts = []
            window_samples = int(self.window_size * sampling_rate)
            
            for ch in range(channels):
                # 计算通道数据的均值和标准差
                ch_mean = np.mean(data[ch])
                ch_std = np.std(data[ch])
                
                # 设置检测阈值
                threshold_value = ch_mean + self.threshold * ch_std
                
                # 找出超过阈值的样本
                outliers = np.where(np.abs(data[ch] - ch_mean) > threshold_value)[0]
                
                # 将连续的伪迹样本分组
                if len(outliers) > 0:
                    # 分组连续的索引
                    groups = np.split(outliers, np.where(np.diff(outliers) > 1)[0] + 1)
                    
                    for group in groups:
                        if len(group) > 0:
                            # 扩展伪迹窗口
                            start = max(0, group[0] - window_samples // 2)
                            end = min(samples, group[-1] + window_samples // 2 + 1)
                            
                            artifacts.append((start, end, f'threshold_ch{ch}'))
            
            # 合并重叠的伪迹
            if artifacts:
                artifacts.sort(key=lambda x: x[0])
                merged_artifacts = [artifacts[0]]
                
                for start, end, artifact_type in artifacts[1:]:
                    prev_start, prev_end, prev_type = merged_artifacts[-1]
                    
                    if start <= prev_end:
                        # 重叠，合并
                        merged_artifacts[-1] = (prev_start, max(end, prev_end), prev_type)
                    else:
                        # 不重叠，添加新伪迹
                        merged_artifacts.append((start, end, artifact_type))
                
                logger.debug(f"检测到 {len(merged_artifacts)} 个基于阈值的伪迹")
                return merged_artifacts
            else:
                logger.debug(f"未检测到基于阈值的伪迹")
                return []
                
        except Exception as e:
            logger.error(f"伪迹检测失败: {e}")
            return []
    
    def remove(self, data, sampling_rate):
        """移除超过阈值的伪迹"""
        if data is None or data.size == 0:
            logger.error("无法去除伪迹: 数据为空")
            return data
            
        try:
            # 检测伪迹
            artifacts = self.detect(data, sampling_rate)
            
            if not artifacts:
                logger.debug("未检测到需要去除的伪迹")
                return data.copy()
                
            # 复制数据进行处理
            clean_data = data.copy()
            channels, samples = clean_data.shape
            
            # 处理每个检测到的伪迹
            for start, end, artifact_type in artifacts:
                # 提取通道索引
                if 'ch' in artifact_type:
                    ch = int(artifact_type.split('ch')[1])
                    channels_to_clean = [ch]
                else:
                    channels_to_clean = range(channels)
                
                # 对每个受影响的通道应用替换方法
                for ch in channels_to_clean:
                    # 根据替换方法处理伪迹
                    if self.replace_method == 'linear':
                        # 线性插值
                        if start > 0 and end < samples:
                            # 使用伪迹前后的值进行线性插值
                            x = np.array([start-1, end])
                            y = np.array([clean_data[ch, start-1], clean_data[ch, end]])
                            xnew = np.arange(start, end)
                            clean_data[ch, start:end] = np.interp(xnew, x, y)
                        else:
                            # 边界情况，用均值替代
                            clean_data[ch, start:end] = np.mean(clean_data[ch])
                            
                    elif self.replace_method == 'mean':
                        # 用通道均值替代
                        channel_mean = np.mean(clean_data[ch])
                        clean_data[ch, start:end] = channel_mean
                        
                    elif self.replace_method == 'median':
                        # 用通道中位数替代
                        channel_median = np.median(clean_data[ch])
                        clean_data[ch, start:end] = channel_median
                        
                    elif self.replace_method == 'zero':
                        # 直接置零
                        clean_data[ch, start:end] = 0
            
            logger.debug(f"使用 {self.replace_method} 方法去除了 {len(artifacts)} 个伪迹")
            return clean_data
            
        except Exception as e:
            logger.error(f"伪迹去除失败: {e}")
            return data

class ICARemover(ArtifactRemover):
    """基于ICA的伪迹去除 - 对于眨眼和其他伪迹较为有效"""
    
    def __init__(self, n_components=None, random_state=42, threshold=3.0):
        """
        初始化基于ICA的伪迹去除器
        
        参数:
            n_components (int): ICA组件数量，如果为None则自动确定
            random_state (int): 随机种子，确保结果可重复
            threshold (float): 检测伪迹组件的阈值(标准差的倍数)
        """
        super().__init__(name=f"ICARemover(threshold={threshold})")
        self.n_components = n_components
        self.random_state = random_state
        self.threshold = threshold
        self.ica = None
        self.artifact_components = None
    
    def fit(self, data, sampling_rate):
        """
        拟合ICA模型
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            self: 返回自身实例
        """
        if data is None or data.size == 0:
            logger.error("无法拟合ICA: 数据为空")
            return self
            
        try:
            channels, samples = data.shape
            
            # 确定组件数量
            if self.n_components is None:
                # 默认使用通道数量
                self.n_components = channels
            
            # 转置数据以适应scikit-learn API (samples, features)
            X = data.T
            
            # 实例化并拟合ICA
            self.ica = FastICA(n_components=self.n_components, random_state=self.random_state)
            self.components_ = self.ica.fit_transform(X)
            self.mixing_ = self.ica.mixing_
            
            logger.debug(f"ICA拟合完成: {self.n_components}个组件")
            return self
            
        except Exception as e:
            logger.error(f"ICA拟合失败: {e}")
            self.ica = None
            return self
    
    def detect_artifact_components(self):
        """
        检测伪迹组件
        
        返回:
            list: 伪迹组件的索引列表
        """
        if self.ica is None or self.components_ is None:
            logger.error("无法检测伪迹组件: 未拟合ICA模型")
            return []
            
        try:
            # 计算每个组件的统计特征
            artifacts = []
            
            # 检查每个组件的峭度(kurtosis)，峭度高表示可能是伪迹
            for i in range(self.n_components):
                component = self.components_[:, i]
                
                # 计算组件的峭度
                k = stats.kurtosis(component)
                
                # 计算组件的峰度比(peak-to-peak)
                p2p = np.max(component) - np.min(component)
                
                # 计算组件的偏度(skewness)
                skew = stats.skew(component)
                
                # 计算与垂直眼电模式的相似度 (前几个通道权重大)
                weights = np.abs(self.ica.mixing_[:, i])
                front_weight_ratio = np.sum(weights[:4]) / np.sum(weights) if len(weights) > 4 else 0
                
                # 基于多个特征的组合判断是否为伪迹组件
                # 高峭度表示非高斯性，可能是伪迹
                # 眨眼通常在前额叶通道有更高的权重
                if (k > self.threshold) or (front_weight_ratio > 0.6 and p2p > 2.0) or np.abs(skew) > 2.0:
                    artifacts.append(i)
                    logger.debug(f"组件 {i} 被标记为伪迹: 峭度={k:.2f}, 前额叶权重比={front_weight_ratio:.2f}, 偏度={skew:.2f}")
            
            self.artifact_components = artifacts
            logger.debug(f"检测到 {len(artifacts)} 个伪迹组件: {artifacts}")
            return artifacts
            
        except Exception as e:
            logger.error(f"伪迹组件检测失败: {e}")
            return []
    
    def reconstruct(self, data, exclude_components=None):
        """
        使用ICA重建数据，同时排除伪迹组件
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            exclude_components (list): 要排除的组件索引列表，如果为None则使用检测到的伪迹组件
            
        返回:
            numpy.ndarray: 重建后的数据
        """
        if self.ica is None:
            logger.error("无法重建数据: 未拟合ICA模型")
            return data
            
        if exclude_components is None:
            if self.artifact_components is None:
                # 自动检测伪迹组件
                exclude_components = self.detect_artifact_components()
            else:
                exclude_components = self.artifact_components
        
        try:
            channels, samples = data.shape
            
            # 转置数据以适应scikit-learn API (samples, features)
            X = data.T
            
            # 变换数据到组件空间
            components = self.ica.transform(X)
            
            # 将伪迹组件置零
            for component_idx in exclude_components:
                components[:, component_idx] = 0
            
            # 重建数据
            X_reconstructed = components @ self.mixing_.T
            
            # 转置回原始形状 (channels, samples)
            reconstructed_data = X_reconstructed.T
            
            logger.debug(f"使用ICA重建数据，排除了 {len(exclude_components)} 个组件")
            return reconstructed_data
            
        except Exception as e:
            logger.error(f"数据重建失败: {e}")
            return data
    
    def remove(self, data, sampling_rate):
        """使用ICA去除伪迹"""
        if data is None or data.size == 0:
            logger.error("无法去除伪迹: 数据为空")
            return data
            
        try:
            # 拟合ICA模型
            self.fit(data, sampling_rate)
            
            # 检测伪迹组件
            artifact_components = self.detect_artifact_components()
            
            if not artifact_components:
                logger.debug("未检测到伪迹组件")
                return data.copy()
            
            # 重建数据，排除伪迹组件
            clean_data = self.reconstruct(data, exclude_components=artifact_components)
            
            logger.debug(f"ICA去除伪迹完成，排除了 {len(artifact_components)} 个组件")
            return clean_data
            
        except Exception as e:
            logger.error(f"ICA伪迹去除失败: {e}")
            return data

class WaveletRemover(ArtifactRemover):
    """基于小波变换的伪迹去除 - 适用于短时伪迹"""
    
    def __init__(self, wavelet='db4', level=5, threshold_mult=3.0):
        """
        初始化基于小波变换的伪迹去除器
        
        参数:
            wavelet (str): 小波类型，如'db4', 'sym8'等
            level (int): 分解级别
            threshold_mult (float): 阈值乘数，用于确定去噪阈值
        """
        super().__init__(name=f"WaveletRemover(wavelet={wavelet}, level={level})")
        self.wavelet = wavelet
        self.level = level
        self.threshold_mult = threshold_mult
    
    def remove(self, data, sampling_rate):
        """使用小波变换去除伪迹"""
        if data is None or data.size == 0:
            logger.error("无法去除伪迹: 数据为空")
            return data
        
        try:
            # 尝试导入PyWavelets库
            try:
                import pywt
            except ImportError:
                logger.error("未安装PyWavelets库，无法使用小波伪迹去除")
                return data
            
            channels, samples = data.shape
            clean_data = np.zeros_like(data)
            
            # 对每个通道应用小波去噪
            for ch in range(channels):
                # 小波分解
                coeffs = pywt.wavedec(data[ch], self.wavelet, level=self.level)
                
                # 阈值处理
                for i in range(1, len(coeffs)):
                    # 计算高频系数的阈值
                    sigma = np.median(np.abs(coeffs[i])) / 0.6745  # MAD估计标准差
                    threshold = sigma * self.threshold_mult * np.sqrt(2 * np.log(len(coeffs[i])))
                    
                    # 软阈值处理
                    coeffs[i] = pywt.threshold(coeffs[i], threshold, mode='soft')
                
                # 小波重构
                clean_data[ch] = pywt.waverec(coeffs, self.wavelet)
                
                # 确保长度一致，有时重构后长度可能略有变化
                if len(clean_data[ch]) > samples:
                    clean_data[ch] = clean_data[ch][:samples]
                elif len(clean_data[ch]) < samples:
                    # 填充到原始长度
                    pad_length = samples - len(clean_data[ch])
                    clean_data[ch] = np.pad(clean_data[ch], (0, pad_length), 'constant')
            
            logger.debug(f"小波伪迹去除完成: {self.wavelet}, 级别={self.level}")
            return clean_data
            
        except Exception as e:
            logger.error(f"小波伪迹去除失败: {e}")
            return data

class ArtifactRemovalPipeline:
    """伪迹去除管道，组合多种方法"""
    
    def __init__(self):
        """初始化伪迹去除管道"""
        self.removers = []
        logger.debug("初始化伪迹去除管道")
    
    def add_remover(self, remover):
        """
        添加伪迹去除器到管道
        
        参数:
            remover (ArtifactRemover): 伪迹去除器实例
        """
        if not isinstance(remover, ArtifactRemover):
            logger.error(f"无效的伪迹去除器类型: {type(remover)}")
            return
        
        self.removers.append(remover)
        logger.info(f"添加伪迹去除器到管道: {remover.name}")
    
    def process(self, data, sampling_rate):
        """
        对数据应用整个伪迹去除管道
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            sampling_rate (int): 采样率(Hz)
            
        返回:
            numpy.ndarray: 处理后的数据
        """
        if data is None or data.size == 0:
            logger.error("无法处理数据: 数据为空")
            return data
        
        # 依次应用每个伪迹去除器
        clean_data = data.copy()
        for remover in self.removers:
            clean_data = remover.remove(clean_data, sampling_rate)
        
        logger.debug(f"应用了 {len(self.removers)} 个伪迹去除器")
        return clean_data
    
    def clear(self):
        """清空伪迹去除管道"""
        self.removers = []
        logger.debug("清空伪迹去除管道")
    
    def get_default_pipeline(self):
        """
        获取默认伪迹去除管道
        
        返回:
            ArtifactRemovalPipeline: 配置好的伪迹去除管道
        """
        # 添加常用的伪迹去除器
        # 1. 基于阈值的伪迹去除 - 去除极端值
        self.add_remover(ThresholdArtifactRemover(threshold=5.0, replace_method='linear'))
        
        # 2. 基于ICA的伪迹去除 - 去除眨眼等常见伪迹
        self.add_remover(ICARemover(threshold=3.0))
        
        # 3. 基于小波的伪迹去除 - 处理短时伪迹
        self.add_remover(WaveletRemover(wavelet='db4', level=5))
        
        return self
