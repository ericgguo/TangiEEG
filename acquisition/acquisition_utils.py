"""
数据采集辅助工具 - 提供信号质量检测、通道阻抗测量等功能
"""

import time
import logging
import numpy as np
from scipy import signal
import serial.tools.list_ports

# 创建logger
logger = logging.getLogger('acquisition_utils')

def list_available_ports():
    """
    获取可用的串口列表
    
    返回:
        list: 可用串口设备路径列表
    """
    ports = list(serial.tools.list_ports.comports())
    return [p.device for p in ports]

def detect_openbci_boards():
    """
    检测连接的OpenBCI设备
    
    返回:
        list: 可能是OpenBCI设备的串口设备路径列表
    """
    # 获取所有串口
    all_ports = list(serial.tools.list_ports.comports())
    openbci_ports = []
    
    # 常见的OpenBCI设备特征
    openbci_keywords = ["OpenBCI", "FTDI", "SLAB_USBtoUART", "usbserial", "ttyUSB", "ttyACM"]
    
    for port in all_ports:
        # 检查设备描述和硬件ID是否包含相关关键字
        desc_hw = (port.description + port.hwid).lower()
        for keyword in openbci_keywords:
            if keyword.lower() in desc_hw:
                openbci_ports.append(port.device)
                break
    
    logger.info(f"检测到可能的OpenBCI设备: {openbci_ports}")
    return openbci_ports

def check_signal_quality(data, sampling_rate):
    """
    检查信号质量
    
    参数:
        data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
        sampling_rate (int): 采样率(Hz)
        
    返回:
        dict: 信号质量指标
    """
    if data is None or data.size == 0:
        logger.error("无法检查信号质量: 数据为空")
        return None
    
    try:
        channels, samples = data.shape
        
        # 结果字典
        quality = {
            'overall': 'unknown',  # 整体质量
            'per_channel': {},     # 每个通道的质量
            'metrics': {}          # 具体指标
        }
        
        # 计算整体指标
        # 1. 信噪比估计
        signal_power = np.mean(np.var(data, axis=1))
        # 高通滤波去除低频漂移，保留高频噪声用于估计噪声功率
        b, a = signal.butter(4, 0.5/(sampling_rate/2), 'highpass')
        high_freq = np.zeros_like(data)
        for ch in range(channels):
            high_freq[ch] = signal.filtfilt(b, a, data[ch])
        noise_power = np.mean(np.var(high_freq, axis=1))
        
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
        else:
            snr = float('inf')
        
        # 2. 平均绝对振幅
        mean_abs_amplitude = np.mean(np.abs(data))
        
        # 3. 基线漂移
        baseline_drift = np.mean([np.max(data[ch]) - np.min(data[ch]) for ch in range(channels)])
        
        # 4. 通道间相关性
        channel_corr = np.zeros((channels, channels))
        for i in range(channels):
            for j in range(channels):
                if i != j:
                    corr = np.corrcoef(data[i], data[j])[0, 1]
                    channel_corr[i, j] = corr
        
        mean_corr = np.mean(np.abs(channel_corr[np.triu_indices(channels, k=1)]))
        
        # 存储整体指标
        quality['metrics'] = {
            'snr': float(snr),
            'mean_amplitude': float(mean_abs_amplitude),
            'baseline_drift': float(baseline_drift),
            'mean_correlation': float(mean_corr)
        }
        
        # 计算每个通道的质量
        for ch in range(channels):
            ch_data = data[ch]
            
            # 通道振幅
            amplitude = np.mean(np.abs(ch_data))
            
            # 通道噪声（高频分量）
            ch_noise = high_freq[ch]
            noise_amp = np.mean(np.abs(ch_noise))
            
            # 计算通道频谱特征
            if samples > 10:  # 需要足够的样本来计算频谱
                freqs, psd = signal.welch(ch_data, fs=sampling_rate, nperseg=min(256, samples))
                
                # 计算各频段能量占比
                delta_idx = np.logical_and(freqs >= 0.5, freqs < 4)
                theta_idx = np.logical_and(freqs >= 4, freqs < 8)
                alpha_idx = np.logical_and(freqs >= 8, freqs < 13)
                beta_idx = np.logical_and(freqs >= 13, freqs < 30)
                gamma_idx = np.logical_and(freqs >= 30, freqs <= 50)
                
                delta_power = np.sum(psd[delta_idx]) if np.any(delta_idx) else 0
                theta_power = np.sum(psd[theta_idx]) if np.any(theta_idx) else 0
                alpha_power = np.sum(psd[alpha_idx]) if np.any(alpha_idx) else 0
                beta_power = np.sum(psd[beta_idx]) if np.any(beta_idx) else 0
                gamma_power = np.sum(psd[gamma_idx]) if np.any(gamma_idx) else 0
                
                total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                if total_power > 0:
                    freq_ratios = {
                        'delta_ratio': delta_power / total_power,
                        'theta_ratio': theta_power / total_power,
                        'alpha_ratio': alpha_power / total_power,
                        'beta_ratio': beta_power / total_power,
                        'gamma_ratio': gamma_power / total_power
                    }
                else:
                    freq_ratios = {
                        'delta_ratio': 0,
                        'theta_ratio': 0,
                        'alpha_ratio': 0,
                        'beta_ratio': 0,
                        'gamma_ratio': 0
                    }
            else:
                freq_ratios = {
                    'delta_ratio': 0,
                    'theta_ratio': 0,
                    'alpha_ratio': 0,
                    'beta_ratio': 0,
                    'gamma_ratio': 0
                }
            
            # 通道的信噪比
            ch_snr = float('inf') if noise_amp == 0 else 10 * np.log10(amplitude / noise_amp)
            
            # 综合质量评估
            if ch_snr > 10 and amplitude > 1.0:
                quality_label = 'good'
            elif ch_snr > 5 and amplitude > 0.5:
                quality_label = 'fair'
            else:
                quality_label = 'poor'
            
            # 存储通道指标
            quality['per_channel'][ch] = {
                'quality': quality_label,
                'amplitude': float(amplitude),
                'noise': float(noise_amp),
                'snr': float(ch_snr),
                **{k: float(v) for k, v in freq_ratios.items()}
            }
        
        # 整体信号质量评估
        good_channels = sum(1 for ch in quality['per_channel'] if quality['per_channel'][ch]['quality'] == 'good')
        fair_channels = sum(1 for ch in quality['per_channel'] if quality['per_channel'][ch]['quality'] == 'fair')
        
        if good_channels >= 0.7 * channels:
            quality['overall'] = 'good'
        elif (good_channels + fair_channels) >= 0.7 * channels:
            quality['overall'] = 'fair'
        else:
            quality['overall'] = 'poor'
        
        return quality
        
    except Exception as e:
        logger.error(f"检查信号质量时发生错误: {e}")
        return None

def detect_flat_channels(data, threshold=0.1, window_size=500):
    """
    检测信号平坦的通道，可能表示电极脱落或短路
    
    参数:
        data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
        threshold (float): 平坦检测阈值
        window_size (int): 检测窗口大小
        
    返回:
        list: 平坦通道的索引列表
    """
    if data is None or data.size == 0:
        return []
    
    try:
        channels, samples = data.shape
        flat_channels = []
        
        for ch in range(channels):
            # 计算窗口标准差
            std_values = []
            for i in range(0, samples, window_size):
                end = min(i + window_size, samples)
                if end - i >= 10:  # 需要足够的样本来计算标准差
                    std_values.append(np.std(data[ch, i:end]))
            
            # 如果所有窗口的标准差都小于阈值，则认为是平坦通道
            if std_values and np.mean(std_values) < threshold:
                flat_channels.append(ch)
        
        return flat_channels
        
    except Exception as e:
        logger.error(f"检测平坦通道时发生错误: {e}")
        return []

def detect_artifacts(data, sampling_rate):
    """
    检测数据中的伪迹
    
    参数:
        data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
        sampling_rate (int): 采样率(Hz)
        
    返回:
        list: 伪迹时间点的列表，每个元素为 (开始样本索引, 结束样本索引, 类型)
    """
    if data is None or data.size == 0:
        return []
    
    try:
        channels, samples = data.shape
        artifacts = []
        
        # 1. 检测眨眼伪迹（主要影响前额电极）
        # 前额电极通常是前几个通道
        frontal_channels = min(4, channels)
        for ch in range(frontal_channels):
            # 眨眼特征：短时大振幅波动
            window_size = int(0.5 * sampling_rate)  # 500ms窗口
            
            for i in range(0, samples - window_size, window_size // 2):  # 50%重叠的窗口
                window = data[ch, i:i+window_size]
                
                # 计算窗口内数据的振幅
                window_amp = np.max(window) - np.min(window)
                
                # 计算窗口内数据的标准差
                window_std = np.std(window)
                
                # 大振幅波动可能是眨眼
                if window_amp > 5 * np.std(data[ch]) and window_std > 3 * np.std(data[ch]):
                    # 查找波动的精确位置
                    peak_idx = np.argmax(np.abs(window))
                    start = max(0, i + peak_idx - int(0.1 * sampling_rate))
                    end = min(samples, i + peak_idx + int(0.3 * sampling_rate))
                    
                    # 检查是否与已有伪迹重叠
                    overlap = False
                    for a_start, a_end, a_type in artifacts:
                        if not (end <= a_start or start >= a_end):
                            overlap = True
                            break
                    
                    if not overlap:
                        artifacts.append((start, end, 'blink'))
        
        # 2. 检测肌肉伪迹（高频信号）
        for ch in range(channels):
            # 高通滤波提取高频信号（>30Hz）
            b, a = signal.butter(4, 30/(sampling_rate/2), 'high')
            high_freq = signal.filtfilt(b, a, data[ch])
            
            # 计算高频信号的包络
            envelope = np.abs(high_freq)
            
            # 计算包络的均值和标准差
            env_mean = np.mean(envelope)
            env_std = np.std(envelope)
            
            # 检测高频突发
            window_size = int(0.5 * sampling_rate)
            for i in range(0, samples - window_size, window_size // 2):
                win_env = envelope[i:i+window_size]
                win_env_mean = np.mean(win_env)
                
                # 突发的高频信号可能是肌肉伪迹
                if win_env_mean > env_mean + 3 * env_std:
                    # 找到持续时间
                    onset = i + np.argmax(win_env > env_mean + 2 * env_std)
                    offset = onset
                    while offset < i + window_size and offset < samples and envelope[offset] > env_mean + 2 * env_std:
                        offset += 1
                    
                    # 检查是否与已有伪迹重叠
                    overlap = False
                    for a_start, a_end, a_type in artifacts:
                        if not (offset <= a_start or onset >= a_end):
                            overlap = True
                            break
                    
                    if not overlap and offset - onset > int(0.05 * sampling_rate):  # 至少持续50ms
                        artifacts.append((onset, offset, 'muscle'))
        
        # 3. 检测电极弹出（突然的异常值）
        for ch in range(channels):
            # 计算均值和标准差
            ch_mean = np.mean(data[ch])
            ch_std = np.std(data[ch])
            
            # 设置异常值阈值
            threshold = 5 * ch_std
            
            # 标记超过阈值的样本
            anomalies = np.where(np.abs(data[ch] - ch_mean) > threshold)[0]
            
            # 将连续的异常值分组
            if len(anomalies) > 0:
                groups = np.split(anomalies, np.where(np.diff(anomalies) > 1)[0] + 1)
                
                for group in groups:
                    if len(group) > 0:
                        start = group[0]
                        end = group[-1] + 1
                        
                        # 检查是否与已有伪迹重叠
                        overlap = False
                        for a_start, a_end, a_type in artifacts:
                            if not (end <= a_start or start >= a_end):
                                overlap = True
                                break
                        
                        if not overlap:
                            artifacts.append((start, end, 'electrode_pop'))
        
        # 4. 检测运动伪迹（低频漂移）
        for ch in range(channels):
            # 低通滤波提取低频信号（<1Hz）
            b, a = signal.butter(4, 1/(sampling_rate/2), 'low')
            low_freq = signal.filtfilt(b, a, data[ch])
            
            # 计算低频信号的导数（斜率）
            deriv = np.diff(low_freq)
            deriv = np.append(deriv, 0)  # 保持长度一致
            
            # 计算导数的均值和标准差
            deriv_mean = np.mean(deriv)
            deriv_std = np.std(deriv)
            
            # 检测斜率突变
            window_size = int(1.0 * sampling_rate)  # 1秒窗口
            for i in range(0, samples - window_size, window_size // 2):
                win_deriv = deriv[i:i+window_size]
                
                # 计算窗口内斜率的标准差
                win_deriv_std = np.std(win_deriv)
                
                # 斜率变化大可能是运动伪迹
                if win_deriv_std > 2 * deriv_std:
                    # 检查是否与已有伪迹重叠
                    overlap = False
                    for a_start, a_end, a_type in artifacts:
                        if not (i + window_size <= a_start or i >= a_end):
                            overlap = True
                            break
                    
                    if not overlap:
                        artifacts.append((i, i + window_size, 'movement'))
        
        # 合并重叠的伪迹
        if artifacts:
            artifacts.sort(key=lambda x: x[0])
            merged_artifacts = [artifacts[0]]
            
            for start, end, artifact_type in artifacts[1:]:
                prev_start, prev_end, prev_type = merged_artifacts[-1]
                
                if start <= prev_end:
                    # 重叠，合并
                    new_end = max(end, prev_end)
                    # 如果类型不同，优先选择优先级高的
                    type_priority = {'electrode_pop': 0, 'blink': 1, 'muscle': 2, 'movement': 3}
                    new_type = prev_type if type_priority.get(prev_type, 10) < type_priority.get(artifact_type, 10) else artifact_type
                    merged_artifacts[-1] = (prev_start, new_end, new_type)
                else:
                    # 不重叠，添加新伪迹
                    merged_artifacts.append((start, end, artifact_type))
            
            return merged_artifacts
        else:
            return []
        
    except Exception as e:
        logger.error(f"检测伪迹时发生错误: {e}")
        return []

def compute_impedance_simulation(channels, electrode_quality=None):
    """
    模拟计算电极阻抗（仅用于模拟模式）
    
    参数:
        channels (int): 通道数量
        electrode_quality (list): 每个通道的电极质量，0.0-1.0的浮点数
        
    返回:
        dict: 每个通道的阻抗值(欧姆)
    """
    impedances = {}
    
    if electrode_quality is None:
        # 随机生成电极质量
        electrode_quality = [np.random.uniform(0.3, 1.0) for _ in range(channels)]
    
    for ch in range(channels):
        quality = electrode_quality[ch] if ch < len(electrode_quality) else np.random.uniform(0.3, 1.0)
        
        # 质量越高，阻抗越低
        if quality > 0.8:
            # 优质连接: 5k-20k Ohm
            impedance = np.random.uniform(5000, 20000)
        elif quality > 0.5:
            # 中等连接: 20k-100k Ohm
            impedance = np.random.uniform(20000, 100000)
        else:
            # 差质量连接: 100k-1.5M Ohm
            impedance = np.random.uniform(100000, 1500000)
        
        impedances[ch] = impedance
    
    return impedances
