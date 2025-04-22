"""
数据记录模块 - 将采集的脑电数据保存为标准格式文件(HDF5/CSV)
"""

import os
import time
import logging
import datetime
import numpy as np
import pandas as pd
import h5py
from pathlib import Path

# 创建logger
logger = logging.getLogger('data_recorder')

class DataRecorder:
    """EEG数据记录器，将数据保存为文件"""
    
    def __init__(self, save_dir=None, format='csv', channels=16, channel_names=None):
        """
        初始化数据记录器
        
        参数:
            save_dir (str): 保存目录路径，如果为None则使用默认路径
            format (str): 保存格式，'csv'或'hdf5'
            channels (int): 通道数量
            channel_names (list): 通道名称列表
        """
        self.channels = channels
        self.format = format.lower()
        
        # 设置默认保存目录
        if save_dir is None:
            # 使用项目根目录下的数据目录
            project_root = Path(__file__).parent.parent.absolute()
            save_dir = project_root / 'data' / 'raw'
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置默认通道名称
        if channel_names is None:
            # 使用标准10-20系统电极位置名称
            standard_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                              "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", 
                              "Oz", "O2", "A1", "A2"]
            self.channel_names = standard_names[:channels]
        else:
            self.channel_names = channel_names[:channels]
        
        # 记录会话
        self.current_session = None
        self.session_start_time = None
        self.recording = False
        self.data_buffer = []
        self.timestamp_buffer = []
        
        logger.info(f"数据记录器初始化: 保存目录={save_dir}, 格式={format}, 通道数={channels}")
    
    def start_recording(self, session_name=None, metadata=None):
        """
        开始记录会话
        
        参数:
            session_name (str): 会话名称，如果为None则自动生成
            metadata (dict): 会话元数据
            
        返回:
            str: 会话ID
        """
        if self.recording:
            logger.warning("记录已经在进行中，请先停止当前记录")
            return self.current_session
            
        # 生成会话ID
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        if session_name:
            self.current_session = f"{session_name}_{timestamp}"
        else:
            self.current_session = f"session_{timestamp}"
        
        self.session_start_time = time.time()
        self.recording = True
        self.data_buffer = []
        self.timestamp_buffer = []
        
        # 保存会话元数据
        self.metadata = metadata or {}
        self.metadata.update({
            'session_id': self.current_session,
            'start_time': self.session_start_time,
            'channels': self.channels,
            'channel_names': self.channel_names,
        })
        
        logger.info(f"开始记录会话: {self.current_session}")
        return self.current_session
    
    def add_data(self, data, timestamp=None):
        """
        添加数据到记录缓冲区
        
        参数:
            data (numpy.ndarray): 形状为 (channels, samples) 的数据数组
            timestamp (float): 数据采集时间戳，如果为None则使用当前时间
            
        返回:
            bool: 是否成功添加
        """
        if not self.recording:
            logger.warning("记录未启动，无法添加数据")
            return False
            
        if timestamp is None:
            timestamp = time.time()
            
        # 验证数据形状
        if data.shape[0] != self.channels:
            logger.error(f"数据通道数不匹配: 预期 {self.channels}, 实际 {data.shape[0]}")
            return False
            
        # 添加到缓冲区
        self.data_buffer.append(data)
        self.timestamp_buffer.append(timestamp)
        
        return True
    
    def stop_recording(self, save=True):
        """
        停止记录会话
        
        参数:
            save (bool): 是否保存数据到文件
            
        返回:
            str: 如果保存，则返回保存的文件路径；否则返回None
        """
        if not self.recording:
            logger.warning("记录未启动")
            return None
            
        self.recording = False
        session_duration = time.time() - self.session_start_time
        
        logger.info(f"停止记录会话: {self.current_session}, 持续时间: {session_duration:.2f}秒")
        
        if save and self.data_buffer:
            return self.save_data()
        else:
            logger.info("数据未保存")
            return None
    
    def save_data(self):
        """
        将缓冲区数据保存到文件
        
        返回:
            str: 保存的文件路径
        """
        if not self.data_buffer:
            logger.warning("缓冲区为空，无数据可保存")
            return None
            
        # 确定文件路径和名称
        file_name = f"{self.current_session}.{self.format}"
        file_path = self.save_dir / file_name
        
        try:
            # 合并所有数据
            all_data = np.hstack(self.data_buffer)
            all_timestamps = np.array(self.timestamp_buffer)
            
            # 根据格式保存
            if self.format == 'csv':
                return self._save_as_csv(file_path, all_data, all_timestamps)
            elif self.format == 'hdf5':
                return self._save_as_hdf5(file_path, all_data, all_timestamps)
            else:
                logger.error(f"不支持的文件格式: {self.format}")
                return None
                
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return None
    
    def _save_as_csv(self, file_path, data, timestamps):
        """将数据保存为CSV格式"""
        try:
            # 转置数据使通道成为列
            data_T = data.T
            
            # 创建数据帧
            df = pd.DataFrame(data_T, columns=self.channel_names)
            
            # 添加时间戳列
            time_index = np.arange(data_T.shape[0])
            if len(timestamps) > 1:
                # 插值计算每个样本的时间戳
                first_ts = timestamps[0]
                samples_per_timestamp = data.shape[1] / len(timestamps)
                interp_timestamps = [first_ts + (i / samples_per_timestamp) * (timestamps[-1] - first_ts) 
                                  for i in range(data_T.shape[0])]
                df['timestamp'] = interp_timestamps
            else:
                # 只有一个时间戳，直接使用
                df['timestamp'] = timestamps[0]
            
            # 添加相对时间列（秒）
            df['time'] = (df['timestamp'] - df['timestamp'].iloc[0])
            
            # 保存为CSV
            df.to_csv(file_path, index=False)
            
            # 保存元数据为单独的CSV文件
            meta_path = str(file_path).replace('.csv', '_metadata.csv')
            pd.DataFrame([self.metadata]).to_csv(meta_path, index=False)
            
            logger.info(f"数据已保存为CSV文件: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存CSV文件失败: {e}")
            return None
    
    def _save_as_hdf5(self, file_path, data, timestamps):
        """将数据保存为HDF5格式"""
        try:
            with h5py.File(file_path, 'w') as f:
                # 保存数据
                f.create_dataset('eeg_data', data=data, dtype='float32')
                f.create_dataset('timestamps', data=timestamps, dtype='float64')
                
                # 保存通道名称
                channel_names_ascii = [name.encode('ascii', 'ignore') for name in self.channel_names]
                f.create_dataset('channel_names', data=channel_names_ascii)
                
                # 保存元数据
                meta_group = f.create_group('metadata')
                for key, value in self.metadata.items():
                    if isinstance(value, (str, bool, int, float, list, tuple)):
                        if isinstance(value, str):
                            meta_group.attrs[key] = value.encode('utf-8')
                        else:
                            meta_group.attrs[key] = value
            
            logger.info(f"数据已保存为HDF5文件: {file_path}")
            return str(file_path)
            
        except Exception as e:
            logger.error(f"保存HDF5文件失败: {e}")
            return None
    
    def clear_buffer(self):
        """清空数据缓冲区"""
        self.data_buffer = []
        self.timestamp_buffer = []
        logger.info("数据缓冲区已清空")
    
    def get_buffer_stats(self):
        """
        获取缓冲区统计信息
        
        返回:
            dict: 缓冲区统计信息
        """
        if not self.data_buffer:
            return {
                'chunks': 0,
                'total_samples': 0,
                'duration': 0,
                'memory_usage': 0
            }
            
        total_samples = sum(d.shape[1] for d in self.data_buffer)
        
        if self.timestamp_buffer:
            duration = self.timestamp_buffer[-1] - self.timestamp_buffer[0]
        else:
            duration = 0
            
        # 估计内存使用
        memory_usage = sum(d.nbytes for d in self.data_buffer) / (1024 * 1024)  # MB
        
        return {
            'chunks': len(self.data_buffer),
            'total_samples': total_samples,
            'duration': duration,
            'memory_usage': memory_usage
        }
    
    def load_data(self, file_path):
        """
        加载保存的数据文件
        
        参数:
            file_path (str): 文件路径
            
        返回:
            tuple: (数据数组, 时间戳数组, 元数据)
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return None, None, None
            
        try:
            if file_path.suffix.lower() == '.csv':
                return self._load_from_csv(file_path)
            elif file_path.suffix.lower() in ['.h5', '.hdf5']:
                return self._load_from_hdf5(file_path)
            else:
                logger.error(f"不支持的文件格式: {file_path.suffix}")
                return None, None, None
                
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return None, None, None
    
    def _load_from_csv(self, file_path):
        """从CSV文件加载数据"""
        try:
            # 加载数据
            df = pd.read_csv(file_path)
            
            # 提取时间戳
            timestamps = df['timestamp'].values if 'timestamp' in df.columns else None
            
            # 提取EEG数据（排除非通道列）
            non_eeg_cols = ['timestamp', 'time']
            eeg_cols = [col for col in df.columns if col not in non_eeg_cols]
            
            # 验证通道数量
            if len(eeg_cols) != self.channels:
                logger.warning(f"通道数量不匹配: 文件中 {len(eeg_cols)}, 预期 {self.channels}")
            
            # 提取EEG数据并转置为 (channels, samples) 格式
            eeg_data = df[eeg_cols].values.T
            
            # 加载元数据文件（如果存在）
            meta_path = str(file_path).replace('.csv', '_metadata.csv')
            metadata = {}
            if os.path.exists(meta_path):
                try:
                    meta_df = pd.read_csv(meta_path)
                    metadata = meta_df.iloc[0].to_dict()
                except:
                    logger.warning(f"无法加载元数据文件: {meta_path}")
            
            logger.info(f"从CSV文件加载数据: {file_path}")
            return eeg_data, timestamps, metadata
            
        except Exception as e:
            logger.error(f"从CSV文件加载数据失败: {e}")
            return None, None, None
    
    def _load_from_hdf5(self, file_path):
        """从HDF5文件加载数据"""
        try:
            with h5py.File(file_path, 'r') as f:
                # 加载数据
                eeg_data = f['eeg_data'][:]
                timestamps = f['timestamps'][:]
                
                # 加载元数据
                metadata = {}
                if 'metadata' in f:
                    for key, value in f['metadata'].attrs.items():
                        if isinstance(value, bytes):
                            metadata[key] = value.decode('utf-8')
                        else:
                            metadata[key] = value
            
            logger.info(f"从HDF5文件加载数据: {file_path}")
            return eeg_data, timestamps, metadata
            
        except Exception as e:
            logger.error(f"从HDF5文件加载数据失败: {e}")
            return None, None, None
