"""
会话管理器 - 管理TangiEEG会话状态和数据
"""

import time
import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd

class SessionManager:
    """会话管理器类，用于管理TangiEEG会话状态和数据"""
    
    def __init__(self):
        """初始化会话管理器"""
        self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.start_time = time.time()
        self.session_data = {
            "metadata": {
                "session_id": self.session_id,
                "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "device_type": None,
                "channel_count": 0,
                "sample_rate": 0,
                "notes": ""
            },
            "configuration": {},
            "events": [],
            "markers": [],
            "decoding_results": []
        }
        self.data_buffer = {}
        self.is_recording = False
        self.recording_start_time = None
        self.is_processing = False
        self.last_save_time = time.time()
        
        # 创建会话目录
        self.session_dir = Path(f"data/raw/{self.session_id}")
        
    def start_recording(self, device_type, channel_count, sample_rate, configuration=None):
        """
        开始数据记录
        
        Args:
            device_type: 设备类型
            channel_count: 通道数量
            sample_rate: 采样率
            configuration: 设备配置
        """
        if self.is_recording:
            return False
        
        self.is_recording = True
        self.recording_start_time = time.time()
        
        # 更新元数据
        self.session_data["metadata"]["device_type"] = device_type
        self.session_data["metadata"]["channel_count"] = channel_count
        self.session_data["metadata"]["sample_rate"] = sample_rate
        
        if configuration:
            self.session_data["configuration"] = configuration
        
        # 记录开始事件
        self.add_event("recording_started", "开始数据记录")
        
        # 确保会话目录存在
        os.makedirs(self.session_dir, exist_ok=True)
        
        return True
    
    def stop_recording(self):
        """停止数据记录"""
        if not self.is_recording:
            return False
        
        self.is_recording = False
        duration = time.time() - self.recording_start_time
        
        # 记录停止事件
        self.add_event("recording_stopped", f"停止数据记录，持续时间: {duration:.2f} 秒")
        
        # 保存会话数据
        self.save_session()
        
        return True
    
    def add_data(self, timestamp, channel_data):
        """
        添加采集到的数据
        
        Args:
            timestamp: 数据时间戳
            channel_data: 通道数据字典 {channel_id: value}
        """
        if not self.is_recording:
            return False
        
        # 将数据添加到缓冲区
        self.data_buffer[timestamp] = channel_data
        
        # 如果缓冲区过大，保存到文件
        if len(self.data_buffer) > 1000:  # 每1000个样本保存一次
            self.flush_data_buffer()
        
        return True
    
    def flush_data_buffer(self):
        """将数据缓冲区保存到文件"""
        if not self.data_buffer:
            return False
        
        # 将数据转换为DataFrame
        df = pd.DataFrame.from_dict(self.data_buffer, orient='index')
        
        # 保存到CSV文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.session_dir / f"data_{timestamp}.csv"
        df.to_csv(file_path)
        
        # 清空缓冲区
        self.data_buffer = {}
        
        return True
    
    def add_event(self, event_type, description, data=None):
        """
        添加事件
        
        Args:
            event_type: 事件类型
            description: 事件描述
            data: 事件相关数据
        """
        event = {
            "timestamp": time.time(),
            "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": event_type,
            "description": description
        }
        
        if data:
            event["data"] = data
        
        self.session_data["events"].append(event)
        
        return True
    
    def add_marker(self, marker_type, label, timestamp=None):
        """
        添加数据标记
        
        Args:
            marker_type: 标记类型
            label: 标记标签
            timestamp: 标记时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        marker = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "type": marker_type,
            "label": label
        }
        
        self.session_data["markers"].append(marker)
        
        return True
    
    def add_decoding_result(self, text, confidence, timestamp=None):
        """
        添加解码结果
        
        Args:
            text: 解码文本
            confidence: 置信度
            timestamp: 解码时间戳，如果为None则使用当前时间
        """
        if timestamp is None:
            timestamp = time.time()
        
        result = {
            "timestamp": timestamp,
            "datetime": datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "confidence": confidence
        }
        
        self.session_data["decoding_results"].append(result)
        
        return True
    
    def save_session(self):
        """保存会话数据"""
        # 确保会话目录存在
        os.makedirs(self.session_dir, exist_ok=True)
        
        # 保存会话元数据
        metadata_file = self.session_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        # 保存缓冲区数据
        self.flush_data_buffer()
        
        # 更新最后保存时间
        self.last_save_time = time.time()
        
        return True
    
    def load_session(self, session_id):
        """
        加载已有会话
        
        Args:
            session_id: 会话ID
        """
        session_dir = Path(f"data/raw/{session_id}")
        
        if not session_dir.exists():
            return False
        
        # 加载会话元数据
        metadata_file = session_dir / "metadata.json"
        if not metadata_file.exists():
            return False
        
        with open(metadata_file, 'r') as f:
            self.session_data = json.load(f)
        
        self.session_id = session_id
        self.session_dir = session_dir
        
        return True
    
    def get_session_info(self):
        """获取会话信息摘要"""
        info = {
            "session_id": self.session_id,
            "start_time": self.session_data["metadata"]["start_time"],
            "device_type": self.session_data["metadata"]["device_type"],
            "recording_status": "正在录制" if self.is_recording else "未录制",
            "event_count": len(self.session_data["events"]),
            "marker_count": len(self.session_data["markers"]),
            "decoding_result_count": len(self.session_data["decoding_results"])
        }
        
        # 计算会话持续时间
        start_timestamp = self.start_time
        duration = time.time() - start_timestamp
        hours, remainder = divmod(duration, 3600)
        minutes, seconds = divmod(remainder, 60)
        info["duration"] = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
        
        return info
    
    def get_recent_events(self, count=10):
        """获取最近的事件"""
        events = self.session_data["events"]
        return events[-count:] if len(events) > count else events
    
    def get_recent_decoding_results(self, count=10):
        """获取最近的解码结果"""
        results = self.session_data["decoding_results"]
        return results[-count:] if len(results) > count else results
    
    def set_session_note(self, note):
        """设置会话备注"""
        self.session_data["metadata"]["notes"] = note
        return True
    
    def get_session_note(self):
        """获取会话备注"""
        return self.session_data["metadata"]["notes"]
