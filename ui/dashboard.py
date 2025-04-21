"""
仪表盘组件 - 提供系统概览界面
"""

import time
import random
from datetime import datetime

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd

def generate_sample_data(num_points=100, num_channels=8):
    """生成样例脑电数据用于演示"""
    # 创建时间轴
    t = np.linspace(0, 10, num_points)
    
    # 为每个通道生成随机信号
    channels = {}
    for i in range(1, num_channels + 1):
        # 生成正弦波加上随机噪声
        signal = np.sin(i * t) + np.random.normal(0, 0.2, num_points)
        channels[f"Channel {i}"] = signal
    
    return pd.DataFrame(channels, index=t)

def render_dashboard():
    """渲染仪表盘界面"""
    st.markdown("<h2 class='sub-header'>系统仪表盘</h2>", unsafe_allow_html=True)
    
    # 系统状态卡片
    col1, col2, col3 = st.columns(3)
    
    with col1:
        render_system_status_card()
    
    with col2:
        render_signal_quality_card()
    
    with col3:
        render_decoding_stats_card()
    
    # 信号预览和解码结果
    col1, col2 = st.columns([2, 1])
    
    with col1:
        render_signal_preview()
    
    with col2:
        render_text_output()
    
    # 系统活动日志
    render_activity_log()

def render_system_status_card():
    """渲染系统状态卡片"""
    st.markdown("### 系统状态")
    
    # 设备状态
    device_status = "在线" if st.session_state.device_connected else "离线"
    device_color = "#4CAF50" if st.session_state.device_connected else "#f44336"
    
    # 会话信息
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 内存使用
    memory_usage = random.randint(20, 80)  # 模拟内存使用率
    
    # 构建状态卡片
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 10px; height: 10px; border-radius: 50%; background-color: {device_color}; margin-right: 10px;"></div>
            <div><b>设备状态:</b> {device_status}</div>
        </div>
        <div style="margin-bottom: 5px;"><b>当前时间:</b> {current_time}</div>
        <div style="margin-bottom: 5px;"><b>操作模式:</b> {st.session_state.current_mode}</div>
        <div style="margin-bottom: 5px;"><b>内存使用:</b> {memory_usage}%</div>
        <div><b>运行时间:</b> 00:45:32</div>
    </div>
    """, unsafe_allow_html=True)

def render_signal_quality_card():
    """渲染信号质量卡片"""
    st.markdown("### 信号质量")
    
    # 根据设备类型确定通道数和名称
    device_type = st.session_state.get('device_type', 'Daisy')  # 默认为Daisy
    
    # 定义标准10-20系统电极位置名称
    channel_positions = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                       "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", 
                       "Oz", "O2", "A1", "A2"]
    
    # 根据设备类型确定通道数
    channel_count = 8 if device_type == "Cyton" else 4 if device_type == "Ganglion" else 16 if device_type == "Daisy" else 8
    
    # 选择通道名称
    channels = channel_positions[:channel_count]
    
    # 模拟信号质量数据
    quality_values = [random.randint(60, 100) for _ in range(len(channels))]
    
    # 创建水平条形图
    fig = go.Figure()
    
    for i, (channel, quality) in enumerate(zip(channels, quality_values)):
        color = "#4CAF50" if quality >= 80 else "#FFC107" if quality >= 60 else "#F44336"
        
        fig.add_trace(go.Bar(
            y=[channel],
            x=[quality],
            orientation='h',
            marker=dict(color=color),
            text=[f"{quality}%"],
            textposition='auto',
            name=channel
        ))
    
    # 为16通道时提供更大的高度
    chart_height = 300 if channel_count <= 8 else 450 if channel_count <= 16 else 600
    
    fig.update_layout(
        height=chart_height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="质量 (%)",
            range=[0, 100]
        ),
        yaxis=dict(
            title=""
        ),
        barmode='group',
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_decoding_stats_card():
    """渲染解码统计卡片"""
    st.markdown("### 解码统计")
    
    # 模拟解码统计数据
    stats = {
        "解码准确率": f"{random.randint(75, 95)}%",
        "信噪比": f"{random.randint(15, 30)} dB",
        "处理延迟": f"{random.randint(50, 200)} ms",
        "解码速度": f"{random.randint(8, 25)} 字符/分钟",
        "置信度": f"{random.randint(60, 95)}%"
    }
    
    # 构建统计卡片
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px;">
        <div style="margin-bottom: 8px;"><b>解码准确率:</b> {stats['解码准确率']}</div>
        <div style="margin-bottom: 8px;"><b>信噪比:</b> {stats['信噪比']}</div>
        <div style="margin-bottom: 8px;"><b>处理延迟:</b> {stats['处理延迟']}</div>
        <div style="margin-bottom: 8px;"><b>解码速度:</b> {stats['解码速度']}</div>
        <div><b>置信度:</b> {stats['置信度']}</div>
    </div>
    """, unsafe_allow_html=True)

def render_signal_preview():
    """渲染信号预览"""
    st.markdown("### 实时信号预览")
    
    # 根据设备类型确定通道数
    device_type = st.session_state.get('device_type', 'Daisy')  # 默认为Daisy
    channel_count = 8 if device_type == "Cyton" else 4 if device_type == "Ganglion" else 16 if device_type == "Daisy" else 8
    
    # 生成样例数据
    df = generate_sample_data(num_points=200, num_channels=channel_count)
    
    # 创建多通道线图
    fig = go.Figure()
    
    # 定义标准10-20系统电极位置名称
    channel_positions = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                       "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", 
                       "Oz", "O2", "A1", "A2"]
    
    # 为了更好的可视化效果，对通道进行偏移
    offset = 0
    for i, col in enumerate(df.columns):
        offset += 2
        # 使用标准通道名称（如果有）
        channel_name = channel_positions[i] if i < len(channel_positions) else col
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df[col] + offset,
            mode='lines',
            name=channel_name
        ))
    
    # 为16通道时提供更大的高度
    chart_height = 400 if channel_count <= 8 else 600 if channel_count <= 16 else 800
    
    fig.update_layout(
        height=chart_height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="时间 (秒)"
        ),
        yaxis=dict(
            title="振幅 (µV)",
            showticklabels=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_text_output():
    """渲染文本输出"""
    st.markdown("### 解码结果")
    
    # 创建文本显示区域
    text_container = st.container()
    
    with text_container:
        st.markdown("""
        <div style="border: 1px solid #ddd; border-radius: 5px; padding: 15px; height: 350px; overflow-y: auto;">
            <p style="margin-bottom: 10px;">我想要...</p>
            <p style="margin-bottom: 10px;">帮助我...</p>
            <p style="margin-bottom: 10px; color: #2196F3;">正在解码...</p>
            <p style="margin-bottom: 10px;">打开灯...</p>
            <p style="margin-bottom: 10px;">打开窗户...</p>
            <p style="margin-bottom: 0px; color: #2196F3;">正在解码...</p>
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.button("清除文本", key="clear_text")
    with col2:
        st.button("保存文本", key="save_text")

def render_activity_log():
    """渲染活动日志"""
    st.markdown("### 系统活动")
    
    # 模拟活动日志
    activities = [
        {"timestamp": "10:45:32", "message": "已连接到设备", "type": "success"},
        {"timestamp": "10:45:35", "message": "已开始数据采集", "type": "info"},
        {"timestamp": "10:46:02", "message": "检测到信号质量较低", "type": "warning"},
        {"timestamp": "10:46:15", "message": "应用带通滤波器 (1-50 Hz)", "type": "info"},
        {"timestamp": "10:46:30", "message": "开始解码", "type": "info"},
        {"timestamp": "10:47:45", "message": "解码结果: '我想要...'", "type": "success"},
        {"timestamp": "10:48:10", "message": "解码结果: '帮助我...'", "type": "success"},
        {"timestamp": "10:48:55", "message": "解码结果: '打开灯...'", "type": "success"}
    ]
    
    # 创建活动日志表格
    df = pd.DataFrame(activities)
    
    # 给不同类型的消息添加不同的样式
    def highlight_type(row):
        if row['type'] == 'success':
            return ['background-color: #e8f5e9'] * len(row)
        elif row['type'] == 'warning':
            return ['background-color: #fff8e1'] * len(row)
        elif row['type'] == 'error':
            return ['background-color: #ffebee'] * len(row)
        else:
            return ['background-color: #e1f5fe'] * len(row)
    
    # 显示表格，不显示类型列
    styled_df = df[['timestamp', 'message']].style.apply(highlight_type, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=200)
