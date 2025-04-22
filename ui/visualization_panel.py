"""
可视化面板 - 提供脑电信号和解码结果的可视化界面
"""

import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

def render_visualization_panel():
    """渲染可视化面板"""
    st.markdown("<h2 class='sub-header'>数据可视化</h2>", unsafe_allow_html=True)
    
    # 检查设备连接状态或数据加载状态
    if not st.session_state.device_connected and not st.session_state.data_loaded:
        render_no_data_message()
        return
    
    # 选择可视化内容的选项卡
    tabs = st.tabs(["实时数据", "频谱分析", "脑地形图", "连接性分析", "解码结果"])
    
    # 实时数据选项卡
    with tabs[0]:
        render_realtime_visualization()
    
    # 频谱分析选项卡
    with tabs[1]:
        render_spectral_visualization()
    
    # 脑地形图选项卡
    with tabs[2]:
        render_topographic_visualization()
    
    # 连接性分析选项卡
    with tabs[3]:
        render_connectivity_visualization()
    
    # 解码结果选项卡
    with tabs[4]:
        render_decoding_visualization()

def render_no_data_message():
    """当没有数据时渲染提示信息"""
    st.warning("⚠️ 没有可用的EEG数据可视化")
    
    st.markdown("""
    请通过以下方式获取数据:
    
    1. **连接设备** - 在"设备连接"页面连接OpenBCI设备
    2. **载入数据** - 从"信号处理"页面上传已记录的数据
    """)
    
    # 添加一些示例图像
    st.markdown("### 示例可视化")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 脑电信号示例")
        # 使用Plotly生成一个示例图
        t = np.linspace(0, 10, 1000)
        y = np.sin(t) + np.random.normal(0, 0.1, 1000)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=y, mode='lines', name='EEG'))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("#### 频谱分析示例")
        # 使用Plotly生成一个示例频谱图
        freqs = np.linspace(0, 50, 500)
        power = 5 * np.exp(-((freqs - 10) ** 2) / 20) + np.random.normal(0, 0.05, 500)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freqs, y=power, mode='lines', fill='tozeroy'))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
        
        st.plotly_chart(fig, use_container_width=True)

def render_realtime_visualization():
    """渲染实时数据可视化"""
    st.markdown("### 实时脑电信号")
    
    # 配置选项
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 获取真实的通道数
        num_channels = 8
        if 'device_manager' in st.session_state:
            num_channels = st.session_state.device_manager.channels
        
        selected_channels = st.multiselect(
            "显示通道",
            options=["全部"] + [f"通道 {i+1}" for i in range(num_channels)],
            default=["全部"]
        )
    
    with col2:
        time_window = st.slider(
            "时间窗口 (秒)",
            min_value=1,
            max_value=30,
            value=10,
            step=1
        )
    
    with col3:
        scale = st.slider(
            "振幅缩放",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
    
    # 实时信号显示
    st.markdown("#### 原始信号")
    
    # 获取真实的模拟数据
    if 'device_manager' in st.session_state and st.session_state.device_connected:
        device_manager = st.session_state.device_manager
        sample_rate = device_manager.sample_rate
        num_samples = int(time_window * sample_rate)
        eeg_data = device_manager.get_latest_data(samples=num_samples)
        
        if eeg_data is None or eeg_data.size == 0:
            st.warning("无法获取数据，显示模拟数据")
            # 回退到模拟数据
            num_channels = 8 if 'device_manager' not in st.session_state else device_manager.channels
            num_samples = 1000
            sample_rate = 250
            time_points = np.linspace(0, num_samples / sample_rate, num_samples)
            eeg_data = np.zeros((num_channels, num_samples))
            
            for i in range(num_channels):
                # 生成基础信号
                signal = np.sin(2 * np.pi * (i + 1) * time_points) * 10
                # 添加随机噪声
                signal += np.random.normal(0, 1, num_samples)
                # 添加时变振幅
                envelope = 1 + 0.5 * np.sin(2 * np.pi * 0.2 * time_points)
                signal *= envelope
                
                eeg_data[i, :] = signal
        else:
            # 如果获取到的数据不足，做一些处理
            if eeg_data.shape[1] < num_samples:
                # 填充不足的部分
                padding = np.zeros((eeg_data.shape[0], num_samples - eeg_data.shape[1]))
                eeg_data = np.hstack((padding, eeg_data))
                
            time_points = np.linspace(0, num_samples / sample_rate, num_samples)
    else:
        # 如果设备未连接，显示模拟数据
        num_channels = 8
        num_samples = 1000
        sample_rate = 250  # Hz
        time_points = np.linspace(0, num_samples / sample_rate, num_samples)
        
        eeg_data = np.zeros((num_channels, num_samples))
        
        for i in range(num_channels):
            # 生成基础信号
            signal = np.sin(2 * np.pi * (i + 1) * time_points) * 10
            # 添加随机噪声
            signal += np.random.normal(0, 1, num_samples)
            # 添加时变振幅
            envelope = 1 + 0.5 * np.sin(2 * np.pi * 0.2 * time_points)
            signal *= envelope
            
            eeg_data[i, :] = signal
    
    # 创建一个多通道图
    fig = go.Figure()
    
    # 为每个通道添加数据
    for i in range(min(num_channels, eeg_data.shape[0])):
        if "全部" in selected_channels or f"通道 {i+1}" in selected_channels:
            # 应用缩放
            signal = eeg_data[i, :] * scale
            
            # 应用偏移以便于可视化
            offset = i * 30
            
            # 将通道数据添加到图中
            fig.add_trace(go.Scatter(
                x=time_points,
                y=signal + offset,
                mode='lines',
                name=f'通道 {i+1}',
                line=dict(width=1)
            ))
    
    # 更新布局
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="时间 (秒)",
            range=[time_points[-1] - time_window, time_points[-1]]
        ),
        yaxis=dict(
            title="振幅 (μV)",
            showticklabels=False
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 信号质量指标
    st.markdown("#### 信号质量指标")
    quality_metrics = st.columns(5)
    
    # 计算真实的信号质量指标
    if 'device_manager' in st.session_state and st.session_state.device_connected and eeg_data is not None:
        # 计算信噪比 (估计值)
        from scipy import signal as sp_signal
        # 去除低频趋势，保留高频噪声作为噪声估计
        filtered_data = np.zeros_like(eeg_data)
        for ch in range(eeg_data.shape[0]):
            b, a = sp_signal.butter(4, 0.5/(sample_rate/2), 'highpass')
            filtered_data[ch] = sp_signal.filtfilt(b, a, eeg_data[ch])
        
        signal_power = np.mean(np.var(eeg_data, axis=1))
        noise_power = np.mean(np.var(filtered_data, axis=1))
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else 0
        
        # 计算最大振幅
        max_amp = np.max(np.abs(eeg_data)) * scale
        
        # 计算抗干扰值 (模拟值)
        resistance = 85 + np.random.normal(0, 5)
        
        with quality_metrics[0]:
            st.metric("信噪比", f"{snr:.1f} dB", delta=f"{np.random.normal(0, 1):.1f} dB")
        
        with quality_metrics[1]:
            st.metric("抗干扰值", f"{resistance:.1f}%", delta=f"{np.random.normal(0, 3):.1f}%")
        
        with quality_metrics[2]:
            st.metric("最大振幅", f"{max_amp:.1f} μV", delta=None)
        
        with quality_metrics[3]:
            st.metric("采样率", f"{sample_rate} Hz", delta=None)
        
        with quality_metrics[4]:
            # 模拟阻抗值
            impedance = 10 + np.random.normal(0, 2)
            st.metric("阻抗平均值", f"{impedance:.1f} kΩ", delta=f"{np.random.normal(0, 2):.1f} kΩ")
    else:
        # 使用模拟的信号质量指标
        with quality_metrics[0]:
            st.metric("信噪比", "24.5 dB", delta="1.2 dB")
        
        with quality_metrics[1]:
            st.metric("抗干扰值", "86%", delta="3%")
        
        with quality_metrics[2]:
            st.metric("最大振幅", f"{37.2*scale:.1f} μV", delta=None)
        
        with quality_metrics[3]:
            st.metric("采样率", "250 Hz", delta=None)
        
        with quality_metrics[4]:
            st.metric("阻抗平均值", "12.7 kΩ", delta="-2.1 kΩ")
    
    # 事件标记
    st.markdown("#### 事件标记")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        event_label = st.text_input("事件标签", placeholder="输入事件描述...")
    
    with col2:
        if st.button("添加标记", use_container_width=True):
            # 初始化事件标记列表
            if 'event_markers' not in st.session_state:
                st.session_state.event_markers = []
            
            # 添加新的事件标记
            if event_label:
                new_marker = {
                    "time": datetime.now().strftime('%H:%M:%S'),
                    "label": event_label
                }
                st.session_state.event_markers.append(new_marker)
                st.success(f"已添加标记: '{event_label}' @ {new_marker['time']}")
    
    # 显示已有标记
    if 'event_markers' in st.session_state and st.session_state.event_markers:
        markers_df = pd.DataFrame(st.session_state.event_markers)
    else:
        # 使用默认的示例标记
        markers_df = pd.DataFrame([
            {"time": "10:45:32", "label": "眼睛闭合"},
            {"time": "10:45:55", "label": "眼睛睁开"},
            {"time": "10:46:21", "label": "开始想象左手运动"},
            {"time": "10:46:45", "label": "结束想象"}
        ])
    
    st.dataframe(markers_df, use_container_width=True, height=150)

def render_spectral_visualization():
    """渲染频谱分析可视化"""
    st.markdown("### 频谱分析")
    
    # 配置选项
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 获取通道数
        num_channels = 8
        if 'device_manager' in st.session_state:
            num_channels = st.session_state.device_manager.channels
            
        selected_channel = st.selectbox(
            "选择通道",
            options=[f"通道 {i+1}" for i in range(num_channels)],
            index=0
        )
    
    with col2:
        fft_window = st.selectbox(
            "FFT窗口函数",
            options=["汉宁窗", "海明窗", "布莱克曼窗", "矩形窗"],
            index=0
        )
    
    with col3:
        frequency_range = st.slider(
            "频率范围 (Hz)",
            min_value=0,
            max_value=100,
            value=(0, 50),
            step=1
        )
    
    # 频谱图
    st.markdown("#### 功率谱密度")
    
    # 获取真实数据并计算频谱
    if 'device_manager' in st.session_state and st.session_state.device_connected:
        # 获取设备管理器和采样率
        device_manager = st.session_state.device_manager
        sample_rate = device_manager.sample_rate
        
        # 获取至少2秒的数据用于频谱分析
        num_samples = int(2 * sample_rate)
        eeg_data = device_manager.get_latest_data(samples=num_samples)
        
        if eeg_data is not None and eeg_data.size > 0:
            # 将通道索引转换为数字
            channel_idx = int(selected_channel.split()[-1]) - 1
            
            # 限制通道索引在有效范围内
            if channel_idx >= 0 and channel_idx < eeg_data.shape[0]:
                # 获取所选通道的数据
                channel_data = eeg_data[channel_idx]
                
                # 计算功率谱
                from scipy import signal as sp_signal
                
                # 根据选择的窗口函数设置窗口
                if fft_window == "汉宁窗":
                    window = 'hann'
                elif fft_window == "海明窗":
                    window = 'hamming'
                elif fft_window == "布莱克曼窗":
                    window = 'blackman'
                else:  # 矩形窗
                    window = 'boxcar'
                
                # 计算功率谱密度
                freqs, power = sp_signal.welch(channel_data, fs=sample_rate, nperseg=min(256, len(channel_data)), window=window)
                
                # 限制频率范围
                mask = (freqs >= frequency_range[0]) & (freqs <= frequency_range[1])
                freqs = freqs[mask]
                power = power[mask]
                
                # 计算各频段功率
                delta_mask = (freqs >= 0.5) & (freqs < 4)
                theta_mask = (freqs >= 4) & (freqs < 8)
                alpha_mask = (freqs >= 8) & (freqs < 13)
                beta_mask = (freqs >= 13) & (freqs < 30)
                gamma_mask = (freqs >= 30)
                
                delta_power = np.sum(power[delta_mask]) if np.any(delta_mask) else 0
                theta_power = np.sum(power[theta_mask]) if np.any(theta_mask) else 0
                alpha_power = np.sum(power[alpha_mask]) if np.any(alpha_mask) else 0
                beta_power = np.sum(power[beta_mask]) if np.any(beta_mask) else 0
                gamma_power = np.sum(power[gamma_mask]) if np.any(gamma_mask) else 0
                
                total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                
                # 创建频谱图
                fig = go.Figure()
                
                # 添加频带
                if total_power > 0:
                    # 计算各频段百分比
                    delta_percent = delta_power / total_power * 100 if total_power > 0 else 0
                    theta_percent = theta_power / total_power * 100 if total_power > 0 else 0
                    alpha_percent = alpha_power / total_power * 100 if total_power > 0 else 0
                    beta_percent = beta_power / total_power * 100 if total_power > 0 else 0
                    gamma_percent = gamma_power / total_power * 100 if total_power > 0 else 0
                    
                    # 更新频带能量值
                    delta_label = f"Delta ({delta_percent:.1f}%)"
                    theta_label = f"Theta ({theta_percent:.1f}%)"
                    alpha_label = f"Alpha ({alpha_percent:.1f}%)"
                    beta_label = f"Beta ({beta_percent:.1f}%)"
                    gamma_label = f"Gamma ({gamma_percent:.1f}%)"
                    
                    # 使用各频带的数据添加频带能量
                    delta_freqs = freqs[delta_mask]
                    delta_psd = power[delta_mask]
                    if len(delta_freqs) > 0:
                        fig.add_trace(go.Scatter(
                            x=delta_freqs, y=delta_psd,
                            mode='lines',
                            fill='tozeroy',
                            name=delta_label,
                            line=dict(color='rgba(142,124,197,0.8)')
                        ))
                    
                    theta_freqs = freqs[theta_mask]
                    theta_psd = power[theta_mask]
                    if len(theta_freqs) > 0:
                        fig.add_trace(go.Scatter(
                            x=theta_freqs, y=theta_psd,
                            mode='lines',
                            fill='tozeroy',
                            name=theta_label,
                            line=dict(color='rgba(76,175,80,0.8)')
                        ))
                    
                    alpha_freqs = freqs[alpha_mask]
                    alpha_psd = power[alpha_mask]
                    if len(alpha_freqs) > 0:
                        fig.add_trace(go.Scatter(
                            x=alpha_freqs, y=alpha_psd,
                            mode='lines',
                            fill='tozeroy',
                            name=alpha_label,
                            line=dict(color='rgba(33,150,243,0.8)')
                        ))
                    
                    beta_freqs = freqs[beta_mask]
                    beta_psd = power[beta_mask]
                    if len(beta_freqs) > 0:
                        fig.add_trace(go.Scatter(
                            x=beta_freqs, y=beta_psd,
                            mode='lines',
                            fill='tozeroy',
                            name=beta_label,
                            line=dict(color='rgba(255,152,0,0.8)')
                        ))
                    
                    gamma_freqs = freqs[gamma_mask]
                    gamma_psd = power[gamma_mask]
                    if len(gamma_freqs) > 0:
                        fig.add_trace(go.Scatter(
                            x=gamma_freqs, y=gamma_psd,
                            mode='lines',
                            fill='tozeroy',
                            name=gamma_label,
                            line=dict(color='rgba(244,67,54,0.8)')
                        ))
                
                # 添加总功率
                fig.add_trace(go.Scatter(
                    x=freqs, y=power,
                    mode='lines',
                    name='总功率',
                    line=dict(color='black', width=2)
                ))
                
                # 更新布局
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=30, b=0),
                    xaxis_title="频率 (Hz)",
                    yaxis_title="功率 (μV²/Hz)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # 频带能量分布
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Delta", f"{delta_percent:.1f}%")
                
                with col2:
                    st.metric("Theta", f"{theta_percent:.1f}%")
                
                with col3:
                    st.metric("Alpha", f"{alpha_percent:.1f}%")
                
                with col4:
                    st.metric("Beta", f"{beta_percent:.1f}%")
                
                with col5:
                    st.metric("Gamma", f"{gamma_percent:.1f}%")
                
                return
    
    # 如果没有数据或设备未连接，显示模拟数据
    # 生成频率数据
    freqs = np.linspace(0, 100, 1000)
    
    # 创建典型的脑电频率节律
    delta_power = 30 * np.exp(-((freqs - 2) ** 2) / 2)   # 0.5-4 Hz
    theta_power = 15 * np.exp(-((freqs - 6) ** 2) / 3)   # 4-8 Hz
    alpha_power = 25 * np.exp(-((freqs - 10) ** 2) / 4)  # 8-13 Hz
    beta_power = 10 * np.exp(-((freqs - 20) ** 2) / 30)  # 13-30 Hz
    gamma_power = 5 * np.exp(-((freqs - 40) ** 2) / 50)  # 30-100 Hz
    
    # 添加一些随机波动
    np.random.seed(42)  # 设置随机种子以获得可重现的结果
    noise = np.random.normal(0, 0.5, freqs.shape)
    
    # 组合所有频带
    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power + noise
    
    # 创建频谱图
    fig = go.Figure()
    
    # 添加各个频带
    fig.add_trace(go.Scatter(
        x=freqs, y=delta_power,
        mode='lines', 
        fill='tozeroy',
        name='Delta (0.5-4 Hz)',
        line=dict(color='rgba(142,124,197,0.8)')
    ))
    
    fig.add_trace(go.Scatter(
        x=freqs, y=theta_power,
        mode='lines', 
        fill='tozeroy',
        name='Theta (4-8 Hz)',
        line=dict(color='rgba(76,175,80,0.8)')
    ))
    
    fig.add_trace(go.Scatter(
        x=freqs, y=alpha_power,
        mode='lines', 
        fill='tozeroy',
        name='Alpha (8-13 Hz)',
        line=dict(color='rgba(33,150,243,0.8)')
    ))
    
    fig.add_trace(go.Scatter(
        x=freqs, y=beta_power,
        mode='lines', 
        fill='tozeroy',
        name='Beta (13-30 Hz)',
        line=dict(color='rgba(255,152,0,0.8)')
    ))
    
    fig.add_trace(go.Scatter(
        x=freqs, y=gamma_power,
        mode='lines', 
        fill='tozeroy',
        name='Gamma (30-100 Hz)',
        line=dict(color='rgba(244,67,54,0.8)')
    ))
    
    # 添加总功率
    fig.add_trace(go.Scatter(
        x=freqs, y=total_power,
        mode='lines',
        name='总功率',
        line=dict(color='black', width=2)
    ))
    
    # 更新布局
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="频率 (Hz)",
        yaxis_title="功率 (μV²/Hz)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 频带能量分布
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Delta", "32.4%")
    
    with col2:
        st.metric("Theta", "18.7%")
    
    with col3:
        st.metric("Alpha", "28.5%")
    
    with col4:
        st.metric("Beta", "14.6%")
    
    with col5:
        st.metric("Gamma", "5.8%")

def render_topographic_visualization():
    """渲染脑地形图可视化"""
    st.markdown("### 脑地形图")
    
    # 配置选项
    col1, col2 = st.columns(2)
    
    with col1:
        band = st.selectbox(
            "频带",
            options=["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", 
                     "Beta (13-30 Hz)", "Gamma (30-100 Hz)", "全频带"],
            index=2
        )
    
    with col2:
        time_point = st.slider(
            "时间点 (秒)",
            min_value=0.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )
    
    # 10-20系统电极位置（简化版，2D平面投影）
    electrode_positions = {
        'Fp1': [-0.2, 0.8], 'Fp2': [0.2, 0.8],
        'F7': [-0.7, 0.4], 'F3': [-0.4, 0.4], 'Fz': [0.0, 0.4], 'F4': [0.4, 0.4], 'F8': [0.7, 0.4],
        'T3': [-0.8, 0.0], 'C3': [-0.4, 0.0], 'Cz': [0.0, 0.0], 'C4': [0.4, 0.0], 'T4': [0.8, 0.0],
        'T5': [-0.7, -0.4], 'P3': [-0.4, -0.4], 'Pz': [0.0, -0.4], 'P4': [0.4, -0.4], 'T6': [0.7, -0.4],
        'O1': [-0.2, -0.8], 'O2': [0.2, -0.8]
    }
    
    # 根据所选频带生成不同的电极值
    np.random.seed(42)
    
    if band == "Alpha (8-13 Hz)":
        # Alpha通常在后部区域更强
        values = {
            'Fp1': np.random.uniform(0, 3), 'Fp2': np.random.uniform(0, 3),
            'F7': np.random.uniform(0, 4), 'F3': np.random.uniform(0, 4), 
            'Fz': np.random.uniform(0, 4), 'F4': np.random.uniform(0, 4), 
            'F8': np.random.uniform(0, 4),
            'T3': np.random.uniform(2, 6), 'C3': np.random.uniform(2, 6), 
            'Cz': np.random.uniform(2, 6), 'C4': np.random.uniform(2, 6), 
            'T4': np.random.uniform(2, 6),
            'T5': np.random.uniform(5, 10), 'P3': np.random.uniform(5, 10), 
            'Pz': np.random.uniform(5, 10), 'P4': np.random.uniform(5, 10), 
            'T6': np.random.uniform(5, 10),
            'O1': np.random.uniform(8, 12), 'O2': np.random.uniform(8, 12)
        }
    elif band == "Beta (13-30 Hz)":
        # Beta通常在前部区域更强
        values = {
            'Fp1': np.random.uniform(5, 10), 'Fp2': np.random.uniform(5, 10),
            'F7': np.random.uniform(5, 10), 'F3': np.random.uniform(5, 10), 
            'Fz': np.random.uniform(5, 10), 'F4': np.random.uniform(5, 10), 
            'F8': np.random.uniform(5, 10),
            'T3': np.random.uniform(3, 8), 'C3': np.random.uniform(3, 8), 
            'Cz': np.random.uniform(3, 8), 'C4': np.random.uniform(3, 8), 
            'T4': np.random.uniform(3, 8),
            'T5': np.random.uniform(1, 5), 'P3': np.random.uniform(1, 5), 
            'Pz': np.random.uniform(1, 5), 'P4': np.random.uniform(1, 5), 
            'T6': np.random.uniform(1, 5),
            'O1': np.random.uniform(0, 3), 'O2': np.random.uniform(0, 3)
        }
    else:
        # 其他频带随机分布
        values = {electrode: np.random.uniform(0, 10) for electrode in electrode_positions}
    
    # 提取电极位置和值
    x = [pos[0] for pos in electrode_positions.values()]
    y = [pos[1] for pos in electrode_positions.values()]
    z = list(values.values())
    
    # 创建网格数据进行插值
    xi = np.linspace(-1, 1, 100)
    yi = np.linspace(-1, 1, 100)
    
    # 创建基于插值的脑地形图
    # 注意：实际应用中应使用MNE库中的更准确插值方法
    from scipy.interpolate import griddata
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method='cubic')
    
    # 创建头部轮廓
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    
    # 创建鼻子形状
    nose_x = [0, 0, 0.2, 0, -0.2, 0]
    nose_y = [1, 1.1, 1, 1, 1, 1]
    
    # 创建耳朵形状
    left_ear_x = [-1, -1.1, -1.05, -1]
    left_ear_y = [0.1, 0, -0.1, -0.2]
    right_ear_x = [1, 1.1, 1.05, 1]
    right_ear_y = [0.1, 0, -0.1, -0.2]
    
    # 创建heatmap
    fig = go.Figure()
    
    # 添加插值后的脑电活动
    fig.add_trace(go.Contour(
        z=zi,
        x=xi,
        y=yi,
        colorscale='Jet',
        colorbar=dict(title="功率 (μV²)"),
        contours=dict(
            start=min(z),
            end=max(z),
            size=(max(z)-min(z))/20,
            showlabels=True
        )
    ))
    
    # 添加电极点
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(size=10, color='black'),
        text=list(electrode_positions.keys()),
        textposition="top center",
        name='电极'
    ))
    
    # 添加头部轮廓
    fig.add_trace(go.Scatter(
        x=head_x,
        y=head_y,
        mode='lines',
        line=dict(color='black', width=2),
        name='头部轮廓'
    ))
    
    # 添加鼻子
    fig.add_trace(go.Scatter(
        x=nose_x,
        y=nose_y,
        mode='lines',
        line=dict(color='black', width=2),
        name='鼻子'
    ))
    
    # 添加耳朵
    fig.add_trace(go.Scatter(
        x=left_ear_x,
        y=left_ear_y,
        mode='lines',
        line=dict(color='black', width=2),
        name='左耳'
    ))
    
    fig.add_trace(go.Scatter(
        x=right_ear_x,
        y=right_ear_y,
        mode='lines',
        line=dict(color='black', width=2),
        name='右耳'
    ))
    
    # 更新布局
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{band} 在 {time_point} 秒时的脑地形图",
        showlegend=False,
        xaxis=dict(
            range=[-1.2, 1.2],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_connectivity_visualization():
    """渲染连接性分析可视化"""
    st.markdown("### 连接性分析")
    
    # 配置选项
    col1, col2 = st.columns(2)
    
    with col1:
        method = st.selectbox(
            "连接性分析方法",
            options=["相干性", "格兰杰因果关系", "相位锁定值", "相位延迟指数"],
            index=0
        )
    
    with col2:
        band = st.selectbox(
            "频带",
            options=["Delta (0.5-4 Hz)", "Theta (4-8 Hz)", "Alpha (8-13 Hz)", 
                    "Beta (13-30 Hz)", "Gamma (30-100 Hz)", "全频带"],
            index=2
        )
    
    # 连接性矩阵
    st.markdown("#### 连接性矩阵")
    
    # 电极名称
    electrodes = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3']
    
    # 生成随机连接性矩阵 (应根据所选方法和频带呈现不同的模式)
    np.random.seed(42)
    
    # 为Alpha频带创建特定的连接模式
    if band == "Alpha (8-13 Hz)":
        # Alpha通常表现为更强的后部连接
        base = np.random.uniform(0.2, 0.4, (len(electrodes), len(electrodes)))
        # 增强后部电极间的连接
        posterior_idx = [5, 6, 7]  # 假设这些是后部电极的索引
        for i in posterior_idx:
            for j in posterior_idx:
                if i != j:
                    base[i, j] = np.random.uniform(0.6, 0.9)
        conn_matrix = base
    elif band == "Beta (13-30 Hz)":
        # Beta通常表现为更强的前部连接
        base = np.random.uniform(0.2, 0.4, (len(electrodes), len(electrodes)))
        # 增强前部电极间的连接
        anterior_idx = [0, 1, 2, 3]  # 假设这些是前部电极的索引
        for i in anterior_idx:
            for j in anterior_idx:
                if i != j:
                    base[i, j] = np.random.uniform(0.6, 0.9)
        conn_matrix = base
    else:
        # 其他频带使用随机模式
        conn_matrix = np.random.uniform(0.1, 0.9, (len(electrodes), len(electrodes)))
    
    # 确保矩阵是对称的（对于一些连接性度量）
    if method in ["相干性", "相位锁定值"]:
        conn_matrix = (conn_matrix + conn_matrix.T) / 2
    
    # 设置对角线为0（自连接）
    np.fill_diagonal(conn_matrix, 0)
    
    # 创建热图
    fig = go.Figure(data=go.Heatmap(
        z=conn_matrix,
        x=electrodes,
        y=electrodes,
        colorscale='Viridis',
        colorbar=dict(title="连接强度"),
        zmin=0,
        zmax=1
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{band} 频带的 {method} 连接性",
        xaxis=dict(title=""),
        yaxis=dict(title="")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 连接性网络图
    st.markdown("#### 连接性网络图")
    
    # 10-20系统电极位置（简化版，2D平面投影）
    electrode_positions = {
        'Fp1': [-0.2, 0.8], 'Fp2': [0.2, 0.8],
        'F7': [-0.7, 0.4], 'F3': [-0.4, 0.4], 'Fz': [0.0, 0.4], 'F4': [0.4, 0.4], 'F8': [0.7, 0.4],
        'T3': [-0.8, 0.0]
    }
    
    # 获取电极位置
    x = [electrode_positions[e][0] for e in electrodes]
    y = [electrode_positions[e][1] for e in electrodes]
    
    # 创建网络图
    fig = go.Figure()
    
    # 添加电极点
    fig.add_trace(go.Scatter(
        x=x,
        y=y,
        mode='markers+text',
        marker=dict(
            size=15,
            color='rgba(0, 0, 255, 0.8)',
            line=dict(color='black', width=1)
        ),
        text=electrodes,
        textposition="top center",
        name='电极'
    ))
    
    # 添加连接线（只显示强连接）
    for i in range(len(electrodes)):
        for j in range(i+1, len(electrodes)):
            if conn_matrix[i, j] > 0.5:  # 只显示强连接
                # 线宽表示连接强度
                width = 1 + 4 * conn_matrix[i, j]
                # 颜色透明度也表示连接强度
                opacity = 0.3 + 0.7 * conn_matrix[i, j]
                
                fig.add_trace(go.Scatter(
                    x=[x[i], x[j]],
                    y=[y[i], y[j]],
                    mode='lines',
                    line=dict(
                        width=width,
                        color=f'rgba(255, 0, 0, {opacity})'
                    ),
                    showlegend=False
                ))
    
    # 创建头部轮廓
    theta = np.linspace(0, 2*np.pi, 100)
    head_x = np.cos(theta)
    head_y = np.sin(theta)
    
    # 添加头部轮廓
    fig.add_trace(go.Scatter(
        x=head_x,
        y=head_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    
    # 添加鼻子指示
    nose_x = [0, 0, 0.2, 0, -0.2, 0]
    nose_y = [1, 1.1, 1, 1, 1, 1]
    
    fig.add_trace(go.Scatter(
        x=nose_x,
        y=nose_y,
        mode='lines',
        line=dict(color='black', width=2),
        showlegend=False
    ))
    
    # 更新布局
    fig.update_layout(
        height=600,
        margin=dict(l=0, r=0, t=30, b=0),
        title=f"{band} 频带的 {method} 连接网络",
        showlegend=False,
        xaxis=dict(
            range=[-1.2, 1.2],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-1.2, 1.2],
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            scaleanchor="x",
            scaleratio=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 连接性指标
    st.markdown("#### 连接性指标")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("全局连接性", f"{np.mean(conn_matrix):.3f}", delta="+0.12")
    
    with col2:
        st.metric("聚类系数", "0.68", delta="-0.05")
    
    with col3:
        st.metric("最强连接", f"{np.max(conn_matrix):.3f}", delta="+0.08")

def render_decoding_visualization():
    """渲染解码结果可视化"""
    st.markdown("### 解码结果可视化")
    
    # 配置选项
    col1, col2 = st.columns(2)
    
    with col1:
        decode_method = st.selectbox(
            "解码方法",
            options=["CNN", "RNN", "混合模型", "DeWave"],
            index=2
        )
    
    with col2:
        display_mode = st.selectbox(
            "显示模式",
            options=["单词概率", "字符概率", "解码过程", "时间序列"],
            index=0
        )
    
    # 解码结果
    st.markdown("#### 当前解码结果")
    
    # 模拟一些解码结果
    decoded_text = "我想要打开灯"
    
    # 大号显示解码结果
    st.markdown(f"""
    <div style="border: 1px solid #ddd; border-radius: 5px; padding: 20px; text-align: center; font-size: 2rem; margin-bottom: 20px;">
        {decoded_text}
    </div>
    """, unsafe_allow_html=True)
    
    # 根据所选显示模式展示不同的可视化
    if display_mode == "单词概率":
        render_word_probabilities()
    elif display_mode == "字符概率":
        render_character_probabilities()
    elif display_mode == "解码过程":
        render_decoding_process()
    else:  # 时间序列
        render_decoding_timeseries()
    
    # 解码历史
    st.markdown("#### 解码历史")
    
    # 模拟解码历史数据
    history = [
        {"时间": "10:45:32", "解码结果": "我想要", "置信度": "87%"},
        {"时间": "10:46:15", "解码结果": "帮助我", "置信度": "92%"},
        {"时间": "10:47:03", "解码结果": "打开窗户", "置信度": "76%"},
        {"时间": "10:48:22", "解码结果": "关闭音乐", "置信度": "85%"},
        {"时间": "10:49:45", "解码结果": "调高音量", "置信度": "81%"},
        {"时间": "10:50:30", "解码结果": "打开灯", "置信度": "94%"}
    ]
    
    st.dataframe(pd.DataFrame(history), use_container_width=True, height=200)
    
    # 解码指标
    st.markdown("#### 解码性能指标")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("准确率", "86.5%", delta="+2.3%")
    
    with col2:
        st.metric("解码速度", "15.2 字符/分钟", delta="+1.7")
    
    with col3:
        st.metric("拒识率", "8.3%", delta="-1.5%")
    
    with col4:
        st.metric("延迟", "1.2 秒", delta="-0.3 秒")

def render_word_probabilities():
    """渲染单词概率可视化"""
    # 模拟一些单词及其概率
    words = ["打开灯", "关闭灯", "调暗灯", "打开音乐", "关闭音乐"]
    probabilities = [0.82, 0.12, 0.03, 0.02, 0.01]
    
    # 创建条形图
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=probabilities,
        y=words,
        orientation='h',
        marker=dict(
            color=['rgba(33,150,243,0.8)' if p == max(probabilities) else 'rgba(0,0,0,0.2)' for p in probabilities],
            line=dict(color='rgba(0,0,0,0.5)', width=1)
        )
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="概率",
            range=[0, 1]
        ),
        yaxis=dict(
            title=""
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_character_probabilities():
    """渲染字符概率可视化"""
    # 假设当前编码结果是"打开灯"
    characters = ["打", "开", "灯"]
    
    # 为每个字符创建概率分布
    char_candidates = [
        ["打", "拍", "拿", "找", "放"],
        ["开", "关", "动", "启", "带"],
        ["灯", "窗", "门", "灶", "车"]
    ]
    
    # 为每个位置生成概率
    probabilities = [
        [0.85, 0.08, 0.04, 0.02, 0.01],
        [0.92, 0.04, 0.02, 0.01, 0.01],
        [0.78, 0.15, 0.05, 0.01, 0.01]
    ]
    
    # 创建子图
    fig = make_subplots(rows=1, cols=len(characters), subplot_titles=[f"位置 {i+1}" for i in range(len(characters))])
    
    # 添加每个字符位置的概率分布
    for i, (chars, probs) in enumerate(zip(char_candidates, probabilities)):
        fig.add_trace(
            go.Bar(
                x=chars,
                y=probs,
                marker=dict(
                    color=['rgba(33,150,243,0.8)' if p == max(probs) else 'rgba(0,0,0,0.2)' for p in probs],
                    line=dict(color='rgba(0,0,0,0.5)', width=1)
                )
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
    )
    
    # 更新y轴范围为0到1
    for i in range(len(characters)):
        fig.update_yaxes(range=[0, 1], title="概率", row=1, col=i+1)
        fig.update_xaxes(title="候选字符", row=1, col=i+1)
    
    st.plotly_chart(fig, use_container_width=True)

def render_decoding_process():
    """渲染解码过程可视化"""
    # 模拟解码过程
    steps = ["信号获取", "预处理", "特征提取", "模型解码", "语言模型修正", "输出生成"]
    durations = [15, 35, 50, 110, 40, 10]  # 毫秒
    status = ["完成", "完成", "完成", "完成", "进行中", "等待"]
    
    # 创建甘特图数据
    df = pd.DataFrame({
        "任务": steps,
        "开始": np.cumsum([0] + durations[:-1]),
        "持续时间": durations,
        "状态": status
    })
    
    # 添加结束时间列
    df["结束"] = df["开始"] + df["持续时间"]
    
    # 创建甘特图
    fig = go.Figure()
    
    for i, task in enumerate(steps):
        status_color = {
            "完成": "rgba(76,175,80,0.8)",
            "进行中": "rgba(33,150,243,0.8)",
            "等待": "rgba(189,189,189,0.8)"
        }[status[i]]
        
        # 添加任务条
        fig.add_trace(go.Bar(
            x=[durations[i]],
            y=[task],
            orientation='h',
            marker=dict(
                color=status_color,
                line=dict(color='rgba(0,0,0,0.5)', width=1)
            ),
            base=df.loc[i, "开始"],
            showlegend=False
        ))
    
    # 更新布局
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="时间 (毫秒)"
        ),
        yaxis=dict(
            title="",
            autorange="reversed"  # 反转y轴，使第一个任务显示在顶部
        ),
        barmode='overlay'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 性能分解
    st.markdown("#### 性能分解")
    
    # 创建饼图，显示各步骤占用的时间比例
    fig = go.Figure(data=[go.Pie(
        labels=steps,
        values=durations,
        hole=.3
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_decoding_timeseries():
    """渲染解码时间序列可视化"""
    # 生成时间点
    time_points = np.linspace(0, 10, 100)
    
    # 生成置信度随时间的变化
    confidence = 0.5 + 0.3 * np.sin(time_points) + 0.1 * np.cos(5 * time_points) + 0.05 * np.random.randn(len(time_points))
    confidence = np.clip(confidence, 0, 1)  # 将置信度限制在0到1之间
    
    # 生成准确率随时间的变化
    accuracy = 0.6 + 0.25 * np.sin(time_points + 1) + 0.1 * np.cos(3 * time_points) + 0.05 * np.random.randn(len(time_points))
    accuracy = np.clip(accuracy, 0, 1)  # 将准确率限制在0到1之间
    
    # 创建时间序列图
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=confidence,
        mode='lines',
        name='置信度',
        line=dict(color='rgba(33,150,243,0.8)', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=time_points,
        y=accuracy,
        mode='lines',
        name='准确率',
        line=dict(color='rgba(76,175,80,0.8)', width=2)
    ))
    
    # 添加解码事件标记
    decode_events = [
        {"time": 2.3, "text": "我"},
        {"time": 4.1, "text": "我想"},
        {"time": 5.8, "text": "我想要"},
        {"time": 8.2, "text": "我想要打开灯"}
    ]
    
    for event in decode_events:
        fig.add_trace(go.Scatter(
            x=[event["time"]],
            y=[confidence[int(event["time"] / 10 * len(time_points))]],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=[event["text"]],
            textposition="top center",
            showlegend=False
        ))
    
    # 更新布局
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="时间 (秒)"
        ),
        yaxis=dict(
            title="值",
            range=[0, 1]
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