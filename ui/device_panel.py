"""
设备连接面板 - 提供设备连接和配置界面
"""

import time
import serial.tools.list_ports
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from streamlit_extras.metric_cards import style_metric_cards

# 模拟OpenBCI设备
def get_available_ports():
    """获取可用的串口列表"""
    ports = list(serial.tools.list_ports.comports())
    return [p.device for p in ports]

def render_device_panel():
    """渲染设备连接面板"""
    st.markdown("<h2 class='sub-header'>设备连接</h2>", unsafe_allow_html=True)
    
    # 创建两列布局
    col1, col2 = st.columns([2, 3])
    
    with col1:
        render_connection_settings()
    
    with col2:
        render_signal_preview()
    
    # 通道配置和电极阻抗
    col1, col2 = st.columns([1, 1])
    
    with col1:
        render_channel_config()
    
    with col2:
        render_impedance_check()
        
    # 设置初始会话状态
    if 'device_type' not in st.session_state:
        st.session_state.device_type = "Daisy"  # 默认为16通道Daisy设备

def render_connection_settings():
    """渲染连接设置面板"""
    st.markdown("### 连接设置")
    
    # 设备类型选择
    st.selectbox(
        "设备类型",
        options=["Cyton", "Ganglion", "Daisy", "Custom"],
        key="device_type",
        index=2  # 设置默认选项为"Daisy"(索引2)
    )
    
    # 连接方式
    conn_type = st.radio(
        "连接方式",
        options=["串口", "蓝牙", "WiFi", "模拟数据"],
        key="connection_type",
        horizontal=True
    )
    
    # 根据连接方式显示不同的选项
    if conn_type == "串口":
        ports = get_available_ports()
        if ports:
            st.selectbox("串口", options=ports, key="serial_port")
        else:
            st.warning("未检测到可用串口")
            st.text("请检查设备连接或驱动程序安装")
        
        st.selectbox("波特率", options=[115200, 230400, 921600], key="baud_rate")
    
    elif conn_type == "蓝牙":
        st.text_input("MAC地址", key="mac_address", placeholder="00:00:00:00:00:00")
    
    elif conn_type == "WiFi":
        st.text_input("IP地址", key="ip_address", placeholder="192.168.4.1")
        st.number_input("端口", value=5000, key="port_number")
    
    # 连接/断开按钮
    if not st.session_state.device_connected:
        if st.button("连接设备", key="connect_button", use_container_width=True, type="primary"):
            # 模拟连接过程
            with st.spinner("正在连接设备..."):
                time.sleep(2)  # 模拟连接延迟
                st.session_state.device_connected = True
                st.experimental_rerun()
    else:
        if st.button("断开设备", key="disconnect_button", use_container_width=True):
            # 模拟断开过程
            with st.spinner("正在断开设备..."):
                time.sleep(1)  # 模拟断开延迟
                st.session_state.device_connected = False
                st.experimental_rerun()
    
    # 设备连接状态指示
    if st.session_state.device_connected:
        st.success("设备已连接")
        
        # 显示设备信息
        st.markdown("### 设备信息")
        
        # 根据设备类型显示不同的信息
        if st.session_state.device_type == "Cyton":
            info = {
                "序列号": "CY-2023XA45",
                "固件版本": "v3.1.2",
                "通道数": "8",
                "采样率": "250 Hz",
                "电池电量": "85%"
            }
        elif st.session_state.device_type == "Ganglion":
            info = {
                "序列号": "GA-2023B32",
                "固件版本": "v2.0.1",
                "通道数": "4",
                "采样率": "200 Hz",
                "电池电量": "72%"
            }
        elif st.session_state.device_type == "Daisy":
            info = {
                "序列号": "CYD-2023X78",
                "固件版本": "v3.2.0",
                "通道数": "16",
                "采样率": "125 Hz",
                "电池电量": "78%"
            }
        else:
            info = {
                "序列号": "UNKNOWN",
                "固件版本": "UNKNOWN",
                "通道数": "UNKNOWN",
                "采样率": "UNKNOWN",
                "电池电量": "UNKNOWN"
            }
        
        # 显示设备信息
        for key, value in info.items():
            st.text(f"{key}: {value}")
    else:
        st.warning("设备未连接")

def render_signal_preview():
    """渲染信号预览面板"""
    st.markdown("### 信号质量监测")
    
    # 生成模拟数据
    time_points = 200
    time = np.linspace(0, 4, time_points)
    
    # 根据设备类型确定通道数
    device_type = st.session_state.get('device_type', 'Daisy')  # 默认为Daisy
    channels = 8 if device_type == "Cyton" else 4 if device_type == "Ganglion" else 16 if device_type == "Daisy" else 8
    
    # 创建图表
    fig = go.Figure()
    
    # 定义标准10-20系统电极位置名称
    channel_positions = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                      "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", 
                      "Oz", "O2", "A1", "A2"]
    
    if st.session_state.device_connected:
        # 为每个通道添加一条线
        for i in range(channels):
            # 生成模拟的脑电信号，添加一些随机噪声和偏移
            freq = 10  # 频率，单位Hz
            amplitude = 10  # 振幅
            
            # 基本信号：正弦波 + 随机噪声 + 通道偏移
            signal = amplitude * np.sin(2 * np.pi * freq * time + i * np.pi / 4)
            signal += np.random.normal(0, amplitude * 0.1, time_points)  # 添加噪声
            signal += i * amplitude * 2  # 添加通道间的偏移
            
            # 添加随机的眨眼伪迹
            if i == 0 or i == 1:  # 只在前两个通道添加眨眼伪迹（通常是Fp1和Fp2）
                for blink_time in [0.5, 2.0, 3.5]:
                    blink_idx = int(blink_time * time_points / 4)
                    blink_width = int(0.1 * time_points / 4)
                    blink_amplitude = amplitude * 5
                    
                    # 创建眨眼形状（高斯曲线）
                    for j in range(max(0, blink_idx - blink_width), min(time_points, blink_idx + blink_width)):
                        distance = abs(j - blink_idx)
                        attenuation = np.exp(-(distance ** 2) / (2 * (blink_width / 2) ** 2))
                        signal[j] += blink_amplitude * attenuation
            
            # 获取通道名称
            channel_name = channel_positions[i] if i < len(channel_positions) else f"通道 {i+1}"
            
            # 为通道添加一条线
            fig.add_trace(go.Scatter(
                x=time,
                y=signal,
                mode='lines',
                name=channel_name
            ))
        
        # 更新布局
        # 为16通道时提供更大的高度
        chart_height = 300 if channels <= 8 else 450 if channels <= 16 else 600
        
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
        
        # 信号指标
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(label="噪声水平", value="12.3 µV")
        
        with col2:
            st.metric(label="信噪比", value="24.5 dB")
        
        with col3:
            st.metric(label="基线漂移", value="0.8 µV/s")
        
        style_metric_cards()
    else:
        # 如果设备未连接，显示示例数据
        # 为每个通道添加一条线 - 显示示例静态信号
        for i in range(channels):
            # 生成不同的静态示例信号
            amplitude = 10
            freq = 5 + i % 3  # 不同频率
            
            # 基本示例信号：较小振幅的正弦波
            signal = amplitude * 0.5 * np.sin(2 * np.pi * freq * time + i * np.pi / 4)
            signal += i * amplitude * 1.5  # 通道间隔
            
            # 减少随机性，显示更平滑的示例波形
            if np.random.random() < 0.3:  # 随机为一些通道添加特征
                for feature_time in [1.0, 3.0]:
                    feature_idx = int(feature_time * time_points / 4)
                    feature_width = int(0.2 * time_points / 4)
                    feature_amplitude = amplitude * 2
                    
                    # 添加特征
                    for j in range(max(0, feature_idx - feature_width), min(time_points, feature_idx + feature_width)):
                        distance = abs(j - feature_idx)
                        attenuation = np.exp(-(distance ** 2) / (2 * (feature_width / 2) ** 2))
                        signal[j] += feature_amplitude * attenuation * 0.5
            
            # 获取通道名称
            channel_name = channel_positions[i] if i < len(channel_positions) else f"通道 {i+1}"
            
            # 为通道添加一条线
            fig.add_trace(go.Scatter(
                x=time,
                y=signal,
                mode='lines',
                name=channel_name,
                line=dict(color=f'rgba(0, 0, {150 + i*5}, 0.7)')  # 使用蓝色系，降低透明度
            ))
        
        # 更新布局
        chart_height = 300 if channels <= 8 else 450 if channels <= 16 else 600
        
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
        
        # 添加说明性文本
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
            <p style="color: #666; margin-bottom: 0;">这是16通道示例数据预览 - 连接设备后可查看实时信号</p>
        </div>
        """, unsafe_allow_html=True)

def render_channel_config():
    """渲染通道配置面板"""
    st.markdown("### 通道配置")
    
    # 获取设备类型和通道数
    device_type = st.session_state.get('device_type', 'Daisy')  # 默认为Daisy
    channel_count = 8 if device_type == "Cyton" else 4 if device_type == "Ganglion" else 16 if device_type == "Daisy" else 8
    
    # 创建一个表格，显示通道配置
    channels_data = []
    
    # 定义标准10-20系统电极位置名称
    channel_positions = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                      "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", 
                      "Oz", "O2", "A1", "A2"]
    
    for i in range(1, channel_count + 1):
        position = channel_positions[i-1] if i <= len(channel_positions) else f"CH{i}"
        channels_data.append({
            "id": i,
            "name": f"Channel {i}",
            "position": position,
            "enabled": True,
            "gain": 24
        })
    
    # 将数据转换为DataFrame，使用Streamlit的data_editor进行编辑
    df = pd.DataFrame(channels_data)
    
    # 根据通道数量调整编辑器高度
    editor_height = 300 if channel_count <= 8 else 450 if channel_count <= 16 else 600
    
    if st.session_state.device_connected:
        # 创建编辑器
        edited_df = st.data_editor(
            df,
            column_config={
                "id": st.column_config.NumberColumn(
                    "ID",
                    help="通道ID",
                    disabled=True,
                    width="small"
                ),
                "name": st.column_config.TextColumn(
                    "名称",
                    help="通道名称",
                    width="medium"
                ),
                "position": st.column_config.SelectboxColumn(
                    "位置",
                    help="电极位置",
                    width="medium",
                    options=[
                        "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
                        "T3", "C3", "Cz", "C4", "T4",
                        "T5", "P3", "Pz", "P4", "T6",
                        "O1", "Oz", "O2",
                        "A1", "A2", "自定义"
                    ]
                ),
                "enabled": st.column_config.CheckboxColumn(
                    "启用",
                    help="是否启用该通道",
                    width="small"
                ),
                "gain": st.column_config.SelectboxColumn(
                    "增益",
                    help="通道增益",
                    width="small",
                    options=[1, 2, 4, 6, 8, 12, 24]
                )
            },
            hide_index=True,
            use_container_width=True,
            height=editor_height
        )
        
        # 通道配置按钮
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("应用配置", key="apply_channel_config", use_container_width=True, type="primary"):
                st.success("通道配置已应用")
        
        with col2:
            if st.button("恢复默认值", key="reset_channel_config", use_container_width=True):
                st.info("已恢复默认通道配置")
    else:
        # 如果设备未连接，显示只读表格
        st.dataframe(
            df,
            column_config={
                "id": st.column_config.NumberColumn(
                    "ID",
                    help="通道ID",
                    width="small"
                ),
                "name": st.column_config.TextColumn(
                    "名称",
                    help="通道名称",
                    width="medium"
                ),
                "position": st.column_config.TextColumn(
                    "位置",
                    help="电极位置",
                    width="medium"
                ),
                "enabled": st.column_config.CheckboxColumn(
                    "启用",
                    help="是否启用该通道",
                    width="small"
                ),
                "gain": st.column_config.NumberColumn(
                    "增益",
                    help="通道增益",
                    width="small"
                )
            },
            hide_index=True,
            use_container_width=True,
            height=editor_height
        )
        
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
            <p style="color: #666; margin-bottom: 0;">示例16通道配置 - 连接设备后可编辑配置</p>
        </div>
        """, unsafe_allow_html=True)

def render_impedance_check():
    """渲染阻抗检查面板"""
    st.markdown("### 电极阻抗检查")
    
    # 获取设备类型和通道数
    device_type = st.session_state.get('device_type', 'Daisy')  # 默认为Daisy
    channel_count = 8 if device_type == "Cyton" else 4 if device_type == "Ganglion" else 16 if device_type == "Daisy" else 8
    
    # 定义标准10-20系统电极位置名称
    channel_positions = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                      "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1", 
                      "Oz", "O2", "A1", "A2"]
    
    if st.session_state.device_connected:
        # 阻抗检查按钮
        if st.button("开始阻抗检查", key="start_impedance_check", use_container_width=True, type="primary"):
            # 模拟阻抗检查过程
            with st.spinner("正在进行阻抗检查..."):
                time.sleep(2)  # 模拟检查延迟
                st.success("阻抗检查完成")
        
        # 模拟阻抗数据 - 随机数据
        impedance_data = []
        
        for i in range(1, channel_count + 1):
            # 随机生成一些阻抗值
            impedance = round(np.random.uniform(5, 100), 1)
            status = "良好" if impedance < 20 else "一般" if impedance < 50 else "较差"
            color = "#4CAF50" if impedance < 20 else "#FFC107" if impedance < 50 else "#F44336"
            
            position = channel_positions[i-1] if i <= len(channel_positions) else f"CH{i}"
            impedance_data.append({
                "channel": f"通道 {i}",
                "position": position,
                "impedance": impedance,
                "status": status,
                "color": color
            })
    else:
        # 如果设备未连接，显示示例阻抗数据
        impedance_data = []
        
        # 生成固定的示例阻抗数据，保持一些模式
        for i in range(1, channel_count + 1):
            # 使用一种基于位置的模式，而不是完全随机
            if i <= 4:  # 前额叶电极通常阻抗较低
                impedance = round(10 + i * 2, 1)
            elif i >= channel_count - 3:  # 后部电极通常阻抗较高
                impedance = round(40 + (i % 3) * 15, 1)
            else:  # 中间电极阻抗适中
                impedance = round(20 + (i % 5) * 6, 1)
                
            status = "良好" if impedance < 20 else "一般" if impedance < 50 else "较差"
            color = "#4CAF50" if impedance < 20 else "#FFC107" if impedance < 50 else "#F44336"
            
            position = channel_positions[i-1] if i <= len(channel_positions) else f"CH{i}"
            impedance_data.append({
                "channel": f"通道 {i}",
                "position": position,
                "impedance": impedance,
                "status": status,
                "color": color
            })
    
    # 创建阻抗数据的可视化
    fig = go.Figure()
    
    for data in impedance_data:
        fig.add_trace(go.Bar(
            x=[data["impedance"]],
            y=[data["channel"]],
            orientation='h',
            marker=dict(color=data["color"]),
            text=[f"{data['impedance']} kΩ"],
            textposition='auto',
            name=data["channel"]
        ))
    
    # 为16通道时提供更大的高度
    chart_height = 300 if channel_count <= 8 else 450 if channel_count <= 16 else 600
    
    fig.update_layout(
        height=chart_height,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(
            title="阻抗 (kΩ)",
            range=[0, 120]
        ),
        yaxis=dict(
            title=""
        ),
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 阻抗范围参考
    st.markdown("""
    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
        <div><span style="color: #4CAF50; font-weight: bold;">●</span> 良好: < 20 kΩ</div>
        <div><span style="color: #FFC107; font-weight: bold;">●</span> 一般: 20-50 kΩ</div>
        <div><span style="color: #F44336; font-weight: bold;">●</span> 较差: > 50 kΩ</div>
    </div>
    """, unsafe_allow_html=True)
    
    # 如果设备未连接，添加说明文本
    if not st.session_state.device_connected:
        st.markdown("""
        <div style="text-align: center; padding: 10px; background-color: #f8f9fa; border-radius: 5px; margin-top: 10px;">
            <p style="color: #666; margin-bottom: 0;">示例16通道阻抗数据 - 连接设备后可进行实时检测</p>
        </div>
        """, unsafe_allow_html=True)
