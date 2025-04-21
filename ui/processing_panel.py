"""
信号处理面板 - 提供信号处理控制界面
"""

import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def render_processing_panel():
    """渲染信号处理面板"""
    st.markdown("<h2 class='sub-header'>信号处理</h2>", unsafe_allow_html=True)
    
    # 检查设备连接状态
    if not st.session_state.device_connected and not st.session_state.data_loaded:
        render_no_data_warning()
        return
    
    # 顶部控制按钮
    render_control_buttons()
    
    # 处理流水线配置
    col1, col2 = st.columns([2, 3])
    
    with col1:
        render_pipeline_config()
    
    with col2:
        render_signal_comparison()
    
    # 特征提取和频谱分析
    col1, col2 = st.columns(2)
    
    with col1:
        render_feature_extraction()
    
    with col2:
        render_spectral_analysis()
    
    # 处理日志
    render_processing_log()

def render_no_data_warning():
    """当没有数据时渲染警告"""
    st.warning("⚠️ 没有可用的EEG数据")
    
    st.markdown("""
    请通过以下方式获取数据:
    
    1. **连接设备** - 在"设备连接"页面连接OpenBCI设备
    2. **载入数据** - 从下方上传已记录的数据
    """)
    
    # 数据上传区域
    uploaded_file = st.file_uploader("上传EEG数据文件", type=["csv", "txt", "edf", "bdf", "gdf"])
    
    if uploaded_file is not None:
        st.success("数据上传成功！正在处理...")
        
        # 模拟数据加载过程
        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i + 1)
        
        # 更新会话状态
        st.session_state.data_loaded = True
        st.session_state.upload_file_name = uploaded_file.name
        
        # 重新加载页面以显示数据处理界面
        st.experimental_rerun()

def render_control_buttons():
    """渲染顶部控制按钮"""
    # 数据来源信息
    if st.session_state.device_connected:
        data_source = "实时设备数据"
    elif st.session_state.data_loaded:
        data_source = f"文件: {st.session_state.get('upload_file_name', '未知文件')}"
    else:
        data_source = "无数据"
    
    st.markdown(f"**数据来源:** {data_source}")
    
    # 控制按钮行
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if not st.session_state.processing_active:
            if st.button("开始处理", type="primary", use_container_width=True):
                st.session_state.processing_active = True
                st.experimental_rerun()
        else:
            if st.button("停止处理", use_container_width=True):
                st.session_state.processing_active = False
                st.experimental_rerun()
    
    with col2:
        if st.button("保存设置", use_container_width=True):
            st.success("处理设置已保存")
    
    with col3:
        if st.button("加载设置", use_container_width=True):
            st.info("处理设置已加载")
    
    with col4:
        if st.button("重置设置", use_container_width=True):
            st.info("处理设置已重置为默认值")
    
    # 处理状态指示
    if st.session_state.processing_active:
        st.success("✅ 信号处理进行中")
    else:
        st.warning("⏸️ 信号处理已暂停")

def render_pipeline_config():
    """渲染处理流水线配置"""
    st.markdown("### 处理流水线配置")
    
    # 预设配置选择
    st.selectbox(
        "预设配置",
        options=["默认", "运动想象", "P300", "SSVEP", "睡眠分析", "自定义"],
        key="processing_preset"
    )
    
    # 处理步骤配置
    with st.expander("预处理", expanded=True):
        st.checkbox("带通滤波", value=True, key="bp_filter_enabled")
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("低频截止 (Hz)", value=1.0, min_value=0.1, max_value=100.0, step=0.1, key="bp_low_cutoff")
        with col2:
            st.number_input("高频截止 (Hz)", value=50.0, min_value=1.0, max_value=200.0, step=1.0, key="bp_high_cutoff")
        
        st.checkbox("陷波滤波器", value=True, key="notch_filter_enabled")
        st.selectbox("陷波频率 (Hz)", options=[50, 60], index=0, key="notch_freq")
        
        st.checkbox("基线校正", value=True, key="baseline_correction")
        st.checkbox("重采样", value=False, key="resampling_enabled")
        if st.session_state.resampling_enabled:
            st.number_input("目标采样率 (Hz)", value=125, min_value=10, max_value=1000, step=1, key="target_sample_rate")
    
    with st.expander("伪迹处理", expanded=True):
        st.checkbox("自动伪迹检测", value=True, key="artifact_detection")
        st.select_slider("阈值", options=["低", "中", "高"], value="中", key="artifact_threshold")
        
        st.checkbox("眼电伪迹 (EOG) 移除", value=True, key="eog_removal")
        st.checkbox("肌电伪迹 (EMG) 移除", value=False, key="emg_removal")
        
        st.checkbox("使用ICA分解", value=True, key="use_ica")
        if st.session_state.use_ica:
            st.number_input("ICA组件数", value=8, min_value=1, max_value=64, key="ica_components")
    
    with st.expander("特征提取", expanded=True):
        st.multiselect(
            "时域特征",
            options=["平均值", "标准差", "峰峰值", "均方根", "过零率", "熵"],
            default=["平均值", "标准差", "峰峰值"],
            key="time_domain_features"
        )
        
        st.multiselect(
            "频域特征",
            options=["绝对功率", "相对功率", "平均频率", "中位频率", "频带能量", "功率谱密度"],
            default=["绝对功率", "相对功率", "功率谱密度"],
            key="freq_domain_features"
        )
        
        st.checkbox("提取时频特征", value=True, key="extract_tf_features")
        if st.session_state.extract_tf_features:
            st.selectbox(
                "时频分析方法",
                options=["短时傅里叶变换 (STFT)", "小波变换 (WT)", "希尔伯特-黄变换 (HHT)"],
                index=0,
                key="tf_method"
            )
    
    with st.expander("其他设置", expanded=False):
        st.checkbox("应用空间滤波", value=False, key="spatial_filter")
        if st.session_state.spatial_filter:
            st.selectbox(
                "空间滤波方法",
                options=["共同空间模式 (CSP)", "表面拉普拉斯", "共同平均参考 (CAR)", "双极参考"],
                index=0,
                key="spatial_filter_method"
            )
        
        st.checkbox("数据分段", value=True, key="segmentation_enabled")
        if st.session_state.segmentation_enabled:
            st.number_input("时间窗口长度 (秒)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, key="window_length")
            st.number_input("窗口重叠 (%)", value=50, min_value=0, max_value=90, step=10, key="window_overlap")
        
        st.checkbox("特征归一化", value=True, key="feature_normalization")
        if st.session_state.feature_normalization:
            st.selectbox(
                "归一化方法",
                options=["标准化 (Z-score)", "最小-最大缩放", "鲁棒缩放"],
                index=0,
                key="normalization_method"
            )

def render_signal_comparison():
    """渲染信号对比图"""
    st.markdown("### 处理前后信号对比")
    
    # 创建样例数据
    time_points = 500
    time = np.linspace(0, 2, time_points)
    
    # 原始信号 (带有噪声和伪迹的正弦波)
    raw_signal = 10 * np.sin(2 * np.pi * 10 * time)  # 10Hz的正弦波
    
    # 添加随机噪声
    np.random.seed(0)  # 使结果可重复
    noise = np.random.normal(0, 2, time_points)
    raw_signal += noise
    
    # 添加电源线干扰 (50Hz)
    raw_signal += 3 * np.sin(2 * np.pi * 50 * time)
    
    # 添加基线漂移
    raw_signal += 5 * np.sin(2 * np.pi * 0.3 * time)
    
    # 添加眨眼伪迹
    blink_indices = [100, 300]
    for idx in blink_indices:
        blink_width = 25
        for j in range(max(0, idx - blink_width), min(time_points, idx + blink_width)):
            distance = abs(j - idx)
            attenuation = np.exp(-(distance ** 2) / (2 * (blink_width / 3) ** 2))
            raw_signal[j] += 20 * attenuation
    
    # 处理后的信号 (滤波后的干净信号)
    if st.session_state.processing_active:
        # 模拟处理后的信号 (去除噪声和伪迹)
        processed_signal = 10 * np.sin(2 * np.pi * 10 * time)  # 保留原始的10Hz信号
        
        # 添加少量残余噪声
        residual_noise = np.random.normal(0, 0.5, time_points)
        processed_signal += residual_noise
    else:
        processed_signal = raw_signal.copy()
    
    # 使用plotly创建对比图
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    
    # 添加原始信号
    fig.add_trace(
        go.Scatter(x=time, y=raw_signal, mode='lines', name='原始信号'),
        row=1, col=1
    )
    
    # 添加处理后的信号
    fig.add_trace(
        go.Scatter(x=time, y=processed_signal, mode='lines', name='处理后信号'),
        row=2, col=1
    )
    
    # 更新布局
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(title_text="时间 (秒)", row=2, col=1)
    fig.update_yaxes(title_text="振幅 (μV)", row=1, col=1)
    fig.update_yaxes(title_text="振幅 (μV)", row=2, col=1)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 信号质量指标
    if st.session_state.processing_active:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("信噪比改善", "+18.5 dB")
        
        with col2:
            st.metric("伪迹抑制率", "92.3%")
        
        with col3:
            st.metric("有效带宽", "1-40 Hz")

def render_feature_extraction():
    """渲染特征提取面板"""
    st.markdown("### 特征提取")
    
    # 选择特征提取参数
    col1, col2 = st.columns(2)
    
    with col1:
        selected_channel = st.selectbox(
            "选择通道",
            options=["所有通道", "Fp1", "Fp2", "F7", "F8", "T3", "T4", "O1", "O2"],
            index=0
        )
    
    with col2:
        selected_feature = st.selectbox(
            "选择特征类型",
            options=["时域特征", "频域特征", "时频特征"],
            index=1
        )
    
    # 根据选择的特征类型显示不同的图表
    if selected_feature == "时域特征":
        render_time_domain_features()
    elif selected_feature == "频域特征":
        render_frequency_domain_features()
    else:
        render_time_frequency_features()

def render_time_domain_features():
    """渲染时域特征图表"""
    # 模拟时域特征数据
    channels = ["Fp1", "Fp2", "F7", "F8", "T3", "T4", "O1", "O2"]
    features = ["平均值", "标准差", "峰峰值", "均方根", "过零率"]
    
    # 生成随机特征值
    np.random.seed(0)
    feature_values = np.random.rand(len(channels), len(features))
    
    # 创建热图
    fig = go.Figure(data=go.Heatmap(
        z=feature_values,
        x=features,
        y=channels,
        colorscale='Viridis',
        colorbar=dict(title="特征值")
    ))
    
    fig.update_layout(
        title="时域特征提取结果",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_frequency_domain_features():
    """渲染频域特征图表"""
    # 创建频率范围
    freq = np.linspace(0, 60, 120)
    
    # 创建多个频带的功率值
    delta_power = 20 * np.exp(-((freq - 2) ** 2) / 8)
    theta_power = 15 * np.exp(-((freq - 6) ** 2) / 8)
    alpha_power = 25 * np.exp(-((freq - 10) ** 2) / 8)
    beta_power = 10 * np.exp(-((freq - 20) ** 2) / 32)
    gamma_power = 5 * np.exp(-((freq - 40) ** 2) / 50)
    
    # 结合所有频带
    total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
    
    # 创建频谱图
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=freq, y=total_power,
        mode='lines',
        line=dict(color='rgba(0,0,0,0.2)', width=2),
        name='全频谱'
    ))
    
    fig.add_trace(go.Scatter(
        x=freq, y=delta_power,
        mode='lines', 
        fill='tozeroy',
        line=dict(color='rgba(142,124,197,0.8)'),
        name='Delta (0.5-4 Hz)'
    ))
    
    fig.add_trace(go.Scatter(
        x=freq, y=theta_power,
        mode='lines', 
        fill='tozeroy',
        line=dict(color='rgba(76,175,80,0.8)'),
        name='Theta (4-8 Hz)'
    ))
    
    fig.add_trace(go.Scatter(
        x=freq, y=alpha_power,
        mode='lines', 
        fill='tozeroy',
        line=dict(color='rgba(33,150,243,0.8)'),
        name='Alpha (8-13 Hz)'
    ))
    
    fig.add_trace(go.Scatter(
        x=freq, y=beta_power,
        mode='lines', 
        fill='tozeroy',
        line=dict(color='rgba(255,152,0,0.8)'),
        name='Beta (13-30 Hz)'
    ))
    
    fig.add_trace(go.Scatter(
        x=freq, y=gamma_power,
        mode='lines', 
        fill='tozeroy',
        line=dict(color='rgba(244,67,54,0.8)'),
        name='Gamma (30-50 Hz)'
    ))
    
    fig.update_layout(
        title="频域特征 - 功率谱密度",
        xaxis_title="频率 (Hz)",
        yaxis_title="功率 (μV²/Hz)",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0),
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

def render_time_frequency_features():
    """渲染时频特征图表"""
    # 创建时间和频率数据
    time = np.linspace(0, 4, 100)
    freq = np.linspace(1, 50, 50)
    
    # 创建时频数据 (模拟小波变换或STFT的结果)
    time_freq_data = np.zeros((len(freq), len(time)))
    
    # 添加一些模拟的时变频率模式
    for i, t in enumerate(time):
        # 低频段的持续活动
        time_freq_data[0:10, i] = 15 * np.exp(-((freq[0:10] - 3) ** 2) / 10)
        
        # Alpha (8-13 Hz) 在中间时间段增强
        if 1 < t < 3:
            alpha_idx = np.where((freq >= 8) & (freq <= 13))[0]
            time_freq_data[alpha_idx, i] = 25 * np.exp(-((t - 2) ** 2) / 0.5) * np.exp(-((freq[alpha_idx] - 10) ** 2) / 10)
        
        # 瞬时的高频活动
        if 2.5 < t < 2.7:
            beta_idx = np.where((freq >= 15) & (freq <= 30))[0]
            time_freq_data[beta_idx, i] = 20 * np.exp(-((t - 2.6) ** 2) / 0.01) * np.exp(-((freq[beta_idx] - 20) ** 2) / 20)
    
    # 创建热图
    fig = go.Figure(data=go.Heatmap(
        z=time_freq_data,
        x=time,
        y=freq,
        colorscale='Jet',
        colorbar=dict(title="功率 (dB)")
    ))
    
    fig.update_layout(
        title="时频分析 - 小波变换",
        xaxis_title="时间 (秒)",
        yaxis_title="频率 (Hz)",
        height=400,
        margin=dict(l=0, r=0, t=30, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_spectral_analysis():
    """渲染频谱分析面板"""
    st.markdown("### 频谱分析")
    
    # 频谱分析参数
    col1, col2 = st.columns(2)
    
    with col1:
        selected_method = st.selectbox(
            "分析方法",
            options=["功率谱密度 (PSD)", "时频分析", "相干性分析", "频带能量"],
            index=0
        )
    
    with col2:
        window_size = st.slider(
            "窗口大小 (秒)",
            min_value=0.5,
            max_value=5.0,
            value=2.0,
            step=0.5
        )
    
    # 生成脑电频率节律数据
    freq = np.linspace(0, 50, 500)
    
    # 常见的脑电节律
    rhythms = {
        "Delta (0.5-4 Hz)": [0.5, 4, 10],
        "Theta (4-8 Hz)": [4, 8, 6],
        "Alpha (8-13 Hz)": [8, 13, 12],
        "Beta (13-30 Hz)": [13, 30, 5],
        "Gamma (30-50 Hz)": [30, 50, 2]
    }
    
    # 创建频谱图
    fig = go.Figure()
    
    # 总的PSD
    total_psd = np.zeros_like(freq)
    
    # 添加每个节律的PSD
    for rhythm_name, params in rhythms.items():
        low_freq, high_freq, amplitude = params
        
        # 为该节律创建高斯PSD
        center_freq = (low_freq + high_freq) / 2
        sigma = (high_freq - low_freq) / 3
        
        rhythm_psd = amplitude * np.exp(-((freq - center_freq) ** 2) / (2 * sigma ** 2))
        total_psd += rhythm_psd
        
        # 将节律添加到图表
        fig.add_trace(go.Scatter(
            x=freq,
            y=rhythm_psd,
            mode='lines',
            name=rhythm_name,
            line=dict(width=1),
            fill='tozeroy',
            fillcolor=f'rgba({np.random.randint(0, 256)}, {np.random.randint(0, 256)}, {np.random.randint(0, 256)}, 0.3)'
        ))
    
    # 添加一些随机噪声
    noise = np.random.normal(0, 0.5, len(freq))
    noisy_psd = total_psd + noise
    
    # 添加总PSD
    fig.add_trace(go.Scatter(
        x=freq,
        y=noisy_psd,
        mode='lines',
        name='总PSD',
        line=dict(color='black', width=2)
    ))
    
    # 更新布局
    fig.update_layout(
        xaxis_title="频率 (Hz)",
        yaxis_title="功率谱密度 (μV²/Hz)",
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 频带分布
    st.markdown("#### 频带能量分布")
    
    # 计算每个频带的能量
    band_energy = {}
    for rhythm_name, params in rhythms.items():
        low_freq, high_freq, _ = params
        mask = (freq >= low_freq) & (freq <= high_freq)
        energy = np.sum(noisy_psd[mask])
        band_energy[rhythm_name] = energy
    
    # 归一化能量
    total_energy = sum(band_energy.values())
    normalized_energy = {k: (v / total_energy) * 100 for k, v in band_energy.items()}
    
    # 创建条形图
    fig = go.Figure(data=[
        go.Bar(
            x=list(normalized_energy.keys()),
            y=list(normalized_energy.values()),
            marker_color=['rgba(142,124,197,0.8)', 'rgba(76,175,80,0.8)', 
                        'rgba(33,150,243,0.8)', 'rgba(255,152,0,0.8)', 
                        'rgba(244,67,54,0.8)']
        )
    ])
    
    fig.update_layout(
        yaxis_title="能量百分比 (%)",
        height=300,
        margin=dict(l=0, r=0, t=0, b=0)
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_processing_log():
    """渲染处理日志"""
    st.markdown("### 处理日志")
    
    # 模拟处理日志
    logs = [
        {"timestamp": "10:45:32", "level": "INFO", "message": "开始信号处理"},
        {"timestamp": "10:45:32", "level": "INFO", "message": "应用带通滤波器 (1-50 Hz)"},
        {"timestamp": "10:45:33", "level": "INFO", "message": "应用50Hz陷波滤波器"},
        {"timestamp": "10:45:33", "level": "WARNING", "message": "通道'T3'信号质量较低"},
        {"timestamp": "10:45:34", "level": "INFO", "message": "运行自动伪迹检测"},
        {"timestamp": "10:45:35", "level": "INFO", "message": "检测到5个眨眼伪迹"},
        {"timestamp": "10:45:36", "level": "INFO", "message": "移除伪迹"},
        {"timestamp": "10:45:37", "level": "INFO", "message": "提取频带能量特征"},
        {"timestamp": "10:45:38", "level": "SUCCESS", "message": "处理完成，信噪比提高了18.5 dB"}
    ]
    
    # 显示日志表格
    df = pd.DataFrame(logs)
    
    # 根据日志级别设置行颜色
    def highlight_level(row):
        if row['level'] == 'WARNING':
            return ['background-color: #fff8e1'] * len(row)
        elif row['level'] == 'ERROR':
            return ['background-color: #ffebee'] * len(row)
        elif row['level'] == 'SUCCESS':
            return ['background-color: #e8f5e9'] * len(row)
        else:
            return ['background-color: white'] * len(row)
    
    # 显示表格
    styled_df = df.style.apply(highlight_level, axis=1)
    st.dataframe(styled_df, use_container_width=True, height=250)
