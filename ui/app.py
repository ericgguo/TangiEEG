"""
TangiEEG用户界面主程序
提供基于Streamlit的Web界面，用于控制和监控TangiEEG系统
"""

import os
import sys
import time
from pathlib import Path

import streamlit as st
from streamlit_option_menu import option_menu
import hydralit_components as hc
import plotly.graph_objects as go
from annotated_text import annotated_text

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入其他UI组件
from ui.dashboard import render_dashboard
from ui.device_panel import render_device_panel
from ui.processing_panel import render_processing_panel
from ui.visualization_panel import render_visualization_panel
from ui.session_manager import SessionManager

# 导入系统模块
from config.system_config import get_system_config, OPERATIONAL_MODES

# 设置页面配置
st.set_page_config(
    page_title="TangiEEG - 脑机接口系统",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 自定义CSS样式
def load_css():
    """加载自定义CSS样式"""
    st.markdown("""
    <style>
        .main-header {
            font-size: 2.5rem !important;
            color: #4CAF50;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.5rem !important;
            color: #2196F3;
            margin-bottom: 1rem;
        }
        .status-card {
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .connect-button {
            background-color: #4CAF50 !important;
            color: white !important;
        }
        .disconnect-button {
            background-color: #f44336 !important;
            color: white !important;
        }
        .stButton button {
            width: 100%;
        }
        .info-box {
            background-color: #e1f5fe;
            border-left: 5px solid #03a9f4;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .warning-box {
            background-color: #fff8e1;
            border-left: 5px solid #ffc107;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .success-box {
            background-color: #e8f5e9;
            border-left: 5px solid #4caf50;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
        .error-box {
            background-color: #ffebee;
            border-left: 5px solid #f44336;
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """初始化Streamlit会话状态"""
    if 'device_connected' not in st.session_state:
        st.session_state.device_connected = False
    if 'device_type' not in st.session_state:
        st.session_state.device_type = 'Daisy'  # 默认为16通道Daisy设备
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'offline_analysis'
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'channel_data' not in st.session_state:
        st.session_state.channel_data = {}
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()

def render_header():
    """渲染页面标题和介绍"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.image("https://raw.githubusercontent.com/ericgguo/TangiEEG/main/logo.png", width=100)
    
    with col2:
        st.markdown("<h1 class='main-header'>TangiEEG 脑机接口系统</h1>", unsafe_allow_html=True)
        st.markdown("基于脑电信号(EEG)的文本生成系统，将脑电意图转化为文本输出")

def render_navigation():
    """渲染导航菜单"""
    selected = option_menu(
        menu_title=None,
        options=["仪表盘", "设备连接", "信号处理", "可视化", "配置", "帮助"],
        icons=["speedometer2", "cpu", "braces-asterisk", "graph-up", "gear", "question-circle"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "16px"}, 
            "nav-link": {"font-size": "16px", "text-align": "center", "margin":"0px", "--hover-color": "#eee"},
            "nav-link-selected": {"background-color": "#4CAF50"},
        }
    )
    return selected

def render_status_bar():
    """渲染状态栏"""
    cols = st.columns(4)
    
    with cols[0]:
        device_status = "已连接" if st.session_state.device_connected else "未连接"
        device_color = "#4CAF50" if st.session_state.device_connected else "#f44336"
        st.markdown(f"""
        <div class="status-card" style="background-color: {device_color}20; border-left: 5px solid {device_color}">
            <h4>设备状态: {device_status}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        mode_name = OPERATIONAL_MODES[st.session_state.current_mode]['description'].split('-')[0].strip()
        st.markdown(f"""
        <div class="status-card" style="background-color: #2196F320; border-left: 5px solid #2196F3">
            <h4>当前模式: {mode_name}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        recording_status = "进行中" if st.session_state.recording_active else "未录制"
        recording_color = "#4CAF50" if st.session_state.recording_active else "#f44336"
        st.markdown(f"""
        <div class="status-card" style="background-color: {recording_color}20; border-left: 5px solid {recording_color}">
            <h4>数据录制: {recording_status}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        processing_status = "进行中" if st.session_state.processing_active else "未处理"
        processing_color = "#4CAF50" if st.session_state.processing_active else "#f44336"
        st.markdown(f"""
        <div class="status-card" style="background-color: {processing_color}20; border-left: 5px solid {processing_color}">
            <h4>信号处理: {processing_status}</h4>
        </div>
        """, unsafe_allow_html=True)

def render_system_messages():
    """渲染系统消息"""
    if st.session_state.system_messages:
        with st.expander("系统消息", expanded=False):
            for msg in st.session_state.system_messages[-10:]:
                message_type = msg.get('type', 'info')
                message_text = msg.get('text', '')
                
                if message_type == 'info':
                    st.info(message_text)
                elif message_type == 'success':
                    st.success(message_text)
                elif message_type == 'warning':
                    st.warning(message_text)
                elif message_type == 'error':
                    st.error(message_text)

def add_system_message(message_text, message_type='info'):
    """添加系统消息"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.system_messages.append({
        'text': f"[{timestamp}] {message_text}",
        'type': message_type,
        'timestamp': time.time()
    })

def main():
    """主函数"""
    # 加载CSS样式
    load_css()
    
    # 初始化会话状态
    initialize_session_state()
    
    # 渲染页面标题
    render_header()
    
    # 渲染状态栏
    render_status_bar()
    
    # 渲染导航菜单
    selected_tab = render_navigation()
    
    # 渲染系统消息
    render_system_messages()
    
    # 根据选择的选项卡渲染不同的内容
    st.markdown("---")
    
    if selected_tab == "仪表盘":
        render_dashboard()
    elif selected_tab == "设备连接":
        render_device_panel()
    elif selected_tab == "信号处理":
        render_processing_panel()
    elif selected_tab == "可视化":
        render_visualization_panel()
    elif selected_tab == "配置":
        render_config_panel()
    elif selected_tab == "帮助":
        render_help_panel()

def render_config_panel():
    """渲染配置面板"""
    st.markdown("<h2 class='sub-header'>系统配置</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["基本配置", "设备配置", "处理配置", "模型配置", "系统配置"])
    
    with tabs[0]:
        st.markdown("### 基本配置")
        col1, col2 = st.columns(2)
        
        with col1:
            st.selectbox("操作模式", options=list(OPERATIONAL_MODES.keys()), 
                         format_func=lambda x: OPERATIONAL_MODES[x]['description'],
                         key="selected_mode")
            
            if st.button("应用模式"):
                st.session_state.current_mode = st.session_state.selected_mode
                add_system_message(f"已切换到{OPERATIONAL_MODES[st.session_state.current_mode]['description']}模式", "success")
        
        with col2:
            st.checkbox("启用调试模式", key="debug_mode")
            st.checkbox("自动保存会话", key="auto_save")
    
    with tabs[1]:
        st.markdown("### 设备配置")
        device_type = st.selectbox("设备类型", ["Cyton", "Ganglion", "Daisy", "Custom"], 
                               index=2,  # 默认选择Daisy (16通道)
                               key="config_device_type")
        
        # 根据设备类型调整通道数
        channel_count = 8 if device_type == "Cyton" else 4 if device_type == "Ganglion" else 16 if device_type == "Daisy" else 8
        
        # 根据设备类型调整采样率选项和默认值
        sample_rate_options = [250, 500, 1000] if device_type == "Cyton" else [200] if device_type == "Ganglion" else [125, 250] if device_type == "Daisy" else [250, 500, 1000]
        default_sample_rate = 250 if device_type == "Cyton" else 200 if device_type == "Ganglion" else 125 if device_type == "Daisy" else 250
        
        st.selectbox("采样率 (Hz)", options=sample_rate_options, index=sample_rate_options.index(default_sample_rate) if default_sample_rate in sample_rate_options else 0)
        
        # 根据设备类型生成通道列表
        channel_options = [f"Ch{i+1}" for i in range(channel_count)]
        st.multiselect("启用通道", options=channel_options, default=channel_options)
    
    with tabs[2]:
        st.markdown("### 处理配置")
        st.checkbox("启用带通滤波", value=True)
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("低频截止 (Hz)", value=1.0, min_value=0.1, max_value=100.0)
        with col2:
            st.number_input("高频截止 (Hz)", value=50.0, min_value=1.0, max_value=200.0)
        
        st.checkbox("启用陷波滤波器", value=True)
        st.selectbox("陷波频率 (Hz)", options=[50, 60], help="50Hz (欧洲) 或 60Hz (美国)")
        
        st.checkbox("自动去除伪迹", value=True)
        st.slider("伪迹阈值", min_value=0, max_value=100, value=50)

def render_help_panel():
    """渲染帮助面板"""
    st.markdown("<h2 class='sub-header'>帮助 & 文档</h2>", unsafe_allow_html=True)
    
    tabs = st.tabs(["快速入门", "使用指南", "常见问题", "关于"])
    
    with tabs[0]:
        st.markdown("### 快速入门")
        st.markdown("""
        1. 在**设备连接**选项卡中连接您的OpenBCI设备
        2. 检查信号质量并调整电极
        3. 在**信号处理**选项卡中配置处理管道
        4. 使用**可视化**选项卡查看实时脑电信号和解码结果
        """)
        
        with st.expander("硬件连接指南"):
            st.markdown("""
            1. 确保您的OpenBCI设备已充电
            2. 通过USB连接设备或使用蓝牙连接
            3. 在软件中选择正确的端口
            4. 点击"连接设备"按钮
            """)
            
            st.image("https://docs.openbci.com/assets/images/ganglion_head_shot-ef218d46ea2d7c9ecfd223ca74d83da7.jpg", 
                     caption="OpenBCI Ganglion示例")
    
    with tabs[1]:
        st.markdown("### 使用指南")
        st.markdown("""
        #### 数据采集
        - 使用**设备连接**选项卡连接和配置您的设备
        - 调整采样率和通道设置
        - 使用阻抗检查确保良好的电极接触
        
        #### 信号处理
        - 配置滤波器参数以减少噪声
        - 启用自动伪迹检测和去除
        - 调整特征提取参数
        
        #### 实时解码
        - 选择适当的解码模型
        - 启动解码过程
        - 查看解码结果和置信度
        """)
    
    with tabs[2]:
        st.markdown("### 常见问题")
        
        with st.expander("设备无法连接怎么办？"):
            st.markdown("""
            1. 检查设备是否开启并充电
            2. 确认正确的串行端口或蓝牙地址
            3. 重启设备和软件
            4. 检查驱动程序是否正确安装
            """)
        
        with st.expander("信号质量差如何改善？"):
            st.markdown("""
            1. 确保电极正确放置并有良好接触
            2. 使用导电凝胶提高接触质量
            3. 减少环境电磁干扰
            4. 保持被试静止，减少肌肉活动
            """)
        
        with st.expander("解码准确率低怎么办？"):
            st.markdown("""
            1. 重新校准系统
            2. 确保有足够的训练数据
            3. 调整预处理参数
            4. 尝试不同的特征提取方法和解码算法
            """)
    
    with tabs[3]:
        st.markdown("### 关于TangiEEG")
        
        st.markdown("""
        TangiEEG是一个开源的脑机接口系统，旨在将脑电信号转换为文本输出。该系统使用OpenBCI硬件采集脑电数据，并通过深度学习模型实现EEG到文本的转换。
        
        - **版本**: 0.1.0
        - **许可证**: MIT
        - **代码库**: [GitHub](https://github.com/ericgguo/TangiEEG)
        - **问题反馈**: [Issues](https://github.com/ericgguo/TangiEEG/issues)
        """)
        
        st.markdown("#### 贡献者")
        st.markdown("- Eric Guo")

if __name__ == "__main__":
    main()
