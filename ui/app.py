"""
TangiEEG用户界面主程序
提供基于Streamlit的Web界面，用于控制和监控TangiEEG系统
基于流程化设计的新版UI
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

import streamlit as st
import hydralit_components as hc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from annotated_text import annotated_text

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 导入会话管理器
from ui.session_manager import SessionManager

# 导入系统配置
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
        .step-card {
            background-color: #f5f5f5;
            border-radius: 0.5rem;
            padding: 1.5rem;
            margin-bottom: 1rem;
            border-left: 5px solid #2196F3;
        }
        .step-card.active {
            border-left: 5px solid #4CAF50;
            background-color: #e8f5e9;
        }
        .step-card.completed {
            border-left: 5px solid #9E9E9E;
            opacity: 0.8;
        }
        .data-flow-arrow {
            text-align: center;
            font-size: 2rem;
            color: #2196F3;
            margin: 1rem 0;
        }
        /* 隐藏Streamlit默认页脚 */
        footer {visibility: hidden;}
        #MainMenu {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """初始化Streamlit会话状态"""
    # 设备状态
    if 'device_connected' not in st.session_state:
        st.session_state.device_connected = False
    if 'device_type' not in st.session_state:
        st.session_state.device_type = 'Daisy'  # 默认为16通道Daisy设备
    
    # 会话状态
    if 'session_active' not in st.session_state:
        st.session_state.session_active = False
    if 'current_session_id' not in st.session_state:
        st.session_state.current_session_id = None
    
    # 用户信息状态 - 新增
    if 'username' not in st.session_state:
        st.session_state.username = "默认用户"
    if 'institution' not in st.session_state:
        st.session_state.institution = "研究机构"
    if 'session_name' not in st.session_state:
        st.session_state.session_name = f"会话_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if 'remarks' not in st.session_state:
        st.session_state.remarks = ""
    
    # 处理状态
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = 'offline_analysis'
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'recording_active' not in st.session_state:
        st.session_state.recording_active = False
    if 'decoding_active' not in st.session_state:
        st.session_state.decoding_active = False
    
    # 数据状态
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'channel_data' not in st.session_state:
        st.session_state.channel_data = {}
    
    # 流程状态 - 新增
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'session'  # 默认首个步骤是会话管理
    if 'step_status' not in st.session_state:
        st.session_state.step_status = {
            'session': 'active',      # 会话管理
            'device': 'pending',      # 设备连接
            'acquisition': 'pending', # 数据采集
            'processing': 'pending',  # 信号处理
            'decoding': 'pending'     # 解码分析
        }
    
    # 系统消息
    if 'system_messages' not in st.session_state:
        st.session_state.system_messages = []
    
    # 会话管理器
    if 'session_manager' not in st.session_state:
        st.session_state.session_manager = SessionManager()
        
    # 设备管理器
    if 'device_manager' not in st.session_state:
        from acquisition.device_manager import DeviceManager
        # 默认使用模拟数据模式
        st.session_state.device_manager = DeviceManager(
            simulate=True,
            device_type='daisy',
            connection_params={
                'sample_rate': 250,
                'noise_level': 0.1,
                'artifact_prob': 0.05
            }
        )

def add_system_message(message_text, message_type='info'):
    """添加系统消息"""
    timestamp = time.strftime("%H:%M:%S")
    st.session_state.system_messages.append({
        'text': f"[{timestamp}] {message_text}",
        'type': message_type,
        'timestamp': time.time()
    })

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

def render_header():
    """渲染页面标题和介绍"""
    col1, col2 = st.columns([1, 5])
    
    with col1:
        st.image("https://raw.githubusercontent.com/ericgguo/TangiEEG/main/logo.png", width=100)
    
    with col2:
        st.markdown("<h1 class='main-header'>TangiEEG 脑机接口系统</h1>", unsafe_allow_html=True)
        st.markdown("基于脑电信号(EEG)的文本生成系统，将脑电意图转化为文本输出")

def render_status_bar():
    """渲染顶部状态栏"""
    # 创建状态指示
    status_items = []
    
    # 设备状态
    device_color = "success" if st.session_state.device_connected else "danger"
    device_icon = "✓" if st.session_state.device_connected else "✗"
    device_status = "已连接" if st.session_state.device_connected else "未连接"
    status_items.append({"icon": device_icon, "name": f"设备: {device_status}", "theme": device_color})
    
    # 会话状态
    session_color = "success" if st.session_state.session_active else "warning"
    session_icon = "📝" if st.session_state.session_active else "📝"
    session_status = f"会话: {st.session_state.current_session_id}" if st.session_state.session_active else "无活动会话"
    status_items.append({"icon": session_icon, "name": session_status, "theme": session_color})
    
    # 记录状态
    recording_color = "success" if st.session_state.recording_active else "danger"
    recording_icon = "⏺" if st.session_state.recording_active else "⏹"
    recording_status = "记录中" if st.session_state.recording_active else "未记录"
    status_items.append({"icon": recording_icon, "name": f"数据: {recording_status}", "theme": recording_color})
    
    # 处理状态
    processing_color = "success" if st.session_state.processing_active else "danger"
    processing_icon = "⚙" if st.session_state.processing_active else "⚙"
    processing_status = "进行中" if st.session_state.processing_active else "未处理"
    status_items.append({"icon": processing_icon, "name": f"处理: {processing_status}", "theme": processing_color})
    
    # 解码状态
    decoding_color = "success" if st.session_state.decoding_active else "danger"
    decoding_icon = "🔄" if st.session_state.decoding_active else "🔄"
    decoding_status = "进行中" if st.session_state.decoding_active else "未解码"
    status_items.append({"icon": decoding_icon, "name": f"解码: {decoding_status}", "theme": decoding_color})
    
    # 使用简单的列来显示状态栏
    cols = st.columns(len(status_items))
    
    for i, item in enumerate(status_items):
        icon = item["icon"]
        name = item["name"]
        theme = item["theme"]
        
        # 根据主题设置不同的颜色
        if theme == "success":
            color = "green"
        elif theme == "warning":
            color = "orange"
        elif theme == "danger":
            color = "red"
        else:
            color = "blue"
        
        cols[i].markdown(f"<div style='text-align: center; color: {color};'><b>{icon} {name}</b></div>", unsafe_allow_html=True)

def render_progress_tracker():
    """渲染流程进度跟踪器"""
    st.markdown("### 实验流程")
    
    # 流程步骤
    steps = [
        {"id": "session", "title": "实验会话", "description": "创建和管理实验会话"},
        {"id": "device", "title": "设备连接", "description": "连接和配置脑电设备"},
        {"id": "acquisition", "title": "数据采集", "description": "采集脑电数据"},
        {"id": "processing", "title": "信号处理", "description": "预处理和特征提取"},
        {"id": "decoding", "title": "解码分析", "description": "模型解码和结果分析"}
    ]
    
    # 使用列显示进度
    cols = st.columns(len(steps))
    
    # 渲染进度跟踪器
    for i, step in enumerate(steps):
        status = st.session_state.step_status[step["id"]]
        
        if step["id"] == st.session_state.current_step:
            icon = "🔍"
            color = "blue"
            bg_color = "rgba(0, 100, 255, 0.1)"
        elif status == "completed":
            icon = "✓"
            color = "green"
            bg_color = "rgba(0, 200, 0, 0.1)"
        else:
            icon = "○"
            color = "gray"
            bg_color = "rgba(200, 200, 200, 0.1)"
        
        # 生成HTML以美化显示
        html = f"""
        <div style="padding: 10px; background-color: {bg_color}; border-radius: 5px; height: 85px; text-align: center;">
            <div style="font-size: 24px; color: {color};">{icon}</div>
            <div style="font-weight: bold; margin-bottom: 3px;">{step["title"]}</div>
            <div style="font-size: 0.8em; opacity: 0.8;">{step["description"]}</div>
        </div>
        """
        
        cols[i].markdown(html, unsafe_allow_html=True)

def navigate_to_step(step):
    """导航到指定步骤并更新状态"""
    # 更新当前步骤
    st.session_state.current_step = step
    
    # 更新步骤状态 - 当前步骤激活
    for s in st.session_state.step_status:
        if s == step:
            st.session_state.step_status[s] = "active"
        elif st.session_state.step_status[s] == "active":
            st.session_state.step_status[s] = "pending"
    
    # 重新加载页面
    st.rerun()

def render_sidebar():
    """渲染侧边栏导航"""
    with st.sidebar:
        st.markdown("## 导航菜单")
        
        # 主导航菜单
        selected = option_menu(
            menu_title=None,
            options=[
                "实验会话", 
                "设备连接", 
                "数据采集", 
                "信号处理", 
                "解码分析", 
                "系统设置"
            ],
            icons=[
                "journal-text", 
                "cpu", 
                "file-earmark-medical", 
                "sliders", 
                "braces-asterisk", 
                "gear"
            ],
            default_index=["session", "device", "acquisition", "processing", "decoding", "settings"].index(st.session_state.current_step) if st.session_state.current_step in ["session", "device", "acquisition", "processing", "decoding", "settings"] else 0,
            styles={
                "container": {"padding": "0px", "background-color": "#fafafa"},
                "icon": {"color": "#4CAF50", "font-size": "16px"}, 
                "nav-link": {
                    "font-size": "16px", 
                    "text-align": "left", 
                    "margin": "0px",
                    "padding": "10px",
                    "--hover-color": "#eee"
                },
                "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
                "menu-title": {"display": "none"}
            }
        )
        
        # 更新当前步骤
        if selected == "实验会话":
            if st.session_state.current_step != "session":
                navigate_to_step("session")
        elif selected == "设备连接":
            if st.session_state.current_step != "device":
                navigate_to_step("device")
        elif selected == "数据采集":
            if st.session_state.current_step != "acquisition":
                navigate_to_step("acquisition")
        elif selected == "信号处理":
            if st.session_state.current_step != "processing":
                navigate_to_step("processing")
        elif selected == "解码分析":
            if st.session_state.current_step != "decoding":
                navigate_to_step("decoding")
        elif selected == "系统设置":
            if st.session_state.current_step != "settings":
                navigate_to_step("settings")
        
        # 分隔线
        st.markdown("---")
        
        # 系统信息和帮助
        st.markdown("### 系统信息")
        st.info(f"设备类型: {st.session_state.device_type}")
        st.info(f"运行模式: {OPERATIONAL_MODES[st.session_state.current_mode]['description']}")
        
        # 帮助按钮
        if st.button("📚 查看帮助文档", use_container_width=True):
            st.session_state.show_help = True
        
        # 关于信息
        st.markdown("### 关于")
        st.markdown("TangiEEG v1.0.0")
        st.markdown("© 2023 [Eric Guo](https://github.com/ericgguo)")

def render_main_content():
    """根据当前步骤渲染主要内容"""
    # 渲染流程进度跟踪器
    render_progress_tracker()
    
    # 渲染系统状态栏
    render_status_bar()
    
    # 分隔线
    st.markdown("---")
    
    # 根据当前步骤渲染相应内容
    if st.session_state.current_step == "session":
        render_session_panel()
    elif st.session_state.current_step == "device":
        render_device_panel()
    elif st.session_state.current_step == "acquisition":
        render_acquisition_panel()
    elif st.session_state.current_step == "processing":
        render_processing_panel()
    elif st.session_state.current_step == "decoding":
        render_decoding_panel()
    elif st.session_state.current_step == "settings":
        render_settings_panel()
    
    # 渲染系统消息
    render_system_messages()

def render_session_panel():
    """渲染会话管理面板"""
    st.markdown("### 会话管理")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        # 用户信息输入
        username = st.text_input("用户名", value="默认用户", key="username")
        institution = st.text_input("机构", value="研究机构", key="institution")
        
        # 会话信息输入
        session_name = st.text_input("会话名称", 
            value=f"Session_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
            key="session_name"
        )
        
        # 备注信息
        session_notes = st.text_area("备注", 
            value="", 
            height=100,
            key="session_notes",
            help="请输入本次会话的相关备注信息"
        )
        
        # 会话信息
        session_info = st.session_state.session_manager.get_session_info()
        
        st.markdown("#### 会话信息")
        st.markdown(f"""
            - **开始时间:** {session_info.get('start_time', '未知')}
            - **设备类型:** {session_info.get('device_type', '未设置')}
            - **事件数量:** {session_info.get('event_count', 0)}
            - **标记数量:** {session_info.get('marker_count', 0)}
            - **解码结果数:** {session_info.get('decoding_result_count', 0)}
            """)
            
            # 会话管理按钮
        col1a, col1b = st.columns(2)
        with col1a:
                if st.button("结束会话", use_container_width=True):
                    # 结束当前会话
                    st.session_state.session_active = False
                    st.session_state.current_session_id = None
                    add_system_message("会话已结束", "info")
                    st.rerun()
            
        with col1b:
                if st.button("添加会话备注", use_container_width=True):
                    st.session_state.show_add_note = True
            
            # 会话备注
        if st.session_state.get('show_add_note', False):
                with st.form("add_note_form"):
                    note = st.text_area("会话备注", height=100)
                    if st.form_submit_button("保存备注"):
                        st.session_state.session_manager.set_session_note(note)
                        st.session_state.show_add_note = False
                        add_system_message("会话备注已添加", "success")
                        st.rerun()
        
        else:
            # 创建新会话
            st.markdown("#### 创建新会话")
            
            with st.form("create_session_form"):
                # 实验名称
                experiment_name = st.text_input("实验名称", value="默认实验")
                
                # 实验类型
                experiment_type = st.selectbox(
                    "实验类型",
                    options=["运动想象", "P300", "SSVEP", "默读", "自由探索", "其他"],
                    index=3
                )
                
                # 被试ID
                subject_id = st.text_input("被试ID", value="S001")
                
                # 运行模式
                mode = st.selectbox(
                    "运行模式",
                    options=list(OPERATIONAL_MODES.keys()),
                    format_func=lambda x: OPERATIONAL_MODES[x]["description"],
                    index=list(OPERATIONAL_MODES.keys()).index(st.session_state.current_mode)
                )
                
                # 提交按钮
                submit_button = st.form_submit_button("创建会话", use_container_width=True)
                
                if submit_button:
                    # 生成会话ID
                    session_id = f"{experiment_type}_{subject_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    
                    # 更新会话状态
                    st.session_state.session_active = True
                    st.session_state.current_session_id = session_id
                    st.session_state.current_mode = mode
                    
                    # 标记会话步骤为已完成
                    st.session_state.step_status["session"] = "completed"
                    
                    # 添加系统消息
                    add_system_message(f"会话 '{session_id}' 已创建", "success")
                    
                    # 重新加载页面
                    st.rerun()
            
            # 加载现有会话
            st.markdown("#### 或加载现有会话")
            
            # 模拟获取现有会话列表
            existing_sessions = [
                {"id": "默读_S001_20230501_123045", "type": "默读", "subject": "S001", "date": "2023-05-01"},
                {"id": "运动想象_S002_20230502_143022", "type": "运动想象", "subject": "S002", "date": "2023-05-02"},
                {"id": "SSVEP_S001_20230503_095612", "type": "SSVEP", "subject": "S001", "date": "2023-05-03"}
            ]
            
            if existing_sessions:
                session_df = pd.DataFrame(existing_sessions)
                selected_session = st.selectbox("选择会话", options=session_df["id"].tolist())
                
                if st.button("加载会话", use_container_width=True):
                    # 更新会话状态
                    st.session_state.session_active = True
                    st.session_state.current_session_id = selected_session
                    
                    # 标记会话步骤为已完成
                    st.session_state.step_status["session"] = "completed"
                    
                    # 添加系统消息
                    add_system_message(f"会话 '{selected_session}' 已加载", "success")
                    
                    # 重新加载页面
                    st.rerun()
            else:
                st.info("没有找到现有会话")
        
        with col2:
        # 会话配置和信息
            st.markdown("### 会话设置")
        
        # 如果有活动会话，显示会话配置
        if st.session_state.session_active:
            # 数据存储设置
            st.markdown("#### 数据存储设置")
            storage_options = st.multiselect(
                "选择要保存的数据类型",
                options=["原始脑电数据", "预处理后数据", "特征数据", "解码结果", "事件标记"],
                default=["原始脑电数据", "预处理后数据", "解码结果"]
            )
            
            # 文件格式
            file_format = st.selectbox(
                "文件格式",
                options=["CSV", "HDF5", "MAT", "EDF"],
                index=1
            )
            
            if st.button("应用存储设置", use_container_width=True):
                add_system_message("数据存储设置已更新", "success")
            
            # 硬件设置
            st.markdown("#### 硬件设置")
            
            # 设备类型
            device_type = st.selectbox(
                "设备类型",
                options=["Cyton", "Ganglion", "Daisy", "Custom"],
                index=2  # 默认选择Daisy
            )
            
            # 采样率
            sample_rate = st.selectbox(
                "采样率 (Hz)",
                options=[125, 250, 500, 1000],
                index=0
            )
            
            if st.button("应用硬件设置", use_container_width=True):
                st.session_state.device_type = device_type
                add_system_message(f"硬件设置已更新: {device_type}, {sample_rate}Hz", "success")
            
            # 下一步操作
            st.markdown("#### 下一步")
            if st.button("进入设备连接", use_container_width=True, type="primary"):
                navigate_to_step("device")
        
        else:
            # 如果没有活动会话，显示提示
            st.info("请先创建或加载会话，然后才能配置会话设置")
        
        # 会话操作指南
        with st.expander("会话管理指南", expanded=False):
            st.markdown("""
            ### 如何管理实验会话
            
            1. **创建新会话** - 填写实验相关信息创建一个新的会话
            2. **加载现有会话** - 从之前的记录中继续一个已有会话
            3. **会话设置** - 配置数据保存和硬件参数
            4. **会话备注** - 添加实验相关说明或备注
            5. **进入下一步** - 会话创建后，进入设备连接阶段
            
            > 实验会话会记录所有数据采集、处理和解码的结果，便于后续分析和恢复
            """)

def render_device_panel():
    """渲染设备连接面板"""
    import time  # 添加time模块导入
    
    st.markdown("<h2 class='sub-header'>设备连接</h2>", unsafe_allow_html=True)
    
    # 检查是否有活动会话
    if not st.session_state.session_active:
        st.warning("请先创建或加载一个会话，然后再连接设备")
        if st.button("返回会话管理", use_container_width=True):
            navigate_to_step("session")
        return
    
    # 创建三列布局
    col1, col2 = st.columns([1, 2])
    
    with col1:
        # 连接设置
        st.markdown("### 连接设置")
        
        # 设备类型选择
        st.selectbox(
            "设备类型",
            options=["Cyton", "Ganglion", "Daisy", "Custom"],
            key="device_type",
            index=2  # 设置默认选项为"Daisy"(索引2)
        )
        
        # 获取可用串口列表的函数
        def get_available_ports():
            """模拟获取可用的串口列表"""
            return [f"COM{i}" for i in range(1, 10)] + [f"/dev/ttyUSB{i}" for i in range(3)]
        
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
        
        elif conn_type == "模拟数据":
            st.success("将使用模拟数据进行测试")
            st.number_input("模拟采样率 (Hz)", value=250, min_value=100, max_value=1000, step=50)
            st.number_input("噪声水平", value=0.1, min_value=0.0, max_value=1.0, step=0.1)
        
        # 连接/断开按钮
        if not st.session_state.device_connected:
            if st.button("连接设备", key="connect_button", use_container_width=True, type="primary"):
                # 使用设备管理器进行连接
                with st.spinner("正在连接设备..."):
                    # 更新连接参数
                    connection_params = {}
                    if conn_type == "串口":
                        connection_params.update({
                            'port': st.session_state.get('serial_port'),
                            'baud_rate': st.session_state.get('baud_rate')
                        })
                    elif conn_type == "蓝牙":
                        connection_params.update({
                            'mac_address': st.session_state.get('mac_address')
                        })
                    elif conn_type == "WiFi":
                        connection_params.update({
                            'ip_address': st.session_state.get('ip_address'),
                            'port': st.session_state.get('port_number')
                        })
                    elif conn_type == "模拟数据":
                        connection_params.update({
                            'sample_rate': st.session_state.get('sample_rate', 250),
                            'noise_level': st.session_state.get('noise_level', 0.1)
                        })
                    
                    # 更新设备类型
                    device_type = st.session_state.device_type.lower()
                    
                    # 重新初始化设备管理器
                    from acquisition.device_manager import DeviceManager
                    st.session_state.device_manager = DeviceManager(
                        simulate=(conn_type == "模拟数据"),
                        device_type=device_type,
                        connection_params=connection_params
                    )
                    
                    # 连接设备
                    if st.session_state.device_manager.connect():
                        st.session_state.device_connected = True
                        # 启动数据流
                        st.session_state.device_manager.start_stream()
                        add_system_message("设备连接成功", "success")
                        # 标记设备连接步骤为已完成
                        st.session_state.step_status["device"] = "completed"
                        st.rerun()
                    else:
                        add_system_message("设备连接失败", "error")
        else:
            if st.button("断开设备", key="disconnect_button", use_container_width=True):
                # 使用设备管理器断开连接
                with st.spinner("正在断开设备..."):
                    if st.session_state.device_manager.disconnect():
                        st.session_state.device_connected = False
                        add_system_message("设备已断开", "info")
                        st.rerun()
                    else:
                        add_system_message("断开设备失败", "error")
        
        # 设备连接状态指示
        if st.session_state.device_connected:
            st.success("✅ 设备已连接")
            
            # 显示设备信息
            st.markdown("### 设备信息")
            
            # 获取真实的设备信息
            device_manager = st.session_state.device_manager
            
            info = {
                "序列号": f"{st.session_state.device_type}-{str(int(time.time()))[5:10]}",
                "固件版本": f"v{device_manager.sample_rate / 100:.1f}",
                "通道数": str(device_manager.channels),
                "采样率": f"{device_manager.sample_rate} Hz",
                "电池电量": "模拟模式" if device_manager.simulate else "78%"
            }
            
            # 显示设备信息
            for key, value in info.items():
                st.text(f"{key}: {value}")
                
            # 下一步操作
            st.markdown("#### 下一步")
            if st.button("进入数据采集", use_container_width=True, type="primary"):
                navigate_to_step("acquisition")
        else:
            st.warning("⚠️ 设备未连接")
    
    with col2:
        # 信号质量监测
        st.markdown("### 信号质量监测")
        
        if st.session_state.device_connected:
            # 显示通道阻抗和信号质量
            # 定义通道标签 - 使用国际10-20系统
            channel_labels = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                             "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1"]
            
            # 根据设备类型选择通道数
            if st.session_state.device_type == "Cyton":
                num_channels = 8
            elif st.session_state.device_type == "Ganglion":
                num_channels = 4
            elif st.session_state.device_type == "Daisy":
                num_channels = 16
            else:
                num_channels = 8
            
            channels = channel_labels[:num_channels]
            
            # 生成随机阻抗值 (单位: kΩ)，真实情况下应从设备获取
            import random
            impedances = [random.uniform(5, 50) for _ in range(num_channels)]
            
            # 计算信号质量得分 (0-100)
            quality_scores = [max(0, min(100, int(100 - (imp - 5) * 2))) for imp in impedances]
            
            # 创建阻抗和信号质量数据框
            df = pd.DataFrame({
                "通道": channels,
                "阻抗 (kΩ)": [f"{imp:.1f}" for imp in impedances],
                "信号质量": quality_scores
            })
            
            # 显示通道状态表格
            st.dataframe(df, use_container_width=True, hide_index=True)
            
            # 通道阻抗图
            fig = go.Figure()
            
            # 添加阻抗条形图
            fig.add_trace(go.Bar(
                x=channels,
                y=impedances,
                name="阻抗",
                marker_color=[
                    'green' if imp < 10 else 'orange' if imp < 30 else 'red'
                    for imp in impedances
                ]
            ))
            
            # 更新布局
            fig.update_layout(
                title="通道阻抗分布",
                xaxis_title="通道",
                yaxis_title="阻抗 (kΩ)",
                height=300,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # 显示图表
            st.plotly_chart(fig, use_container_width=True)
            
            # 添加信号质量指示器
            st.markdown("#### 整体信号质量")
            
            # 计算平均信号质量
            avg_quality = sum(quality_scores) / len(quality_scores)
            
            # 创建进度条样式的质量指示器
            quality_color = (
                "red" if avg_quality < 60
                else "orange" if avg_quality < 80
                else "green"
            )
            
            st.progress(avg_quality / 100)
            st.markdown(
                f"<p style='text-align: center; color: {quality_color};'>"
                f"信号质量: {avg_quality:.1f}%</p>",
                unsafe_allow_html=True
            )
            
            # 添加实时信号预览
            st.markdown("#### 实时信号预览")
            
            # 获取最新的信号数据 (这里使用模拟数据)
            preview_duration = 5  # 预览5秒的数据
            sample_rate = st.session_state.device_manager.sample_rate
            t = np.linspace(0, preview_duration, int(preview_duration * sample_rate))
            
            # 创建模拟信号 (实际应用中应从设备获取真实数据)
            signals = []
            for i in range(num_channels):
                base_freq = 10 + i  # 每个通道使用不同的基频
                signal = np.sin(2 * np.pi * base_freq * t)
                signal += 0.2 * np.random.randn(len(t))  # 添加噪声
                signals.append(signal)
            
            # 创建信号预览图
            fig = go.Figure()
            
            # 为每个通道添加一条线
            for i, (channel, signal) in enumerate(zip(channels, signals)):
                # 对信号进行缩放和偏移，以便在图表中清晰显示
                scaled_signal = signal + i * 3
                
                fig.add_trace(go.Scatter(
                    x=t,
                    y=scaled_signal,
                    name=channel,
                    line=dict(width=1)
                ))
            
            # 更新布局
            fig.update_layout(
                title="实时信号预览",
                xaxis_title="时间 (秒)",
                yaxis_title="信号幅度 (μV)",
                height=400,
                showlegend=True,
                margin=dict(l=0, r=0, t=40, b=0)
            )
            
            # 显示图表
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("连接设备后将显示信号质量信息")
            
            # 添加设备连接指南
            st.markdown("### 设备连接指南")
            st.markdown("""
                1. 选择合适的设备类型
                2. 选择连接方式（串口/蓝牙/WiFi）
                3. 配置连接参数
                4. 点击"连接设备"按钮
                5. 等待连接成功提示
                """)
            
            # 添加故障排除提示
            with st.expander("故障排除"):
                st.markdown("""
                    如果无法连接设备，请检查：
                    - 设备是否已开启
                    - 连接方式是否正确
                    - 驱动程序是否已安装
                    - 设备是否被其他程序占用
                    - USB端口是否正常工作
                    
                    如果问题仍然存在，请尝试：
                    1. 重启设备
                    2. 更换USB端口
                    3. 重新安装驱动程序
                    4. 检查设备固件版本
                    """)

def render_acquisition_panel():
    """渲染数据采集面板"""
    st.markdown("<h2 class='sub-header'>数据采集</h2>", unsafe_allow_html=True)
    
    # 检查是否有活动会话
    if not st.session_state.session_active:
        st.warning("请先创建或加载一个会话，然后再进行数据采集")
        if st.button("返回会话管理", use_container_width=True):
            navigate_to_step("session")
        return
    
    # 检查设备是否已连接
    if not st.session_state.device_connected:
        st.warning("请先连接设备，然后再进行数据采集")
        if st.button("前往设备连接", use_container_width=True):
            navigate_to_step("device")
        return
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 数据录制控制
        st.markdown("### 数据录制控制")
        
        # 根据录制状态显示不同的按钮
        if not st.session_state.recording_active:
            # 录制设置
            st.markdown("#### 录制设置")
            
            # 录制名称
            recording_name = st.text_input("录制名称", value=f"Recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # 录制时长
            recording_duration = st.number_input(
                "录制时长 (秒, 0表示无限)",
                min_value=0,
                max_value=3600,
                value=60,
                step=10
            )
            
            # 数据存储选项
            save_options = st.multiselect(
                "保存数据",
                options=["原始数据", "滤波后数据", "事件标记", "通道状态"],
                default=["原始数据", "事件标记"]
            )
            
            # 文件格式
            file_format = st.selectbox(
                "文件格式",
                options=["CSV", "HDF5", "EDF", "BDF"],
                index=0
            )
            
            # 开始录制按钮
            if st.button("开始录制", use_container_width=True, type="primary"):
                st.session_state.recording_active = True
                st.session_state.recording_start_time = time.time()
                st.session_state.recording_name = recording_name
                st.session_state.recording_duration = recording_duration
                
                # 添加系统消息
                add_system_message(f"开始录制: {recording_name}", "success")
                
                # 更新步骤状态
                st.session_state.step_status["acquisition"] = "active"
                
                st.rerun()
        else:
            # 显示当前录制状态
            elapsed_time = time.time() - st.session_state.recording_start_time
            remaining_time = max(0, st.session_state.recording_duration - elapsed_time) if st.session_state.recording_duration > 0 else float('inf')
            
            st.markdown("#### 当前录制状态")
            
            # 信息展示
            st.markdown(f"""
            - **录制名称:** {st.session_state.recording_name}
            - **已录制时间:** {int(elapsed_time)} 秒
            - **剩余时间:** {"无限" if st.session_state.recording_duration == 0 else f"{int(remaining_time)} 秒"}
            - **估计数据大小:** {int(elapsed_time * 16 * 125 * 4 / 1024 / 1024)} MB
            """)
            
            # 进度条 - 只在有限时长时显示
            if st.session_state.recording_duration > 0:
                progress = min(1.0, elapsed_time / st.session_state.recording_duration)
                st.progress(progress)
            
            # 停止录制按钮
            if st.button("停止录制", use_container_width=True, type="primary"):
                st.session_state.recording_active = False
                
                # 添加系统消息
                add_system_message(f"录制已停止: {st.session_state.recording_name}", "info")
                
                # 标记数据采集步骤为完成
                st.session_state.step_status["acquisition"] = "completed"
                
                st.rerun()
            
            # 添加标记按钮
            if st.button("添加事件标记", use_container_width=True):
                st.session_state.show_add_marker = True
            
            # 添加标记表单
            if st.session_state.get('show_add_marker', False):
                with st.form("add_marker_form"):
                    marker_label = st.text_input("标记标签")
                    marker_type = st.selectbox(
                        "标记类型",
                        options=["刺激开始", "刺激结束", "任务切换", "被试反应", "伪迹", "自定义"]
                    )
                    
                    if st.form_submit_button("保存标记"):
                        # 在真实情况下，这里会将标记保存到数据流中
                        st.session_state.session_manager.add_marker(marker_type, marker_label)
                        st.session_state.show_add_marker = False
                        add_system_message(f"已添加标记: {marker_label}", "success")
                        st.rerun()
        
        # 数据预览设置
        st.markdown("### 数据预览设置")
        
        # 选择要显示的通道
        preview_channels = st.multiselect(
            "显示通道",
            options=["所有通道"] + [f"通道 {i+1}" for i in range(16)],
            default=["所有通道"]
        )
        
        # 时间窗口设置
        time_window = st.slider("时间窗口 (秒)", 1, 30, 10)
        
        # 刷新率
        refresh_rate = st.select_slider(
            "刷新率",
            options=["低 (1Hz)", "中 (5Hz)", "高 (10Hz)"],
            value="中 (5Hz)"
        )
        
        # 应用设置按钮
        if st.button("应用预览设置", use_container_width=True):
            add_system_message("预览设置已更新", "success")
    
    with col2:
        # 实时脑电信号预览
        st.markdown("### 实时脑电信号")
        
        # 生成示例数据
        num_points = 1000
        time_points = np.linspace(0, 4, num_points)
        
        # 获取通道数
        if st.session_state.device_type == "Cyton":
            num_channels = 8
        elif st.session_state.device_type == "Ganglion":
            num_channels = 4
        elif st.session_state.device_type == "Daisy":
            num_channels = 16
        else:
            num_channels = 8
        
        # 创建多通道图
        fig = go.Figure()
        
        # 定义通道名称
        channel_names = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", 
                         "C3", "Cz", "C4", "T4", "P3", "Pz", "P4", "O1"][:num_channels]
        
        # 为每个通道生成数据
        for i in range(num_channels):
            # 如果选择了"所有通道"或特定通道
            if "所有通道" in preview_channels or f"通道 {i+1}" in preview_channels:
                # 生成基础信号
                # 不同频率的正弦波 + 随机噪声 + 时间偏移
                signal = np.sin(2 * np.pi * (i + 5) * time_points / 10)
                signal += 0.2 * np.sin(2 * np.pi * 50 * time_points)  # 50Hz电源噪声
                signal += np.random.normal(0, 0.1, num_points)  # 添加随机噪声
                
                # 添加一些更复杂的特征
                if i < 2:  # 在额叶通道添加眨眼伪迹
                    for blink_time in [1.0, 2.5, 3.8]:
                        blink_idx = int(blink_time * num_points / 4)
                        blink_width = int(0.1 * num_points / 4)
                        for j in range(max(0, blink_idx - blink_width), min(num_points, blink_idx + blink_width)):
                            dist = abs(j - blink_idx) / blink_width
                            signal[j] += 2.0 * np.exp(-dist * dist * 4)
                
                # 偏移显示
                signal = signal + (num_channels - i) * 2
                
                # 添加到图表
                fig.add_trace(go.Scatter(
                    x=time_points,
                    y=signal,
                    mode='lines',
                    name=channel_names[i]
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
                title="振幅 (µV)",
                showticklabels=False
            )
        )
        
        # 显示图表
        st.plotly_chart(fig, use_container_width=True)
        
        # 频谱预览
        st.markdown("### 频谱预览")
        
        # 选择一个通道进行频谱分析
        spectrum_channel = st.selectbox("选择通道", options=channel_names)
        
        # 为所选通道生成频谱数据
        channel_idx = channel_names.index(spectrum_channel)
        
        # 生成简单的通道数据
        channel_data = np.sin(2 * np.pi * 10 * time_points)  # 10Hz成分
        channel_data += 0.5 * np.sin(2 * np.pi * 20 * time_points)  # 20Hz成分
        channel_data += 0.3 * np.sin(2 * np.pi * 30 * time_points)  # 30Hz成分
        channel_data += 0.2 * np.sin(2 * np.pi * 50 * time_points)  # 50Hz成分
        channel_data += np.random.normal(0, 0.1, num_points)  # 随机噪声
        
        # 计算功率谱
        from scipy import signal as sp_signal
        fs = 250  # 采样率
        f, Pxx = sp_signal.welch(channel_data, fs, nperseg=256)
        
        # 绘制频谱
        fig_spectrum = go.Figure()
        fig_spectrum.add_trace(go.Scatter(
            x=f,
            y=Pxx,
            mode='lines',
            fill='tozeroy',
            name='功率谱'
        ))
        
        # 添加频带指示
        bands = [
            {"name": "Delta", "range": [0.5, 4], "color": "rgba(255, 0, 0, 0.2)"},
            {"name": "Theta", "range": [4, 8], "color": "rgba(0, 255, 0, 0.2)"},
            {"name": "Alpha", "range": [8, 13], "color": "rgba(0, 0, 255, 0.2)"},
            {"name": "Beta", "range": [13, 30], "color": "rgba(255, 0, 255, 0.2)"},
            {"name": "Gamma", "range": [30, 100], "color": "rgba(255, 255, 0, 0.2)"}
        ]
        
        for band in bands:
            # 添加频带区域
            fig_spectrum.add_vrect(
                x0=band["range"][0], 
                x1=band["range"][1],
                fillcolor=band["color"],
                opacity=0.5,
                layer="below",
                line_width=0,
                annotation_text=band["name"],
                annotation_position="top left"
            )
        
        fig_spectrum.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis=dict(
                title="频率 (Hz)",
                range=[0, 60]
            ),
            yaxis=dict(
                title="功率 (µV²/Hz)"
            ),
            title=f"通道 {spectrum_channel} 频谱分析"
        )
        
        st.plotly_chart(fig_spectrum, use_container_width=True)
        
        # 导出/下一步操作
        if st.session_state.step_status["acquisition"] == "completed":
            st.markdown("### 下一步操作")
            if st.button("进入信号处理", use_container_width=True, type="primary"):
                navigate_to_step("processing")

        # 事件标记列表
        with st.expander("事件标记列表", expanded=False):
            if st.session_state.session_active:
                # 获取当前会话的标记
                markers = [
                    {"time": "10:45:32", "type": "刺激开始", "label": "视觉刺激1"},
                    {"time": "10:45:55", "type": "被试反应", "label": "按键响应"},
                    {"time": "10:46:21", "type": "刺激开始", "label": "视觉刺激2"},
                    {"time": "10:46:45", "type": "伪迹", "label": "眨眼"},
                    {"time": "10:47:10", "type": "任务切换", "label": "开始默读任务"}
                ]
                
                if markers:
                    st.dataframe(pd.DataFrame(markers), use_container_width=True, hide_index=True)
                else:
                    st.info("当前会话没有标记")

def render_processing_panel():
    """渲染信号处理面板"""
    st.markdown("<h2 class='sub-header'>信号处理</h2>", unsafe_allow_html=True)
    
    # 检查前置条件
    if not st.session_state.session_active:
        st.warning("请先创建或加载一个会话，然后再进行信号处理")
        if st.button("返回会话管理", use_container_width=True):
            navigate_to_step("session")
        return
    
    # 数据来源状态展示
    acquisition_complete = st.session_state.step_status["acquisition"] == "completed"
    data_source = "已采集的数据" if acquisition_complete else "模拟数据"
    
    st.info(f"数据来源: {data_source}")
    
    # 如果数据采集未完成，显示警告
    if not acquisition_complete:
        st.warning("数据采集未完成，将使用模拟数据进行信号处理演示")
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 处理控制面板
        st.markdown("### 处理控制")
        
        # 处理模式选择
        processing_mode = st.radio(
            "处理模式",
            options=["实时处理", "批处理"],
            horizontal=True
        )
        
        # 处理流水线配置
        st.markdown("#### 处理流水线")
        
        # 预设选择
        preset = st.selectbox(
            "预设配置",
            options=["默认", "噪声消除优化", "频带分析优化", "特征提取优化", "自定义"],
            index=0
        )
        
        if preset == "自定义":
            # 预处理
            with st.expander("预处理配置", expanded=True):
                # 滤波器
                st.checkbox("带通滤波", value=True, key="bandpass_enabled")
                col1a, col1b = st.columns(2)
                with col1a:
                    st.number_input("低频截止 (Hz)", value=1.0, min_value=0.1, max_value=100.0, step=0.1, key="bp_low_cutoff")
                with col1b:
                    st.number_input("高频截止 (Hz)", value=50.0, min_value=1.0, max_value=200.0, step=1.0, key="bp_high_cutoff")
                
                st.checkbox("陷波滤波", value=True, key="notch_enabled")
                st.number_input("陷波频率 (Hz)", value=50.0, min_value=1.0, max_value=200.0, step=1.0, key="notch_freq")
                
                st.checkbox("基线校正", value=True, key="baseline_correction")
                st.checkbox("平滑处理", value=False, key="smoothing_enabled")
            
            # 伪迹去除
            with st.expander("伪迹去除", expanded=True):
                st.checkbox("眨眼伪迹检测与去除", value=True, key="blink_removal")
                st.checkbox("肌电伪迹检测与去除", value=False, key="muscle_removal")
                st.checkbox("运动伪迹检测与去除", value=True, key="motion_removal")
                st.select_slider("伪迹检测阈值", options=["宽松", "标准", "严格"], value="标准", key="artifact_threshold")
                
                st.checkbox("使用ICA进行伪迹去除", value=True, key="use_ica")
                if st.session_state.get("use_ica", False):
                    st.number_input("ICA组件数", value=8, min_value=1, max_value=16, key="ica_components")
            
            # 特征提取
            with st.expander("特征提取", expanded=True):
                # 时域特征
                st.markdown("**时域特征**")
                time_features = st.multiselect(
                    "选择时域特征",
                    options=["均值", "标准差", "方差", "峰峰值", "均方根", "过零率", "峰度", "偏度", "Hjorth参数"],
                    default=["均值", "标准差", "峰峰值", "均方根"]
                )
                
                # 频域特征
                st.markdown("**频域特征**")
                freq_features = st.multiselect(
                    "选择频域特征",
                    options=["频带功率", "相对功率", "绝对功率", "功率谱密度", "频谱熵", "主频", "中值频率", "频谱中心"],
                    default=["频带功率", "功率谱密度", "频谱熵"]
                )
                
                # 时频特征
                st.markdown("**时频特征**")
                st.checkbox("启用时频分析", value=False, key="timefreq_enabled")
                if st.session_state.get("timefreq_enabled", False):
                    tf_method = st.selectbox(
                        "时频分析方法",
                        options=["短时傅里叶变换(STFT)", "连续小波变换(CWT)", "希尔伯特-黄变换(HHT)"],
                        index=0
                    )
        else:
            # 显示预设详情
            if preset == "默认":
                st.success("已加载默认配置: 均衡的处理流水线，适合一般用途")
            elif preset == "噪声消除优化":
                st.success("已加载噪声消除优化配置: 强化滤波和伪迹去除")
            elif preset == "频带分析优化":
                st.success("已加载频带分析优化配置: 精细的频谱分析和频带提取")
            elif preset == "特征提取优化":
                st.success("已加载特征提取优化配置: 全面的特征集合，适合机器学习应用")
        
        # 处理控制按钮
        if not st.session_state.processing_active:
            if st.button("开始处理", use_container_width=True, type="primary"):
                st.session_state.processing_active = True
                
                # 更新步骤状态
                st.session_state.step_status["processing"] = "active"
                
                # 添加系统消息
                add_system_message("开始信号处理", "success")
                
                st.rerun()
        else:
            if st.button("停止处理", use_container_width=True):
                st.session_state.processing_active = False
                
                # 标记处理步骤为完成
                st.session_state.step_status["processing"] = "completed"
                
                # 添加系统消息
                add_system_message("信号处理已停止", "info")
                
                st.rerun()
        
        # 保存配置
        if st.button("保存处理配置", use_container_width=True):
            add_system_message("处理配置已保存", "success")
        
        # 高级选项
        with st.expander("高级选项", expanded=False):
            # 窗口设置
            st.markdown("**窗口设置**")
            st.number_input("处理窗口大小(秒)", value=1.0, min_value=0.1, max_value=10.0, step=0.1, key="proc_window_size")
            st.number_input("窗口重叠率(%)", value=50, min_value=0, max_value=90, step=10, key="proc_window_overlap")
            
            # 并行处理
            st.markdown("**性能**")
            st.checkbox("启用并行处理", value=True, key="parallel_processing")
            st.number_input("处理线程数", value=4, min_value=1, max_value=16, step=1, key="num_threads")
            
            # 缓存设置
            st.markdown("**缓存**")
            st.checkbox("启用处理缓存", value=True, key="enable_cache")
            if st.session_state.get("enable_cache", False):
                st.number_input("缓存大小(MB)", value=500, min_value=100, max_value=5000, step=100, key="cache_size")
    
    with col2:
        # 结果可视化区域
        st.markdown("### 处理前后对比")
        
        # 生成示例数据
        num_points = 1000
        time_points = np.linspace(0, 4, num_points)
        
        # 为对比创建两条曲线
        # 原始信号 - 添加噪声和伪迹
        raw_signal = np.sin(2 * np.pi * 10 * time_points)  # 10Hz基础信号
        raw_signal += 0.5 * np.sin(2 * np.pi * 50 * time_points)  # 50Hz电源干扰
        raw_signal += np.random.normal(0, 0.3, num_points)  # 随机噪声
        
        # 添加眨眼伪迹
        for blink_time in [1.0, 2.5, 3.8]:
            blink_idx = int(blink_time * num_points / 4)
            blink_width = int(0.1 * num_points / 4)
            for j in range(max(0, blink_idx - blink_width), min(num_points, blink_idx + blink_width)):
                dist = abs(j - blink_idx) / blink_width
                raw_signal[j] += 3.0 * np.exp(-dist * dist * 4)
        
        # 处理后的信号 - 干净的10Hz信号
        if st.session_state.processing_active:
            processed_signal = np.sin(2 * np.pi * 10 * time_points)  # 纯净的基础信号
            processed_signal += np.random.normal(0, 0.05, num_points)  # 少量残余噪声
        else:
            processed_signal = raw_signal.copy()  # 如果未启动处理，显示相同信号
        
        # 创建对比图
        fig = go.Figure()
        
        # 原始信号
        fig.add_trace(go.Scatter(
            x=time_points,
            y=raw_signal,
            mode='lines',
            name='原始信号',
            line=dict(color='red')
        ))
        
        # 处理后的信号
        fig.add_trace(go.Scatter(
            x=time_points,
            y=processed_signal,
            mode='lines',
            name='处理后信号',
            line=dict(color='green')
        ))
        
        # 更新布局
        fig.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(title="时间 (秒)"),
            yaxis=dict(title="振幅 (µV)")
        )
        
        # 显示对比图
        st.plotly_chart(fig, use_container_width=True)
        
        # 频谱对比
        st.markdown("### 频谱对比")
        
        # 计算原始信号和处理后信号的频谱
        from scipy import signal as sp_signal
        fs = 250  # 假设的采样率
        
        # 原始信号频谱
        f_raw, Pxx_raw = sp_signal.welch(raw_signal, fs, nperseg=256)
        
        # 处理后信号频谱
        f_proc, Pxx_proc = sp_signal.welch(processed_signal, fs, nperseg=256)
        
        # 创建频谱对比图
        fig_spectrum = go.Figure()
        
        # 原始信号频谱
        fig_spectrum.add_trace(go.Scatter(
            x=f_raw,
            y=Pxx_raw,
            mode='lines',
            name='原始信号',
            line=dict(color='red')
        ))
        
        # 处理后信号频谱
        fig_spectrum.add_trace(go.Scatter(
            x=f_proc,
            y=Pxx_proc,
            mode='lines',
            name='处理后信号',
            line=dict(color='green')
        ))
        
        # 添加频带区域
        bands = [
            {"name": "Delta", "range": [0.5, 4], "color": "rgba(255, 0, 0, 0.1)"},
            {"name": "Theta", "range": [4, 8], "color": "rgba(0, 255, 0, 0.1)"},
            {"name": "Alpha", "range": [8, 13], "color": "rgba(0, 0, 255, 0.1)"},
            {"name": "Beta", "range": [13, 30], "color": "rgba(255, 0, 255, 0.1)"},
            {"name": "Gamma", "range": [30, 100], "color": "rgba(255, 255, 0, 0.1)"}
        ]
        
        for band in bands:
            fig_spectrum.add_vrect(
                x0=band["range"][0], 
                x1=band["range"][1],
                fillcolor=band["color"],
                opacity=0.5,
                layer="below",
                line_width=0,
                annotation_text=band["name"],
                annotation_position="top left"
            )
        
        # 更新布局
        fig_spectrum.update_layout(
            height=250,
            margin=dict(l=0, r=0, t=0, b=0),
            xaxis=dict(
                title="频率 (Hz)",
                range=[0, 60]
            ),
            yaxis=dict(
                title="功率 (µV²/Hz)"
            )
        )
        
        # 显示频谱对比图
        st.plotly_chart(fig_spectrum, use_container_width=True)
        
        # 提取的特征预览
        st.markdown("### 提取的特征")
        
        if st.session_state.processing_active:
            # 模拟特征提取结果
            features = {
                "时域特征": {
                    "通道1": {"均值": 0.02, "标准差": 1.05, "峰峰值": 6.31, "均方根": 1.07},
                    "通道2": {"均值": 0.04, "标准差": 1.12, "峰峰值": 6.55, "均方根": 1.15}
                },
                "频域特征": {
                    "通道1": {"Delta功率": 0.23, "Theta功率": 0.18, "Alpha功率": 4.56, "Beta功率": 1.21, "Alpha/Beta比值": 3.77},
                    "通道2": {"Delta功率": 0.27, "Theta功率": 0.19, "Alpha功率": 4.89, "Beta功率": 1.35, "Alpha/Beta比值": 3.62}
                }
            }
            
            # 转换为DataFrame以便显示
            time_features_df = pd.DataFrame.from_dict(features["时域特征"], orient="index")
            freq_features_df = pd.DataFrame.from_dict(features["频域特征"], orient="index")
            
            # 创建标签页来组织特征显示
            feature_tabs = st.tabs(["时域特征", "频域特征"])
            
            with feature_tabs[0]:
                st.dataframe(time_features_df, use_container_width=True)
            
            with feature_tabs[1]:
                st.dataframe(freq_features_df, use_container_width=True)
            
            # 保存特征按钮
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("导出特征", use_container_width=True):
                    add_system_message("特征已导出为CSV文件", "success")
            
            with col2b:
                if st.button("保存到会话", use_container_width=True):
                    add_system_message("特征已保存到当前会话", "success")
        else:
            st.info("请先开始处理以提取特征")
        
        # 下一步操作
        if st.session_state.step_status["processing"] == "completed":
            st.markdown("### 下一步操作")
            if st.button("进入解码分析", use_container_width=True, type="primary"):
                navigate_to_step("decoding")

def render_decoding_panel():
    """渲染解码分析面板"""
    st.markdown("<h2 class='sub-header'>解码分析</h2>", unsafe_allow_html=True)
    
    # 检查前置条件
    if not st.session_state.session_active:
        st.warning("请先创建或加载一个会话，然后再进行解码分析")
        if st.button("返回会话管理", use_container_width=True):
            navigate_to_step("session")
        return
    
    # 检查信号处理是否完成
    processing_complete = st.session_state.step_status["processing"] == "completed"
    if not processing_complete:
        st.warning("请先完成信号处理步骤，然后再进行解码分析")
        if st.button("返回信号处理", use_container_width=True):
            navigate_to_step("processing")
        return
    
    # 创建两列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 解码控制面板
        st.markdown("### 解码控制")
        
        # 解码模型选择
        model_type = st.selectbox(
            "解码模型",
            options=["CNN模型", "RNN模型", "CNN-RNN混合模型", "DeWave模型", "自定义模型"],
            index=3  # 默认为DeWave模型
        )
        
        # 解码任务类型
        task_type = st.radio(
            "任务类型",
            options=["字母解码", "单词解码", "意图分类"],
            horizontal=True
        )
        
        # 解码候选集
        st.markdown("#### 解码候选集")
        
        if task_type == "字母解码":
            # 字母集
            letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            letter_set = st.multiselect(
                "字母集",
                options=list(letters),
                default=list(letters)
            )
            
            # 额外的符号
            symbols = "0123456789,.?!_-"
            symbol_set = st.multiselect(
                "额外符号",
                options=list(symbols),
                default=[]
            )
            
        elif task_type == "单词解码":
            # 单词词典大小
            vocabulary_size = st.slider("词典大小", 10, 1000, 100, step=10)
            
            # 词典类型
            vocabulary_type = st.selectbox(
                "词典类型",
                options=["常用单词", "任务相关词汇", "默读实验词汇", "自定义词典"]
            )
            
            if vocabulary_type == "自定义词典":
                st.file_uploader("上传词典文件", type=["txt", "csv"])
                
        elif task_type == "意图分类":
            # 意图类别
            intents = st.text_area("意图类别 (每行一个)", "默读\n运动想象\n休息\n注意力集中")
            
        # 解码参数
        st.markdown("#### 解码参数")
        
        # 时间窗口设置
        window_size = st.number_input("时间窗口 (秒)", value=1.0, min_value=0.1, max_value=10.0, step=0.1)
        
        # 窗口步长
        step_size = st.number_input("窗口步长 (秒)", value=0.2, min_value=0.05, max_value=5.0, step=0.05)
        
        # 置信度阈值
        confidence_threshold = st.slider("置信度阈值", 0.0, 1.0, 0.5, step=0.05)
        
        # 模型参数
        with st.expander("模型参数", expanded=False):
            if model_type == "CNN模型":
                st.number_input("卷积层数", value=3, min_value=1, max_value=10, step=1)
                st.number_input("过滤器数量", value=64, min_value=8, max_value=256, step=8)
                st.number_input("池化大小", value=2, min_value=1, max_value=4, step=1)
                st.number_input("丢弃率", value=0.3, min_value=0.0, max_value=0.9, step=0.1)
            
            elif model_type == "RNN模型":
                st.selectbox("RNN类型", options=["LSTM", "GRU", "SimpleRNN"])
                st.number_input("隐藏单元数", value=128, min_value=16, max_value=512, step=16)
                st.number_input("RNN层数", value=2, min_value=1, max_value=5, step=1)
                st.checkbox("双向RNN", value=True)
            
            elif model_type == "DeWave模型":
                st.number_input("编码维度", value=128, min_value=32, max_value=512, step=16)
                st.number_input("注意力头数", value=8, min_value=1, max_value=16, step=1)
                st.number_input("Transformer层数", value=6, min_value=1, max_value=12, step=1)
        
        # 解码控制按钮
        if not st.session_state.decoding_active:
            if st.button("开始解码", use_container_width=True, type="primary"):
                st.session_state.decoding_active = True
                
                # 更新步骤状态
                st.session_state.step_status["decoding"] = "active"
                
                # 添加系统消息
                add_system_message("开始解码分析", "success")
                
                st.rerun()
        else:
            if st.button("停止解码", use_container_width=True):
                st.session_state.decoding_active = False
                
                # 标记解码步骤为完成
                st.session_state.step_status["decoding"] = "completed"
                
                # 添加系统消息
                add_system_message("解码分析已停止", "info")
                
                st.rerun()
    
    with col2:
        # 解码结果显示
        st.markdown("### 解码结果")
        
        if st.session_state.decoding_active:
            # 显示解码文本
            st.markdown("#### 识别文本")
            
            # 解码结果文本框
            result_text = ""
            if task_type == "字母解码":
                result_text = "HELLO WORLD"
            elif task_type == "单词解码":
                result_text = "打开 灯光 关闭 窗户"
            elif task_type == "意图分类":
                result_text = "意图: 默读"
            
            st.text_area("解码结果", value=result_text, height=100)
            
            # 候选概率可视化
            st.markdown("#### 候选概率")
            
            # 生成演示概率数据
            if task_type == "字母解码":
                candidates = list("HELOWRDT")
                probabilities = [0.92, 0.87, 0.76, 0.83, 0.91, 0.85, 0.77, 0.69, 0.42]
            elif task_type == "单词解码":
                candidates = ["打开", "关闭", "灯光", "窗户", "门", "音乐", "电视", "空调"]
                probabilities = [0.89, 0.83, 0.78, 0.75, 0.52, 0.48, 0.37, 0.29]
            elif task_type == "意图分类":
                candidates = ["默读", "运动想象", "休息", "注意力集中"]
                probabilities = [0.82, 0.43, 0.36, 0.21]
            
            # 创建概率条形图
            fig = go.Figure()
            
            # 添加条形图
            fig.add_trace(go.Bar(
                x=probabilities,
                y=candidates,
                orientation='h',
                marker=dict(
                    color=probabilities,
                    colorscale='Viridis',
                    line=dict(width=1)
                )
            ))
            
            # 更新布局
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(
                    title="概率",
                    range=[0, 1]
                ),
                yaxis=dict(
                    title=""
                )
            )
            
            # 显示条形图
            st.plotly_chart(fig, use_container_width=True)
            
            # 实时解码流
            st.markdown("#### 解码过程")
            
            # 创建时间轴数据
            timestamps = list(range(10))  # 最近10个时间点
            decoded_letters = [c for c in "...HELLO..."]  # 对应的解码结果
            confidences = [0.3, 0.4, 0.5, 0.8, 0.9, 0.85, 0.82, 0.78, 0.4, 0.3]  # 对应的置信度
            
            # 创建解码流表格
            df = pd.DataFrame({
                "时间点": [f"t-{9-i}" for i in range(10)],
                "解码结果": decoded_letters,
                "置信度": confidences
            })
            
            # 为置信度添加颜色
            def highlight_confidence(val):
                if val >= 0.7:
                    color = 'green'
                elif val >= 0.5:
                    color = 'orange'
                else:
                    color = 'red'
                return f'color: {color}'
            
            # 显示表格
            st.dataframe(
                df.style.applymap(
                    highlight_confidence, 
                    subset=['置信度']
                ),
                use_container_width=True,
                hide_index=True
            )
            
            # 保存/导出按钮
            col2a, col2b = st.columns(2)
            with col2a:
                if st.button("保存解码结果", use_container_width=True):
                    add_system_message("解码结果已保存到会话", "success")
            
            with col2b:
                if st.button("导出为文本", use_container_width=True):
                    add_system_message("解码结果已导出为文本文件", "success")
        
        else:
            # 如果未开始解码，显示提示
            st.info("请点击'开始解码'按钮开始解码分析")
        
        # 解码性能指标
        st.markdown("### 解码性能")
        
        if st.session_state.decoding_active:
            # 计算真实性能指标
            if 'device_manager' in st.session_state and st.session_state.device_connected:
                # 获取最新的模拟数据以计算准确率等指标
                # 这里我们使用随机模拟，但在真实情况下应该基于解码结果计算
                accuracy = np.random.uniform(70, 95)
                speed = np.random.uniform(8, 15)
                latency = np.random.uniform(200, 500)
                
                # 变化趋势，通常会随着时间改善
                accuracy_delta = np.random.uniform(0, 5)
                speed_delta = np.random.uniform(0, 2)
                latency_delta = -np.random.uniform(0, 100)  # 负值表示延迟减少
                
                # 显示性能指标
                col_metrics = st.columns(3)
                
                with col_metrics[0]:
                    st.metric("准确率", f"{accuracy:.1f}%", delta=f"{accuracy_delta:.1f}%")
                
                with col_metrics[1]:
                    st.metric("解码速度", f"{speed:.1f}字符/分钟", delta=f"{speed_delta:.1f}字符/分钟")
                
                with col_metrics[2]:
                    st.metric("延迟", f"{latency:.0f}毫秒", delta=f"{latency_delta:.0f}毫秒")
            else:
                # 使用硬编码的示例值
                col_metrics = st.columns(3)
                
                with col_metrics[0]:
                    st.metric("准确率", "84%", delta="2%")
                
                with col_metrics[1]:
                    st.metric("解码速度", "12字符/分钟", delta="1.5字符/分钟")
                
                with col_metrics[2]:
                    st.metric("延迟", "350毫秒", delta="-50毫秒")
        
        # 解码指南
        with st.expander("解码分析指南", expanded=False):
            st.markdown("""
            ### 解码分析使用指南
            
            1. **选择合适的解码模型**
               - CNN模型: 适合空间特征提取
               - RNN模型: 适合序列数据处理
               - DeWave模型: 结合了transformer架构，性能更好
            
            2. **参数调整建议**
               - 时间窗口: 较大窗口提供更多上下文但增加延迟
               - 置信度阈值: 提高可减少错误，但可能增加未识别率
               - 对于字母解码，建议使用较小步长
            
            3. **提高解码性能的技巧**
               - 确保高质量的信号输入
               - 用足够的数据训练模型
               - 针对特定用户进行模型微调
               - 尝试不同的特征组合
            """)

def render_settings_panel():
    """渲染系统设置面板"""
    st.markdown("<h2 class='sub-header'>系统设置</h2>", unsafe_allow_html=True)
    
    # 创建多列布局
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # 通用设置
        st.markdown("### 通用设置")
        
        # 用户信息
        with st.expander("用户信息", expanded=True):
            st.text_input("用户名", value="默认用户")
            st.text_input("机构", value="研究机构")
            st.text_area("备注", value="", height=100)
            
            if st.button("保存用户信息", use_container_width=True):
                add_system_message("用户信息已保存", "success")
        
        # UI设置
        with st.expander("界面设置", expanded=True):
            st.selectbox("主题", options=["亮色", "暗色", "自动"])
            st.selectbox("语言", options=["简体中文", "English", "日本語"])
            st.checkbox("显示高级选项", value=False)
            st.checkbox("启用动画效果", value=True)
            
            if st.button("应用界面设置", use_container_width=True):
                add_system_message("界面设置已应用", "success")
        
        # 数据设置
        with st.expander("数据设置", expanded=True):
            st.text_input("数据存储路径", value="./data")
            st.number_input("自动保存间隔 (分钟)", value=5, min_value=1, max_value=60, step=1)
            st.checkbox("启用自动备份", value=True)
            st.selectbox("默认数据格式", options=["CSV", "HDF5", "EDF", "BDF"])
            
            if st.button("应用数据设置", use_container_width=True):
                add_system_message("数据设置已应用", "success")
        
        # 硬件设置
        with st.expander("硬件设置", expanded=True):
            st.selectbox("默认设备类型", options=["Cyton", "Ganglion", "Daisy", "Custom"], index=2)
            st.selectbox("默认连接方式", options=["串口", "蓝牙", "WiFi", "模拟数据"])
            st.selectbox("默认采样率", options=["125 Hz", "250 Hz", "500 Hz", "1000 Hz"])
            
            if st.button("应用硬件设置", use_container_width=True):
                add_system_message("硬件设置已应用", "success")
    
    with col2:
        # 高级设置
        st.markdown("### 高级设置")
        
        # 模型管理
        with st.expander("模型管理", expanded=True):
            st.selectbox(
                "默认模型",
                options=["CNN模型", "RNN模型", "DeWave模型", "混合模型", "自定义模型"],
                index=2
            )
            
            # 模型列表
            model_list = [
                {"名称": "DeWave模型", "类型": "Transformer", "训练集": "通用数据集", "准确率": "89%"},
                {"名称": "CNN-LSTM模型", "类型": "混合", "训练集": "运动想象数据", "准确率": "84%"},
                {"名称": "EEGNet", "类型": "CNN", "训练集": "BCI竞赛数据", "准确率": "82%"}
            ]
            
            st.dataframe(pd.DataFrame(model_list), use_container_width=True)
            
            col2a, col2b = st.columns(2)
            with col2a:
                st.button("导入模型", use_container_width=True)
            with col2b:
                st.button("导出模型", use_container_width=True)
        
        # 处理流水线管理
        with st.expander("处理流水线", expanded=True):
            pipeline_list = [
                {"名称": "默认", "步骤": "带通滤波→伪迹去除→特征提取", "适用场景": "通用"},
                {"名称": "高噪声环境", "步骤": "高级滤波→ICA→伪迹去除→特征提取", "适用场景": "嘈杂环境"},
                {"名称": "实时处理", "步骤": "轻量级滤波→快速特征提取", "适用场景": "实时反馈"}
            ]
            
            st.dataframe(pd.DataFrame(pipeline_list), use_container_width=True)
            
            col2c, col2d = st.columns(2)
            with col2c:
                st.button("创建流水线", use_container_width=True)
            with col2d:
                st.button("删除流水线", use_container_width=True)
        
        # 系统信息
        with st.expander("系统信息", expanded=True):
            st.markdown("""
            **软件版本:** TangiEEG v1.0.0
            
            **Python版本:** 3.8.15
            
            **主要依赖:**
            - MNE 1.2.3
            - NumPy 1.23.5
            - SciPy 1.9.3
            - Streamlit 1.21.0
            
            **设备支持:**
            - OpenBCI Cyton
            - OpenBCI Ganglion
            - OpenBCI Cyton+Daisy
            
            **内存使用:** 1.2 GB / 8.0 GB
            
            **存储使用:** 234 MB / 523 GB
            """)
            
        if st.button("检查更新", use_container_width=True):
            st.success("您使用的是最新版本")
        
        # 重置与备份
        st.markdown("### 重置与备份")
        
        col2e, col2f = st.columns(2)
        with col2e:
            if st.button("创建备份", use_container_width=True):
                add_system_message("已创建系统备份", "success")
        
        with col2f:
            if st.button("恢复设置", use_container_width=True):
                st.warning("您确定要恢复到默认设置吗?")
                
                col2g, col2h = st.columns(2)
                with col2g:
                    if st.button("确认", key="confirm_reset", use_container_width=True):
                        add_system_message("已恢复默认设置", "success")
                with col2h:
                    st.button("取消", key="cancel_reset", use_container_width=True)

def main():
    """主函数，程序入口点"""
    # 加载CSS样式
    load_css()
    
    # 初始化会话状态
    initialize_session_state()
    
    # 渲染页面标题
    render_header()
    
    # 渲染侧边栏导航
    render_sidebar()
    
    # 渲染主要内容
    render_main_content()

if __name__ == "__main__":
    main()
