"""
Streamlit网页界面模块 - 提供基于Streamlit的网页界面
这是一个独立的脚本，由UI管理器通过子进程启动
"""

import os
import sys
import time
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
import socket

# 添加项目根目录到Python路径
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# 尝试导入Streamlit
try:
    import streamlit as st
    from streamlit.web.server.websocket_headers import _get_websocket_headers
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import altair as alt
    
    HAS_STREAMLIT = True
except ImportError:
    HAS_STREAMLIT = False
    print("错误: 未安装Streamlit，请使用 'pip install streamlit plotly altair' 安装所需依赖")
    sys.exit(1)

# 常量定义
SAMPLING_RATE = 250  # 采样率
DATA_DIR = Path(PROJECT_ROOT) / "data"
SOCKET_HOST = '127.0.0.1'
SOCKET_PORT = 8765  # 用于与主程序通信的套接字端口

# 数据通信类
class DataInterface:
    """数据接口类 - 负责与主程序通信"""
    
    def __init__(self):
        """初始化数据接口"""
        self.connected = False
        self.socket = None
        self.client_socket = None
        self.buffer_size = 4096
        self.data_buffer = None
        self.running = False
        self.receive_thread = None
        self.last_command_result = None
        
        # 初始化数据缓冲区
        self.init_buffer()
    
    def init_buffer(self):
        """初始化数据缓冲区"""
        # 8通道，2500样本（10秒@250Hz）的缓冲区
        self.data_buffer = np.zeros((8, 2500))
        self.decoded_text = ""
    
    def start_server(self):
        """启动通信服务器"""
        try:
            # 创建套接字
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.socket.bind((SOCKET_HOST, SOCKET_PORT))
            self.socket.listen(1)
            self.running = True
            
            # 启动接收线程
            self.receive_thread = threading.Thread(target=self._receive_data, daemon=True)
            self.receive_thread.start()
            
            print(f"数据服务器启动在 {SOCKET_HOST}:{SOCKET_PORT}")
            return True
        except Exception as e:
            print(f"启动数据服务器失败: {e}")
            return False
    
    def _receive_data(self):
        """数据接收线程"""
        print("等待主程序连接...")
        while self.running:
            try:
                # 等待连接
                self.client_socket, addr = self.socket.accept()
                print(f"连接来自: {addr}")
                self.connected = True
                
                # 处理数据
                while self.connected and self.running:
                    try:
                        data = self.client_socket.recv(self.buffer_size)
                        if not data:
                            print("连接断开")
                            self.connected = False
                            break
                        
                        # 解析数据
                        self._process_data(data)
                    except Exception as e:
                        print(f"接收数据错误: {e}")
                        self.connected = False
                        break
                
                # 关闭客户端连接
                if self.client_socket:
                    self.client_socket.close()
                    self.client_socket = None
            
            except Exception as e:
                print(f"连接错误: {e}")
                time.sleep(1)  # 避免CPU过载
    
    def _process_data(self, data):
        """处理接收到的数据"""
        try:
            # 解析JSON数据
            message = json.loads(data.decode('utf-8'))
            
            # 处理不同类型的消息
            msg_type = message.get('type')
            msg_data = message.get('data')
            
            if msg_type == 'eeg_data':
                # 处理EEG数据
                if isinstance(msg_data, list) and len(msg_data) > 0:
                    # 转换数据格式
                    eeg_data = np.array(msg_data)
                    
                    # 更新缓冲区
                    samples = eeg_data.shape[1] if len(eeg_data.shape) > 1 else 1
                    self.data_buffer = np.roll(self.data_buffer, -samples, axis=1)
                    
                    if len(eeg_data.shape) > 1:
                        self.data_buffer[:, -samples:] = eeg_data
                    else:
                        self.data_buffer[:, -1] = eeg_data
            
            elif msg_type == 'decoded_text':
                # 处理解码文本
                self.decoded_text = msg_data
            
            elif msg_type == 'command_result':
                # 处理命令执行结果
                self.last_command_result = msg_data
        
        except Exception as e:
            print(f"处理数据错误: {e}")
    
    def send_command(self, command, params=None):
        """发送命令到主程序"""
        if not self.connected or not self.client_socket:
            print("未连接到主程序，无法发送命令")
            return False
        
        try:
            # 构建命令消息
            message = {
                'type': 'command',
                'command': command,
                'params': params or {}
            }
            
            # 发送消息
            data = json.dumps(message).encode('utf-8')
            self.client_socket.sendall(data)
            
            # 等待响应
            timeout = 3.0  # 超时时间
            start_time = time.time()
            self.last_command_result = None
            
            while self.last_command_result is None and time.time() - start_time < timeout:
                time.sleep(0.1)
            
            return self.last_command_result
        
        except Exception as e:
            print(f"发送命令错误: {e}")
            return False
    
    def get_eeg_data(self):
        """获取当前EEG数据"""
        return self.data_buffer.copy() if self.data_buffer is not None else None
    
    def get_decoded_text(self):
        """获取当前解码文本"""
        return self.decoded_text
    
    def stop(self):
        """停止数据接口"""
        self.running = False
        
        if self.client_socket:
            try:
                self.client_socket.close()
            except:
                pass
        
        if self.socket:
            try:
                self.socket.close()
            except:
                pass
        
        if self.receive_thread and self.receive_thread.is_alive():
            self.receive_thread.join(timeout=1.0)

# 全局数据接口实例
data_interface = None

# 检查是否已开启数据接口
def ensure_data_interface():
    """确保数据接口已初始化"""
    global data_interface
    if data_interface is None:
        data_interface = DataInterface()
        data_interface.start_server()
    return data_interface

# 保存会话状态的函数
def save_session_state():
    """保存会话状态到文件"""
    state_data = {
        'mode': st.session_state.get('mode', 'idle'),
        'device_connected': st.session_state.get('device_connected', False),
        'recording': st.session_state.get('recording', False),
        'last_update': datetime.now().isoformat()
    }
    
    try:
        with open(DATA_DIR / 'session_state.pkl', 'wb') as f:
            pickle.dump(state_data, f)
    except Exception as e:
        st.error(f"保存会话状态失败: {e}")

# 加载会话状态的函数
def load_session_state():
    """从文件加载会话状态"""
    try:
        state_file = DATA_DIR / 'session_state.pkl'
        if state_file.exists():
            with open(state_file, 'rb') as f:
                state_data = pickle.load(f)
            
            # 恢复状态
            for key, value in state_data.items():
                st.session_state[key] = value
    except Exception as e:
        st.warning(f"加载会话状态失败: {e}")

# 模式切换函数
def change_mode(new_mode):
    """切换系统模式"""
    # 保存之前的模式
    old_mode = st.session_state.get('mode', 'idle')
    
    # 如果模式没变，不执行任何操作
    if old_mode == new_mode:
        return
    
    # 更新模式
    st.session_state['mode'] = new_mode
    
    # 发送模式切换命令
    data_if = ensure_data_interface()
    result = data_if.send_command('change_mode', {'mode': new_mode})
    
    # 保存会话状态
    save_session_state()
    
    # 刷新页面
    st.experimental_rerun()

# 设备连接函数
def toggle_device_connection():
    """切换设备连接状态"""
    connected = st.session_state.get('device_connected', False)
    
    data_if = ensure_data_interface()
    if connected:
        # 断开连接
        result = data_if.send_command('disconnect_device')
        if result:
            st.session_state['device_connected'] = False
            st.success("设备已断开连接")
        else:
            st.error("断开设备连接失败")
    else:
        # 连接设备
        result = data_if.send_command('connect_device')
        if result:
            st.session_state['device_connected'] = True
            st.success("设备连接成功")
        else:
            st.error("设备连接失败")
    
    # 保存会话状态
    save_session_state()

# 录制控制函数
def toggle_recording():
    """切换录制状态"""
    recording = st.session_state.get('recording', False)
    
    data_if = ensure_data_interface()
    if recording:
        # 停止录制
        result = data_if.send_command('stop_recording')
        if result:
            st.session_state['recording'] = False
            st.success("已停止录制")
        else:
            st.error("停止录制失败")
    else:
        # 开始录制
        params = {
            'filename': st.session_state.get('record_filename', 'session'),
            'format': st.session_state.get('record_format', 'csv')
        }
        result = data_if.send_command('start_recording', params)
        if result:
            st.session_state['recording'] = True
            st.success("开始录制数据")
        else:
            st.error("开始录制失败")
    
    # 保存会话状态
    save_session_state()

# 脑电信号可视化函数
def plot_eeg_signals(data):
    """绘制脑电信号图表"""
    if data is None:
        return go.Figure().update_layout(title="无数据")
    
    # 创建子图
    fig = make_subplots(rows=4, cols=2, 
                        subplot_titles=[f"通道 {i+1}" for i in range(8)],
                        shared_xaxes=True)
    
    # 时间轴（10秒）
    sample_count = data.shape[1]
    time_axis = np.linspace(-sample_count/SAMPLING_RATE, 0, sample_count)
    
    # 为每个通道添加曲线
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan', 'magenta', 'black']
    
    for i in range(min(8, data.shape[0])):
        row = (i // 2) + 1
        col = (i % 2) + 1
        
        fig.add_trace(
            go.Scatter(
                x=time_axis, 
                y=data[i], 
                mode='lines',
                line=dict(color=colors[i], width=1),
                name=f"通道 {i+1}"
            ),
            row=row, col=col
        )
    
    # 更新布局
    fig.update_layout(
        height=800,
        title_text="脑电信号实时显示",
        showlegend=False,
    )
    
    # 更新所有X轴
    for i in range(1, 5):
        for j in range(1, 3):
            fig.update_xaxes(title_text="时间 (秒)", row=i, col=j)
            fig.update_yaxes(title_text="幅度 (μV)", row=i, col=j)
    
    return fig

# 频谱分析函数
def plot_spectrum(data):
    """绘制频谱分析图表"""
    if data is None or data.shape[0] == 0:
        return go.Figure().update_layout(title="无数据用于频谱分析")
    
    # 使用第一个通道的数据
    signal = data[0]
    
    # 应用窗函数
    windowed_signal = signal * np.hanning(len(signal))
    
    # 计算FFT
    fft = np.abs(np.fft.rfft(windowed_signal))
    freq = np.fft.rfftfreq(len(signal), 1/SAMPLING_RATE)
    
    # 只显示50Hz以下的频率
    mask = freq <= 50
    
    # 转换为dB
    power_db = 20 * np.log10(fft + 1e-10)
    
    # 创建图表
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=freq[mask], 
            y=power_db[mask], 
            mode='lines',
            line=dict(color='blue', width=2),
            name="功率谱"
        )
    )
    
    # 更新布局
    fig.update_layout(
        title="频谱分析 (0-50 Hz)",
        xaxis_title="频率 (Hz)",
        yaxis_title="功率 (dB)",
        height=400
    )
    
    return fig

# 界面的主页面
def main_page():
    """主页面"""
    st.title("TangiEEG - 脑机接口系统")
    
    # 初始化会话状态
    if 'mode' not in st.session_state:
        st.session_state['mode'] = 'idle'
    if 'device_connected' not in st.session_state:
        st.session_state['device_connected'] = False
    if 'recording' not in st.session_state:
        st.session_state['recording'] = False
    
    # 加载之前的会话状态
    load_session_state()
    
    # 确保数据接口已启动
    data_if = ensure_data_interface()
    
    # 创建顶部控制栏
    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        
        # 模式选择
        with col1:
            mode = st.selectbox(
                "操作模式",
                ["idle", "acquire", "monitor", "analyze", "decode", "simulate"],
                index=["idle", "acquire", "monitor", "analyze", "decode", "simulate"].index(st.session_state.get('mode', 'idle')),
                format_func=lambda x: {
                    "idle": "空闲", 
                    "acquire": "数据采集", 
                    "monitor": "信号监测",
                    "analyze": "离线分析", 
                    "decode": "在线解码", 
                    "simulate": "模拟数据"
                }.get(x, x),
                on_change=lambda: change_mode(st.session_state['mode_selector'])
            )
            st.session_state['mode_selector'] = mode
        
        # 设备连接按钮
        with col2:
            if st.session_state.get('device_connected', False):
                st.button("断开设备", on_click=toggle_device_connection)
            else:
                st.button("连接设备", on_click=toggle_device_connection)
        
        # 录制控制
        with col3:
            if st.session_state.get('recording', False):
                st.button("停止录制", on_click=toggle_recording)
            else:
                st.button("开始录制", on_click=toggle_recording)
        
        # 状态显示
        with col4:
            st.metric(
                "设备状态", 
                "已连接" if st.session_state.get('device_connected', False) else "未连接"
            )
    
    # 内容区域使用选项卡组织
    tab1, tab2, tab3 = st.tabs(["信号监测", "文本输出", "系统状态"])
    
    # 信号监测选项卡
    with tab1:
        # 获取最新的EEG数据
        eeg_data = data_if.get_eeg_data()
        
        # 显示脑电信号
        st.subheader("脑电信号")
        eeg_fig = plot_eeg_signals(eeg_data)
        st.plotly_chart(eeg_fig, use_container_width=True)
        
        # 显示频谱分析
        st.subheader("频谱分析")
        spectrum_fig = plot_spectrum(eeg_data)
        st.plotly_chart(spectrum_fig, use_container_width=True)
    
    # 文本输出选项卡
    with tab2:
        st.subheader("解码结果")
        
        # 获取最新的解码文本
        decoded_text = data_if.get_decoded_text()
        
        # 显示解码结果
        if decoded_text:
            st.markdown(f"""
            <div style="padding: 20px; 
                        border-radius: 10px; 
                        border: 1px solid #ccc; 
                        font-size: 24px; 
                        text-align: center;
                        background-color: #f8f9fa;">
                {decoded_text}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("等待解码结果...")
        
        # 添加预测区域
        st.subheader("预测")
        st.text("预测功能尚未实现")
    
    # 系统状态选项卡
    with tab3:
        st.subheader("系统状态")
        
        # 显示基本状态信息
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("当前模式", st.session_state.get('mode', 'idle'))
            st.metric("数据录制", "进行中" if st.session_state.get('recording', False) else "未录制")
        
        with col2:
            st.metric("设备连接", "已连接" if st.session_state.get('device_connected', False) else "未连接")
            st.metric("数据采样率", f"{SAMPLING_RATE} Hz")
        
        # 显示更多系统状态
        st.subheader("高级状态")
        
        # 从主程序获取更多状态信息
        status_info = data_if.send_command('get_status')
        if isinstance(status_info, dict):
            for key, value in status_info.items():
                st.text(f"{key}: {value}")
        else:
            st.text("无高级状态信息")
        
        # 添加系统控制区域
        st.subheader("系统控制")
        
        # 重置系统按钮
        if st.button("重置系统"):
            if data_if.send_command('reset_system'):
                st.success("系统已重置")
                # 重置会话状态
                st.session_state['mode'] = 'idle'
                st.session_state['device_connected'] = False
                st.session_state['recording'] = False
                save_session_state()
                st.experimental_rerun()
            else:
                st.error("系统重置失败")

# 程序入口
def main():
    """程序入口函数"""
    # 设置页面配置
    st.set_page_config(
        page_title="TangiEEG - 脑机接口系统",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # 确保数据目录存在
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # 启动主页面
    main_page()
    
    # 设置自动刷新
    st.markdown(
        """
        <script>
            var timeout = setTimeout(function() {
                window.location.reload();
            }, 2000);
        </script>
        """,
        unsafe_allow_html=True
    )

# 执行入口
if __name__ == "__main__":
    main() 