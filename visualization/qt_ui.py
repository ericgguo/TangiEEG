"""
Qt图形用户界面模块 - 提供基于PyQt5的图形界面
"""

import sys
import os
import numpy as np
from pathlib import Path

# 防止在导入PyQt5失败时导致整个系统崩溃
try:
    from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                                QTabWidget, QPushButton, QLabel, QComboBox, 
                                QGroupBox, QCheckBox, QFileDialog, QMessageBox,
                                QSplitter, QStatusBar, QAction, QMenu, QToolBar)
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal, pyqtSlot
    from PyQt5.QtGui import QIcon, QPixmap
    import pyqtgraph as pg
    
    # 设置pyqtgraph显示主题
    pg.setConfigOption('background', 'w')
    pg.setConfigOption('foreground', 'k')
    
    HAS_PYQT = True
except ImportError:
    HAS_PYQT = False
    # 创建一个虚拟的QMainWindow类，在没有PyQt5时避免导入错误
    class QMainWindow:
        def __init__(self, *args, **kwargs):
            pass

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from utils.logging_utils import get_logger

class MainWindow(QMainWindow):
    """TangiEEG主窗口类"""
    
    # 自定义信号
    mode_changed = pyqtSignal(str)
    connection_changed = pyqtSignal(bool)
    data_received = pyqtSignal(object)
    
    def __init__(self, system):
        """
        初始化主窗口
        
        Args:
            system: TangiEEGSystem实例
        """
        if not HAS_PYQT:
            raise ImportError("无法导入PyQt5，请安装后再试")
            
        super().__init__()
        
        self.logger = get_logger("qt_ui")
        self.system = system
        
        # 数据缓冲区
        self.data_buffer = None
        
        # 设置窗口属性
        self.setWindowTitle("TangiEEG - 脑机接口系统")
        self.resize(1200, 800)
        
        # 初始化UI
        self._init_ui()
        
        # 信号槽连接
        self._connect_signals()
        
        # 设置更新定时器
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_ui)
        self.update_timer.start(100)  # 10Hz刷新率
        
        self.logger.info("Qt界面初始化完成")
    
    def _init_ui(self):
        """初始化UI组件"""
        # 主窗口布局
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建菜单栏
        self._create_menu_bar()
        
        # 创建工具栏
        self._create_tool_bar()
        
        # 创建主分割器
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.main_splitter)
        
        # 创建控制面板
        self.control_panel = self._create_control_panel()
        self.main_splitter.addWidget(self.control_panel)
        
        # 创建视图面板
        self.view_panel = self._create_view_panel()
        self.main_splitter.addWidget(self.view_panel)
        
        # 设置分割器初始比例
        self.main_splitter.setSizes([300, 900])
        
        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统就绪")
        
        # 状态指示器
        self.status_label = QLabel()
        self.statusBar.addPermanentWidget(self.status_label)
        self.status_label.setText("模式: 空闲 | 设备: 未连接")
    
    def _create_menu_bar(self):
        """创建菜单栏"""
        # 文件菜单
        file_menu = self.menuBar().addMenu("文件")
        
        open_action = QAction("打开数据文件", self)
        open_action.triggered.connect(self._on_open_file)
        file_menu.addAction(open_action)
        
        save_action = QAction("保存当前数据", self)
        save_action.triggered.connect(self._on_save_file)
        file_menu.addAction(save_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # 设备菜单
        device_menu = self.menuBar().addMenu("设备")
        
        connect_action = QAction("连接设备", self)
        connect_action.triggered.connect(self._on_connect_device)
        device_menu.addAction(connect_action)
        
        disconnect_action = QAction("断开连接", self)
        disconnect_action.triggered.connect(self._on_disconnect_device)
        device_menu.addAction(disconnect_action)
        
        device_menu.addSeparator()
        
        settings_action = QAction("设备设置", self)
        settings_action.triggered.connect(self._on_device_settings)
        device_menu.addAction(settings_action)
        
        # 模式菜单
        mode_menu = self.menuBar().addMenu("模式")
        
        for mode_name in ["idle", "acquire", "monitor", "analyze", "decode", "simulate"]:
            mode_action = QAction(self._get_mode_display_name(mode_name), self)
            mode_action.setData(mode_name)
            mode_action.triggered.connect(lambda checked, m=mode_name: self._on_mode_changed(m))
            mode_menu.addAction(mode_action)
        
        # 帮助菜单
        help_menu = self.menuBar().addMenu("帮助")
        
        about_action = QAction("关于", self)
        about_action.triggered.connect(self._on_about)
        help_menu.addAction(about_action)
        
        docs_action = QAction("文档", self)
        docs_action.triggered.connect(self._on_docs)
        help_menu.addAction(docs_action)
    
    def _create_tool_bar(self):
        """创建工具栏"""
        self.toolbar = QToolBar("主工具栏")
        self.addToolBar(self.toolbar)
        
        # 连接/断开按钮
        self.connect_btn = QPushButton("连接设备")
        self.connect_btn.clicked.connect(self._on_connect_device)
        self.toolbar.addWidget(self.connect_btn)
        
        self.toolbar.addSeparator()
        
        # 模式选择下拉框
        self.toolbar.addWidget(QLabel("模式:"))
        self.mode_combo = QComboBox()
        modes = [
            ("idle", "空闲"),
            ("acquire", "数据采集"),
            ("monitor", "信号监测"),
            ("analyze", "离线分析"),
            ("decode", "在线解码"),
            ("simulate", "模拟数据")
        ]
        for mode_id, mode_name in modes:
            self.mode_combo.addItem(mode_name, mode_id)
        self.mode_combo.currentIndexChanged.connect(
            lambda idx: self._on_mode_changed(self.mode_combo.itemData(idx))
        )
        self.toolbar.addWidget(self.mode_combo)
        
        self.toolbar.addSeparator()
        
        # 录制按钮
        self.record_btn = QPushButton("开始录制")
        self.record_btn.setCheckable(True)
        self.record_btn.clicked.connect(self._on_record_toggle)
        self.toolbar.addWidget(self.record_btn)
    
    def _create_control_panel(self):
        """创建控制面板"""
        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        
        # 设备状态组
        device_group = QGroupBox("设备状态")
        device_layout = QVBoxLayout(device_group)
        
        self.device_status_label = QLabel("未连接")
        device_layout.addWidget(self.device_status_label)
        
        self.device_battery_label = QLabel("电池: N/A")
        device_layout.addWidget(self.device_battery_label)
        
        self.device_quality_label = QLabel("信号质量: N/A")
        device_layout.addWidget(self.device_quality_label)
        
        control_layout.addWidget(device_group)
        
        # 采集设置组
        acquisition_group = QGroupBox("采集设置")
        acquisition_layout = QVBoxLayout(acquisition_group)
        
        # 采样率设置
        sample_rate_layout = QHBoxLayout()
        sample_rate_layout.addWidget(QLabel("采样率:"))
        self.sample_rate_combo = QComboBox()
        for rate in ["125 Hz", "250 Hz", "500 Hz", "1000 Hz"]:
            self.sample_rate_combo.addItem(rate)
        self.sample_rate_combo.setCurrentText("250 Hz")
        sample_rate_layout.addWidget(self.sample_rate_combo)
        acquisition_layout.addLayout(sample_rate_layout)
        
        # 通道选择
        channel_layout = QHBoxLayout()
        channel_layout.addWidget(QLabel("通道:"))
        self.channel_combo = QComboBox()
        for ch in ["全部", "1-4", "5-8", "自定义"]:
            self.channel_combo.addItem(ch)
        channel_layout.addWidget(self.channel_combo)
        acquisition_layout.addLayout(channel_layout)
        
        # 滤波器设置
        self.notch_filter_check = QCheckBox("50Hz陷波滤波器")
        self.notch_filter_check.setChecked(True)
        acquisition_layout.addWidget(self.notch_filter_check)
        
        self.bandpass_filter_check = QCheckBox("带通滤波器 (1-50Hz)")
        self.bandpass_filter_check.setChecked(True)
        acquisition_layout.addWidget(self.bandpass_filter_check)
        
        control_layout.addWidget(acquisition_group)
        
        # 解码设置组
        decoding_group = QGroupBox("解码设置")
        decoding_layout = QVBoxLayout(decoding_group)
        
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("模型:"))
        self.model_combo = QComboBox()
        for model in ["DeWave", "CNN", "RNN", "混合模型"]:
            self.model_combo.addItem(model)
        model_layout.addWidget(self.model_combo)
        decoding_layout.addLayout(model_layout)
        
        self.intent_detection_check = QCheckBox("意图检测")
        self.intent_detection_check.setChecked(True)
        decoding_layout.addWidget(self.intent_detection_check)
        
        self.language_model_check = QCheckBox("语言模型辅助")
        self.language_model_check.setChecked(True)
        decoding_layout.addWidget(self.language_model_check)
        
        control_layout.addWidget(decoding_group)
        
        # 填充剩余空间
        control_layout.addStretch()
        
        return control_widget
    
    def _create_view_panel(self):
        """创建视图面板"""
        view_widget = QWidget()
        view_layout = QVBoxLayout(view_widget)
        
        # 创建标签页
        self.tab_widget = QTabWidget()
        view_layout.addWidget(self.tab_widget)
        
        # 信号视图页
        signal_tab = QWidget()
        signal_layout = QVBoxLayout(signal_tab)
        
        # 创建信号图表
        self.signal_plot = pg.PlotWidget(title="脑电信号")
        self.signal_plot.setLabel('left', "幅度", units='μV')
        self.signal_plot.setLabel('bottom', "时间", units='s')
        self.signal_plot.showGrid(x=True, y=True)
        
        # 创建8个通道的曲线
        self.channel_curves = []
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w', 'k']
        for i, color in enumerate(colors):
            curve = self.signal_plot.plot(pen=color, name=f"通道 {i+1}")
            self.channel_curves.append(curve)
        
        signal_layout.addWidget(self.signal_plot)
        
        # 频谱视图
        self.spectrum_plot = pg.PlotWidget(title="频谱分析")
        self.spectrum_plot.setLabel('left', "功率", units='dB')
        self.spectrum_plot.setLabel('bottom', "频率", units='Hz')
        self.spectrum_plot.showGrid(x=True, y=True)
        self.spectrum_curve = self.spectrum_plot.plot(pen='b')
        
        signal_layout.addWidget(self.spectrum_plot)
        
        self.tab_widget.addTab(signal_tab, "信号视图")
        
        # 文本输出页
        text_tab = QWidget()
        text_layout = QVBoxLayout(text_tab)
        
        # 解码输出显示
        self.decoded_text = QLabel("等待解码结果...")
        self.decoded_text.setAlignment(Qt.AlignCenter)
        self.decoded_text.setStyleSheet("font-size: 24px; font-weight: bold;")
        text_layout.addWidget(self.decoded_text)
        
        # 实时预测
        self.prediction_label = QLabel("预测:")
        self.prediction_label.setStyleSheet("font-size: 18px;")
        text_layout.addWidget(self.prediction_label)
        
        text_layout.addStretch()
        
        self.tab_widget.addTab(text_tab, "文本输出")
        
        # 状态页
        status_tab = QWidget()
        status_layout = QVBoxLayout(status_tab)
        
        self.status_text = QLabel("系统状态信息将显示在这里")
        status_layout.addWidget(self.status_text)
        
        self.tab_widget.addTab(status_tab, "系统状态")
        
        return view_widget
    
    def _connect_signals(self):
        """连接信号与槽"""
        # 连接自定义信号
        self.mode_changed.connect(self._update_mode_display)
        self.connection_changed.connect(self._update_connection_status)
        self.data_received.connect(self._update_signal_display)
    
    def update_ui(self):
        """定时更新UI"""
        if not hasattr(self, 'system') or not self.system:
            return
            
        # 更新设备状态
        if self.system.device_manager:
            is_connected = self.system.device_manager.is_connected() 
            if is_connected:
                self.device_status_label.setText("已连接")
                self.connect_btn.setText("断开连接")
                
                # 获取并显示电池状态
                battery = self.system.device_manager.get_battery_level() if hasattr(self.system.device_manager, 'get_battery_level') else None
                if battery is not None:
                    self.device_battery_label.setText(f"电池: {battery}%")
                
                # 获取并显示信号质量
                quality = self.system.device_manager.get_signal_quality() if hasattr(self.system.device_manager, 'get_signal_quality') else None
                if quality is not None:
                    self.device_quality_label.setText(f"信号质量: {quality}%")
            else:
                self.device_status_label.setText("未连接")
                self.connect_btn.setText("连接设备")
                self.device_battery_label.setText("电池: N/A")
                self.device_quality_label.setText("信号质量: N/A")
        
        # 更新模式显示
        current_mode = self.system.mode.value if hasattr(self.system, 'mode') else "idle"
        self._update_mode_display(current_mode)
        
        # 获取最新数据并更新显示
        if self.system.device_manager and self.system.device_manager.is_connected():
            data = self.system.device_manager.get_latest_data()
            if data is not None:
                self.data_received.emit(data)
        
        # 获取解码结果并更新文本显示
        if self.system.decoder and self.system.mode.value == "decode":
            result = self.system.decoder.get_latest_result() if hasattr(self.system.decoder, 'get_latest_result') else None
            if result:
                self.decoded_text.setText(result)
                
        # 更新系统状态
        status_text = f"当前模式: {self._get_mode_display_name(self.system.mode.value)}\n"
        status_text += f"设备状态: {'已连接' if self.system.device_manager and self.system.device_manager.is_connected() else '未连接'}\n"
        
        # 其他状态信息
        if hasattr(self.system, 'get_status_info'):
            status_info = self.system.get_status_info()
            if status_info:
                for key, value in status_info.items():
                    status_text += f"{key}: {value}\n"
        
        self.status_text.setText(status_text)
    
    @pyqtSlot(str)
    def _update_mode_display(self, mode):
        """更新模式显示"""
        # 更新组合框
        index = self.mode_combo.findData(mode)
        if index >= 0:
            self.mode_combo.setCurrentIndex(index)
        
        # 更新状态栏
        mode_text = self._get_mode_display_name(mode)
        connection_text = "已连接" if self.system.device_manager and self.system.device_manager.is_connected() else "未连接"
        self.status_label.setText(f"模式: {mode_text} | 设备: {connection_text}")
    
    @pyqtSlot(bool)
    def _update_connection_status(self, connected):
        """更新连接状态"""
        if connected:
            self.device_status_label.setText("已连接")
            self.connect_btn.setText("断开连接")
        else:
            self.device_status_label.setText("未连接")
            self.connect_btn.setText("连接设备")
            self.device_battery_label.setText("电池: N/A")
            self.device_quality_label.setText("信号质量: N/A")
        
        # 更新状态栏
        mode_text = self._get_mode_display_name(self.system.mode.value)
        connection_text = "已连接" if connected else "未连接"
        self.status_label.setText(f"模式: {mode_text} | 设备: {connection_text}")
    
    @pyqtSlot(object)
    def _update_signal_display(self, data):
        """更新信号显示"""
        if data is None:
            return
            
        # 存储数据到缓冲区
        if self.data_buffer is None:
            # 初始化数据缓冲区，假设数据形状为(通道数, 样本数)
            buffer_size = 1000  # 缓冲区大小
            num_channels = data.shape[0] if hasattr(data, 'shape') and len(data.shape) > 0 else 8
            self.data_buffer = np.zeros((num_channels, buffer_size))
        
        # 更新数据缓冲区
        # 这里假设data的形状为(通道数, 新样本数)
        if hasattr(data, 'shape') and len(data.shape) > 0:
            # 移动现有数据
            new_samples = data.shape[1] if len(data.shape) > 1 else 1
            self.data_buffer = np.roll(self.data_buffer, -new_samples, axis=1)
            
            # 添加新数据
            if len(data.shape) > 1:
                self.data_buffer[:, -new_samples:] = data
            else:
                self.data_buffer[:, -1] = data
        
        # 更新信号图表
        sample_rate = 250  # 采样率，应该从设备配置中获取
        time_axis = np.linspace(-self.data_buffer.shape[1]/sample_rate, 0, self.data_buffer.shape[1])
        
        for i, curve in enumerate(self.channel_curves):
            if i < self.data_buffer.shape[0]:
                # 为了显示清晰，不同通道添加偏移
                offset = i * 100
                curve.setData(time_axis, self.data_buffer[i] + offset)
        
        # 更新频谱图
        # 选择第一个通道进行FFT分析
        if self.data_buffer.shape[0] > 0:
            # 计算FFT
            signal = self.data_buffer[0]
            fft = np.abs(np.fft.rfft(signal * np.hanning(len(signal))))
            # 计算频率轴
            freq = np.fft.rfftfreq(len(signal), 1/sample_rate)
            # 只显示50Hz以下的频率
            mask = freq <= 50
            self.spectrum_curve.setData(freq[mask], 20*np.log10(fft[mask] + 1e-10))
    
    def _on_connect_device(self):
        """连接设备按钮点击事件"""
        if not self.system.device_manager:
            QMessageBox.warning(self, "错误", "设备管理器未初始化")
            return
            
        if self.system.device_manager.is_connected():
            # 如果已连接，则断开
            if self.system.device_manager.disconnect():
                self.connection_changed.emit(False)
                self.statusBar.showMessage("设备已断开连接")
            else:
                QMessageBox.warning(self, "错误", "断开设备连接失败")
        else:
            # 如果未连接，则连接
            if self.system.device_manager.connect():
                self.connection_changed.emit(True)
                self.statusBar.showMessage("设备连接成功")
            else:
                QMessageBox.warning(self, "错误", "设备连接失败")
    
    def _on_disconnect_device(self):
        """断开设备连接"""
        if not self.system.device_manager:
            return
            
        if self.system.device_manager.is_connected():
            if self.system.device_manager.disconnect():
                self.connection_changed.emit(False)
                self.statusBar.showMessage("设备已断开连接")
    
    def _on_device_settings(self):
        """设备设置按钮点击事件"""
        # 这里可以实现设备设置对话框
        QMessageBox.information(self, "设备设置", "设备设置功能尚未实现")
    
    def _on_mode_changed(self, mode):
        """模式切换事件"""
        if self.system.change_mode(mode):
            self.mode_changed.emit(mode)
            self.statusBar.showMessage(f"已切换到{self._get_mode_display_name(mode)}模式")
        else:
            QMessageBox.warning(self, "错误", f"切换到{mode}模式失败")
    
    def _on_record_toggle(self, checked):
        """录制按钮切换事件"""
        if checked:
            # 开始录制
            self.record_btn.setText("停止录制")
            # TODO: 调用系统的录制功能
            self.statusBar.showMessage("开始录制数据")
        else:
            # 停止录制
            self.record_btn.setText("开始录制")
            # TODO: 调用系统的停止录制功能
            self.statusBar.showMessage("停止录制数据")
    
    def _on_open_file(self):
        """打开文件事件"""
        filename, _ = QFileDialog.getOpenFileName(
            self, 
            "打开数据文件", 
            "", 
            "CSV文件 (*.csv);;EDF文件 (*.edf);;全部文件 (*)"
        )
        
        if filename:
            # TODO: 调用系统的文件加载功能
            self.statusBar.showMessage(f"已加载文件: {filename}")
    
    def _on_save_file(self):
        """保存文件事件"""
        filename, _ = QFileDialog.getSaveFileName(
            self, 
            "保存数据", 
            "", 
            "CSV文件 (*.csv);;EDF文件 (*.edf);;全部文件 (*)"
        )
        
        if filename:
            # TODO: 调用系统的文件保存功能
            self.statusBar.showMessage(f"数据已保存到: {filename}")
    
    def _on_about(self):
        """关于对话框"""
        QMessageBox.about(
            self,
            "关于 TangiEEG",
            "TangiEEG - 脑机接口文本转换系统\n\n"
            "版本: 1.0.0\n"
            "基于Python和OpenBCI硬件的脑电信号采集和解码系统\n\n"
            "Copyright © 2023"
        )
    
    def _on_docs(self):
        """文档链接"""
        QMessageBox.information(
            self,
            "文档",
            "请访问项目GitHub页面获取完整文档:\n"
            "https://github.com/ericgguo/TangiEEG"
        )
    
    def _get_mode_display_name(self, mode_id):
        """获取模式的显示名称"""
        mode_names = {
            "idle": "空闲",
            "acquire": "数据采集",
            "monitor": "信号监测",
            "analyze": "离线分析",
            "decode": "在线解码",
            "simulate": "模拟数据"
        }
        return mode_names.get(mode_id, mode_id)
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确认对话框
        reply = QMessageBox.question(
            self, 
            '确认退出', 
            "确定要退出程序吗?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )
        
        if reply == QMessageBox.Yes:
            # 停止系统
            if self.system:
                self.system.stop()
            event.accept()
        else:
            event.ignore() 