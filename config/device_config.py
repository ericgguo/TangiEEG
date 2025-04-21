"""
设备配置模块 - 包含OpenBCI硬件的配置参数
"""

# OpenBCI设备基本配置
OPENBCI_CONFIG = {
    # 基本设备参数
    'board_type': 'cyton_daisy',  # 'cyton'、'cyton_daisy'或'ganglion'
    'serial_port': None,    # 自动检测，或指定如'/dev/ttyUSB0'或'COM3'
    'mac_address': None,    # 蓝牙设备地址（仅Ganglion）
    'serial_timeout': 5,    # 串口连接超时时间(秒)
    
    # 数据流参数
    'sample_rate': 250,     # 采样率(Hz): 250Hz是Cyton默认值
    'streaming_package_size': 100,  # 每次数据包大小（样本数）
    
    # 通道配置
    'channels': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],  # 使用的通道号列表
    'channel_names': ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'P3', 'Pz', 'P4', 'O1'],  # 通道名称
    'gain': 24,             # 24 = x24 增益值，影响信号量程
    
    # 滤波器设置（硬件级别）
    'use_filters': True,    # 是否启用板载滤波器
    'notch_filter': 50,     # 电源线路噪声陷波滤波器(Hz)，50Hz(欧洲)/60Hz(美国)
    'bandpass_filter': {    # 带通滤波器设置
        'enabled': True,
        'low_cutoff': 1.0,  # 高通滤波器截止频率(Hz)
        'high_cutoff': 50.0 # 低通滤波器截止频率(Hz)
    },
    
    # 阻抗检测设置
    'impedance_test': {
        'enabled': True,    # 是否自动检测通道阻抗
        'test_duration': 5, # 阻抗测试持续时间(秒)
        'threshold': 1e6    # 阻抗警告阈值(欧姆)
    }
}

# 电极布局配置
ELECTRODE_LAYOUT = {
    'layout_type': '10-20',  # 使用国际10-20系统
    'custom_positions': {},  # 自定义电极位置（如果需要）
    'reference': 'A1',       # 参考电极位置
    'ground': 'A2'           # 接地电极位置
}

# 数据保存配置
DATA_SAVE_CONFIG = {
    'enabled': True,         # 是否保存原始数据
    'format': 'csv',         # 保存格式: 'csv', 'hdf5'
    'save_path': '../data/raw',  # 保存路径
    'file_prefix': 'session', # 文件名前缀
    'add_timestamp': True     # 是否在文件名中添加时间戳
}

# EOG眼电图采集设置
EOG_CONFIG = {
    'enabled': True,          # 是否采集EOG信号
    'channels': [15, 16],     # EOG通道编号，使用最后两个通道
    'positions': ['HEOG', 'VEOG'],  # 水平和垂直EOG
    'sample_rate': 250        # EOG采样率
}

def get_device_config():
    """返回当前设备配置"""
    return {
        'openbci': OPENBCI_CONFIG,
        'electrode_layout': ELECTRODE_LAYOUT,
        'data_save': DATA_SAVE_CONFIG,
        'eog': EOG_CONFIG
    }
