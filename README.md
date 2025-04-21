# TangiEEG：脑机接口文本转换系统

TangiEEG是一个基于脑电信号(EEG)的文本生成系统，旨在将脑电意图转化为文本输出。该系统使用OpenBCI硬件采集脑电数据，并通过深度学习模型实现EEG到文本的转换。

## 项目结构

```
TangiEEG/
├── README.md                      # 项目说明文档
├── requirements.txt               # 项目依赖包列表
├── config/                        # 配置文件目录
│   ├── __init__.py
│   ├── device_config.py           # OpenBCI设备配置参数
│   ├── preprocessing_config.py    # 信号预处理参数配置
│   ├── model_config.py            # 解码模型配置参数
│   └── system_config.py           # 系统全局配置
│
├── data/                          # 数据存储目录
│   ├── raw/                       # 原始脑电数据
│   ├── processed/                 # 预处理后的数据
│   └── models/                    # 训练好的模型
│
├── acquisition/                   # 脑波数据获取模块
│   ├── __init__.py
│   ├── device_manager.py          # 设备连接和管理类
│   │   # 负责与OpenBCI Cyton硬件建立连接、配置参数、启动/停止数据流
│   │
│   ├── data_streamer.py           # 数据流处理类
│   │   # 处理实时数据流，包括数据缓存、时间戳同步和数据格式转换
│   │
│   ├── data_recorder.py           # 数据记录类
│   │   # 将采集的脑电数据保存为标准格式文件(HDF5/CSV)
│   │
│   └── acquisition_utils.py       # 数据采集辅助工具
│       # 提供信号质量检测、通道阻抗测量等功能
│
├── preprocessing/                 # 数据预处理模块
│   ├── __init__.py
│   ├── filters.py                 # 滤波器集合
│   │   # 实现带通、陷波等数字滤波器，去除电源噪声和基线漂移
│   │
│   ├── artifact_removal.py        # 伪迹去除
│   │   # 眨眼、肌电等伪迹的自动检测与去除，使用ICA或阈值法
│   │
│   ├── signal_enhancer.py         # 信号增强
│   │   # 实现空间滤波、信号平滑等增强信号质量的方法
│   │
│   ├── feature_extraction.py      # 特征提取
│   │   # 提取时域、频域和时频域特征
│   │
│   └── data_segmentation.py       # 数据分段
│       # 将连续数据分割为适合解码的时间窗口
│
├── decoding/                      # 信号解码模块
│   ├── __init__.py
│   ├── models/                    # 解码模型子目录
│   │   ├── __init__.py
│   │   ├── cnn_model.py           # CNN模型架构
│   │   │   # 使用卷积神经网络处理时空特征
│   │   │
│   │   ├── rnn_model.py           # RNN模型架构
│   │   │   # 处理序列信息，捕捉时间依赖性
│   │   │
│   │   ├── hybrid_model.py        # 混合模型架构
│   │   │   # 组合CNN和RNN的优势
│   │   │
│   │   └── dewave_model.py        # DeWave模型实现
│   │       # 基于离散编码的EEG到文本转换模型
│   │
│   ├── trainer.py                 # 模型训练器
│   │   # 管理模型训练过程、验证和超参数优化
│   │
│   ├── decoder.py                 # 解码器主类
│   │   # 使用训练好的模型将预处理后的EEG解码为原始表示
│   │
│   ├── evaluator.py               # 模型评估
│   │   # 评估解码性能的指标和方法
│   │
│   └── intent_detector.py         # 意图检测器
│       # 检测用户是否有通信意图，避免误触发
│
├── generation/                    # 文本生成模块
│   ├── __init__.py
│   ├── language_model.py          # 语言模型整合
│   │   # 使用预训练语言模型增强解码结果
│   │
│   ├── text_formatter.py          # 文本格式化
│   │   # 将解码结果格式化为自然文本
│   │
│   ├── prediction_engine.py       # 文本预测引擎
│   │   # 根据部分解码结果预测可能的完整文本
│   │
│   └── output_manager.py          # 输出管理器
│       # 管理文本输出到界面或其他目标
│
├── visualization/                 # 可视化模块
│   ├── __init__.py
│   ├── signal_viewer.py           # 信号可视化
│   │   # 实时显示原始和处理后的EEG信号
│   │
│   ├── spectrogram_viewer.py      # 频谱图可视化
│   │   # 显示信号的时频特性
│   │
│   ├── decoding_visualizer.py     # 解码过程可视化
│   │   # 可视化解码模型的中间结果
│   │
│   └── text_display.py            # 文本显示界面
│       # 简单的文本输出面板，显示最终生成的文本
│
├── utils/                         # 工具函数模块
│   ├── __init__.py
│   ├── data_utils.py              # 数据处理工具
│   │   # 通用数据格式转换、加载和保存函数
│   │
│   ├── math_utils.py              # 数学工具函数
│   │   # 信号处理所需的数学函数
│   │
│   ├── io_utils.py                # 输入输出工具
│   │   # 文件读写、数据导入导出函数
│   │
│   └── logging_utils.py           # 日志工具
│       # 系统日志记录功能
│
├── tests/                         # 测试模块
│   ├── __init__.py
│   ├── test_acquisition.py        # 数据采集测试
│   ├── test_preprocessing.py      # 预处理测试
│   ├── test_decoding.py           # 解码测试
│   └── test_generation.py         # 生成测试
│
└── main.py                        # 主程序入口
    # 整合所有模块，提供命令行接口
```

## 安装与使用

### 环境要求
- Python 3.8+
- OpenBCI Cyton 硬件或兼容设备

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行系统

```bash
python main.py
```

## 许可证

MIT

## 贡献

欢迎提交问题和贡献代码。
