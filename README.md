# TangiEEG：脑机接口文本转换系统

TangiEEG是一个基于脑电信号(EEG)的文本生成系统，旨在将脑电意图转化为文本输出。该系统使用OpenBCI硬件采集使用者默读某个文字时的脑电数据，并通过训练深度学习模型分析默读不同文字时脑电波的细微差异，最终实现EEG到文本的转换。
这个项目当前处于概念和基础结构的构建阶段，由于技术和资金的缺乏，整个研究开发（特别是信号的伪迹去除，特征增强等部分）可能会长达数年时间。不过，相信所有开发者甚至是普通人都对这项科技抱有充足的热情，未来精准的脑控系统也必将成为流行趋势。在此感谢所有可能会为该项目提供任何支持的人，我们可能正在开启人类历史上最重要的进程。
初期的词汇库可能会在中文的基础上构建，如果你有更好的想法，欢迎提供。开发者目前只是一个大一本科生，想交流，欢迎，Wechat：Ailikethatsme526

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
├── ui/                            # 用户界面模块
│   ├── __init__.py
│   ├── app.py                     # Streamlit应用程序主入口
│   ├── dashboard.py               # 仪表盘界面组件
│   ├── device_panel.py            # 设备连接和配置面板
│   ├── processing_panel.py        # 信号处理配置面板
│   ├── visualization_panel.py     # 信号可视化面板
│   └── session_manager.py         # 会话管理组件
│
├── visualization/                 # 数据可视化模块
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
│   └── text_display.py           # 文本显示
│       # 生成文本输出面板
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
- OpenBCI Cyton 或 Cyton+Daisy (16通道) 硬件

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行系统

TangiEEG提供两种运行方式：

#### 1. 命令行模式

命令行模式适合进行脚本化处理和自动化任务：

```bash
# 基本运行
python main.py

# 带参数运行
python main.py --config path/to/config.yaml --mode online_decoding --debug
```

主要参数:
- `--config`: 配置文件路径
- `--mode`: 运行模式 (offline_analysis, online_recording, online_decoding, simulation)
- `--debug`: 启用调试模式
- `--simulate`: 使用模拟数据而不是真实设备
- `--record`: 记录会话数据到文件

#### 2. 图形界面模式

图形界面模式提供了直观的可视化和交互功能：

```bash
# 启动Web界面
streamlit run ui/app.py
```

启动后，会自动打开浏览器访问本地Web界面(默认为http://localhost:8501)。

图形界面提供以下功能：
- 设备连接和配置
- 实时信号可视化
- 通道设置和阻抗检查
- 信号处理参数调整
- 解码结果显示和导出

要使用16通道配置，请在界面中选择"Daisy"设备类型。

## 项目状态

### 当前进展
- 完成了基础系统架构设计
- 实现了16通道EEG数据采集和可视化系统
- 完成了Streamlit用户界面的开发，支持实时信号监测和处理
- 实现了基本的信号预处理流程，包括滤波和伪迹检测

### 正在进行
- 优化信号处理算法，提高伪迹去除效率
- 开发特征提取和机器学习模型训练框架
- 收集和标注训练数据集
- 改进实时处理性能

### 未来计划
- 实现基于深度学习的EEG解码模型
- 开发适应性学习算法，提高系统对个体差异的适应能力
- 扩展支持更多硬件设备
- 创建开放数据集，促进社区研究

## 许可证

MIT

## 贡献

欢迎提交问题和贡献代码。如对项目有任何建议或想法，请通过以下方式联系：

Email: Ericguo526@gmail.com
Wechat: Ailikethatsme526
