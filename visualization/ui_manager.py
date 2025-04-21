"""
用户界面管理器 - 负责协调和管理所有图形界面组件
支持多种UI模式：Qt界面、Streamlit界面、命令行界面等
"""

import logging
import sys
import threading
import time
from enum import Enum
from pathlib import Path

# 导入工具模块
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from utils.logging_utils import get_logger

# UI类型枚举
class UIType(Enum):
    """支持的用户界面类型"""
    CLI = "cli"           # 命令行界面
    QT = "qt"             # Qt图形界面
    STREAMLIT = "web"     # Streamlit网页界面
    NONE = "none"         # 无界面

class UIManager:
    """用户界面管理器类"""
    
    def __init__(self, system, ui_type=None):
        """
        初始化UI管理器
        
        Args:
            system: TangiEEGSystem实例，用于系统交互
            ui_type: UI类型，支持'cli', 'qt', 'web', 'none'
        """
        self.logger = get_logger("ui")
        self.system = system
        self.ui_thread = None
        self.running = False
        self.ui_components = {}
        self.ui_app = None
        
        # 确定UI类型
        if ui_type is None:
            # 如果未指定，根据命令行参数和可用性决定
            if system.args.no_ui:
                ui_type = UIType.CLI
            else:
                # 尝试按Qt > Streamlit > CLI的顺序加载
                try:
                    import PyQt5
                    ui_type = UIType.QT
                except ImportError:
                    try:
                        import streamlit
                        ui_type = UIType.STREAMLIT
                    except ImportError:
                        ui_type = UIType.CLI
        elif isinstance(ui_type, str):
            try:
                ui_type = UIType(ui_type)
            except ValueError:
                self.logger.warning(f"未知UI类型: {ui_type}，回退到CLI模式")
                ui_type = UIType.CLI
                
        self.ui_type = ui_type
        self.logger.info(f"UI管理器初始化，使用 {self.ui_type.value} 界面模式")
    
    def start(self):
        """启动UI"""
        if self.running:
            self.logger.warning("UI已经在运行中")
            return
            
        try:
            if self.ui_type == UIType.NONE:
                self.logger.info("无界面模式，跳过UI启动")
                return
                
            elif self.ui_type == UIType.CLI:
                # 命令行界面，在主线程中运行简单的状态输出
                self.logger.info("启动命令行界面")
                self._init_cli_ui()
                self.running = True
                
            elif self.ui_type == UIType.QT:
                # Qt图形界面，在单独线程中运行
                self.logger.info("启动Qt图形界面")
                self.ui_thread = threading.Thread(target=self._run_qt_ui, daemon=True)
                self.ui_thread.start()
                self.running = True
                
            elif self.ui_type == UIType.STREAMLIT:
                # Streamlit网页界面，在单独线程中运行
                self.logger.info("启动Streamlit网页界面")
                self.ui_thread = threading.Thread(target=self._run_streamlit_ui, daemon=True)
                self.ui_thread.start()
                self.running = True
            
            self.logger.info("UI启动完成")
            
        except Exception as e:
            self.logger.exception(f"UI启动失败: {e}")
            # 启动失败，回退到命令行模式
            self.ui_type = UIType.CLI
            self._init_cli_ui()
            self.running = True
    
    def stop(self):
        """停止UI"""
        if not self.running:
            return
            
        self.logger.info("停止UI...")
        self.running = False
        
        # 等待UI线程结束
        if self.ui_thread and self.ui_thread.is_alive():
            self.ui_thread.join(timeout=2.0)
        
        # 关闭特定UI相关资源
        if self.ui_type == UIType.QT and self.ui_app:
            try:
                self.ui_app.quit()
            except:
                pass
        
        self.logger.info("UI已停止")
    
    def update(self):
        """更新UI显示"""
        if not self.running:
            return
            
        if self.ui_type == UIType.CLI:
            # 命令行界面定期刷新状态
            self._update_cli_ui()
            
        # Qt和Streamlit界面有自己的更新机制，不需要在此处理
    
    def show_message(self, message, level="info"):
        """显示消息到界面"""
        if not self.running:
            return
            
        if self.ui_type == UIType.CLI:
            # 在命令行中打印消息
            if level == "error":
                print(f"\033[91m[错误] {message}\033[0m")
            elif level == "warning":
                print(f"\033[93m[警告] {message}\033[0m")
            else:
                print(f"[信息] {message}")
        
        # 其他UI类型的消息显示在相应的UI组件中实现
    
    def get_component(self, name):
        """获取指定名称的UI组件"""
        return self.ui_components.get(name)
    
    def register_component(self, name, component):
        """注册UI组件"""
        self.ui_components[name] = component
        return component
    
    def _init_cli_ui(self):
        """初始化命令行界面"""
        # 简单的命令行UI，显示状态和支持基本命令
        self.last_cli_update = 0
        print("===== TangiEEG 命令行界面 =====")
        print("可用命令:")
        print("  help            - 显示帮助信息")
        print("  status          - 显示系统状态")
        print("  mode <模式名>    - 切换系统模式")
        print("  connect         - 连接设备")
        print("  disconnect      - 断开设备连接")
        print("  record <文件名>  - 开始记录数据")
        print("  stop            - 停止记录")
        print("  quit/exit       - 退出程序")
        print("============================")
        
        # 创建命令处理线程
        self.cli_thread = threading.Thread(target=self._run_cli_command_handler, daemon=True)
        self.cli_thread.start()
    
    def _run_cli_command_handler(self):
        """CLI命令处理线程"""
        while self.running:
            try:
                # 获取用户输入
                cmd = input("TangiEEG> ").strip()
                
                # 解析和执行命令
                if cmd in ["quit", "exit"]:
                    print("正在退出...")
                    self.system.stop()
                    break
                    
                elif cmd == "help":
                    print("可用命令:")
                    print("  help            - 显示帮助信息")
                    print("  status          - 显示系统状态")
                    print("  mode <模式名>    - 切换系统模式")
                    print("  connect         - 连接设备")
                    print("  disconnect      - 断开设备连接")
                    print("  record <文件名>  - 开始记录数据")
                    print("  stop            - 停止记录")
                    print("  quit/exit       - 退出程序")
                
                elif cmd == "status":
                    print(f"系统状态:")
                    print(f"  当前模式: {self.system.mode.value}")
                    print(f"  设备连接: {'已连接' if self.system.device_manager and self.system.device_manager.is_connected() else '未连接'}")
                    # 添加更多状态信息
                
                elif cmd.startswith("mode "):
                    # 切换模式
                    new_mode = cmd[5:].strip()
                    if self.system.change_mode(new_mode):
                        print(f"模式已切换到: {new_mode}")
                    else:
                        print(f"无法切换到模式: {new_mode}")
                
                elif cmd == "connect":
                    # 连接设备
                    if self.system.device_manager:
                        if self.system.device_manager.connect():
                            print("设备连接成功")
                        else:
                            print("设备连接失败")
                    else:
                        print("设备管理器未初始化")
                
                elif cmd == "disconnect":
                    # 断开设备
                    if self.system.device_manager and self.system.device_manager.is_connected():
                        self.system.device_manager.disconnect()
                        print("设备已断开连接")
                    else:
                        print("设备未连接")
                
                # 添加更多命令处理
                        
                else:
                    print(f"未知命令: {cmd}")
                
            except EOFError:
                # Ctrl+D
                print("\n正在退出...")
                self.system.stop()
                break
                
            except Exception as e:
                print(f"命令处理错误: {e}")
    
    def _update_cli_ui(self):
        """更新命令行界面状态"""
        # 每5秒更新一次状态
        current_time = time.time()
        if current_time - self.last_cli_update > 5:
            self.last_cli_update = current_time
            # 更新状态信息（不打扰用户输入）
            # 这里可以实现更复杂的终端UI，例如使用curses库
    
    def _run_qt_ui(self):
        """启动Qt图形界面"""
        try:
            from PyQt5.QtWidgets import QApplication
            from .qt_ui import MainWindow
            
            # 创建Qt应用
            self.ui_app = QApplication([])
            
            # 创建主窗口
            main_window = MainWindow(self.system)
            self.register_component("main_window", main_window)
            
            # 显示窗口
            main_window.show()
            
            # 运行事件循环
            self.ui_app.exec_()
            
        except ImportError:
            self.logger.error("无法导入PyQt5模块，请安装后再试")
        except Exception as e:
            self.logger.exception(f"启动Qt界面失败: {e}")
    
    def _run_streamlit_ui(self):
        """启动Streamlit网页界面"""
        try:
            import streamlit as st
            import subprocess
            
            # Streamlit需要作为单独的进程运行
            # 这里我们启动一个子进程来运行它
            streamlit_script = Path(__file__).parent / "streamlit_ui.py"
            
            if not streamlit_script.exists():
                self.logger.error(f"找不到Streamlit UI脚本: {streamlit_script}")
                return
                
            # 构建命令并启动
            cmd = [sys.executable, "-m", "streamlit", "run", str(streamlit_script)]
            process = subprocess.Popen(cmd)
            
            # 注册进程以便后续清理
            self.register_component("streamlit_process", process)
            
            # 等待进程结束
            process.wait()
            
        except ImportError:
            self.logger.error("无法导入streamlit模块，请安装后再试")
        except Exception as e:
            self.logger.exception(f"启动Streamlit界面失败: {e}") 