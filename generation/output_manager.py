"""
输出管理器 - 管理文本输出到界面或其他目标
"""

import logging
import time
import json
import numpy as np
from pathlib import Path
import os

# 创建logger
logger = logging.getLogger('output_manager')

class OutputManager:
    """输出管理器主类 - 处理解码结果的显示和导出"""
    
    def __init__(self, output_mode='text', history_length=10, confidence_threshold=0.6):
        """
        初始化输出管理器
        
        参数:
            output_mode (str): 输出模式，'text'(文本)或'command'(命令)
            history_length (int): 保存历史记录的最大长度
            confidence_threshold (float): 最小置信度阈值
        """
        self.output_mode = output_mode
        self.history_length = history_length
        self.confidence_threshold = confidence_threshold
        
        # 初始化输出历史
        self.output_history = []
        
        # 上次输出时间
        self.last_output_time = 0
        
        # 默认去抖时间(秒)
        self.debounce_time = 1.0
        
        logger.info(f"输出管理器初始化: 模式={output_mode}, 历史长度={history_length}")
    
    def display(self, decoded_result, timestamp=None):
        """
        显示解码结果
        
        参数:
            decoded_result (dict): 解码结果字典
            timestamp (float): 时间戳，如果为None则使用当前时间
            
        返回:
            str: 格式化的输出文本
        """
        if decoded_result is None or not isinstance(decoded_result, dict):
            logger.warning("无效的解码结果，无法显示")
            return None
        
        # 设置时间戳
        if timestamp is None:
            timestamp = time.time()
        
        # 去抖动 - 避免短时间内多次输出
        if timestamp - self.last_output_time < self.debounce_time:
            logger.debug(f"去抖过滤: 距上次输出仅{timestamp - self.last_output_time:.2f}秒")
            return None
        
        try:
            # 提取最高置信度的结果
            best_result = self._get_best_result(decoded_result)
            
            if best_result is None:
                logger.debug("没有超过置信度阈值的结果")
                return None
            
            # 根据输出模式格式化结果
            if self.output_mode == 'text':
                formatted_output = self._format_text_output(best_result)
            elif self.output_mode == 'command':
                formatted_output = self._format_command_output(best_result)
            else:
                formatted_output = str(best_result)
            
            # 更新历史记录
            self._update_history(best_result, formatted_output, timestamp)
            
            # 更新最后输出时间
            self.last_output_time = timestamp
            
            # 记录输出
            logger.info(f"输出: {formatted_output} (置信度: {best_result.get('confidence', 0):.2f})")
            
            return formatted_output
            
        except Exception as e:
            logger.error(f"显示解码结果时出错: {e}")
            return None
    
    def _get_best_result(self, decoded_result):
        """提取置信度最高的解码结果"""
        best_confidence = 0
        best_result = None
        
        for segment_id, result in decoded_result.items():
            confidence = result.get('confidence', 0)
            
            # 检查置信度阈值
            if confidence >= self.confidence_threshold and confidence > best_confidence:
                best_confidence = confidence
                best_result = result.copy()  # 创建副本以避免修改原始数据
                best_result['segment_id'] = segment_id
        
        return best_result
    
    def _format_text_output(self, result):
        """格式化文本输出"""
        # 提取标签或意图
        if 'label' in result:
            output = result['label']
        elif 'intent' in result:
            output = result['intent']
        else:
            output = "未知输出"
        
        return output
    
    def _format_command_output(self, result):
        """格式化命令输出"""
        # 将意图转换为命令格式
        if 'intent' in result:
            intent = result['intent']
            
            # 根据意图类型构建命令
            if intent in ['左', '右', '上', '下']:
                return f"移动_{intent}"
            elif intent in ['选择', '确认']:
                return "选择"
            elif intent in ['取消', '返回']:
                return "取消"
            else:
                return f"命令_{intent}"
        
        elif 'label' in result:
            return f"状态_{result['label']}"
        
        else:
            return "未知命令"
    
    def _update_history(self, result, formatted_output, timestamp):
        """更新输出历史记录"""
        # 创建历史记录条目
        history_entry = {
            'time': timestamp,
            'time_str': time.strftime('%H:%M:%S', time.localtime(timestamp)),
            'output': formatted_output,
            'confidence': result.get('confidence', 0),
            'original_result': result
        }
        
        # 添加到历史记录
        self.output_history.append(history_entry)
        
        # 保持历史记录长度限制
        if len(self.output_history) > self.history_length:
            self.output_history = self.output_history[-self.history_length:]
    
    def get_history(self, count=None):
        """
        获取历史记录
        
        参数:
            count (int): 要获取的记录数量，如果为None则返回全部
            
        返回:
            list: 历史记录列表
        """
        if count is None or count >= len(self.output_history):
            return self.output_history.copy()
        else:
            return self.output_history[-count:].copy()
    
    def clear_history(self):
        """清空历史记录"""
        self.output_history = []
        logger.info("已清空输出历史记录")
    
    def export_history(self, file_path, format='json'):
        """
        导出历史记录到文件
        
        参数:
            file_path (str): 输出文件路径
            format (str): 输出格式，'json'或'csv'
            
        返回:
            bool: 导出是否成功
        """
        if not self.output_history:
            logger.warning("历史记录为空，无法导出")
            return False
        
        try:
            # 创建目录
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            if format.lower() == 'json':
                # 导出为JSON格式
                with open(file_path, 'w', encoding='utf-8') as f:
                    # 转换为可序列化的格式
                    export_data = []
                    for entry in self.output_history:
                        export_entry = dict(entry)
                        # 移除不可序列化的对象
                        if 'original_result' in export_entry:
                            export_entry['original_result'] = str(export_entry['original_result'])
                        export_data.append(export_entry)
                    
                    json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            elif format.lower() == 'csv':
                # 导出为CSV格式
                import csv
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    # 写入表头
                    writer.writerow(['时间', '输出', '置信度'])
                    # 写入数据
                    for entry in self.output_history:
                        writer.writerow([
                            entry['time_str'],
                            entry['output'],
                            entry['confidence']
                        ])
            
            else:
                logger.error(f"不支持的导出格式: {format}")
                return False
            
            logger.info(f"历史记录已成功导出到: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"导出历史记录失败: {e}")
            return False
    
    def set_confidence_threshold(self, threshold):
        """设置置信度阈值"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"置信度阈值已设置为: {self.confidence_threshold}")
    
    def set_debounce_time(self, seconds):
        """设置去抖时间"""
        self.debounce_time = max(0.0, seconds)
        logger.info(f"去抖时间已设置为: {self.debounce_time}秒")

class TextFormatter:
    """文本格式化器 - 增强输出文本的自然度和可读性"""
    
    def __init__(self, language='zh'):
        """
        初始化文本格式化器
        
        参数:
            language (str): 语言代码，'zh'为中文，'en'为英文
        """
        self.language = language
        
        # 语言特定的格式化参数
        self.lang_params = {
            'zh': {
                'sentence_end': ['。', '！', '？', '…'],
                'connecting_words': ['和', '与', '并且', '而', '但是', '然而', '因此'],
                'punctuation': ['，', '；', '：', '。', '！', '？']
            },
            'en': {
                'sentence_end': ['.', '!', '?', '...'],
                'connecting_words': ['and', 'or', 'but', 'however', 'therefore', 'thus'],
                'punctuation': [',', ';', ':', '.', '!', '?']
            }
        }
        
        # 获取当前语言的参数
        self.params = self.lang_params.get(language, self.lang_params['en'])
        
        logger.info(f"文本格式化器初始化: 语言={language}")
    
    def format_text(self, text, add_punctuation=True, capitalize=True):
        """
        格式化文本以提高可读性
        
        参数:
            text (str): 需要格式化的文本
            add_punctuation (bool): 是否添加标点符号
            capitalize (bool): 是否大写句首字母（英文）
            
        返回:
            str: 格式化后的文本
        """
        if not text:
            return text
        
        try:
            # 去除多余空格
            formatted = ' '.join(text.split())
            
            # 添加标点符号
            if add_punctuation and not self._ends_with_punctuation(formatted):
                formatted = self._add_end_punctuation(formatted)
            
            # 大写句首字母（仅英文）
            if capitalize and self.language == 'en':
                formatted = self._capitalize_sentences(formatted)
            
            return formatted
            
        except Exception as e:
            logger.error(f"格式化文本失败: {e}")
            return text
    
    def combine_texts(self, text_list, as_sentences=True):
        """
        将多个文本片段组合成一段连贯的文本
        
        参数:
            text_list (list): 文本片段列表
            as_sentences (bool): 是否作为独立句子组合
            
        返回:
            str: 组合后的文本
        """
        if not text_list:
            return ""
        
        try:
            formatted_texts = []
            
            for text in text_list:
                # 格式化每个文本片段
                formatted = self.format_text(text, add_punctuation=as_sentences)
                formatted_texts.append(formatted)
            
            # 根据需要组合文本
            if as_sentences:
                # 作为独立句子组合
                combined = ' '.join(formatted_texts)
            else:
                # 使用连接词组合
                if len(formatted_texts) == 1:
                    combined = formatted_texts[0]
                elif len(formatted_texts) == 2:
                    # 随机选择一个连接词
                    connector = np.random.choice(self.params['connecting_words'])
                    combined = f"{formatted_texts[0]} {connector} {formatted_texts[1]}"
                else:
                    # 对于多个片段，使用逗号和连接词
                    combined = ', '.join(formatted_texts[:-1])
                    connector = np.random.choice(self.params['connecting_words'])
                    combined = f"{combined} {connector} {formatted_texts[-1]}"
                
                # 确保组合后的文本有结束标点
                if not self._ends_with_punctuation(combined):
                    combined = self._add_end_punctuation(combined)
            
            return combined
            
        except Exception as e:
            logger.error(f"组合文本失败: {e}")
            return ' '.join(text_list)
    
    def _ends_with_punctuation(self, text):
        """检查文本是否以标点符号结束"""
        if not text:
            return False
        
        return any(text.endswith(p) for p in self.params['punctuation'])
    
    def _add_end_punctuation(self, text):
        """添加结束标点"""
        # 根据文本内容决定添加什么标点
        if '?' in text or text.lower().startswith(('what', 'why', 'how', 'when', 'where', 'who', '什么', '为什么', '如何', '何时', '哪里', '谁')):
            return text + self.params['sentence_end'][2]  # 问号
        elif any(exclamation in text.lower() for exclamation in ('!', 'wow', 'amazing', 'great', 'terrible', '太好了', '真棒', '厉害', '不可思议')):
            return text + self.params['sentence_end'][1]  # 感叹号
        else:
            return text + self.params['sentence_end'][0]  # 句号
    
    def _capitalize_sentences(self, text):
        """将句子首字母大写（英文专用）"""
        if self.language != 'en':
            return text
        
        # 按句子分割
        sentences = []
        current = []
        
        for char in text:
            current.append(char)
            if char in self.params['sentence_end']:
                sentences.append(''.join(current))
                current = []
        
        if current:
            sentences.append(''.join(current))
        
        # 大写每个句子的首字母
        capitalized = []
        for sentence in sentences:
            trimmed = sentence.lstrip()
            if trimmed and trimmed[0].isalpha():
                # 保留前导空格
                leading_spaces = sentence[:len(sentence) - len(trimmed)]
                capitalized.append(leading_spaces + trimmed[0].upper() + trimmed[1:])
            else:
                capitalized.append(sentence)
        
        return ''.join(capitalized)


class PredictionEngine:
    """文本预测引擎 - 根据部分解码结果预测可能的完整文本"""
    
    def __init__(self, prediction_mode='simple', language='zh'):
        """
        初始化文本预测引擎
        
        参数:
            prediction_mode (str): 预测模式 'simple'或'model'
            language (str): 语言代码，'zh'为中文，'en'为英文
        """
        self.prediction_mode = prediction_mode
        self.language = language
        
        # 预定义的常用短语和词语
        self.common_phrases = {
            'zh': {
                '你好': ['你好', '你好吗', '你好啊', '你好呀'],
                '我想': ['我想说', '我想要', '我想问', '我想知道'],
                '谢谢': ['谢谢你', '非常感谢', '谢谢帮助', '谢谢理解'],
                '请问': ['请问一下', '请问可以', '请问能否', '请问如何']
            },
            'en': {
                'hello': ['hello there', 'hello world', 'hello everyone', 'hello and welcome'],
                'I want': ['I want to', 'I want some', 'I want you to', 'I want this'],
                'thank': ['thank you', 'thank you very much', 'thanks for your help', 'thanks for understanding'],
                'please': ['please help', 'please consider', 'please allow me', 'please understand']
            }
        }
        
        # 获取当前语言的短语
        self.phrases = self.common_phrases.get(language, self.common_phrases['en'])
        
        # 上下文记忆
        self.context_memory = []
        self.memory_size = 5
        
        logger.info(f"文本预测引擎初始化: 模式={prediction_mode}, 语言={language}")
    
    def predict_completion(self, partial_text, top_k=3):
        """
        预测文本的可能完成形式
        
        参数:
            partial_text (str): 部分解码文本
            top_k (int): 返回的预测数量
            
        返回:
            list: 可能的文本完成列表，按概率从高到低排序
        """
        if not partial_text:
            return []
        
        try:
            # 根据预测模式选择方法
            if self.prediction_mode == 'simple':
                predictions = self._simple_prediction(partial_text, top_k)
            elif self.prediction_mode == 'model':
                predictions = self._model_prediction(partial_text, top_k)
            else:
                predictions = self._simple_prediction(partial_text, top_k)
            
            # 更新上下文记忆
            self._update_context(partial_text)
            
            return predictions
            
        except Exception as e:
            logger.error(f"预测文本完成失败: {e}")
            return []
    
    def _simple_prediction(self, partial_text, top_k):
        """简单预测 - 基于预定义短语和上下文匹配"""
        predictions = []
        
        # 检查是否匹配预定义短语的开头
        for prefix, completions in self.phrases.items():
            if partial_text.startswith(prefix):
                # 添加匹配的完成形式
                predictions.extend(completions)
        
        # 根据上下文记忆添加预测
        for past_text in self.context_memory:
            if past_text.startswith(partial_text) and past_text != partial_text:
                predictions.append(past_text)
        
        # 去重并限制数量
        unique_predictions = []
        for pred in predictions:
            if pred not in unique_predictions:
                unique_predictions.append(pred)
                if len(unique_predictions) >= top_k:
                    break
        
        return unique_predictions
    
    def _model_prediction(self, partial_text, top_k):
        """模型预测 - 使用语言模型进行预测（需要外部模型）"""
        # 注意：此方法需要集成外部语言模型实现
        # 这里仅作为占位实现
        logger.warning("模型预测功能尚未实现，使用简单预测代替")
        return self._simple_prediction(partial_text, top_k)
    
    def _update_context(self, text):
        """更新上下文记忆"""
        if text and text not in self.context_memory:
            self.context_memory.append(text)
            
            # 保持记忆大小限制
            if len(self.context_memory) > self.memory_size:
                self.context_memory = self.context_memory[-self.memory_size:]
    
    def clear_context(self):
        """清空上下文记忆"""
        self.context_memory = []
        logger.info("已清空上下文记忆")
    
    def set_memory_size(self, size):
        """设置记忆大小"""
        self.memory_size = max(1, size)
        # 修剪现有记忆
        if len(self.context_memory) > self.memory_size:
            self.context_memory = self.context_memory[-self.memory_size:]
        logger.info(f"记忆大小已设置为: {self.memory_size}")
    
    def add_custom_phrases(self, phrases_dict):
        """
        添加自定义短语
        
        参数:
            phrases_dict (dict): 格式为{前缀: [完成列表]}的短语字典
        """
        try:
            for prefix, completions in phrases_dict.items():
                if prefix in self.phrases:
                    # 合并现有短语
                    self.phrases[prefix].extend(completions)
                    # 去重
                    self.phrases[prefix] = list(set(self.phrases[prefix]))
                else:
                    # 添加新短语
                    self.phrases[prefix] = completions
            
            logger.info(f"已添加{len(phrases_dict)}个自定义短语")
            
        except Exception as e:
            logger.error(f"添加自定义短语失败: {e}")
