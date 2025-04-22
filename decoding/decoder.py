"""
解码器主类 - 将预处理后的EEG特征解码为文本意图
"""

import logging
import numpy as np
import joblib
from pathlib import Path
import os
import pickle

# 创建logger
logger = logging.getLogger('decoder')

class EEGDecoder:
    """EEG解码器基类"""
    
    def __init__(self, model_path=None, threshold=0.5):
        """
        初始化EEG解码器
        
        参数:
            model_path (str): 模型文件路径，如果为None则使用默认模型
            threshold (float): 解码阈值，用于二分类决策
        """
        self.model = None
        self.model_path = model_path
        self.threshold = threshold
        self.classes = []
        self.is_loaded = False
        
        logger.info("EEG解码器初始化")
    
    def load_model(self, model_path=None):
        """
        加载解码模型
        
        参数:
            model_path (str): 模型文件路径，如果为None则使用初始化时设置的路径
        
        返回:
            bool: 模型加载是否成功
        """
        try:
            path = model_path or self.model_path
            
            if path is None:
                logger.warning("未指定模型路径，使用简单分类器")
                self.is_loaded = True
                return True
            
            if not os.path.exists(path):
                logger.error(f"模型文件不存在: {path}")
                return False
            
            # 根据文件扩展名选择加载方法
            if path.endswith('.pkl') or path.endswith('.pickle'):
                with open(path, 'rb') as f:
                    self.model = pickle.load(f)
            elif path.endswith('.joblib'):
                self.model = joblib.load(path)
            else:
                logger.error(f"不支持的模型文件格式: {path}")
                return False
            
            # 尝试获取类别标签
            if hasattr(self.model, 'classes_'):
                self.classes = self.model.classes_
            
            logger.info(f"模型加载成功: {path}")
            self.is_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            self.is_loaded = False
            return False
    
    def decode(self, features, return_probabilities=False):
        """
        解码EEG特征
        
        参数:
            features (numpy.ndarray): EEG特征向量或矩阵
            return_probabilities (bool): 是否返回概率而不是类别标签
        
        返回:
            dict: 解码结果，包含预测标签和置信度
        """
        if features is None or (isinstance(features, np.ndarray) and features.size == 0):
            logger.warning("特征为空，无法解码")
            return None
        
        try:
            # 检查特征维度
            if len(features.shape) == 1:
                features = features.reshape(1, -1)  # 转换为2D格式 (1, n_features)
            
            # 检查模型是否加载
            if not self.is_loaded:
                success = self.load_model()
                if not success:
                    logger.error("模型未加载，无法解码")
                    return None
            
            # 如果没有模型，使用简单规则
            if self.model is None:
                return self._simple_decode(features)
            
            # 使用模型预测
            if hasattr(self.model, 'predict_proba'):
                # 获取类别概率
                probabilities = self.model.predict_proba(features)
                
                # 获取预测标签
                if return_probabilities:
                    predictions = probabilities
                else:
                    predictions = self.model.predict(features)
                
                # 提取最大概率
                max_probs = np.max(probabilities, axis=1)
                
                # 创建结果字典
                results = {}
                for i in range(len(predictions)):
                    pred_label = predictions[i] if not return_probabilities else self.classes[np.argmax(probabilities[i])]
                    confidence = max_probs[i]
                    
                    # 检查是否超过阈值
                    if confidence >= self.threshold:
                        results[f'segment_{i}'] = {
                            'label': str(pred_label),
                            'confidence': float(confidence),
                            'probabilities': {str(cls): float(prob) for cls, prob in zip(self.classes, probabilities[i])} if return_probabilities else None
                        }
                
                return results
            
            else:
                # 只能获取预测标签
                predictions = self.model.predict(features)
                
                # 创建结果字典
                results = {}
                for i in range(len(predictions)):
                    results[f'segment_{i}'] = {
                        'label': str(predictions[i]),
                        'confidence': 1.0,  # 没有概率信息，默认为1.0
                        'probabilities': None
                    }
                
                return results
                
        except Exception as e:
            logger.error(f"解码失败: {e}")
            return None
    
    def _simple_decode(self, features):
        """
        当没有加载模型时使用的简单解码方法
        基于特征的简单规则进行分类

        参数:
            features (numpy.ndarray): EEG特征向量或矩阵
        
        返回:
            dict: 解码结果
        """
        # 这是一个占位实现，使用特征的统计特性进行简单决策
        # 真实场景下应该使用训练好的机器学习模型
        
        try:
            results = {}
            
            # 假设前5个特征与alpha, beta, theta, delta, gamma频带能量有关
            # 基于频带能量比例做简单判断
            for i in range(features.shape[0]):
                feature_vec = features[i]
                
                # 使用前5个特征（如果存在）
                n_features = min(5, feature_vec.shape[0])
                band_powers = feature_vec[:n_features] if n_features > 0 else np.array([0.2, 0.2, 0.2, 0.2, 0.2])
                
                # 标准化使总和为1
                if np.sum(band_powers) > 0:
                    band_powers = band_powers / np.sum(band_powers)
                
                # 简单规则: 使用频带能量比例确定状态
                # 实际应用中应替换为机器学习模型
                if band_powers[0] > 0.3:  # Alpha强度高
                    label = "放松"
                    confidence = min(1.0, band_powers[0] * 2)
                elif band_powers[1] > 0.3:  # Beta强度高
                    label = "专注"
                    confidence = min(1.0, band_powers[1] * 2)
                elif band_powers[2] > 0.3:  # Theta强度高
                    label = "冥想"
                    confidence = min(1.0, band_powers[2] * 2)
                elif band_powers[3] > 0.3:  # Delta强度高
                    label = "睡眠"
                    confidence = min(1.0, band_powers[3] * 2)
                else:
                    label = "中性"
                    confidence = 0.5
                
                # 创建结果字典
                results[f'segment_{i}'] = {
                    'label': label,
                    'confidence': float(confidence),
                    'probabilities': {
                        '放松': float(band_powers[0]),
                        '专注': float(band_powers[1]),
                        '冥想': float(band_powers[2]),
                        '睡眠': float(band_powers[3]),
                        '中性': float(1.0 - np.sum(band_powers[:4]))
                    }
                }
            
            return results
            
        except Exception as e:
            logger.error(f"简单解码失败: {e}")
            return None
    
    def evaluate(self, features, labels):
        """
        评估解码性能
        
        参数:
            features (numpy.ndarray): 测试特征
            labels (numpy.ndarray): 真实标签
            
        返回:
            dict: 性能指标字典
        """
        if not self.is_loaded or self.model is None:
            logger.error("模型未加载，无法评估")
            return None
        
        try:
            # 预测标签
            predictions = self.model.predict(features)
            
            # 计算准确率
            accuracy = np.mean(predictions == labels)
            
            # 如果可能，计算更多指标
            metrics = {'accuracy': float(accuracy)}
            
            try:
                from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
                
                # 转换为类别标签
                if len(labels.shape) > 1 and labels.shape[1] > 1:
                    true_labels = np.argmax(labels, axis=1)
                else:
                    true_labels = labels
                
                # 计算其他指标
                metrics['precision'] = float(precision_score(true_labels, predictions, average='weighted'))
                metrics['recall'] = float(recall_score(true_labels, predictions, average='weighted'))
                metrics['f1_score'] = float(f1_score(true_labels, predictions, average='weighted'))
                metrics['confusion_matrix'] = confusion_matrix(true_labels, predictions).tolist()
                
            except Exception as metric_error:
                logger.warning(f"计算详细指标时出错: {metric_error}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"评估解码性能时出错: {e}")
            return None
    
    def save_model(self, save_path):
        """
        保存解码模型
        
        参数:
            save_path (str): 保存路径
            
        返回:
            bool: 保存是否成功
        """
        if self.model is None:
            logger.error("没有模型可保存")
            return False
        
        try:
            # 创建目录
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 根据文件扩展名选择保存方法
            if save_path.endswith('.pkl') or save_path.endswith('.pickle'):
                with open(save_path, 'wb') as f:
                    pickle.dump(self.model, f)
            elif save_path.endswith('.joblib'):
                joblib.dump(self.model, save_path)
            else:
                # 默认使用joblib
                joblib.dump(self.model, save_path)
            
            logger.info(f"模型保存成功: {save_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存模型失败: {e}")
            return False

class IntentDecoder(EEGDecoder):
    """意图解码器 - 识别用户的意图类型"""
    
    def __init__(self, model_path=None, threshold=0.5, intent_types=None):
        """
        初始化意图解码器
        
        参数:
            model_path (str): 模型文件路径
            threshold (float): 解码阈值
            intent_types (list): 支持的意图类型列表
        """
        super().__init__(model_path, threshold)
        
        # 默认支持的意图类型
        self.default_intent_types = [
            "注意", "放松", "左", "右", "前进", "后退", "选择", "取消", "无意图"
        ]
        
        # 设置意图类型
        self.intent_types = intent_types if intent_types is not None else self.default_intent_types
        
        logger.info(f"意图解码器初始化: {len(self.intent_types)}种意图类型")
    
    def decode_intent(self, features):
        """
        解码用户意图
        
        参数:
            features (numpy.ndarray): 特征向量
            
        返回:
            dict: 意图解码结果，包含意图类型和置信度
        """
        # 调用基类的解码方法
        raw_results = self.decode(features, return_probabilities=True)
        
        if raw_results is None:
            return None
        
        # 转换为意图格式
        intent_results = {}
        
        for segment_id, result in raw_results.items():
            # 如果预测标签不在意图类型列表中，检查是否有映射关系
            pred_label = result['label']
            if pred_label not in self.intent_types:
                # 尝试将数值标签映射到意图名称
                try:
                    idx = int(pred_label)
                    if 0 <= idx < len(self.intent_types):
                        pred_label = self.intent_types[idx]
                except (ValueError, TypeError):
                    # 如果转换失败，使用置信度最高的类别
                    if result['probabilities']:
                        probs = result['probabilities']
                        # 找到置信度最高的意图
                        max_intent = max(probs, key=probs.get)
                        pred_label = max_intent
            
            # 仅保留有效的意图（超过阈值）
            if result['confidence'] >= self.threshold:
                intent_results[segment_id] = {
                    'intent': pred_label,
                    'confidence': result['confidence'],
                    'probabilities': result['probabilities']
                }
        
        return intent_results if intent_results else None


class P300Decoder(EEGDecoder):
    """P300解码器 - 专注于检测P300事件相关电位"""
    
    def __init__(self, model_path=None, threshold=0.6, window_size=0.6):
        """
        初始化P300解码器
        
        参数:
            model_path (str): 模型文件路径
            threshold (float): 解码阈值
            window_size (float): 分析窗口大小(秒)
        """
        super().__init__(model_path, threshold)
        self.window_size = window_size
        
        # P300通常在刺激后300ms左右出现，设置相关时间窗口
        self.p300_start_ms = 250  # P300开始的典型时间(ms)
        self.p300_end_ms = 500    # P300结束的典型时间(ms)
        
        logger.info(f"P300解码器初始化: 窗口大小={window_size}秒")
    
    def detect_p300(self, eeg_data, sampling_rate, stimulus_times=None):
        """
        从EEG数据中检测P300反应
        
        参数:
            eeg_data (numpy.ndarray): 形状为(channels, samples)的EEG数据
            sampling_rate (int): 采样率(Hz)
            stimulus_times (list): 刺激出现的时间点列表(秒)
            
        返回:
            dict: P300检测结果
        """
        if eeg_data is None or eeg_data.size == 0:
            logger.warning("EEG数据为空，无法检测P300")
            return None
        
        if stimulus_times is None or len(stimulus_times) == 0:
            logger.warning("未提供刺激时间点，尝试从整个数据中检测P300")
            # 如果没有提供刺激时间，将整个信号作为特征提取
            features = self._extract_p300_features(eeg_data, sampling_rate)
            results = self.decode(features)
            
            # 标记为无法确定时间的P300检测
            if results:
                for segment_id, result in results.items():
                    result['stimulus_time'] = None
                    result['p300_detected'] = result['confidence'] >= self.threshold
            
            return results
        
        try:
            detection_results = {}
            
            # 对每个刺激时间点进行处理
            for i, stim_time in enumerate(stimulus_times):
                # 转换时间为样本索引
                stim_sample = int(stim_time * sampling_rate)
                
                # 定义P300窗口
                p300_start = stim_sample + int(self.p300_start_ms * sampling_rate / 1000)
                p300_end = stim_sample + int(self.p300_end_ms * sampling_rate / 1000)
                
                # 确保索引在有效范围内
                if p300_end > eeg_data.shape[1]:
                    logger.warning(f"刺激时间点{stim_time}对应的P300窗口超出数据范围")
                    continue
                
                # 提取P300窗口的数据
                p300_window = eeg_data[:, p300_start:p300_end]
                
                # 提取特征
                features = self._extract_p300_features(p300_window, sampling_rate)
                
                # 进行解码
                segment_results = self.decode(features)
                
                if segment_results:
                    # 将刺激时间和P300检测结果添加到结果中
                    for segment_id, result in segment_results.items():
                        result['stimulus_time'] = float(stim_time)
                        result['p300_detected'] = result['confidence'] >= self.threshold
                        
                        # 重命名键以反映刺激索引
                        new_key = f'stimulus_{i}_{segment_id}'
                        detection_results[new_key] = result
            
            return detection_results
            
        except Exception as e:
            logger.error(f"P300检测失败: {e}")
            return None
    
    def _extract_p300_features(self, data, sampling_rate):
        """
        从EEG数据中提取P300相关特征
        
        参数:
            data (numpy.ndarray): EEG数据
            sampling_rate (int): 采样率
            
        返回:
            numpy.ndarray: 特征向量
        """
        # 这里实现P300特定的特征提取
        # 在实际应用中，可能包括:
        # - 峰值检测
        # - 时域特征（平均幅度、标准差等）
        # - 滤波后的样本值
        # - 主成分或独立成分的投影
        
        try:
            channels, samples = data.shape
            
            # 获取中心通道的数据 (例如Cz, Pz)
            # 简化起见，使用所有通道的平均
            avg_data = np.mean(data, axis=0)
            
            # 提取简单特征
            features = []
            
            # 1. 峰值幅度及其位置
            max_amp = np.max(avg_data)
            max_idx = np.argmax(avg_data)
            max_time = max_idx / sampling_rate * 1000  # 毫秒
            
            # 2. 平均幅度
            mean_amp = np.mean(avg_data)
            
            # 3. 标准差
            std_amp = np.std(avg_data)
            
            # 4. 窗口内积分（面积）
            area = np.sum(avg_data) / sampling_rate
            
            # 5. 添加所有特征
            features.extend([max_amp, max_time, mean_amp, std_amp, area])
            
            # 6. 可选：添加降采样的原始数据点（最多10个点）
            n_points = min(10, samples)
            indices = np.linspace(0, samples-1, n_points, dtype=int)
            sample_points = avg_data[indices]
            features.extend(sample_points)
            
            # 7. 可选：对每个通道重复以上过程
            for ch in range(channels):
                ch_data = data[ch, :]
                
                max_amp_ch = np.max(ch_data)
                mean_amp_ch = np.mean(ch_data)
                std_amp_ch = np.std(ch_data)
                
                features.extend([max_amp_ch, mean_amp_ch, std_amp_ch])
            
            return np.array(features).reshape(1, -1)
            
        except Exception as e:
            logger.error(f"P300特征提取失败: {e}")
            return np.array([])
