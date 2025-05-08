from typing import Dict, Any
from simulation.base import SimulationStrategy

class RooflineStrategy(SimulationStrategy):
    """Roofline模型模拟策略"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("Roofline", params)
    
    def estimate_execution_time(self, operator, hardware, batch_size: int, 
                              seq_length: int, **kwargs) -> float:
        """
        使用Roofline模型估算执行时间
        
        计算时间 = max(计算时间, 内存访问时间)
        计算时间 = 计算量 / 计算能力
        内存访问时间 = 内存访问量 / 内存带宽
        
        Args:
            operator: 算子对象
            hardware: 硬件对象
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        # 获取算子的计算量和访存量，具体算法根据算子类型而定
        compute_ops = self._get_compute_operations(operator, batch_size, seq_length, **kwargs)
        memory_bytes = self._get_memory_access(operator, batch_size, seq_length, **kwargs)
        
        # 根据硬件性能计算时间
        operation_type = kwargs.get("operation_type", "fp16")  # 默认使用fp16
        compute_capability = hardware.get_compute_capability(operation_type)
        memory_bandwidth = hardware.get_memory_bandwidth()
        
        # 计算实际执行时间（秒）
        compute_time = compute_ops / (compute_capability * 1e12)  # TFLOPS转换为FLOPS
        memory_time = memory_bytes / (memory_bandwidth * 1e9)  # GB/s转换为B/s
        
        # Roofline模型：取计算和访存的最大值作为执行时间
        return max(compute_time, memory_time)
    
    def _get_compute_operations(self, operator, batch_size: int, seq_length: int, **kwargs) -> float:
        """计算算子的计算量（浮点操作数）"""
        # 根据不同算子类型计算
        if operator.name == "MatMul":
            input_size = kwargs.get("input_size", 1)
            output_size = kwargs.get("output_size", 1)
            # 矩阵乘法计算量: 2 * B * S * I * O (每个元素需要一次乘法和一次加法)
            return 2 * batch_size * seq_length * input_size * output_size
        
        elif operator.name == "Attention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            # 自注意力计算量: 2 * B * S * S * H * D (QK乘法+加权求和)
            return 2 * batch_size * seq_length * seq_length * num_heads * head_dim
        
        elif operator.name == "FFN":
            hidden_size = kwargs.get("hidden_size", 1)
            intermediate_size = kwargs.get("intermediate_size", 1)
            # FFN计算量: 2 * B * S * H * (4H) + 2 * B * S * (4H) * H
            return 2 * batch_size * seq_length * hidden_size * intermediate_size + \
                   2 * batch_size * seq_length * intermediate_size * hidden_size
        
        elif operator.name == "LayerNorm":
            hidden_size = kwargs.get("hidden_size", 1)
            # LayerNorm计算量: 4 * B * S * H (平均值、方差、归一化、缩放)
            return 4 * batch_size * seq_length * hidden_size
        
        else:
            return 1.0  # 默认值
    
    def _get_memory_access(self, operator, batch_size: int, seq_length: int, **kwargs) -> float:
        """计算算子的内存访问量（字节）"""
        # 假设每个浮点数占4字节
        bytes_per_element = 2 if kwargs.get("operation_type", "fp16") == "fp16" else 4
        
        if operator.name == "MatMul":
            input_size = kwargs.get("input_size", 1)
            output_size = kwargs.get("output_size", 1)
            # 输入矩阵读取 + 权重矩阵读取 + 输出矩阵写入
            return (batch_size * seq_length * input_size + input_size * output_size + 
                    batch_size * seq_length * output_size) * bytes_per_element
        
        elif operator.name == "Attention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            # Q, K, V矩阵读取 + 注意力权重读写 + 输出写入
            return (3 * batch_size * seq_length * hidden_size + 
                    2 * batch_size * num_heads * seq_length * seq_length + 
                    batch_size * seq_length * hidden_size) * bytes_per_element
        
        elif operator.name == "FFN":
            hidden_size = kwargs.get("hidden_size", 1)
            intermediate_size = kwargs.get("intermediate_size", 1)
            # 输入读取 + 中间结果读写 + 输出写入
            return (batch_size * seq_length * hidden_size + 
                    2 * batch_size * seq_length * intermediate_size + 
                    batch_size * seq_length * hidden_size) * bytes_per_element
        
        elif operator.name == "LayerNorm":
            hidden_size = kwargs.get("hidden_size", 1)
            # 输入读取 + 中间统计量 + 输出写入
            return (batch_size * seq_length * hidden_size + 
                    2 * batch_size * seq_length + 
                    batch_size * seq_length * hidden_size) * bytes_per_element
        
        else:
            return 1.0  # 默认值


class AnalyticalStrategy(SimulationStrategy):
    """基于解析模型的模拟策略"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("Analytical", params)
    
    def estimate_execution_time(self, operator, hardware, batch_size: int, 
                              seq_length: int, **kwargs) -> float:
        """
        使用解析模型估算执行时间
        
        Args:
            operator: 算子对象
            hardware: 硬件对象
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        # 解析模型一般会结合硬件特性和算子特性，使用更复杂的函数来建模
        # 这里简单实现，实际可以基于实验数据拟合更精确的模型
        
        # 基于Roofline模型计算一个基本时间
        roofline = RooflineStrategy()
        base_time = roofline.estimate_execution_time(
            operator, hardware, batch_size, seq_length, **kwargs
        )
        
        # 考虑实际硬件效率因素（可能小于1）
        hardware_efficiency = self.params.get("hardware_efficiency", 0.7)
        
        # 考虑额外开销，如内核启动、同步等
        overhead = self.params.get("overhead", 1e-5)  # 默认10微秒开销
        
        # 最终时间
        return base_time / hardware_efficiency + overhead


class EmpiricalStrategy(SimulationStrategy):
    """基于经验数据的模拟策略"""
    
    def __init__(self, benchmark_data: Dict[str, Any], params: Dict[str, Any] = None):
        """
        初始化经验模型
        
        Args:
            benchmark_data: 基准测试数据，用于查找或插值预测执行时间
            params: 其他参数
        """
        super().__init__("Empirical", params)
        self.benchmark_data = benchmark_data
    
    def estimate_execution_time(self, operator, hardware, batch_size: int, 
                              seq_length: int, **kwargs) -> float:
        """
        使用经验数据估算执行时间
        
        Args:
            operator: 算子对象
            hardware: 硬件对象
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        # 构建查询键，用于在benchmark_data中查找
        key = self._build_lookup_key(operator, hardware, batch_size, seq_length, **kwargs)
        
        # 如果有精确匹配的数据，直接返回
        if key in self.benchmark_data:
            return self.benchmark_data[key]
        
        # 否则进行插值估计（这里简化处理，实际应实现更复杂的插值算法）
        # 找到最接近的数据点
        closest_key = self._find_closest_key(key)
        if closest_key:
            # 根据规模比例进行简单缩放
            scale_factor = self._calculate_scale_factor(key, closest_key)
            return self.benchmark_data[closest_key] * scale_factor
        
        # 如果没有类似数据，回退到Roofline模型
        roofline = RooflineStrategy()
        return roofline.estimate_execution_time(
            operator, hardware, batch_size, seq_length, **kwargs
        )
    
    def _build_lookup_key(self, operator, hardware, batch_size: int, 
                        seq_length: int, **kwargs) -> str:
        """构建用于查找基准数据的键"""
        # 简化实现，实际应考虑更多因素
        return f"{operator.name}_{hardware.name}_{batch_size}_{seq_length}"
    
    def _find_closest_key(self, key: str) -> str:
        """找到最接近的基准数据键"""
        # 简化实现，实际应实现更复杂的相似度计算
        parts = key.split('_')
        operator_name, hardware_name = parts[0], parts[1]
        
        for k in self.benchmark_data.keys():
            k_parts = k.split('_')
            if k_parts[0] == operator_name and k_parts[1] == hardware_name:
                return k
        
        return None
    
    def _calculate_scale_factor(self, target_key: str, source_key: str) -> float:
        """计算缩放因子"""
        # 简化实现，假设时间和batch_size成正比，与seq_length的平方成正比
        target_parts = target_key.split('_')
        source_parts = source_key.split('_')
        
        target_batch = int(target_parts[2])
        source_batch = int(source_parts[2])
        target_seq = int(target_parts[3])
        source_seq = int(source_parts[3])
        
        batch_ratio = target_batch / source_batch
        seq_ratio = (target_seq / source_seq) ** 2
        
        return batch_ratio * seq_ratio 