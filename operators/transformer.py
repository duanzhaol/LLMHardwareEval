from typing import Dict, Any
from operators.base import Operator
from hardware.base import Hardware
from simulation.base import SimulationStrategy

__all__ = [
    "EmbeddingOperator",
    "MultiHeadAttentionOperator",
    "GroupedQueryAttentionOperator",
    "MultiQueryAttentionOperator",
    "FFNOperator",
    "LayerNormOperator",
    "LogitsOperator",
    "MLAOperator"
]

class MatMulOperator(Operator):
    """矩阵乘法算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有input_size和output_size，如果没有才从dimensions获取
        if 'input_size' not in kwargs:
            kwargs['input_size'] = self.dimensions.get("input_size", 1)
        if 'output_size' not in kwargs:
            kwargs['output_size'] = self.dimensions.get("output_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class MultiHeadAttentionOperator(Operator):
    """多头注意力算子 (MHA)"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有num_heads和head_dim，如果没有才从dimensions获取
        if 'num_heads' not in kwargs:
            kwargs['num_heads'] = self.dimensions.get("num_heads", 1)
        if 'head_dim' not in kwargs:
            kwargs['head_dim'] = self.dimensions.get("head_dim", 64)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class GroupedQueryAttentionOperator(Operator):
    """分组查询注意力算子 (GQA)"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有必要的参数
        if 'num_heads' not in kwargs:
            kwargs['num_heads'] = self.dimensions.get("num_heads", 1)
        if 'num_kv_heads' not in kwargs:
            kwargs['num_kv_heads'] = self.dimensions.get("num_kv_heads", 1)
        if 'head_dim' not in kwargs:
            kwargs['head_dim'] = self.dimensions.get("head_dim", 64)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class MultiQueryAttentionOperator(Operator):
    """多查询注意力算子 (MQA)"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有必要的参数
        if 'num_heads' not in kwargs:
            kwargs['num_heads'] = self.dimensions.get("num_heads", 1)
        if 'head_dim' not in kwargs:
            kwargs['head_dim'] = self.dimensions.get("head_dim", 64)
        
        # MQA是GQA的特例，只有一个KV头
        kwargs['num_kv_heads'] = 1
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class FFNOperator(Operator):
    """前馈神经网络算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有hidden_size和intermediate_size，如果没有才从dimensions获取
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = self.dimensions.get("hidden_size", 1)
        if 'intermediate_size' not in kwargs:
            kwargs['intermediate_size'] = self.dimensions.get("intermediate_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class LayerNormOperator(Operator):
    """层归一化算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有hidden_size，如果没有才从dimensions获取
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = self.dimensions.get("hidden_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class EmbeddingOperator(Operator):
    """词嵌入算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有vocab_size和hidden_size，如果没有才从dimensions获取
        if 'vocab_size' not in kwargs:
            kwargs['vocab_size'] = self.dimensions.get("vocab_size", 1)
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = self.dimensions.get("hidden_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class LogitsOperator(Operator):
    """Logits计算算子，用于将隐藏状态投影到词表空间"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                    batch_size: int, seq_length: int, **kwargs) -> float:
        """
        计算logits层的执行时间
        
        Args:
            hardware: 硬件信息
            strategy: 模拟策略
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        # 检查kwargs中是否已有hidden_size和vocab_size，如果没有才从dimensions获取
        if 'hidden_size' not in kwargs:
            kwargs['hidden_size'] = self.dimensions.get("hidden_size", 1)
        if 'vocab_size' not in kwargs:
            kwargs['vocab_size'] = self.dimensions.get("vocab_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        )

class MLAOperator(Operator):
    """Mixture of Linear Attention算子，DeepSeek提出的线性注意力机制"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 检查kwargs中是否已有必要参数，如果没有才从dimensions获取
        if 'num_heads' not in kwargs:
            kwargs['num_heads'] = self.dimensions.get("num_heads", 1)
        if 'head_dim' not in kwargs:
            kwargs['head_dim'] = self.dimensions.get("head_dim", 64)
        if 'mixture_size' not in kwargs:
            kwargs['mixture_size'] = self.dimensions.get("mixture_size", 4)  # DeepSeek MLA默认使用4个mixture components
        if 'hidden_size' not in kwargs:
            num_heads = kwargs['num_heads']
            head_dim = kwargs['head_dim']
            kwargs['hidden_size'] = num_heads * head_dim
        if 'matrix_fusion' not in kwargs:
            kwargs['matrix_fusion'] = self.dimensions.get("matrix_fusion", True)  # 默认开启矩阵吸收

        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            **kwargs
        ) 