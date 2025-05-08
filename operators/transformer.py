from typing import Dict, Any
from operators.base import Operator
from hardware.base import Hardware
from simulation.base import SimulationStrategy

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

class AttentionOperator(Operator):
    """自注意力算子"""
    
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