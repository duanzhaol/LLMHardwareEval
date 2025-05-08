from typing import Dict, Any
from operators.base import Operator
from hardware.base import Hardware
from simulation.base import SimulationStrategy

class MatMulOperator(Operator):
    """矩阵乘法算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 提取算子维度信息
        input_size = self.dimensions.get("input_size", 1)
        output_size = self.dimensions.get("output_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            input_size=input_size,
            output_size=output_size,
            **kwargs
        )

class AttentionOperator(Operator):
    """自注意力算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 提取算子维度信息
        num_heads = self.dimensions.get("num_heads", 1)
        head_dim = self.dimensions.get("head_dim", 64)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            num_heads=num_heads,
            head_dim=head_dim,
            **kwargs
        )

class FFNOperator(Operator):
    """前馈神经网络算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 提取算子维度信息
        hidden_size = self.dimensions.get("hidden_size", 1)
        intermediate_size = self.dimensions.get("intermediate_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            **kwargs
        )

class LayerNormOperator(Operator):
    """层归一化算子"""
    
    def compute_time(self, hardware: Hardware, strategy: SimulationStrategy, 
                   batch_size: int, seq_length: int, **kwargs) -> float:
        # 提取算子维度信息
        hidden_size = self.dimensions.get("hidden_size", 1)
        
        # 使用模拟策略计算执行时间
        return strategy.estimate_execution_time(
            operator=self,
            hardware=hardware,
            batch_size=batch_size,
            seq_length=seq_length,
            hidden_size=hidden_size,
            **kwargs
        ) 