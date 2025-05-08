from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Operator(ABC):
    """算子基类，所有具体算子都应继承此类"""
    
    def __init__(self, name: str, dimensions: Dict[str, Any]):
        """
        初始化算子
        
        Args:
            name: 算子名称
            dimensions: 算子维度，如输入大小、权重、计算复杂度等
        """
        self.name = name
        self.dimensions = dimensions
    
    @abstractmethod
    def compute_time(self, hardware: 'Hardware', strategy: 'SimulationStrategy', 
                    batch_size: int, seq_length: int, **kwargs) -> float:
        """
        计算该算子在指定硬件和模拟策略下的执行时间
        
        Args:
            hardware: 硬件配置
            strategy: 模拟策略
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} Operator with dimensions: {self.dimensions}" 