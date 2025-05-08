from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Hardware(ABC):
    """硬件基类，所有具体硬件配置都应继承此类"""
    
    def __init__(self, name: str, specs: Dict[str, Any], cost_per_hour: float = 0.0):
        """
        初始化硬件
        
        Args:
            name: 硬件名称
            specs: 硬件规格，如计算能力、内存带宽等
            cost_per_hour: 每小时使用成本（美元）
        """
        self.name = name
        self.specs = specs
        self.cost_per_hour = cost_per_hour
    
    @abstractmethod
    def get_compute_capability(self, operation_type: str) -> float:
        """
        获取指定操作类型的计算能力
        
        Args:
            operation_type: 操作类型，如'fp16', 'fp32', 'int8'等
            
        Returns:
            计算能力（FLOPS或TOPS）
        """
        pass
    
    @abstractmethod
    def get_memory_bandwidth(self) -> float:
        """
        获取内存带宽
        
        Returns:
            内存带宽（GB/s）
        """
        pass
    
    @abstractmethod
    def get_communication_bandwidth(self) -> float:
        """
        获取通信带宽
        
        Returns:
            通信带宽（GB/s）
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} Hardware with specs: {self.specs}, cost: ${self.cost_per_hour}/hour" 