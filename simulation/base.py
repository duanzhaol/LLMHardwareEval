from abc import ABC, abstractmethod
from typing import Dict, Any

class SimulationStrategy(ABC):
    """模拟策略基类，所有具体模拟策略都应继承此类"""
    
    def __init__(self, name: str, params: Dict[str, Any] = None):
        """
        初始化模拟策略
        
        Args:
            name: 策略名称
            params: 策略参数
        """
        self.name = name
        self.params = params or {}
    
    @abstractmethod
    def estimate_execution_time(self, operator, hardware, batch_size: int, 
                              seq_length: int, **kwargs) -> float:
        """
        估算算子在指定硬件下的执行时间
        
        Args:
            operator: 算子对象
            hardware: 硬件对象
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} Strategy with params: {self.params}" 