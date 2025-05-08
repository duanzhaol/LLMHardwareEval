from abc import ABC, abstractmethod
from typing import Dict, Any, List
from hardware.base import Hardware

class Cluster(ABC):
    """集群基类，定义计算集群的配置和特性"""
    
    def __init__(self, name: str, hardware_list: List[Hardware], 
                network_bandwidth: float, specs: Dict[str, Any] = None):
        """
        初始化集群
        
        Args:
            name: 集群名称
            hardware_list: 集群中的硬件列表
            network_bandwidth: 集群内网络带宽（GB/s）
            specs: 其他集群规格
        """
        self.name = name
        self.hardware_list = hardware_list
        self.network_bandwidth = network_bandwidth
        self.specs = specs or {}
        
        # 计算集群总成本
        self._calculate_cost()
    
    def _calculate_cost(self):
        """计算集群总成本"""
        self.total_cost_per_hour = sum(hw.cost_per_hour for hw in self.hardware_list)
    
    @abstractmethod
    def allocate_resources(self, model, batch_size: int, 
                         seq_length: int) -> List[Hardware]:
        """
        为给定模型和请求参数分配硬件资源
        
        Args:
            model: 模型对象
            batch_size: 批处理大小
            seq_length: 序列长度
            
        Returns:
            分配的硬件列表
        """
        pass
    
    def get_cost_per_hour(self) -> float:
        """
        获取集群每小时成本
        
        Returns:
            每小时成本（美元）
        """
        return self.total_cost_per_hour
    
    def __str__(self) -> str:
        hardware_names = ", ".join(hw.name for hw in self.hardware_list)
        return f"{self.name} Cluster with {len(self.hardware_list)} units of hardware: [{hardware_names}], cost: ${self.total_cost_per_hour}/hour"