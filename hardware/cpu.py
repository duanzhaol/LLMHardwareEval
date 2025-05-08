from typing import Dict, Any
from hardware.base import Hardware

class CPU(Hardware):
    """CPU硬件类"""
    
    def __init__(self, name: str, specs: Dict[str, Any], cost_per_hour: float = 0.0):
        """
        初始化CPU
        
        Args:
            name: CPU名称
            specs: CPU规格，包括计算能力、内存带宽等
            cost_per_hour: 每小时使用成本（美元）
        """
        super().__init__(name, specs, cost_per_hour)
        
        # 确保必要的规格已提供
        required_specs = ["cores", "frequency_ghz", "memory_bandwidth", "ram_gb", "network_bandwidth"]
        for spec in required_specs:
            if spec not in specs:
                raise ValueError(f"必须提供CPU规格参数: {spec}")
        
        # 计算理论FLOPS
        self.specs["theoretical_flops"] = self.specs["cores"] * self.specs["frequency_ghz"] * 1e9 * 16  # 假设AVX-512，每周期16个FP32操作
    
    def get_compute_capability(self, operation_type: str) -> float:
        """
        获取指定操作类型的计算能力
        
        Args:
            operation_type: 操作类型，暂时只支持'fp32'
            
        Returns:
            计算能力（GFLOPS）
        """
        if operation_type == "fp32":
            return self.specs["theoretical_flops"] / 1e9
        else:
            raise ValueError(f"CPU不支持的操作类型: {operation_type}")
    
    def get_memory_bandwidth(self) -> float:
        """
        获取内存带宽
        
        Returns:
            内存带宽（GB/s）
        """
        return self.specs["memory_bandwidth"]
    
    def get_communication_bandwidth(self) -> float:
        """
        获取CPU间网络通信带宽
        
        Returns:
            通信带宽（GB/s）
        """
        return self.specs["network_bandwidth"]
    
    def get_ram_capacity(self) -> float:
        """
        获取内存容量
        
        Returns:
            内存容量（GB）
        """
        return self.specs["ram_gb"] 