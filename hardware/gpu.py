from typing import Dict, Any
from hardware.base import Hardware

class GPU(Hardware):
    """GPU硬件类"""
    
    def __init__(self, name: str, specs: Dict[str, Any], cost_per_hour: float = 0.0):
        """
        初始化GPU
        
        Args:
            name: GPU名称
            specs: GPU规格，包括计算能力、内存带宽等
            cost_per_hour: 每小时使用成本（美元）
        """
        super().__init__(name, specs, cost_per_hour)
        
        # 确保必要的规格已提供
        required_specs = ["fp16_tflops", "fp32_tflops", "memory_bandwidth", "vram_gb", "interconnect_bandwidth"]
        for spec in required_specs:
            if spec not in specs:
                raise ValueError(f"必须提供GPU规格参数: {spec}")
    
    def get_compute_capability(self, operation_type: str) -> float:
        """
        获取指定操作类型的计算能力
        
        Args:
            operation_type: 操作类型，如'fp16', 'fp32'等
            
        Returns:
            计算能力（TFLOPS）
        """
        key = f"{operation_type}_tflops"
        if key in self.specs:
            return self.specs[key]
        else:
            raise ValueError(f"不支持的操作类型: {operation_type}")
    
    def get_memory_bandwidth(self) -> float:
        """
        获取内存带宽
        
        Returns:
            内存带宽（GB/s）
        """
        return self.specs["memory_bandwidth"]
    
    def get_communication_bandwidth(self) -> float:
        """
        获取GPU间通信带宽
        
        Returns:
            通信带宽（GB/s）
        """
        return self.specs["interconnect_bandwidth"]
    
    def get_vram_capacity(self) -> float:
        """
        获取显存容量
        
        Returns:
            显存容量（GB）
        """
        return self.specs["vram_gb"] 