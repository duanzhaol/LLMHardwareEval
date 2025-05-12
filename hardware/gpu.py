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
        
        # 设置默认值，如果用户未提供这些规格
        # 基于fp16和fp32的性能来估算其他精度的性能
        if "fp8_tflops" not in specs and "fp16_tflops" in specs:
            self.specs["fp8_tflops"] = self.specs["fp16_tflops"] * 2  # fp8通常是fp16性能的2倍
        
        if "bf16_tflops" not in specs and "fp16_tflops" in specs:
            self.specs["bf16_tflops"] = self.specs["fp16_tflops"]  # bf16通常与fp16性能相近
        
        if "int8_tops" not in specs and "fp16_tflops" in specs:
            self.specs["int8_tops"] = self.specs["fp16_tflops"] * 2  # int8通常是fp16性能的2倍
    
    def get_compute_capability(self, operation_type: str) -> float:
        """
        获取指定操作类型的计算能力
        
        Args:
            operation_type: 操作类型，如'fp8', 'fp16', 'bf16', 'fp32', 'int8'等
            
        Returns:
            计算能力（TFLOPS或TOPS）
        """
        if operation_type == "int8":
            key = "int8_tops"
        else:
            key = f"{operation_type}_tflops"
        
        if key in self.specs:
            return self.specs[key]
        else:
            raise ValueError(f"不支持的操作类型: {operation_type}，请在specs中提供{key}参数")
    
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
    
    def get_bytes_per_element(self, operation_type: str) -> int:
        """
        获取指定操作类型每个元素的字节数
        
        Args:
            operation_type: 操作类型，如'fp8', 'fp16', 'bf16', 'fp32', 'int8'等
            
        Returns:
            每个元素的字节数
        """
        bytes_mapping = {
            "fp8": 1,
            "int8": 1,
            "bf16": 2,
            "fp16": 2,
            "fp32": 4,
            "fp64": 8
        }
        
        if operation_type in bytes_mapping:
            return bytes_mapping[operation_type]
        else:
            raise ValueError(f"不支持的操作类型: {operation_type}") 