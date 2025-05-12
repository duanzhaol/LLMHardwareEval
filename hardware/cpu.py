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
        self.specs["theoretical_fp32_flops"] = self.specs["cores"] * self.specs["frequency_ghz"] * 1e9 * 16  # 假设AVX-512，每周期16个FP32操作
        
        # 其他精度默认值（如果未提供）
        if "avx512_support" not in specs:
            self.specs["avx512_support"] = True  # 默认支持AVX-512
            
        if "bf16_support" not in specs:
            self.specs["bf16_support"] = False  # 默认不支持BF16
            
        if "amx_support" not in specs:
            self.specs["amx_support"] = False  # AMX（Advanced Matrix Extensions）支持INT8/BF16操作
    
    def get_compute_capability(self, operation_type: str) -> float:
        """
        获取指定操作类型的计算能力
        
        Args:
            operation_type: 操作类型，支持'fp32', 'fp16', 'bf16', 'int8', 'fp8'
            
        Returns:
            计算能力（TFLOPS）
        """
        base_capability = self.specs["theoretical_fp32_flops"] / 1e12  # 转换为TFLOPS
        
        if operation_type == "fp32":
            return base_capability
        elif operation_type == "fp16":
            return base_capability * 2  # FP16通常是FP32性能的2倍
        elif operation_type == "bf16":
            # BF16性能取决于是否支持专门的BF16指令集
            if self.specs.get("bf16_support", False):
                return base_capability * 2
            else:
                return base_capability  # 如果不支持BF16指令，性能与FP32相当
        elif operation_type == "int8":
            # INT8性能取决于是否支持AMX或VNNI
            if self.specs.get("amx_support", False):
                return base_capability * 4  # AMX可能提供4倍于FP32的INT8性能
            else:
                return base_capability * 2  # 保守估计
        elif operation_type == "fp8":
            # FP8目前在大多数CPU上没有硬件支持
            return base_capability  # 保守估计，与FP32相当
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
        获取网络带宽
        
        Returns:
            网络带宽（GB/s）
        """
        return self.specs["network_bandwidth"]
    
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
    
    def get_ram_capacity(self) -> float:
        """
        获取内存容量
        
        Returns:
            内存容量（GB）
        """
        return self.specs["ram_gb"] 