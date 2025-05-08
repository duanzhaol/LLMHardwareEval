from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from operators.base import Operator

class Model(ABC):
    """模型基类，所有具体模型类型都应继承此类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化模型
        
        Args:
            name: 模型名称
            config: 模型配置，包括模型大小、层数等
        """
        self.name = name
        self.config = config
        self.operators = self._build_operators()
    
    @abstractmethod
    def _build_operators(self) -> List[Operator]:
        """
        根据模型配置构建算子列表
        
        Returns:
            模型中的算子列表
        """
        pass
    
    def prefill_operators(self) -> List[Operator]:
        """
        获取预填充（prefill）阶段使用的算子列表
        
        Returns:
            预填充阶段的算子列表
        """
        # 默认实现：返回所有算子
        return self.operators
    
    def decode_operators(self) -> List[Operator]:
        """
        获取解码（decode）阶段使用的算子列表
        
        Returns:
            解码阶段的算子列表
        """
        # 默认实现：返回所有算子
        return self.operators
    
    def estimate_runtime(self, hardware, strategy, batch_size: int, 
                        seq_length: int, phase: str = "prefill") -> float:
        """
        估算模型在指定条件下的运行时间
        
        Args:
            hardware: 硬件对象
            strategy: 模拟策略
            batch_size: 批处理大小
            seq_length: 序列长度
            phase: 执行阶段，"prefill"或"decode"
            
        Returns:
            运行时间（秒）
        """
        operators = self.prefill_operators() if phase == "prefill" else self.decode_operators()
        
        total_time = 0.0
        for op in operators:
            # 仅传递必要的参数，避免与算子的dimensions字段重复
            # 算子会自己从dimensions获取缺少的参数
            time = op.compute_time(
                hardware=hardware,
                strategy=strategy,
                batch_size=batch_size,
                seq_length=seq_length if phase == "prefill" else 1,  # decode阶段每次只处理一个token
                # 不再传递整个config **self.config
                operation_type=self.config.get("operation_type", "fp16"),  # 只传递通用参数
                parallelization_factor=self.config.get("parallelization_factor", 1.0)
            )
            total_time += time
        
        # 考虑并行化，这里简化处理
        parallelization_factor = self.config.get("parallelization_factor", 1.0)
        if parallelization_factor > 1.0:
            total_time /= parallelization_factor
        
        return total_time
    
    def __str__(self) -> str:
        return f"{self.name} Model with config: {self.config}" 