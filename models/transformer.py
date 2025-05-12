from typing import Dict, Any, List
from models.base import Model
from operators.transformer import (
    MatMulOperator,
    MultiHeadAttentionOperator,
    GroupedQueryAttentionOperator,
    MultiQueryAttentionOperator,
    FFNOperator,
    LayerNormOperator
)

class TransformerModel(Model):
    """Transformer模型基类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化Transformer模型
        
        Args:
            name: 模型名称
            config: 模型配置
        """
        super().__init__(name, config)
        self.operators = self._build_operators()
    
    def _build_operators(self) -> List:
        """
        构建模型的算子列表
        
        Returns:
            模型中的算子列表
        """
        raise NotImplementedError("Subclasses must implement _build_operators")
    
    def estimate_execution_time(self, batch_size: int, seq_length: int, **kwargs) -> float:
        """
        估计模型执行时间
        
        Args:
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        total_time = 0.0
        for operator in self.operators:
            total_time += operator.compute_time(
                hardware=self.hardware,
                strategy=self.strategy,
                batch_size=batch_size,
                seq_length=seq_length,
                **kwargs
            )
        return total_time

    def prefill_operators(self) -> List:
        """
        获取预填充（prefill）阶段使用的算子列表
        在prefill阶段，所有算子都会执行
        
        Returns:
            预填充阶段的算子列表
        """
        return self.operators
    
    def decode_operators(self) -> List:
        """
        获取解码（decode）阶段使用的算子列表
        在decode阶段，每次只处理一个新token，可以复用KV缓存
        
        Returns:
            解码阶段的算子列表
        """
        # 在实际解码时，注意力计算可以复用KV缓存，因此时间复杂度会有所不同
        # 这里简化处理，返回与prefill相同的算子列表，但在estimate_runtime中会区分处理
        return self.operators 