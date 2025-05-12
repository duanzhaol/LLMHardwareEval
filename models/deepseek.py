from typing import Dict, Any, List
from models.transformer import TransformerModel
from operators.transformer import (
    EmbeddingOperator,
    FFNOperator,
    LayerNormOperator,
    LogitsOperator,
    MLAOperator
)

class DeepSeekV3Model(TransformerModel):
    """DeepSeek V3模型类，使用MLA（Mixture of Linear Attention）机制"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化DeepSeek V3模型
        
        Args:
            name: 模型名称
            config: 模型配置
        """
        # 设置必要的属性
        self.num_heads = config.get("num_heads", 32)
        self.head_dim = config.get("head_dim", 64)  # DeepSeek V3默认使用较小的head_dim
        self.hidden_size = self.num_heads * self.head_dim
        self.intermediate_size = config.get("intermediate_size", self.hidden_size * 4)
        self.num_layers = config.get("num_layers", 32)
        self.vocab_size = config.get("vocab_size", 100000)  # DeepSeek的词表通常较大
        self.mixture_size = config.get("mixture_size", 4)  # MLA中混合组件的数量
        self.matrix_fusion = config.get("matrix_fusion", True)  # 默认启用矩阵吸收
        
        # 调用父类初始化
        super().__init__(name, config)
    
    def _build_operators(self) -> List:
        """
        根据模型配置构建DeepSeek V3模型的算子列表
        DeepSeek V3的架构特点:
        1. 使用MLA（Mixture of Linear Attention）替代传统的多头注意力
        2. 使用RMSNorm
        3. 可能使用SwiGLU或GeGLU激活函数
        4. 使用残差连接
        
        Returns:
            模型中的算子列表
        """
        operators = []
        
        # 词嵌入层
        operators.append(EmbeddingOperator(
            name="Embedding",
            dimensions={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size
            }
        ))
        
        # DeepSeek V3 Transformer层
        for _ in range(self.num_layers):
            # 第一个RMSNorm (用LayerNormOperator近似表示)
            operators.append(LayerNormOperator(
                name="RMSNorm",
                dimensions={"hidden_size": self.hidden_size}
            ))
            
            # MLA注意力层（包含残差连接）
            operators.append(MLAOperator(
                name="MLAAttention",
                dimensions={
                    "num_heads": self.num_heads,
                    "head_dim": self.head_dim,
                    "mixture_size": self.mixture_size,
                    "matrix_fusion": self.matrix_fusion
                }
            ))
            
            # 第二个RMSNorm
            operators.append(LayerNormOperator(
                name="RMSNorm",
                dimensions={"hidden_size": self.hidden_size}
            ))
            
            # 前馈网络（包含残差连接）
            operators.append(FFNOperator(
                name="GeLU_FFN",  # DeepSeek可能使用GeLU或SwiGLU
                dimensions={
                    "hidden_size": self.hidden_size,
                    "intermediate_size": self.intermediate_size
                }
            ))
        
        # 最后的RMSNorm
        operators.append(LayerNormOperator(
            name="RMSNorm",
            dimensions={"hidden_size": self.hidden_size}
        ))
        
        # Logits层
        operators.append(LogitsOperator(
            name="Logits",
            dimensions={
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size
            }
        ))
        
        return operators
    
    def prefill_operators(self) -> List:
        """
        获取prefill阶段使用的算子列表
        
        Returns:
            prefill阶段的算子列表
        """
        return self._build_operators()
    
    def decode_operators(self) -> List:
        """
        获取decode阶段使用的算子列表
        
        Returns:
            decode阶段的算子列表
        """
        return self._build_operators()
    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model_type": "deepseek_v3",
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "mixture_size": self.mixture_size,
            "matrix_fusion": self.matrix_fusion
        } 