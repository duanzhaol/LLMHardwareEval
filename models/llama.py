from typing import Dict, Any, List, Optional
from models.transformer import TransformerModel
from operators.transformer import (
    EmbeddingOperator,
    MultiHeadAttentionOperator,
    GroupedQueryAttentionOperator,
    MultiQueryAttentionOperator,
    FFNOperator,
    LayerNormOperator,
    LogitsOperator
)

class LlamaModel(TransformerModel):
    """Llama系列模型类，包括Llama 2和Llama 3"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化Llama模型
        
        Args:
            name: 模型名称
            config: 模型配置
        """
        # 在调用父类初始化之前设置所有必要的属性
        self.attention_type = config.get("attention_type", "mha")  # 可选: "mha", "gqa", "mqa"
        self.num_heads = config.get("num_heads", 32)
        self.num_kv_heads = config.get("num_kv_heads", 1)  # 用于GQA
        self.head_dim = config.get("head_dim", 128)
        self.hidden_size = self.num_heads * self.head_dim
        self.intermediate_size = config.get("intermediate_size", self.hidden_size * 4)
        self.num_layers = config.get("num_layers", 32)
        self.vocab_size = config.get("vocab_size", 32000)
        
        # 调用父类初始化
        super().__init__(name, config)
        
        # 创建算子
        self._create_operators()
    
    def _create_operators(self):
        """创建模型所需的算子"""
        # 创建词嵌入算子
        self.embedding = EmbeddingOperator(
            name="Embedding",
            dimensions={
                "vocab_size": self.vocab_size,
                "hidden_size": self.hidden_size
            }
        )
        
        # 根据注意力类型选择对应的算子
        if self.attention_type == "mha":
            self.attention = MultiHeadAttentionOperator(
                name="MultiHeadAttention",
                dimensions={
                    "num_heads": self.num_heads,
                    "head_dim": self.head_dim
                }
            )
        elif self.attention_type == "gqa":
            self.attention = GroupedQueryAttentionOperator(
                name="GroupedQueryAttention",
                dimensions={
                    "num_heads": self.num_heads,
                    "num_kv_heads": self.num_kv_heads,
                    "head_dim": self.head_dim
                }
            )
        elif self.attention_type == "mqa":
            self.attention = MultiQueryAttentionOperator(
                name="MultiQueryAttention",
                dimensions={
                    "num_heads": self.num_heads,
                    "head_dim": self.head_dim
                }
            )
        else:
            raise ValueError(f"Unsupported attention type: {self.attention_type}")
        
        self.ffn = FFNOperator(
            name="FFN",
            dimensions={
                "hidden_size": self.hidden_size,
                "intermediate_size": self.intermediate_size
            }
        )
        self.layer_norm = LayerNormOperator(
            name="LayerNorm",
            dimensions={"hidden_size": self.hidden_size}
        )
        
        # 创建logits算子
        self.logits = LogitsOperator(
            name="Logits",
            dimensions={
                "hidden_size": self.hidden_size,
                "vocab_size": self.vocab_size
            }
        )
    
    def _build_operators(self) -> List:
        """
        根据模型配置构建Llama模型的算子列表
        Llama架构特点:
        1. 使用RMSNorm而不是LayerNorm
        2. 使用旋转位置编码(RoPE)
        3. SwiGLU激活函数
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
        
        # Llama Transformer层
        for _ in range(self.num_layers):
            # 第一个RMSNorm (用LayerNormOperator近似表示)
            operators.append(LayerNormOperator(
                name="RMSNorm",
                dimensions={"hidden_size": self.hidden_size}
            ))
            
            # 自注意力层（包含残差连接）
            if self.attention_type == "mha":
                operators.append(MultiHeadAttentionOperator(
                    name="MultiHeadAttention",
                    dimensions={
                        "num_heads": self.num_heads,
                        "head_dim": self.head_dim
                    }
                ))
            elif self.attention_type == "gqa":
                operators.append(GroupedQueryAttentionOperator(
                    name="GroupedQueryAttention",
                    dimensions={
                        "num_heads": self.num_heads,
                        "num_kv_heads": self.num_kv_heads,
                        "head_dim": self.head_dim
                    }
                ))
            elif self.attention_type == "mqa":
                operators.append(MultiQueryAttentionOperator(
                    name="MultiQueryAttention",
                    dimensions={
                        "num_heads": self.num_heads,
                        "head_dim": self.head_dim
                    }
                ))
            
            # 第二个RMSNorm
            operators.append(LayerNormOperator(
                name="RMSNorm",
                dimensions={"hidden_size": self.hidden_size}
            ))
            
            # SwiGLU前馈网络（包含残差连接）
            operators.append(FFNOperator(
                name="SwiGLU_FFN",
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

    
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        return {
            "model_type": "llama",
            "attention_type": self.attention_type,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_layers,
            "vocab_size": self.vocab_size,
            "max_position_embeddings": self.max_position_embeddings
        } 