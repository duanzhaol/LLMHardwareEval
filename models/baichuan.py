from typing import Dict, Any, List
from models.transformer import TransformerModel
from operators.transformer import MatMulOperator, AttentionOperator, FFNOperator, LayerNormOperator

class BaichuanModel(TransformerModel):
    """百川系列模型类，包括Baichuan和Baichuan2"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化Baichuan模型
        
        Args:
            name: 模型名称
            config: 模型配置
        """
        super().__init__(name, config)
    
    def _build_operators(self) -> List:
        """
        根据模型配置构建Baichuan模型的算子列表
        Baichuan架构特点:
        1. 使用RMSNorm
        2. 使用ALiBi位置编码
        3. 使用SiLU激活函数
        
        Returns:
            模型中的算子列表
        """
        operators = []
        
        # 模型配置参数
        hidden_size = self.config["hidden_size"]
        num_layers = self.config["num_layers"]
        num_heads = self.config["num_heads"]
        head_dim = self.config["head_dim"]
        intermediate_size = self.config["intermediate_size"]
        vocab_size = self.config["vocab_size"]
        
        # 词嵌入
        operators.append(MatMulOperator(
            name="Embedding",
            dimensions={"input_size": 1, "output_size": hidden_size}
        ))
        
        # Baichuan Transformer层
        for i in range(num_layers):
            # RMSNorm (用LayerNormOperator近似表示)
            operators.append(LayerNormOperator(
                name="RMSNorm",
                dimensions={"hidden_size": hidden_size}
            ))
            
            # 自注意力中的Q, K, V投影
            operators.append(MatMulOperator(
                name="Query_Proj",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            operators.append(MatMulOperator(
                name="Key_Proj",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            operators.append(MatMulOperator(
                name="Value_Proj",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            
            # ALiBi注意力计算
            operators.append(AttentionOperator(
                name="ALiBi_Attention",
                dimensions={"num_heads": num_heads, "head_dim": head_dim}
            ))
            
            # 自注意力输出投影
            operators.append(MatMulOperator(
                name="Output_Proj",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            
            # RMSNorm
            operators.append(LayerNormOperator(
                name="RMSNorm",
                dimensions={"hidden_size": hidden_size}
            ))
            
            # SiLU前馈网络
            operators.append(FFNOperator(
                name="SiLU_FFN",
                dimensions={"hidden_size": hidden_size, "intermediate_size": intermediate_size}
            ))
        
        # 最后的RMSNorm
        operators.append(LayerNormOperator(
            name="RMSNorm",
            dimensions={"hidden_size": hidden_size}
        ))
        
        # 输出层
        operators.append(MatMulOperator(
            name="LM_Head",
            dimensions={"input_size": hidden_size, "output_size": vocab_size}
        ))
        
        return operators 