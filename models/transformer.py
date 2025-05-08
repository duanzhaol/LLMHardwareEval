from typing import Dict, Any, List
from models.base import Model
from operators.transformer import MatMulOperator, AttentionOperator, FFNOperator, LayerNormOperator

class TransformerModel(Model):
    """Transformer模型类"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """
        初始化Transformer模型
        
        Args:
            name: 模型名称
            config: 模型配置，应包含如下参数：
                - hidden_size: 隐藏层大小
                - num_layers: 层数
                - num_heads: 注意力头数
                - intermediate_size: 前馈网络中间层大小
                - vocab_size: 词表大小
                - max_seq_length: 最大序列长度
                - head_dim: 每个注意力头的维度（默认为hidden_size/num_heads）
        """
        # 验证必要参数
        required_params = ["hidden_size", "num_layers", "num_heads", "intermediate_size", "vocab_size"]
        for param in required_params:
            if param not in config:
                raise ValueError(f"Transformer模型配置必须包含参数：{param}")
        
        # 设置默认值
        if "head_dim" not in config:
            config["head_dim"] = config["hidden_size"] // config["num_heads"]
        
        super().__init__(name, config)
    
    def _build_operators(self) -> List:
        """
        根据模型配置构建算子列表
        
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
            name="MatMul",
            dimensions={"input_size": 1, "output_size": hidden_size}
        ))
        
        # Transformer层
        for i in range(num_layers):
            # 自注意力前的LayerNorm
            operators.append(LayerNormOperator(
                name="LayerNorm",
                dimensions={"hidden_size": hidden_size}
            ))
            
            # 自注意力中的Q, K, V投影
            operators.append(MatMulOperator(
                name="MatMul",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            operators.append(MatMulOperator(
                name="MatMul",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            operators.append(MatMulOperator(
                name="MatMul",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            
            # 自注意力计算
            operators.append(AttentionOperator(
                name="Attention",
                dimensions={"num_heads": num_heads, "head_dim": head_dim}
            ))
            
            # 自注意力输出投影
            operators.append(MatMulOperator(
                name="MatMul",
                dimensions={"input_size": hidden_size, "output_size": hidden_size}
            ))
            
            # FFN前的LayerNorm
            operators.append(LayerNormOperator(
                name="LayerNorm",
                dimensions={"hidden_size": hidden_size}
            ))
            
            # 前馈神经网络
            operators.append(FFNOperator(
                name="FFN",
                dimensions={"hidden_size": hidden_size, "intermediate_size": intermediate_size}
            ))
        
        # 最后的LayerNorm
        operators.append(LayerNormOperator(
            name="LayerNorm",
            dimensions={"hidden_size": hidden_size}
        ))
        
        # 输出层
        operators.append(MatMulOperator(
            name="MatMul",
            dimensions={"input_size": hidden_size, "output_size": vocab_size}
        ))
        
        return operators
    
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