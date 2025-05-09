"""模型配置文件，包含不同模型架构的具体参数配置"""

# Llama系列模型配置
LLAMA3_CONFIGS = {
    "8B": {
        "name": "Llama3-8B",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 14336,  # 3.5 * hidden_size
        "vocab_size": 128000,
        "max_seq_length": 8192,
        "head_dim": 128,  # hidden_size / num_heads
        "is_decoder": True,
        "use_rotary_position_embeddings": True,
        "activation_function": "silu",
        "norm_epsilon": 1e-5,
        "parallelization_factor": 1.0
    },
    "70B": {
        "name": "Llama3-70B",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "intermediate_size": 28672,  # 3.5 * hidden_size
        "vocab_size": 128000,
        "max_seq_length": 8192,
        "head_dim": 128,  # hidden_size / num_heads
        "is_decoder": True,
        "use_rotary_position_embeddings": True,
        "activation_function": "silu",
        "norm_epsilon": 1e-5,
        "parallelization_factor": 1.0
    }
}

# Qwen系列模型配置
QWEN_CONFIGS = {
    "7B": {
        "name": "Qwen-7B",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 22016,  # 5.37 * hidden_size
        "vocab_size": 151936,
        "max_seq_length": 32768,
        "head_dim": 128,  # hidden_size / num_heads
        "is_decoder": True,
        "use_rotary_position_embeddings": True,
        "activation_function": "swiglu",
        "norm_epsilon": 1e-6,
        "parallelization_factor": 1.0
    },
    "32B": {
        "name": "Qwen2-32B",
        "hidden_size": 6144,
        "num_layers": 60,
        "num_heads": 48,
        "intermediate_size": 33024,  # 5.37 * hidden_size
        "vocab_size": 152064,
        "max_seq_length": 32768,
        "head_dim": 128,  # hidden_size / num_heads
        "is_decoder": True,
        "use_rotary_position_embeddings": True,
        "activation_function": "swiglu",
        "norm_epsilon": 1e-6,
        "parallelization_factor": 1.0
    }
}

# Baichuan系列模型配置
BAICHUAN_CONFIGS = {
    "7B": {
        "name": "Baichuan2-7B",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 11008,  # 2.69 * hidden_size
        "vocab_size": 125696,
        "max_seq_length": 4096,
        "head_dim": 128,  # hidden_size / num_heads
        "is_decoder": True,
        "use_rotary_position_embeddings": True,
        "activation_function": "silu",
        "norm_epsilon": 1e-6,
        "parallelization_factor": 1.0
    },
    "13B": {
        "name": "Baichuan2-13B",
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "intermediate_size": 13696,  # 2.67 * hidden_size
        "vocab_size": 125696,
        "max_seq_length": 4096,
        "head_dim": 128,  # hidden_size / num_heads
        "is_decoder": True,
        "use_rotary_position_embeddings": True,
        "activation_function": "silu",
        "norm_epsilon": 1e-6,
        "parallelization_factor": 1.0
    }
}

# 所有支持的模型配置的映射表
MODEL_CONFIGS = {
    "llama3": LLAMA3_CONFIGS,
    "qwen": QWEN_CONFIGS,
    "baichuan": BAICHUAN_CONFIGS
}

# 获取模型配置的辅助函数
def get_model_config(model_name, model_size=None):
    """
    获取指定模型的配置
    
    Args:
        model_name: 模型名称，如"llama3", "qwen"
        model_size: 模型大小，如"8B", "32B"，如果为None则返回所有大小的配置
        
    Returns:
        模型配置字典
    """
    model_name = model_name.lower()
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"不支持的模型: {model_name}")
    
    if model_size is None:
        return MODEL_CONFIGS[model_name]
    
    if model_size not in MODEL_CONFIGS[model_name]:
        raise ValueError(f"不支持的模型大小: {model_name}-{model_size}")
    
    return MODEL_CONFIGS[model_name][model_size] 