from typing import Dict, Any
from simulation.base import SimulationStrategy

class RooflineStrategy(SimulationStrategy):
    """Roofline模型模拟策略"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("Roofline", params)
    
    def estimate_execution_time(self, operator, hardware, batch_size: int, 
                              seq_length: int, **kwargs) -> float:
        """
        使用Roofline模型估算执行时间
        
        计算时间 = max(计算时间, 内存访问时间)
        计算时间 = 计算量 / 计算能力
        内存访问时间 = 内存访问量 / 内存带宽
        
        Args:
            operator: 算子对象
            hardware: 硬件对象
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        # 获取算子的计算量和访存量，具体算法根据算子类型而定
        compute_ops = self._get_compute_operations(operator, batch_size, seq_length, **kwargs)
        # 将硬件对象传递给_get_memory_access
        kwargs["hardware"] = hardware
        memory_bytes = self._get_memory_access(operator, batch_size, seq_length, **kwargs)
        
        # 根据硬件性能计算时间
        operation_type = kwargs.get("operation_type", "fp16")  # 默认使用fp16
        compute_capability = hardware.get_compute_capability(operation_type)
        memory_bandwidth = hardware.get_memory_bandwidth()
        
        # 计算实际执行时间（秒）
        compute_time = compute_ops / (compute_capability * 1e12)  # TFLOPS转换为FLOPS
        memory_time = memory_bytes / (memory_bandwidth * 1e9)  # GB/s转换为B/s
        
        # Roofline模型：取计算和访存的最大值作为执行时间
        return max(compute_time, memory_time)
    
    def _get_compute_operations(self, operator, batch_size: int, seq_length: int, **kwargs) -> float:
        """计算算子的计算量（浮点操作数）"""
        # 获取阶段信息（prefill或decode）
        is_decode = kwargs.get("is_decode", False)
        
        # 明确区分两种长度：
        # - cur_len: 当前处理的token数量（decode=1, prefill=seq_length）
        # - ctx_len: 上下文长度（历史序列长度，始终为seq_length）
        cur_len = 1 if is_decode else seq_length  # 当前处理的token数量
        ctx_len = seq_length  # 上下文长度
        
        # 根据不同算子类型计算
        if operator.name == "Embedding":
            # Embedding主要是查表操作，计算量很小
            # 每个token只需要一次查表操作
            return batch_size * cur_len
        
        elif operator.name == "MatMul":
            input_size = kwargs.get("input_size", 1)
            output_size = kwargs.get("output_size", 1)
            # 矩阵乘法计算量: 2 * B * S * I * O (每个元素需要一次乘法和一次加法)
            return 2 * batch_size * cur_len * input_size * output_size
        
        elif operator.name == "MultiHeadAttention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            
            if is_decode:
                # Decode阶段（使用KV Cache）：
                # 1. Q投影: 2 * B * cur_len(=1) * H * H (Q头权重矩阵)
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len(=1) * H * H (K,V头权重矩阵)
                # 3. QK注意力: 2 * B * H * cur_len(=1) * ctx_len * D
                # 4. Softmax: 3 * B * H * cur_len(=1) * ctx_len
                # 5. 注意力加权: 2 * B * H * cur_len(=1) * ctx_len * D
                # 6. 输出投影: 2 * B * cur_len(=1) * H * H
                
                # QK与AV计算
                qk_flops = 2 * batch_size * num_heads * cur_len * ctx_len * head_dim
                av_flops = 2 * batch_size * num_heads * cur_len * ctx_len * head_dim
                
                return (2 * batch_size * hidden_size * hidden_size +  # Q投影
                        4 * batch_size * hidden_size * hidden_size +  # KV投影 (2乘加 * 2K,V)
                        qk_flops +  # QK注意力
                        3 * batch_size * num_heads * cur_len * ctx_len +  # Softmax
                        av_flops +  # 注意力加权
                        2 * batch_size * hidden_size * hidden_size)  # 输出投影
            else:
                # Prefill阶段（不使用KV Cache）：
                # 1. QKV投影: 2(乘加) * 3(Q,K,V) * B * cur_len * H * H
                # 2. 输出投影: 2 * B * cur_len * H * H
                # 3. QK注意力: 2 * B * H * cur_len * cur_len * D
                # 4. Softmax: 3 * B * H * cur_len * cur_len
                # 5. 注意力加权: 2 * B * H * cur_len * cur_len * D
                
                # QK与AV计算
                qk_flops = 2 * batch_size * num_heads * cur_len * cur_len * head_dim
                av_flops = 2 * batch_size * num_heads * cur_len * cur_len * head_dim
                
                return (6 * batch_size * cur_len * hidden_size * hidden_size +  # Q,K,V投影 (2乘加 * 3Q,K,V)
                        2 * batch_size * cur_len * hidden_size * hidden_size +  # 输出投影
                        qk_flops + av_flops +  # QK注意力和注意力加权
                        3 * batch_size * num_heads * cur_len * cur_len)  # Softmax
        
        elif operator.name == "GroupedQueryAttention":
            num_heads = kwargs.get("num_heads", 1)
            num_kv_heads = kwargs.get("num_kv_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            kv_hidden_size = num_kv_heads * head_dim  # KV头的隐藏状态大小
            
            if is_decode:
                # Decode阶段（使用KV Cache）：
                # 1. Q投影: 2 * B * cur_len(=1) * H * H
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len(=1) * H * H_kv
                # 3. QK注意力: 2 * B * H * cur_len(=1) * ctx_len * D
                # 4. Softmax: 3 * B * H * cur_len(=1) * ctx_len
                # 5. 注意力加权: 2 * B * H * cur_len(=1) * ctx_len * D
                # 6. 输出投影: 2 * B * cur_len(=1) * H * H
                
                # QK与AV计算
                qk_flops = 2 * batch_size * num_heads * cur_len * ctx_len * head_dim
                av_flops = 2 * batch_size * num_heads * cur_len * ctx_len * head_dim
                
                return (2 * batch_size * hidden_size * hidden_size +  # Q投影
                        4 * batch_size * hidden_size * kv_hidden_size +  # KV投影 (2乘加 * 2K,V)
                        qk_flops +  # QK注意力
                        3 * batch_size * num_heads * cur_len * ctx_len +  # Softmax
                        av_flops +  # 注意力加权
                        2 * batch_size * hidden_size * hidden_size)  # 输出投影
            else:
                # Prefill阶段（不使用KV Cache）：
                # 1. Q投影: 2 * B * cur_len * H * H
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len * H * H_kv
                # 3. QK注意力: 2 * B * H * cur_len * cur_len * D
                # 4. Softmax: 3 * B * H * cur_len * cur_len
                # 5. 注意力加权: 2 * B * H * cur_len * cur_len * D
                # 6. 输出投影: 2 * B * cur_len * H * H
                
                # QK与AV计算
                qk_flops = 2 * batch_size * num_heads * cur_len * cur_len * head_dim
                av_flops = 2 * batch_size * num_heads * cur_len * cur_len * head_dim
                
                return (2 * batch_size * cur_len * hidden_size * hidden_size +  # Q投影
                        4 * batch_size * cur_len * hidden_size * kv_hidden_size +  # KV投影 (2乘加 * 2K,V)
                        qk_flops + av_flops +  # QK注意力和注意力加权
                        3 * batch_size * num_heads * cur_len * cur_len +  # Softmax
                        2 * batch_size * cur_len * hidden_size * hidden_size)  # 输出投影
        
        elif operator.name == "MultiQueryAttention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            kv_hidden_size = head_dim  # MQA只有一个KV头
            
            if is_decode:
                # Decode阶段（使用KV Cache）：
                # MQA是GQA的特例，只有一个KV头
                # 1. Q投影: 2 * B * cur_len(=1) * H * H
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len(=1) * H * H_kv
                # 3. QK注意力: 2 * B * H * cur_len(=1) * ctx_len * D
                # 4. Softmax: 3 * B * H * cur_len(=1) * ctx_len
                # 5. 注意力加权: 2 * B * H * cur_len(=1) * ctx_len * D
                # 6. 输出投影: 2 * B * cur_len(=1) * H * H
                
                # QK与AV计算
                qk_flops = 2 * batch_size * num_heads * cur_len * ctx_len * head_dim
                av_flops = 2 * batch_size * num_heads * cur_len * ctx_len * head_dim
                
                return (2 * batch_size * hidden_size * hidden_size +  # Q投影
                        4 * batch_size * hidden_size * kv_hidden_size +  # KV投影 (2乘加 * 2K,V)
                        qk_flops +  # QK注意力
                        3 * batch_size * num_heads * cur_len * ctx_len +  # Softmax
                        av_flops +  # 注意力加权
                        2 * batch_size * hidden_size * hidden_size)  # 输出投影
            else:
                # Prefill阶段（不使用KV Cache）：
                # 1. Q投影: 2 * B * cur_len * H * H
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len * H * H_kv
                # 3. QK注意力: 2 * B * H * cur_len * cur_len * D
                # 4. Softmax: 3 * B * H * cur_len * cur_len
                # 5. 注意力加权: 2 * B * H * cur_len * cur_len * D
                # 6. 输出投影: 2 * B * cur_len * H * H
                
                # QK与AV计算
                qk_flops = 2 * batch_size * num_heads * cur_len * cur_len * head_dim
                av_flops = 2 * batch_size * num_heads * cur_len * cur_len * head_dim
                
                return (2 * batch_size * cur_len * hidden_size * hidden_size +  # Q投影
                        4 * batch_size * cur_len * hidden_size * kv_hidden_size +  # KV投影 (2乘加 * 2K,V)
                        qk_flops + av_flops +  # QK注意力和注意力加权
                        3 * batch_size * num_heads * cur_len * cur_len +  # Softmax
                        2 * batch_size * cur_len * hidden_size * hidden_size)  # 输出投影
        
        elif operator.name == "FFN":
            hidden_size = kwargs.get("hidden_size", 1)
            intermediate_size = kwargs.get("intermediate_size", 1)
            # FFN计算量: 2 * B * S * H * intermediate_size + 2 * B * S * intermediate_size * H
            return 2 * batch_size * cur_len * hidden_size * intermediate_size + \
                   2 * batch_size * cur_len * intermediate_size * hidden_size
        
        elif operator.name == "LayerNorm":
            hidden_size = kwargs.get("hidden_size", 1)
            # LayerNorm计算量: 4 * B * S * H (平均值、方差、归一化、缩放)
            return 4 * batch_size * cur_len * hidden_size
        
        elif operator.name == "Logits":
            hidden_size = kwargs.get("hidden_size", 1)
            vocab_size = kwargs.get("vocab_size", 1)
            # Logits计算量: 2 * B * S * H * V (每个元素需要一次乘法和一次加法)
            return 2 * batch_size * cur_len * hidden_size * vocab_size
        
        elif operator.name == "MLAAttention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            mixture_size = kwargs.get("mixture_size", 4)  # MLA中的mixture分量数量
            hidden_size = num_heads * head_dim
            
            if is_decode:
                # Decode阶段（使用KV Cache）
                # 1. Q投影: 2 * B * cur_len(=1) * H * H
                q_proj_flops = 2 * batch_size * cur_len * hidden_size * hidden_size
                
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len(=1) * H * H
                kv_proj_flops = 4 * batch_size * cur_len * hidden_size * hidden_size
                
                # 3. φ(·)特征映射: 假设使用ReLU+1，计算量相对较小
                phi_mapping_flops = batch_size * cur_len * hidden_size  # 激活操作
                
                # 4. Mixture gate α (MLP over Q global mean): 2 * B * H * m
                gate_flops = 2 * batch_size * hidden_size * mixture_size
                
                # 5. 内积累加 (流水线): 2 * B * h * d * L * C
                inner_product_flops = 2 * batch_size * num_heads * head_dim * cur_len * ctx_len
                
                # 6. 混合加权: B * m * h * d
                weighted_sum_flops = batch_size * mixture_size * num_heads * head_dim
                
                # 7. 输出 O = A·V: 2 * B * h * d * L * C
                output_flops = 2 * batch_size * num_heads * head_dim * cur_len * ctx_len
                
                # 8. O投影: 2 * B * L * H * H
                o_proj_flops = 2 * batch_size * cur_len * hidden_size * hidden_size
                
                return (q_proj_flops + kv_proj_flops + phi_mapping_flops + gate_flops + 
                        inner_product_flops + weighted_sum_flops + output_flops + o_proj_flops)
            else:
                # Prefill阶段（不使用KV Cache）
                # 1. Q投影: 2 * B * cur_len * H * H
                q_proj_flops = 2 * batch_size * cur_len * hidden_size * hidden_size
                
                # 2. K,V投影: 2(乘加) * 2(K,V) * B * cur_len * H * H
                kv_proj_flops = 4 * batch_size * cur_len * hidden_size * hidden_size
                
                # 3. φ(·)特征映射: 假设使用ReLU+1
                phi_mapping_flops = 2 * batch_size * cur_len * hidden_size  # Q和K各一次
                
                # 4. Mixture gate α (MLP over Q global mean): 2 * B * H * m
                gate_flops = 2 * batch_size * hidden_size * mixture_size
                
                # 5. 内积累加: 2 * B * h * d * cur_len * cur_len
                inner_product_flops = 2 * batch_size * num_heads * head_dim * cur_len * cur_len
                
                # 6. 混合加权: B * m * h * d
                weighted_sum_flops = batch_size * mixture_size * num_heads * head_dim
                
                # 7. 输出 O = A·V: 2 * B * h * d * cur_len * cur_len
                output_flops = 2 * batch_size * num_heads * head_dim * cur_len * cur_len
                
                # 8. O投影: 2 * B * cur_len * H * H
                o_proj_flops = 2 * batch_size * cur_len * hidden_size * hidden_size
                
                return (q_proj_flops + kv_proj_flops + phi_mapping_flops + gate_flops + 
                        inner_product_flops + weighted_sum_flops + output_flops + o_proj_flops)
        
        else:
            return 1.0  # 默认值
    
    def _get_memory_access(self, operator, batch_size: int, seq_length: int, **kwargs) -> float:
        """计算算子的内存访问量（字节）"""
        # 获取操作类型并从硬件获取每个元素的字节数
        operation_type = kwargs.get("operation_type", "fp16")  # 默认使用fp16
        hardware = kwargs.get("hardware", None)
        
        # 如果提供了硬件对象，使用它来获取字节数
        if hardware:
            bytes_per_element = hardware.get_bytes_per_element(operation_type)
        else:
            # 如果没有提供硬件对象，使用默认映射
            bytes_mapping = {
                "fp8": 1,
                "int8": 1,
                "bf16": 2,
                "fp16": 2,
                "fp32": 4,
                "fp64": 8
            }
            bytes_per_element = bytes_mapping.get(operation_type, 2)  # 默认为fp16 (2字节)
        
        # 获取阶段信息（prefill或decode）
        is_decode = kwargs.get("is_decode", False)
        
        # 明确区分两种长度：
        # - cur_len: 当前处理的token数量（decode=1, prefill=seq_length）
        # - ctx_len: 上下文长度（历史序列长度，始终为seq_length）
        cur_len = 1 if is_decode else seq_length  # 当前处理的token数量
        ctx_len = seq_length  # 上下文长度
        
        # 获取是否计算权重访问（在实际工程中，权重通常常驻缓存）
        # 默认不计入权重访问，因为它们通常只加载一次
        include_weight_access = kwargs.get("include_weight_access", False)
        
        if operator.name == "Embedding":
            vocab_size = kwargs.get("vocab_size", 1)
            hidden_size = kwargs.get("hidden_size", 1)
            
            # 激活：查表读取 + 输出写入
            activation_bytes = 2 * batch_size * cur_len * hidden_size * bytes_per_element  # 查表读取+输出写入
            
            # 权重：嵌入表（通常不计入每token访问）
            weight_bytes = vocab_size * hidden_size * bytes_per_element if include_weight_access else 0
            
            # 返回内存访问量
            return activation_bytes + weight_bytes
        
        elif operator.name == "MatMul":
            input_size = kwargs.get("input_size", 1)
            output_size = kwargs.get("output_size", 1)
            
            # 激活：输入读取 + 输出写入
            activation_bytes = (batch_size * cur_len * input_size + 
                               batch_size * cur_len * output_size) * bytes_per_element
            
            # 权重：权重矩阵读取
            weight_bytes = input_size * output_size * bytes_per_element if include_weight_access else 0
            
            # 返回内存访问量
            return activation_bytes + weight_bytes
        
        elif operator.name == "MultiHeadAttention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            kv_hidden_size = hidden_size  # MHA所有头都参与KV
            
            if is_decode:
                # Decode阶段（使用KV Cache）：
                # 激活内存访问：
                # 1. 输入数据: B * cur_len(=1) * H
                # 2. 投影输出: B * cur_len(=1) * H (不计入DRAM访问)
                # 3. KV Cache读取（现有缓存）: 2 * B * ctx_len * H_kv
                # 4. KV Cache写入（当前token）: 2 * B * cur_len(=1) * H_kv
                # 5. 注意力得分: B * H * cur_len(=1) * ctx_len (写入)
                #    + B * H * cur_len(=1) * ctx_len (读回)
                # 6. 注意力输出: B * cur_len(=1) * H
                # 7. 残差连接: 2 * B * cur_len(=1) * H
                
                # 注意力得分写入和读回
                attn_scores_elements = (batch_size * num_heads * cur_len * ctx_len +  # 写入
                                         batch_size * num_heads * cur_len * ctx_len)  # 读回
                
                activation_bytes = (batch_size * cur_len * hidden_size +  # 输入数据
                                   2 * batch_size * ctx_len * kv_hidden_size +  # KV Cache读取
                                   2 * batch_size * cur_len * kv_hidden_size +  # KV Cache写入
                                   attn_scores_elements +  # 注意力得分
                                   batch_size * cur_len * hidden_size +  # 注意力输出
                                   2 * batch_size * cur_len * hidden_size) * bytes_per_element  # 残差连接
                
                # 权重内存访问（如果需要）：
                # 1. Q,K,V投影权重: 3 * H * H
                # 2. 输出投影权重: H * H
                weight_bytes = (3 * hidden_size * hidden_size + 
                               hidden_size * hidden_size) * bytes_per_element if include_weight_access else 0
                
                # 返回内存访问量
                return activation_bytes + weight_bytes
            else:
                # Prefill阶段（不使用KV Cache）：
                # 激活内存访问：
                # 1. 输入数据: B * cur_len * H
                # 2. 投影输出: 3 * B * cur_len * H
                # 3. 注意力得分: B * H * cur_len * cur_len (写入)
                #    + B * H * cur_len * cur_len (读回)
                # 4. 注意力输出: B * cur_len * H
                # 5. 残差连接: 2 * B * cur_len * H
                # 6. KV Cache初始化: 2 * B * cur_len * H_kv
                
                # 注意力得分写入和读回
                attn_scores_elements = (batch_size * num_heads * cur_len * cur_len +  # 写入
                                         batch_size * num_heads * cur_len * cur_len)  # 读回
                
                activation_bytes = (batch_size * cur_len * hidden_size +  # 输入数据
                                   3 * batch_size * cur_len * hidden_size +  # 投影输出
                                   attn_scores_elements +  # 注意力得分
                                   batch_size * cur_len * hidden_size +  # 注意力输出
                                   2 * batch_size * cur_len * hidden_size +  # 残差连接
                                   2 * batch_size * cur_len * kv_hidden_size) * bytes_per_element  # KV Cache初始化
                
                # 权重内存访问（如果需要）：
                # 1. Q,K,V投影权重: 3 * H * H
                # 2. 输出投影权重: H * H
                weight_bytes = (3 * hidden_size * hidden_size + 
                               hidden_size * hidden_size) * bytes_per_element if include_weight_access else 0
                
                # 返回内存访问量
                return activation_bytes + weight_bytes
        
        elif operator.name == "GroupedQueryAttention":
            num_heads = kwargs.get("num_heads", 1)
            num_kv_heads = kwargs.get("num_kv_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            kv_hidden_size = num_kv_heads * head_dim
            
            if is_decode:
                # Decode阶段（使用KV Cache）：
                # 激活内存访问：
                # 1. 输入数据: B * cur_len(=1) * H
                # 2. 投影输出: B * cur_len(=1) * H + 2 * B * cur_len(=1) * H_kv (不计入DRAM访问)
                # 3. KV Cache读取（现有缓存）: 2 * B * ctx_len * H_kv
                # 4. KV Cache写入（当前token）: 2 * B * cur_len(=1) * H_kv
                # 5. 注意力得分: B * H * cur_len(=1) * ctx_len (写入)
                #    + B * H * cur_len(=1) * ctx_len (读回)
                # 6. 注意力输出: B * cur_len(=1) * H
                # 7. 残差连接: 2 * B * cur_len(=1) * H
                
                # 注意力得分写入和读回
                attn_scores_elements = (batch_size * num_heads * cur_len * ctx_len +  # 写入
                                         batch_size * num_heads * cur_len * ctx_len)  # 读回
                
                activation_bytes = (batch_size * cur_len * hidden_size +  # 输入数据
                                   2 * batch_size * ctx_len * kv_hidden_size +  # KV Cache读取
                                   2 * batch_size * cur_len * kv_hidden_size +  # KV Cache写入
                                   attn_scores_elements +  # 注意力得分
                                   batch_size * cur_len * hidden_size +  # 注意力输出
                                   2 * batch_size * cur_len * hidden_size) * bytes_per_element  # 残差连接
                
                # 权重内存访问（如果需要）：
                # 1. Q投影权重: H * H
                # 2. K,V投影权重: 2 * H * H_kv
                # 3. 输出投影权重: H * H
                weight_bytes = (hidden_size * hidden_size + 
                               2 * hidden_size * kv_hidden_size + 
                               hidden_size * hidden_size) * bytes_per_element if include_weight_access else 0
                
                # 返回内存访问量
                return activation_bytes + weight_bytes
            else:
                # Prefill阶段（不使用KV Cache）：
                # 激活内存访问：
                # 1. 输入数据: B * cur_len * H
                # 2. 投影输出: B * cur_len * H + 2 * B * cur_len * H_kv
                # 3. 注意力得分: B * H * cur_len * cur_len (写入)
                #    + B * H * cur_len * cur_len (读回)
                # 4. 注意力输出: B * cur_len * H
                # 5. 残差连接: 2 * B * cur_len * H
                # 6. KV Cache初始化: 2 * B * cur_len * H_kv
                
                # 注意力得分写入和读回
                attn_scores_elements = (batch_size * num_heads * cur_len * cur_len +  # 写入
                                         batch_size * num_heads * cur_len * cur_len)  # 读回
                
                activation_bytes = (batch_size * cur_len * hidden_size +  # 输入数据
                                   batch_size * cur_len * hidden_size +  # Q投影输出
                                   2 * batch_size * cur_len * kv_hidden_size +  # KV投影输出
                                   attn_scores_elements +  # 注意力得分
                                   batch_size * cur_len * hidden_size +  # 注意力输出
                                   2 * batch_size * cur_len * hidden_size +  # 残差连接
                                   2 * batch_size * cur_len * kv_hidden_size) * bytes_per_element  # KV Cache初始化
                
                # 权重内存访问（如果需要）：
                # 1. Q投影权重: H * H
                # 2. K,V投影权重: 2 * H * H_kv
                # 3. 输出投影权重: H * H
                weight_bytes = (hidden_size * hidden_size + 
                               2 * hidden_size * kv_hidden_size + 
                               hidden_size * hidden_size) * bytes_per_element if include_weight_access else 0
                
                # 返回内存访问量
                return activation_bytes + weight_bytes
        
        elif operator.name == "MultiQueryAttention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            hidden_size = num_heads * head_dim
            kv_hidden_size = head_dim  # MQA只有一个KV头
            
            if is_decode:
                # Decode阶段（使用KV Cache）：
                # 激活内存访问：
                # 1. 输入数据: B * cur_len(=1) * H
                # 2. 投影输出: B * cur_len(=1) * H + 2 * B * cur_len(=1) * H_kv (不计入DRAM访问)
                # 3. KV Cache读取（现有缓存）: 2 * B * ctx_len * H_kv
                # 4. KV Cache写入（当前token）: 2 * B * cur_len(=1) * H_kv
                # 5. 注意力得分: B * H * cur_len(=1) * ctx_len (写入)
                #    + B * H * cur_len(=1) * ctx_len (读回)
                # 6. 注意力输出: B * cur_len(=1) * H
                # 7. 残差连接: 2 * B * cur_len(=1) * H
                
                # 注意力得分写入和读回
                attn_scores_elements = (batch_size * num_heads * cur_len * ctx_len +  # 写入
                                         batch_size * num_heads * cur_len * ctx_len)  # 读回
                
                activation_bytes = (batch_size * cur_len * hidden_size +  # 输入数据
                                   2 * batch_size * ctx_len * kv_hidden_size +  # KV Cache读取
                                   2 * batch_size * cur_len * kv_hidden_size +  # KV Cache写入
                                   attn_scores_elements +  # 注意力得分
                                   batch_size * cur_len * hidden_size +  # 注意力输出
                                   2 * batch_size * cur_len * hidden_size) * bytes_per_element  # 残差连接
                
                # 权重内存访问（如果需要）：
                # 1. Q投影权重: H * H
                # 2. K,V投影权重: 2 * H * H_kv
                # 3. 输出投影权重: H * H
                weight_bytes = (hidden_size * hidden_size + 
                               2 * hidden_size * kv_hidden_size + 
                               hidden_size * hidden_size) * bytes_per_element if include_weight_access else 0
                
                # 返回内存访问量
                return activation_bytes + weight_bytes
            else:
                # Prefill阶段（不使用KV Cache）：
                # 激活内存访问：
                # 1. 输入数据: B * cur_len * H
                # 2. 投影输出: B * cur_len * H + 2 * B * cur_len * H_kv
                # 3. 注意力得分: B * H * cur_len * cur_len (写入)
                #    + B * H * cur_len * cur_len (读回)
                # 4. 注意力输出: B * cur_len * H
                # 5. 残差连接: 2 * B * cur_len * H
                # 6. KV Cache初始化: 2 * B * cur_len * H_kv
                
                # 注意力得分写入和读回
                attn_scores_elements = (batch_size * num_heads * cur_len * cur_len +  # 写入
                                         batch_size * num_heads * cur_len * cur_len)  # 读回
                
                activation_bytes = (batch_size * cur_len * hidden_size +  # 输入数据
                                   batch_size * cur_len * hidden_size +  # Q投影输出
                                   2 * batch_size * cur_len * kv_hidden_size +  # KV投影输出
                                   attn_scores_elements +  # 注意力得分
                                   batch_size * cur_len * hidden_size +  # 注意力输出
                                   2 * batch_size * cur_len * hidden_size +  # 残差连接
                                   2 * batch_size * cur_len * kv_hidden_size) * bytes_per_element  # KV Cache初始化
                
                # 权重内存访问（如果需要）：
                # 1. Q投影权重: H * H
                # 2. K,V投影权重: 2 * H * H_kv
                # 3. 输出投影权重: H * H
                weight_bytes = (hidden_size * hidden_size + 
                               2 * hidden_size * kv_hidden_size + 
                               hidden_size * hidden_size) * bytes_per_element if include_weight_access else 0
                
                # 返回内存访问量
                return activation_bytes + weight_bytes
        
        elif operator.name == "FFN":
            hidden_size = kwargs.get("hidden_size", 1)
            intermediate_size = kwargs.get("intermediate_size", 1)
            
            # 激活内存访问：
            # 1. 输入: B * cur_len * H
            # 2. 中间结果: B * cur_len * 4H
            # 3. 输出: B * cur_len * H
            # 4. 残差连接: 2 * B * cur_len * H
            activation_bytes = (batch_size * cur_len * hidden_size +  # 输入
                               batch_size * cur_len * intermediate_size +  # 中间结果
                               batch_size * cur_len * hidden_size +  # 输出
                               2 * batch_size * cur_len * hidden_size) * bytes_per_element  # 残差连接
            
            # 权重内存访问（如果需要）：
            # 1. 上投影权重: H * 4H
            # 2. 下投影权重: 4H * H
            weight_bytes = (hidden_size * intermediate_size + 
                           intermediate_size * hidden_size) * bytes_per_element if include_weight_access else 0
            
            # 返回内存访问量
            return activation_bytes + weight_bytes
        
        elif operator.name == "LayerNorm":
            hidden_size = kwargs.get("hidden_size", 1)
            
            # LayerNorm主要是激活内存访问：
            # 输入读取 + 输出写入
            activation_bytes = 2 * batch_size * cur_len * hidden_size * bytes_per_element
            
            # 权重内存访问（如果需要）：
            # gamma和beta参数
            weight_bytes = 2 * hidden_size * bytes_per_element if include_weight_access else 0
            
            # 返回内存访问量
            return activation_bytes + weight_bytes
        
        elif operator.name == "Logits":
            hidden_size = kwargs.get("hidden_size", 1)
            vocab_size = kwargs.get("vocab_size", 1)
            
            # 激活内存访问：
            # 1. 输入读取 + 输出写入
            activation_bytes = (batch_size * cur_len * hidden_size +  # 输入
                               batch_size * cur_len * vocab_size) * bytes_per_element  # 输出
            
            # 权重内存访问（如果需要）：
            # 权重矩阵读取
            weight_bytes = hidden_size * vocab_size * bytes_per_element if include_weight_access else 0
            
            # 返回内存访问量
            return activation_bytes + weight_bytes
        
        elif operator.name == "MLAAttention":
            num_heads = kwargs.get("num_heads", 1)
            head_dim = kwargs.get("head_dim", 64)
            mixture_size = kwargs.get("mixture_size", 4)
            hidden_size = num_heads * head_dim
            matrix_fusion = kwargs.get("matrix_fusion", True)  # 是否启用矩阵吸收优化
            
            if is_decode:
                # Decode阶段（使用KV Cache）
                if matrix_fusion:
                    # 矩阵吸收后，省去Q/K/V投影输出的一次写
                    # 1. 输入数据: B * cur_len(=1) * H
                    # 2. φK, V缓存读取: 2 * B * ctx_len * h * d
                    # 3. φK, V新token写入: 2 * B * cur_len(=1) * h * d
                    # 4. 注意力得分: B * h * cur_len(=1) * ctx_len (写入+读回)
                    # 5. 注意力输出: B * cur_len(=1) * H
                    # 6. 残差连接: 2 * B * cur_len(=1) * H
                    
                    # 注意力得分写入和读回
                    attn_scores_elements = (batch_size * num_heads * cur_len * ctx_len +  # 写入
                                           batch_size * num_heads * cur_len * ctx_len)    # 读回
                    
                    activation_bytes = (batch_size * cur_len * hidden_size +              # 输入数据
                                       2 * batch_size * ctx_len * hidden_size +          # φK, V缓存读取
                                       2 * batch_size * cur_len * hidden_size +          # φK, V新token写入
                                       attn_scores_elements +                            # 注意力得分
                                       batch_size * cur_len * hidden_size +              # 注意力输出
                                       2 * batch_size * cur_len * hidden_size) * bytes_per_element  # 残差连接
                else:
                    # 未使用矩阵吸收
                    # 1. 输入数据: B * cur_len(=1) * H
                    # 2. 投影输出: 3 * B * cur_len(=1) * H (Q, K, V各一次)
                    # 3. φK, V缓存读取: 2 * B * ctx_len * h * d
                    # 4. φK, V新token写入: 2 * B * cur_len(=1) * h * d
                    # 5. 注意力得分: B * h * cur_len(=1) * ctx_len (写入+读回)
                    # 6. 注意力输出: B * cur_len(=1) * H
                    # 7. 残差连接: 2 * B * cur_len(=1) * H
                    
                    # 注意力得分写入和读回
                    attn_scores_elements = (batch_size * num_heads * cur_len * ctx_len +  # 写入
                                           batch_size * num_heads * cur_len * ctx_len)    # 读回
                    
                    activation_bytes = (batch_size * cur_len * hidden_size +              # 输入数据
                                       3 * batch_size * cur_len * hidden_size +          # Q, K, V投影输出
                                       2 * batch_size * ctx_len * hidden_size +          # φK, V缓存读取
                                       2 * batch_size * cur_len * hidden_size +          # φK, V新token写入
                                       attn_scores_elements +                            # 注意力得分
                                       batch_size * cur_len * hidden_size +              # 注意力输出
                                       2 * batch_size * cur_len * hidden_size) * bytes_per_element  # 残差连接
                
                # 权重内存访问（如果需要）
                if include_weight_access:
                    # 1. Q, K, V投影权重: 3 * H * H
                    # 2. 门控MLP权重: H * m + m
                    # 3. 输出投影权重: H * H
                    weight_bytes = (3 * hidden_size * hidden_size +            # Q, K, V投影
                                   hidden_size * mixture_size + mixture_size + # 门控MLP
                                   hidden_size * hidden_size) * bytes_per_element # 输出投影
                else:
                    weight_bytes = 0
                
                return activation_bytes + weight_bytes
            
            else:
                # Prefill阶段（不使用KV Cache）
                if matrix_fusion:
                    # 矩阵吸收后，省去Q/K/V投影输出的一次写
                    # 1. 输入数据: B * cur_len * H
                    # 2. 投影+φ映射: 直接输出结果
                    # 3. 注意力得分: B * h * cur_len * cur_len (写入+读回)
                    # 4. 注意力输出: B * cur_len * H
                    # 5. 残差连接: 2 * B * cur_len * H
                    # 6. φK, V缓存初始化: 2 * B * cur_len * h * d
                    
                    # 注意力得分写入和读回
                    attn_scores_elements = (batch_size * num_heads * cur_len * cur_len +  # 写入
                                           batch_size * num_heads * cur_len * cur_len)    # 读回
                    
                    activation_bytes = (batch_size * cur_len * hidden_size +              # 输入数据
                                       attn_scores_elements +                            # 注意力得分
                                       batch_size * cur_len * hidden_size +              # 注意力输出
                                       2 * batch_size * cur_len * hidden_size +          # 残差连接
                                       2 * batch_size * cur_len * hidden_size) * bytes_per_element  # φK, V缓存初始化
                else:
                    # 未使用矩阵吸收
                    # 1. 输入数据: B * cur_len * H
                    # 2. 投影输出: 3 * B * cur_len * H (Q, K, V各一次)
                    # 3. φ映射: 2 * B * cur_len * H (Q, K各一次)
                    # 4. 注意力得分: B * h * cur_len * cur_len (写入+读回)
                    # 5. 注意力输出: B * cur_len * H
                    # 6. 残差连接: 2 * B * cur_len * H
                    # 7. φK, V缓存初始化: 2 * B * cur_len * h * d
                    
                    # 注意力得分写入和读回
                    attn_scores_elements = (batch_size * num_heads * cur_len * cur_len +  # 写入
                                           batch_size * num_heads * cur_len * cur_len)    # 读回
                    
                    activation_bytes = (batch_size * cur_len * hidden_size +              # 输入数据
                                       3 * batch_size * cur_len * hidden_size +          # Q, K, V投影输出
                                       2 * batch_size * cur_len * hidden_size +          # φ映射
                                       attn_scores_elements +                            # 注意力得分
                                       batch_size * cur_len * hidden_size +              # 注意力输出
                                       2 * batch_size * cur_len * hidden_size +          # 残差连接
                                       2 * batch_size * cur_len * hidden_size) * bytes_per_element  # φK, V缓存初始化
                
                # 权重内存访问（如果需要）
                if include_weight_access:
                    # 1. Q, K, V投影权重: 3 * H * H
                    # 2. 门控MLP权重: H * m + m
                    # 3. 输出投影权重: H * H
                    weight_bytes = (3 * hidden_size * hidden_size +            # Q, K, V投影
                                   hidden_size * mixture_size + mixture_size + # 门控MLP
                                   hidden_size * hidden_size) * bytes_per_element # 输出投影
                else:
                    weight_bytes = 0
                
                return activation_bytes + weight_bytes
        
        else:
            return 1.0  # 默认值


class SerialStrategy(SimulationStrategy):
    """串行执行模型模拟策略，假设计算和访存完全串行执行"""
    
    def __init__(self, params: Dict[str, Any] = None):
        super().__init__("Serial", params)
    
    def estimate_execution_time(self, operator, hardware, batch_size: int, 
                              seq_length: int, **kwargs) -> float:
        """
        使用串行模型估算执行时间
        
        计算时间 = 计算时间 + 内存访问时间
        计算时间 = 计算量 / 计算能力
        内存访问时间 = 内存访问量 / 内存带宽
        
        Args:
            operator: 算子对象
            hardware: 硬件对象
            batch_size: 批处理大小
            seq_length: 序列长度
            kwargs: 其他参数
            
        Returns:
            执行时间（秒）
        """
        # 获取算子的计算量和访存量
        compute_ops = self._get_compute_operations(operator, batch_size, seq_length, **kwargs)
        # 将硬件对象传递给_get_memory_access
        kwargs["hardware"] = hardware
        memory_bytes = self._get_memory_access(operator, batch_size, seq_length, **kwargs)
        
        # 根据硬件性能计算时间
        operation_type = kwargs.get("operation_type", "fp16")  # 默认使用fp16
        compute_capability = hardware.get_compute_capability(operation_type)
        memory_bandwidth = hardware.get_memory_bandwidth()
        
        # 计算实际执行时间（秒）
        compute_time = compute_ops / (compute_capability * 1e12)  # TFLOPS转换为FLOPS
        memory_time = memory_bytes / (memory_bandwidth * 1e9)  # GB/s转换为B/s
        
        # 串行模型：计算时间和访存时间相加
        return compute_time + memory_time
    
    def _get_compute_operations(self, operator, batch_size: int, seq_length: int, **kwargs) -> float:
        """计算算子的计算量（浮点操作数）"""
        # 复用RooflineStrategy的计算方法
        roofline = RooflineStrategy()
        return roofline._get_compute_operations(operator, batch_size, seq_length, **kwargs)
    
    def _get_memory_access(self, operator, batch_size: int, seq_length: int, **kwargs) -> float:
        """计算算子的内存访问量（字节）"""
        # 复用RooflineStrategy的计算方法
        roofline = RooflineStrategy()
        return roofline._get_memory_access(operator, batch_size, seq_length, **kwargs) 