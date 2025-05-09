#!/usr/bin/env python3
"""
模型比较示例脚本
展示如何使用模型定义功能对不同模型进行性能对比
"""

from utils.factory import Factory
from simulation.strategies import RooflineStrategy
from hardware.gpu import GPU

def main():
    """主函数"""
    print("LLM硬件性能评估: 不同模型架构比较")
    print("-" * 50)
    
    # 创建硬件
    gpu_a100 = Factory.create_hardware("gpu", "A100")
    gpu_h100 = Factory.create_hardware("gpu", "H100")
    
    # 创建模拟策略
    strategy = RooflineStrategy()
    
    # 创建不同的模型
    print("创建模型实例...")
    models = [
        Factory.create_model("llama3", "8B"),
        Factory.create_model("llama3", "70B"),
        Factory.create_model("qwen", "7B"),
        Factory.create_model("qwen", "32B"),
        Factory.create_model("baichuan", "7B"),
        Factory.create_model("baichuan", "13B")
    ]
    
    # 比较参数
    batch_size = 1
    seq_length = 1024
    
    # 打印模型信息
    for model in models:
        print(f"- {model.name}")
    print("-" * 50)
    
    # 在A100上评估不同模型的性能
    print("在A100上的预填充阶段性能 (批大小=1, 序列长度=1024):")
    for model in models:
        time = model.estimate_runtime(
            hardware=gpu_a100,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="prefill"
        )
        print(f"{model.name}: {time:.6f} 秒")
    print()
    
    # 在H100上评估不同模型的性能
    print("在H100上的预填充阶段性能 (批大小=1, 序列长度=1024):")
    for model in models:
        time = model.estimate_runtime(
            hardware=gpu_h100,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="prefill"
        )
        print(f"{model.name}: {time:.6f} 秒")
    print()
    
    # 解码阶段性能评估
    print("在A100上的解码阶段性能 (批大小=1, 序列长度=1):")
    for model in models:
        time = model.estimate_runtime(
            hardware=gpu_a100,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=1,
            phase="decode"
        )
        print(f"{model.name}: {time:.6f} 秒/token")
    print()
    
    # 计算不同模型的吞吐量比较
    print("不同模型在A100上的吞吐量比较 (每秒生成token数):")
    for model in models:
        prefill_time = model.estimate_runtime(
            hardware=gpu_a100,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="prefill"
        )
        decode_time = model.estimate_runtime(
            hardware=gpu_a100,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=1,
            phase="decode"
        )
        
        # 假设生成128个token
        output_tokens = 128
        total_time = prefill_time + decode_time * output_tokens
        throughput = output_tokens / total_time
        
        print(f"{model.name}: {throughput:.2f} tokens/s")

if __name__ == "__main__":
    main() 