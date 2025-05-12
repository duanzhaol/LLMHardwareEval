#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
精度比较测试脚本
此脚本运行不同精度类型的性能比较，包括FP32, FP16, BF16, INT8和FP8
"""

import os
import json
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# 添加项目根目录到Python路径
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hardware.gpu import GPU
from hardware.cpu import CPU 
from simulation.strategies import RooflineStrategy
from models.llama import LlamaModel

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_hardware(hardware_config):
    """根据配置设置硬件"""
    gpu_config = hardware_config.get('gpu', {})
    cpu_config = hardware_config.get('cpu', {})
    
    gpu = GPU(
        name=gpu_config.get('name', 'Default GPU'),
        specs=gpu_config.get('specs', {}),
        cost_per_hour=gpu_config.get('cost_per_hour', 0.0)
    )
    
    cpu = CPU(
        name=cpu_config.get('name', 'Default CPU'),
        specs=cpu_config.get('specs', {}),
        cost_per_hour=cpu_config.get('cost_per_hour', 0.0)
    )
    
    return {'gpu': gpu, 'cpu': cpu}

def create_model(model_config):
    """创建模型"""
    if model_config['name'].lower() == 'llama3':
        return LlamaModel(
            name=f"llama3-{model_config['size']}",
            config={
                'hidden_size': 4096,  # 这些参数应根据实际模型规格调整
                'num_heads': 32,
                'num_kv_heads': 8,
                'head_dim': 128,
                'intermediate_size': 11008,
                'num_layers': 32,
                'vocabulary_size': 32000,
                'operation_type': model_config.get('operation_type', 'fp16')
            }
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_config['name']}")

def run_precision_comparison(config_path):
    """运行精度比较测试"""
    config = load_config(config_path)
    
    # 设置硬件和策略
    hardware = setup_hardware(config['hardware'])
    strategy = RooflineStrategy()
    
    # 创建模型
    base_model = create_model(config['model'])
    
    # 获取工作负载配置
    workload = config['precision_comparison']['workload']
    batch_size = workload['batch_size']
    seq_length = workload['input_length']
    
    # 准备结果
    results = []
    
    # 对每种精度类型进行测试
    for precision in config['precision_comparison']['precision_types']:
        print(f"测试精度类型: {precision['name']} ({precision['operation_type']})")
        
        # 更新模型的操作类型
        model = create_model(config['model'])
        model.config['operation_type'] = precision['operation_type']
        
        # 测量prefill阶段性能
        start_time = time.time()
        prefill_time = model.estimate_runtime(
            hardware=hardware['gpu'],
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="prefill"
        )
        
        # 测量decode阶段性能
        decode_time = model.estimate_runtime(
            hardware=hardware['gpu'],
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="decode"
        )
        
        # 总执行时间
        total_time = prefill_time + decode_time * workload['output_length']
        
        # 收集结果
        result = {
            'precision': precision['name'],
            'operation_type': precision['operation_type'],
            'description': precision['description'],
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'tokens_per_second': (seq_length + workload['output_length']) / total_time * batch_size
        }
        
        results.append(result)
        print(f"  完成时间: {total_time:.4f}秒")
        print(f"  吞吐量: {result['tokens_per_second']:.1f}个token/秒")
        print()
    
    # 保存结果
    if config['output']['save_results']:
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'precision_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 绘制图表
    if config['output']['plot_charts']:
        plot_results(results, output_dir)
    
    return results

def plot_results(results, output_dir):
    """绘制结果图表"""
    # 准备数据
    precisions = [r['precision'] for r in results]
    total_times = [r['total_time'] for r in results]
    throughputs = [r['tokens_per_second'] for r in results]
    
    # 设置图表样式
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 绘制执行时间柱状图
    bars1 = ax1.bar(precisions, total_times, color='skyblue')
    ax1.set_title('不同精度的执行时间比较', fontsize=14)
    ax1.set_xlabel('精度类型')
    ax1.set_ylabel('总执行时间 (秒)')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 绘制吞吐量柱状图
    bars2 = ax2.bar(precisions, throughputs, color='lightgreen')
    ax2.set_title('不同精度的吞吐量比较', fontsize=14)
    ax2.set_xlabel('精度类型')
    ax2.set_ylabel('吞吐量 (tokens/秒)')
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_comparison.png', dpi=300)
    plt.close()
    
    # 绘制性能提升折线图
    plt.figure(figsize=(10, 6))
    
    # 以FP32为基准计算加速比
    fp32_time = next(r['total_time'] for r in results if r['precision'] == 'FP32')
    speedups = [fp32_time / r['total_time'] for r in results]
    
    plt.plot(precisions, speedups, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=1.0, color='red', linestyle='--')
    
    plt.title('不同精度相对于FP32的性能提升', fontsize=14)
    plt.xlabel('精度类型')
    plt.ylabel('加速比 (相对于FP32)')
    plt.grid(True)
    
    # 添加数值标签
    for i, speedup in enumerate(speedups):
        plt.text(i, speedup + 0.05, f'{speedup:.2f}x', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_speedup.png', dpi=300)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行精度比较测试')
    parser.add_argument('--config', type=str, default='config/precision_config.json',
                        help='配置文件路径')
    args = parser.parse_args()
    
    run_precision_comparison(args.config) 