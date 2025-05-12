#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
DeepSeekV3 MLA机制性能对比分析
此脚本比较DeepSeekV3的MLA(Mixture of Linear Attention)机制与传统注意力机制的性能差异
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
from simulation.strategies import RooflineStrategy
from models.deepseek import DeepSeekV3Model
from models.llama import LlamaModel

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def setup_hardware(hardware_config):
    """根据配置设置硬件"""
    gpu_config = hardware_config.get('gpu', {})
    
    gpu = GPU(
        name=gpu_config.get('name', 'Default GPU'),
        specs=gpu_config.get('specs', {}),
        cost_per_hour=gpu_config.get('cost_per_hour', 0.0)
    )
    
    return gpu

def create_model(model_config, model_type, attention_type=None):
    """创建模型"""
    if model_type.lower() == 'deepseek_v3':
        return DeepSeekV3Model(
            name=f"deepseek_v3-{model_config['size']}",
            config={
                'num_heads': model_config.get('num_heads', 32),
                'head_dim': model_config.get('head_dim', 64),
                'mixture_size': model_config.get('mixture_size', 4),
                'matrix_fusion': model_config.get('matrix_fusion', True),
                'num_layers': model_config.get('num_layers', 32),
                'vocab_size': model_config.get('vocabulary_size', 100000),
                'operation_type': model_config.get('operation_type', 'fp16')
            }
        )
    elif model_type.lower() == 'llama':
        return LlamaModel(
            name=f"llama-{model_config['size']}",
            config={
                'attention_type': attention_type or 'gqa',
                'num_heads': model_config.get('num_heads', 32),
                'num_kv_heads': model_config.get('num_kv_heads', 8),
                'head_dim': model_config.get('head_dim', 128),
                'num_layers': model_config.get('num_layers', 32),
                'vocab_size': model_config.get('vocabulary_size', 32000),
                'operation_type': model_config.get('operation_type', 'fp16')
            }
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def run_mla_comparison(config_path):
    """运行MLA与传统注意力机制的性能比较"""
    config = load_config(config_path)
    
    # 设置硬件和策略
    hardware = setup_hardware(config['hardware'])
    strategy = RooflineStrategy()
    
    # 获取工作负载配置
    workload = config['mla_comparison']['workload']
    batch_size = workload['batch_size']
    seq_length = workload['input_length']
    output_length = workload['output_length']
    
    # 准备结果
    results = []
    
    # 对每个比较模型进行测试
    for model_config in config['mla_comparison']['comparison']:
        print(f"测试模型: {model_config['name']}")
        
        # 创建模型
        model = create_model(
            config['model'],
            model_config['model'],
            model_config.get('attention_type')
        )
        
        # 测量prefill阶段性能
        start_time = time.time()
        prefill_time = model.estimate_runtime(
            hardware=hardware,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="prefill"
        )
        
        # 测量decode阶段性能
        decode_time = model.estimate_runtime(
            hardware=hardware,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="decode"
        )
        
        # 总执行时间
        total_time = prefill_time + decode_time * output_length
        
        # 收集结果
        result = {
            'name': model_config['name'],
            'model': model_config['model'],
            'attention_type': model_config.get('attention_type'),
            'description': model_config.get('description', ''),
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'tokens_per_second': (seq_length + output_length) / total_time * batch_size
        }
        
        results.append(result)
        print(f"  Prefill时间: {prefill_time:.4f}秒")
        print(f"  每token Decode时间: {decode_time:.6f}秒")
        print(f"  总完成时间: {total_time:.4f}秒")
        print(f"  吞吐量: {result['tokens_per_second']:.1f}个token/秒")
        print()
    
    # 保存结果
    if config['output']['save_results']:
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'mla_comparison_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 绘制图表
    if config['output']['plot_charts']:
        plot_mla_comparison(results, output_dir)
    
    return results

def run_matrix_fusion_comparison(config_path):
    """运行矩阵吸收优化的性能比较"""
    config = load_config(config_path)
    
    # 设置硬件和策略
    hardware = setup_hardware(config['hardware'])
    strategy = RooflineStrategy()
    
    # 获取工作负载配置
    workload = config['matrix_fusion_comparison']['workload']
    batch_size = workload['batch_size']
    seq_length = workload['input_length']
    output_length = workload['output_length']
    
    # 准备结果
    results = []
    
    # 对矩阵吸收的不同设置进行测试
    for fusion_config in config['matrix_fusion_comparison']['comparison']:
        print(f"测试矩阵吸收设置: {fusion_config['name']}")
        
        # 创建模型（使用相同的基础配置，但更改matrix_fusion设置）
        model_config = config['model'].copy()
        model_config['matrix_fusion'] = fusion_config['matrix_fusion']
        
        model = DeepSeekV3Model(
            name=f"deepseek_v3-{model_config['size']}",
            config=model_config
        )
        
        # 测量prefill阶段性能
        prefill_time = model.estimate_runtime(
            hardware=hardware,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="prefill"
        )
        
        # 测量decode阶段性能
        decode_time = model.estimate_runtime(
            hardware=hardware,
            strategy=strategy,
            batch_size=batch_size,
            seq_length=seq_length,
            phase="decode"
        )
        
        # 总执行时间
        total_time = prefill_time + decode_time * output_length
        
        # 收集结果
        result = {
            'name': fusion_config['name'],
            'matrix_fusion': fusion_config['matrix_fusion'],
            'description': fusion_config.get('description', ''),
            'prefill_time': prefill_time,
            'decode_time': decode_time,
            'total_time': total_time,
            'tokens_per_second': (seq_length + output_length) / total_time * batch_size
        }
        
        results.append(result)
        print(f"  Prefill时间: {prefill_time:.4f}秒")
        print(f"  每token Decode时间: {decode_time:.6f}秒")
        print(f"  总完成时间: {total_time:.4f}秒")
        print(f"  吞吐量: {result['tokens_per_second']:.1f}个token/秒")
        print()
    
    # 保存结果
    if config['output']['save_results']:
        output_dir = Path(config['output']['output_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'matrix_fusion_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 绘制图表
    if config['output']['plot_charts']:
        plot_matrix_fusion_comparison(results, output_dir)
    
    return results

def plot_mla_comparison(results, output_dir):
    """绘制MLA比较结果图表"""
    # 准备数据
    names = [r['name'] for r in results]
    prefill_times = [r['prefill_time'] for r in results]
    decode_times = [r['decode_time'] for r in results]
    total_times = [r['total_time'] for r in results]
    throughputs = [r['tokens_per_second'] for r in results]
    
    # 设置图表样式
    plt.style.use('ggplot')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制Prefill时间柱状图
    bars1 = ax1.bar(names, prefill_times, color='skyblue')
    ax1.set_title('Prefill阶段时间比较', fontsize=14)
    ax1.set_xlabel('注意力机制')
    ax1.set_ylabel('Prefill时间 (秒)')
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 绘制每token Decode时间柱状图
    bars2 = ax2.bar(names, decode_times, color='lightgreen')
    ax2.set_title('每token Decode时间比较', fontsize=14)
    ax2.set_xlabel('注意力机制')
    ax2.set_ylabel('每token Decode时间 (秒)')
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.6f}', ha='center', va='bottom')
    
    # 绘制总执行时间柱状图
    bars3 = ax3.bar(names, total_times, color='salmon')
    ax3.set_title('总执行时间比较', fontsize=14)
    ax3.set_xlabel('注意力机制')
    ax3.set_ylabel('总执行时间 (秒)')
    plt.setp(ax3.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 绘制吞吐量柱状图
    bars4 = ax4.bar(names, throughputs, color='gold')
    ax4.set_title('吞吐量比较', fontsize=14)
    ax4.set_xlabel('注意力机制')
    ax4.set_ylabel('吞吐量 (tokens/秒)')
    plt.setp(ax4.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'mla_comparison.png', dpi=300)
    plt.close()

def plot_matrix_fusion_comparison(results, output_dir):
    """绘制矩阵吸收比较结果图表"""
    # 准备数据
    names = [r['name'] for r in results]
    prefill_times = [r['prefill_time'] for r in results]
    decode_times = [r['decode_time'] for r in results]
    total_times = [r['total_time'] for r in results]
    throughputs = [r['tokens_per_second'] for r in results]
    
    # 设置图表样式
    plt.style.use('ggplot')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 绘制Prefill时间柱状图
    bars1 = ax1.bar(names, prefill_times, color='skyblue')
    ax1.set_title('Prefill阶段时间比较', fontsize=14)
    ax1.set_xlabel('矩阵吸收设置')
    ax1.set_ylabel('Prefill时间 (秒)')
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 绘制每token Decode时间柱状图
    bars2 = ax2.bar(names, decode_times, color='lightgreen')
    ax2.set_title('每token Decode时间比较', fontsize=14)
    ax2.set_xlabel('矩阵吸收设置')
    ax2.set_ylabel('每token Decode时间 (秒)')
    plt.setp(ax2.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.0001,
                f'{height:.6f}', ha='center', va='bottom')
    
    # 绘制总执行时间柱状图
    bars3 = ax3.bar(names, total_times, color='salmon')
    ax3.set_title('总执行时间比较', fontsize=14)
    ax3.set_xlabel('矩阵吸收设置')
    ax3.set_ylabel('总执行时间 (秒)')
    plt.setp(ax3.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom')
    
    # 绘制吞吐量柱状图
    bars4 = ax4.bar(names, throughputs, color='gold')
    ax4.set_title('吞吐量比较', fontsize=14)
    ax4.set_xlabel('矩阵吸收设置')
    ax4.set_ylabel('吞吐量 (tokens/秒)')
    plt.setp(ax4.get_xticklabels(), rotation=15, ha='right')
    
    # 添加数值标签
    for bar in bars4:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'matrix_fusion_comparison.png', dpi=300)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='运行DeepSeekV3 MLA性能测试')
    parser.add_argument('--config', type=str, default='config/deepseek_v3_config.json',
                        help='配置文件路径')
    parser.add_argument('--run-mla', action='store_true', help='运行MLA与传统注意力机制的性能比较')
    parser.add_argument('--run-fusion', action='store_true', help='运行矩阵吸收优化的性能比较')
    args = parser.parse_args()
    
    if args.run_mla:
        run_mla_comparison(args.config)
    elif args.run_fusion:
        run_matrix_fusion_comparison(args.config)
    else:
        # 默认同时运行两种比较
        run_mla_comparison(args.config)
        run_matrix_fusion_comparison(args.config) 