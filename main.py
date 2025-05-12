#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM性能模拟器

模拟不同模型在不同硬件条件下的性能表现，包括吞吐量和延迟。
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, Any, List, Tuple
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

from utils.factory import Factory
from metrics.performance import PerformanceMetrics

class LLMHardwareEval:
    """LLM硬件性能评估器"""
    
    def __init__(self, config_path: str = None):
        """
        初始化评估器
        
        Args:
            config_path: 配置文件路径，如果为None则使用默认配置
        """
        self.config = self._load_config(config_path)
        
        # 创建模型
        if "model" in self.config:
            model_config = self.config["model"]
            
            # 获取模型名称和大小
            if "name" in model_config and "size" in model_config:
                self.model = Factory.create_model(
                    model_config["name"],
                    model_config["size"],
                    model_config.get("custom_config")
                )
            else:
                raise ValueError("模型配置必须包含'name'和'size'字段")
        else:
            # 默认模型
            self.model = Factory.create_model("llama3", "8B")
        
        # 创建硬件集群
        self.clusters = {}
        for cluster_name, cluster_config in self.config["clusters"].items():
            self.clusters[cluster_name] = Factory.create_cluster(cluster_config)
        
        # 创建模拟策略
        self.strategies = {}
        for strategy_name, strategy_config in self.config["strategies"].items():
            self.strategies[strategy_name] = Factory.create_strategy(
                strategy_name,
                strategy_config
            )
        
        # 创建负载
        self.workloads = {}
        for workload_name, workload_config in self.config["workloads"].items():
            self.workloads[workload_name] = Factory.create_workload(workload_config)
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """加载配置"""
        default_config = {
            "model": {
                "name": "llama3",
                "size": "8B"
            },
            "clusters": {
                "gpu_small": "gpu_small",
                "cpu_small": "cpu_small"
            },
            "strategies": {
                "roofline": None,
                "analytical": None
            },
            "workloads": {
                "constant": "constant",
                "realistic": "realistic"
            },
            "output": {
                "save_results": True,
                "output_dir": "results",
                "plot_charts": True
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                user_config = json.load(f)
                # 合并用户配置和默认配置
                for key, value in user_config.items():
                    # 直接替换默认配置中对应的部分
                    default_config[key] = value
        
        return default_config
    
    def run_simulation(self, cluster_name: str, strategy_name: str, 
                      workload_name: str) -> Dict[str, Any]:
        """
        运行单个模拟
        
        Args:
            cluster_name: 集群名称
            strategy_name: 策略名称
            workload_name: 负载名称
            
        Returns:
            模拟结果
        """
        print(f"运行模拟: 集群={cluster_name}, 策略={strategy_name}, 负载={workload_name}")
        
        # 获取集群、策略和负载
        cluster = self.clusters[cluster_name]
        strategy = self.strategies[strategy_name]
        workload = self.workloads[workload_name]
        
        # 生成请求
        requests = workload.generate_requests()
        
        # 模拟结果
        results = []
        
        # 为每个请求分配资源并计算性能
        for i, request in enumerate(requests):
            # 分配资源
            # 很可能不需要这步，先假设请求能完全放下？
            allocated_hardware = cluster.allocate_resources(
                self.model,
                request["batch_size"],
                request["input_length"]
            )
            
            if not allocated_hardware:
                print(f"警告: 请求 {i} 无法分配足够资源，跳过")
                continue
            
            # 预填充阶段
            prefill_time = self.model.estimate_runtime(
                hardware=allocated_hardware[0],  # 简化处理，只使用第一个硬件
                strategy=strategy,
                batch_size=request["batch_size"],
                seq_length=request["input_length"],
                phase="prefill"
            )
            
            # 解码阶段
            decode_time = self.model.estimate_runtime(
                hardware=allocated_hardware[0],  # 简化处理，只使用第一个硬件
                strategy=strategy,
                batch_size=request["batch_size"],
                seq_length=request["input_length"],  # 解码时输入长度已固定
                phase="decode"
            ) * request["output_length"]  # 乘以输出token数
            
            # 计算指标
            ttft = PerformanceMetrics.calculate_ttft(prefill_time)
            tpot = PerformanceMetrics.calculate_tpot(decode_time, request["output_length"])
            e2e_latency = PerformanceMetrics.calculate_e2e_latency(prefill_time, decode_time)
            
            # 记录结果
            result = {
                "request_id": i,
                "batch_size": request["batch_size"],
                "input_length": request["input_length"],
                "output_length": request["output_length"],
                "allocated_hardware": [hw.name for hw in allocated_hardware],
                "prefill_time": prefill_time,
                "decode_time": decode_time,
                "ttft": ttft,
                "tpot": tpot,
                "e2e_latency": e2e_latency
            }
            
            results.append(result)
            
            # 释放资源（实际使用时应在请求完成后释放）
            if hasattr(cluster, "release_resources"):
                cluster.release_resources(str(i))
        
        # 汇总指标
        total_tokens = sum(r["input_length"] + r["output_length"] for r in results)
        total_time = max(r["e2e_latency"] for r in results) if results else 0
        
        throughput = PerformanceMetrics.calculate_throughput(
            len(results), total_tokens, total_time
        ) if total_time > 0 else (0, 0)
        
        cost_efficiency = PerformanceMetrics.calculate_cost_efficiency(
            throughput[1], cluster.get_cost_per_hour()
        ) if cluster.get_cost_per_hour() > 0 else 0
        
        # 聚合结果
        aggregate_metrics = PerformanceMetrics.aggregate_metrics(results)
        
        simulation_result = {
            "config": {
                "cluster": cluster_name,
                "strategy": strategy_name,
                "workload": workload_name,
                "model": self.model.name
            },
            "metrics": {
                "requests_completed": len(results),
                "total_tokens": total_tokens,
                "throughput_rps": throughput[0],
                "throughput_tps": throughput[1],
                "cost_per_hour": cluster.get_cost_per_hour(),
                "tokens_per_dollar": cost_efficiency
            },
            "aggregate_metrics": aggregate_metrics,
            "detailed_results": results
        }
        
        print(f"模拟完成: 吞吐量 = {throughput[1]:.2f} token/s, "
              f"平均TTFT = {aggregate_metrics['avg_ttft']*1000:.2f} ms, "
              f"平均TPOT = {aggregate_metrics['avg_tpot']*1000:.2f} ms")
        
        return simulation_result
    
    def run_all_simulations(self) -> List[Dict[str, Any]]:
        """
        运行所有模拟组合
        
        Returns:
            所有模拟结果列表
        """
        all_results = []
        
        for cluster_name in self.clusters:
            for strategy_name in self.strategies:
                for workload_name in self.workloads:
                    result = self.run_simulation(
                        cluster_name, strategy_name, workload_name
                    )
                    all_results.append(result)
        
        return all_results
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str = None):
        """
        保存模拟结果
        
        Args:
            results: 模拟结果列表
            output_dir: 输出目录，如果为None则使用配置中的目录
        """
        if output_dir is None:
            output_dir = self.config["output"].get("output_dir", "results")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存完整结果为JSON
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        full_results_path = os.path.join(output_dir, f"simulation_results_{timestamp}.json")
        
        with open(full_results_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"已保存完整结果到: {full_results_path}")
        
        # 创建汇总表格数据
        summary_data = []
        for result in results:
            config = result["config"]
            metrics = result["metrics"]
            agg_metrics = result["aggregate_metrics"]
            
            summary_data.append({
                "Model": config["model"],
                "Cluster": config["cluster"],
                "Strategy": config["strategy"],
                "Workload": config["workload"],
                "Throughput (token/s)": metrics["throughput_tps"],
                "Avg TTFT (ms)": agg_metrics["avg_ttft"] * 1000,
                "Avg TPOT (ms)": agg_metrics["avg_tpot"] * 1000,
                "P95 Latency (ms)": agg_metrics.get("p95_latency", 0) * 1000,
                "Tokens/Dollar": metrics["tokens_per_dollar"]
            })
        
        # 保存汇总为CSV
        summary_df = pd.DataFrame(summary_data)
        summary_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"已保存汇总结果到: {summary_path}")
        
        # 打印汇总表格
        print("\n模拟结果汇总:")
        print(tabulate(summary_data, headers="keys", tablefmt="grid"))
        
        # 绘制图表
        if self.config["output"].get("plot_charts", True):
            self._plot_results(summary_df, output_dir, timestamp)
    
    def _plot_results(self, df: pd.DataFrame, output_dir: str, timestamp: str):
        """绘制结果图表"""
        # 创建图表目录
        charts_dir = os.path.join(output_dir, "charts")
        os.makedirs(charts_dir, exist_ok=True)
        
        # 1. 不同集群的吞吐量对比
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby(['Cluster', 'Workload'])['Throughput (token/s)'].mean().unstack()
        df_grouped.plot(kind='bar')
        plt.title('不同集群的吞吐量对比')
        plt.ylabel('吞吐量 (token/s)')
        plt.xlabel('集群')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"throughput_by_cluster_{timestamp}.png"))
        
        # 2. 不同策略的延迟对比
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby(['Strategy', 'Workload'])['Avg TTFT (ms)'].mean().unstack()
        df_grouped.plot(kind='bar')
        plt.title('不同策略的TTFT对比')
        plt.ylabel('平均TTFT (ms)')
        plt.xlabel('策略')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"ttft_by_strategy_{timestamp}.png"))
        
        # 3. 成本效益对比
        plt.figure(figsize=(10, 6))
        df_grouped = df.groupby(['Cluster'])['Tokens/Dollar'].mean()
        df_grouped.plot(kind='bar')
        plt.title('不同集群的成本效益对比')
        plt.ylabel('每美元处理的token数')
        plt.xlabel('集群')
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, f"cost_efficiency_{timestamp}.png"))
        
        print(f"已保存图表到: {charts_dir}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LLM硬件性能评估器")
    parser.add_argument("--config", "-c", type=str, help="配置文件路径")
    args = parser.parse_args()
    
    # 创建评估器
    evaluator = LLMHardwareEval(args.config)
    
    # 运行所有模拟
    results = evaluator.run_all_simulations()
    
    # 保存结果
    if evaluator.config["output"].get("save_results", True):
        evaluator.save_results(results)


if __name__ == "__main__":
    main() 