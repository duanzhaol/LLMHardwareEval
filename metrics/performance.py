from typing import Dict, Any, List, Tuple
import numpy as np

class PerformanceMetrics:
    """性能指标计算类，用于计算各种LLM性能指标"""
    
    @staticmethod
    def calculate_ttft(prefill_time: float) -> float:
        """
        计算TTFT (Time To First Token)
        
        Args:
            prefill_time: 预填充阶段耗时（秒）
            
        Returns:
            TTFT（秒）
        """
        return prefill_time
    
    @staticmethod
    def calculate_tpot(decode_time: float, output_length: int) -> float:
        """
        计算TPOT (Time Per Output Token)
        
        Args:
            decode_time: 解码阶段耗时（秒）
            output_length: 输出token数量
            
        Returns:
            TPOT（秒/token）
        """
        if output_length <= 0:
            raise ValueError("输出token数必须大于0")
        return decode_time / output_length
    
    @staticmethod
    def calculate_e2e_latency(prefill_time: float, decode_time: float) -> float:
        """
        计算端到端延迟
        
        Args:
            prefill_time: 预填充阶段耗时（秒）
            decode_time: 解码阶段耗时（秒）
            
        Returns:
            端到端延迟（秒）
        """
        return prefill_time + decode_time
    
    @staticmethod
    def calculate_throughput(num_requests: int, total_tokens: int, total_time: float) -> Tuple[float, float]:
        """
        计算吞吐量
        
        Args:
            num_requests: 请求数量
            total_tokens: 总token数（输入+输出）
            total_time: 总耗时（秒）
            
        Returns:
            吞吐量元组：(请求/秒, token/秒)
        """
        if total_time <= 0:
            raise ValueError("总时间必须大于0")
        
        rps = num_requests / total_time  # 请求/秒
        tps = total_tokens / total_time  # token/秒
        
        return (rps, tps)
    
    @staticmethod
    def calculate_cost_efficiency(throughput: float, cost_per_hour: float) -> float:
        """
        计算成本效率
        
        Args:
            throughput: 吞吐量（token/秒）
            cost_per_hour: 每小时成本（美元）
            
        Returns:
            成本效率（token/美元）
        """
        if cost_per_hour <= 0:
            raise ValueError("每小时成本必须大于0")
        
        # 转换成本到每秒
        cost_per_second = cost_per_hour / 3600
        
        # 计算每美元处理的token数
        tokens_per_dollar = throughput / cost_per_second * 3600
        
        return tokens_per_dollar
    
    @staticmethod
    def calculate_percentiles(latencies: List[float], percentiles: List[float] = None) -> Dict[float, float]:
        """
        计算延迟的百分位数
        
        Args:
            latencies: 延迟值列表（秒）
            percentiles: 要计算的百分位数列表，默认为[50, 90, 95, 99]
            
        Returns:
            百分位数字典 {百分位: 延迟值}
        """
        if not percentiles:
            percentiles = [50, 90, 95, 99]
        
        result = {}
        for p in percentiles:
            result[p] = np.percentile(latencies, p)
        
        return result
    
    @staticmethod
    def aggregate_metrics(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        聚合多个请求的指标
        
        Args:
            metrics_list: 指标字典列表，每个字典包含单个请求的指标
            
        Returns:
            聚合指标字典
        """
        if not metrics_list:
            return {}
        
        # 提取所有延迟相关指标
        ttfts = [m.get("ttft", 0) for m in metrics_list if "ttft" in m]
        tpots = [m.get("tpot", 0) for m in metrics_list if "tpot" in m]
        e2e_latencies = [m.get("e2e_latency", 0) for m in metrics_list if "e2e_latency" in m]
        
        # 计算总请求数和总token数
        num_requests = len(metrics_list)
        total_input_tokens = sum(m.get("input_length", 0) for m in metrics_list)
        total_output_tokens = sum(m.get("output_length", 0) for m in metrics_list)
        total_tokens = total_input_tokens + total_output_tokens
        
        # 计算总时间（假设所有请求并行处理，取最长时间）
        if e2e_latencies:
            total_time = max(e2e_latencies)
        else:
            total_time = 0
        
        # 计算吞吐量
        throughput = PerformanceMetrics.calculate_throughput(
            num_requests, total_tokens, total_time
        ) if total_time > 0 else (0, 0)
        
        # 汇总指标
        aggregated = {
            "num_requests": num_requests,
            "total_input_tokens": total_input_tokens,
            "total_output_tokens": total_output_tokens,
            "total_tokens": total_tokens,
            "total_time": total_time,
            "throughput_rps": throughput[0],
            "throughput_tps": throughput[1],
            "avg_ttft": np.mean(ttfts) if ttfts else 0,
            "avg_tpot": np.mean(tpots) if tpots else 0,
            "avg_e2e_latency": np.mean(e2e_latencies) if e2e_latencies else 0
        }
        
        # 添加延迟百分位数
        if e2e_latencies:
            percentiles = PerformanceMetrics.calculate_percentiles(e2e_latencies)
            for p, value in percentiles.items():
                aggregated[f"p{int(p)}_latency"] = value
        
        return aggregated 