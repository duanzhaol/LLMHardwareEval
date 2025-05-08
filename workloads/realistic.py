import random
from typing import Dict, Any, List, Tuple
import numpy as np
from workloads.base import Workload

class ConstantWorkload(Workload):
    """恒定负载，所有请求具有相同的参数"""
    
    def __init__(self, batch_size: int, input_length: int, output_length: int, 
                num_requests: int = 1):
        """
        初始化恒定负载
        
        Args:
            batch_size: 批大小
            input_length: 输入序列长度
            output_length: 输出序列长度
            num_requests: 请求数量
        """
        params = {
            "batch_size": batch_size,
            "input_length": input_length,
            "output_length": output_length,
            "num_requests": num_requests
        }
        super().__init__("Constant", params)
    
    def generate_requests(self) -> List[Dict[str, Any]]:
        """
        生成恒定参数的请求列表
        
        Returns:
            请求列表
        """
        return [
            {
                "batch_size": self.params["batch_size"],
                "input_length": self.params["input_length"],
                "output_length": self.params["output_length"]
            }
            for _ in range(self.params["num_requests"])
        ]


class RandomWorkload(Workload):
    """随机负载，请求参数在指定范围内随机生成"""
    
    def __init__(self, batch_size_range: Tuple[int, int], 
                input_length_range: Tuple[int, int],
                output_length_range: Tuple[int, int],
                num_requests: int = 10,
                seed: int = None):
        """
        初始化随机负载
        
        Args:
            batch_size_range: 批大小范围 (min, max)
            input_length_range: 输入长度范围 (min, max)
            output_length_range: 输出长度范围 (min, max)
            num_requests: 请求数量
            seed: 随机数种子
        """
        params = {
            "batch_size_range": batch_size_range,
            "input_length_range": input_length_range,
            "output_length_range": output_length_range,
            "num_requests": num_requests,
            "seed": seed
        }
        super().__init__("Random", params)
        
        # 设置随机数种子
        if seed is not None:
            random.seed(seed)
    
    def generate_requests(self) -> List[Dict[str, Any]]:
        """
        生成随机参数的请求列表
        
        Returns:
            请求列表
        """
        batch_min, batch_max = self.params["batch_size_range"]
        input_min, input_max = self.params["input_length_range"]
        output_min, output_max = self.params["output_length_range"]
        
        return [
            {
                "batch_size": random.randint(batch_min, batch_max),
                "input_length": random.randint(input_min, input_max),
                "output_length": random.randint(output_min, output_max)
            }
            for _ in range(self.params["num_requests"])
        ]


class DistributionWorkload(Workload):
    """分布式负载，请求参数按照指定分布生成"""
    
    def __init__(self, batch_size_dist: Dict[str, Any], 
                input_length_dist: Dict[str, Any],
                output_length_dist: Dict[str, Any],
                num_requests: int = 100,
                seed: int = None):
        """
        初始化分布式负载
        
        Args:
            batch_size_dist: 批大小分布参数，如{"type": "normal", "mean": 4, "std": 1}
            input_length_dist: 输入长度分布参数
            output_length_dist: 输出长度分布参数
            num_requests: 请求数量
            seed: 随机数种子
        """
        params = {
            "batch_size_dist": batch_size_dist,
            "input_length_dist": input_length_dist,
            "output_length_dist": output_length_dist,
            "num_requests": num_requests,
            "seed": seed
        }
        super().__init__("Distribution", params)
        
        # 设置随机数种子
        if seed is not None:
            np.random.seed(seed)
    
    def generate_requests(self) -> List[Dict[str, Any]]:
        """
        生成按分布参数的请求列表
        
        Returns:
            请求列表
        """
        requests = []
        
        batch_sizes = self._generate_from_distribution(
            self.params["batch_size_dist"],
            self.params["num_requests"]
        )
        
        input_lengths = self._generate_from_distribution(
            self.params["input_length_dist"],
            self.params["num_requests"]
        )
        
        output_lengths = self._generate_from_distribution(
            self.params["output_length_dist"],
            self.params["num_requests"]
        )
        
        for i in range(self.params["num_requests"]):
            requests.append({
                "batch_size": max(1, int(batch_sizes[i])),  # 确保批大小至少为1
                "input_length": max(1, int(input_lengths[i])),  # 确保输入长度至少为1
                "output_length": max(1, int(output_lengths[i]))  # 确保输出长度至少为1
            })
        
        return requests
    
    def _generate_from_distribution(self, dist_params: Dict[str, Any], size: int) -> np.ndarray:
        """根据分布参数生成随机数"""
        dist_type = dist_params["type"].lower()
        
        if dist_type == "normal" or dist_type == "gaussian":
            return np.random.normal(dist_params["mean"], dist_params["std"], size)
        
        elif dist_type == "uniform":
            return np.random.uniform(dist_params["min"], dist_params["max"], size)
        
        elif dist_type == "poisson":
            return np.random.poisson(dist_params["lam"], size)
        
        elif dist_type == "exponential":
            return np.random.exponential(dist_params["scale"], size)
        
        elif dist_type == "lognormal":
            return np.random.lognormal(dist_params["mean"], dist_params["sigma"], size)
        
        else:
            raise ValueError(f"不支持的分布类型: {dist_type}")


class RealisticWorkload(Workload):
    """现实负载，模拟真实世界的LLM服务负载模式"""
    
    def __init__(self, concurrent_users: int, 
                arrival_rate: float,
                session_duration: float,
                input_length_profile: Dict[str, float],
                output_length_profile: Dict[str, float],
                max_requests: int = 1000,
                seed: int = None):
        """
        初始化现实负载
        
        Args:
            concurrent_users: 并发用户数
            arrival_rate: 用户到达率（用户/秒）
            session_duration: 平均会话持续时间（秒）
            input_length_profile: 输入长度分布配置，如{"short": 0.3, "medium": 0.5, "long": 0.2}
            output_length_profile: 输出长度分布配置
            max_requests: 最大请求数
            seed: 随机数种子
        """
        # 设置输入和输出长度范围
        self.length_ranges = {
            "input": {
                "short": (1, 256),      # 短输入
                "medium": (257, 1024),  # 中等输入
                "long": (1025, 4096)    # 长输入
            },
            "output": {
                "short": (1, 64),       # 短输出
                "medium": (65, 512),    # 中等输出
                "long": (513, 2048)     # 长输出
            }
        }
        
        params = {
            "concurrent_users": concurrent_users,
            "arrival_rate": arrival_rate,
            "session_duration": session_duration,
            "input_length_profile": input_length_profile,
            "output_length_profile": output_length_profile,
            "max_requests": max_requests,
            "seed": seed
        }
        super().__init__("Realistic", params)
        
        # 设置随机数种子
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_requests(self) -> List[Dict[str, Any]]:
        """
        生成模拟真实世界的请求列表
        
        Returns:
            请求列表
        """
        requests = []
        
        # 模拟用户到达过程（泊松过程）
        arrival_times = []
        current_time = 0
        while len(arrival_times) < self.params["max_requests"]:
            # 生成下一个用户到达的时间间隔（指数分布）
            interval = np.random.exponential(1.0 / self.params["arrival_rate"])
            current_time += interval
            arrival_times.append(current_time)
        
        # 模拟每个用户的请求
        for arrival_time in arrival_times:
            # 根据输入长度配置选择长度类别
            input_category = self._select_category(self.params["input_length_profile"])
            output_category = self._select_category(self.params["output_length_profile"])
            
            # 在选定类别的范围内随机生成长度
            input_min, input_max = self.length_ranges["input"][input_category]
            output_min, output_max = self.length_ranges["output"][output_category]
            
            input_length = random.randint(input_min, input_max)
            output_length = random.randint(output_min, output_max)
            
            # 大多数情况下批大小为1，少数情况为较大值（模拟批处理）
            batch_size = 1
            if random.random() < 0.1:  # 10%的概率使用批处理
                batch_size = random.randint(2, 8)
            
            # 创建请求
            requests.append({
                "arrival_time": arrival_time,
                "batch_size": batch_size,
                "input_length": input_length,
                "output_length": output_length
            })
            
            # 如果达到最大请求数，停止生成
            if len(requests) >= self.params["max_requests"]:
                break
        
        return requests
    
    def _select_category(self, profile: Dict[str, float]) -> str:
        """根据概率分布选择类别"""
        rand = random.random()
        cumulative = 0.0
        for category, prob in profile.items():
            cumulative += prob
            if rand <= cumulative:
                return category
        
        # 默认返回最后一个类别
        return list(profile.keys())[-1] 