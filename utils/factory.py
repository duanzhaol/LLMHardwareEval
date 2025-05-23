from typing import Dict, Any, List, Optional, Union, Type

from hardware.base import Hardware
from hardware.gpu import GPU
from hardware.cpu import CPU

from models.base import Model
from models.transformer import TransformerModel
from models.llama import LlamaModel
from models.qwen import QwenModel

from cluster.base import Cluster
from cluster.gpu import GPUCluster
from cluster.cpu import CPUCluster

from simulation.base import SimulationStrategy
from simulation.strategies import RooflineStrategy, SerialStrategy

from workloads.base import Workload
from workloads.realistic import ConstantWorkload, RandomWorkload, DistributionWorkload, RealisticWorkload

from config.default import (
    DEFAULT_GPU_CONFIGS, 
    DEFAULT_CPU_CONFIGS, 
    DEFAULT_CLUSTER_CONFIGS,
    DEFAULT_STRATEGY_CONFIGS,
    DEFAULT_WORKLOAD_CONFIGS
)
from config.model_configs import MODEL_CONFIGS, get_model_config

class Factory:
    """工厂类，用于创建各种组件实例"""
    
    @staticmethod
    def create_hardware(hw_type: str, hw_model: str, config: Dict[str, Any] = None) -> Hardware:
        """
        创建硬件实例
        
        Args:
            hw_type: 硬件类型，"gpu"或"cpu"
            hw_model: 硬件型号，如"A100"、"EPYC"等
            config: 自定义配置，如果为None则使用默认配置
            
        Returns:
            硬件实例
        """
        if hw_type.lower() == "gpu":
            if config is None:
                if hw_model in DEFAULT_GPU_CONFIGS:
                    config = DEFAULT_GPU_CONFIGS[hw_model]
                else:
                    raise ValueError(f"未知GPU型号: {hw_model}")
            
            return GPU(
                name=config.get("name", hw_model),
                specs=config["specs"],
                cost_per_hour=config.get("cost_per_hour", 0.0)
            )
            
        elif hw_type.lower() == "cpu":
            if config is None:
                if hw_model in DEFAULT_CPU_CONFIGS:
                    config = DEFAULT_CPU_CONFIGS[hw_model]
                else:
                    raise ValueError(f"未知CPU型号: {hw_model}")
            
            return CPU(
                name=config.get("name", hw_model),
                specs=config["specs"],
                cost_per_hour=config.get("cost_per_hour", 0.0)
            )
            
        else:
            raise ValueError(f"不支持的硬件类型: {hw_type}")
    
    @staticmethod
    def create_model(model_name: str, model_size: str, config: Dict[str, Any] = None) -> Model:
        """
        创建模型实例
        
        Args:
            model_name: 模型名称，如"llama3"、"qwen"、"baichuan"等
            model_size: 模型大小，如"8B"、"32B"等
            config: 自定义配置，如果为None则使用默认配置
            
        Returns:
            模型实例
        """
        # 基于模型名称和大小创建具体模型
        if config is None:
            try:
                config = get_model_config(model_name, model_size)
            except ValueError as e:
                raise ValueError(f"未找到模型配置: {model_name}-{model_size}") from e
        
        # 根据模型名称选择对应的模型类
        model_name_lower = model_name.lower()
        if "llama" in model_name_lower:
            return LlamaModel(
                name=config.get("name", f"Llama-{model_size}"),
                config=config
            )
        elif "qwen" in model_name_lower:
            return QwenModel(
                name=config.get("name", f"Qwen-{model_size}"),
                config=config
            )
        elif "baichuan" in model_name_lower:
            return BaichuanModel(
                name=config.get("name", f"Baichuan-{model_size}"),
                config=config
            )
        else:
            # 如果没有匹配的具体模型类，使用通用的TransformerModel
            return TransformerModel(
                name=config.get("name", f"{model_name}-{model_size}"),
                config=config
            )
    
    @staticmethod
    def create_cluster(cluster_config: Union[str, Dict[str, Any]]) -> Cluster:
        """
        创建集群实例
        
        Args:
            cluster_config: 集群配置名称（使用默认配置）或自定义配置字典
            
        Returns:
            集群实例
        """
        if isinstance(cluster_config, str):
            if cluster_config in DEFAULT_CLUSTER_CONFIGS:
                config = DEFAULT_CLUSTER_CONFIGS[cluster_config]
            else:
                raise ValueError(f"未知集群配置: {cluster_config}")
        else:
            config = cluster_config
        
        # 获取集群类型和硬件型号
        hw_type = config["hardware_type"].lower()
        hw_model = config["hardware_model"]
        hw_count = config["hardware_count"]
        network_bandwidth = config["network_bandwidth"]
        allocation_strategy = config.get("allocation_strategy", "greedy")
        
        # 创建硬件列表
        hardware_list = [
            Factory.create_hardware(hw_type, hw_model)
            for _ in range(hw_count)
        ]
        
        # 创建集群实例
        if hw_type == "gpu":
            return GPUCluster(
                name=config["name"],
                gpu_list=hardware_list,
                network_bandwidth=network_bandwidth,
                specs=config.get("specs"),
                allocation_strategy=allocation_strategy
            )
        elif hw_type == "cpu":
            return CPUCluster(
                name=config["name"],
                cpu_list=hardware_list,
                network_bandwidth=network_bandwidth,
                specs=config.get("specs"),
                allocation_strategy=allocation_strategy
            )
        else:
            raise ValueError(f"不支持的集群类型: {hw_type}")
    
    @staticmethod
    def create_strategy(strategy_type: str, config: Dict[str, Any] = None) -> SimulationStrategy:
        """
        创建模拟策略实例
        
        Args:
            strategy_type: 策略类型，如"roofline"、"serial"等
            config: 自定义配置，如果为None则使用默认配置
            
        Returns:
            模拟策略实例
        """
        if config is None:
            if strategy_type in DEFAULT_STRATEGY_CONFIGS:
                config = DEFAULT_STRATEGY_CONFIGS[strategy_type]
            else:
                config = {}
        
        if strategy_type.lower() == "roofline":
            return RooflineStrategy(params=config)
            
        elif strategy_type.lower() == "serial":
            return SerialStrategy(params=config)
            
        else:
            raise ValueError(f"不支持的模拟策略类型: {strategy_type}")
    
    @staticmethod
    def create_workload(workload_config: Union[str, Dict[str, Any]]) -> Workload:
        """
        创建负载实例
        
        Args:
            workload_config: 负载配置名称（使用默认配置）或自定义配置字典
            
        Returns:
            负载实例
        """
        if isinstance(workload_config, str):
            if workload_config in DEFAULT_WORKLOAD_CONFIGS:
                config = DEFAULT_WORKLOAD_CONFIGS[workload_config]
            else:
                raise ValueError(f"未知负载配置: {workload_config}")
        else:
            config = workload_config
        
        workload_type = config["type"].lower()
        
        if workload_type == "constant":
            return ConstantWorkload(
                batch_size=config["batch_size"],
                input_length=config["input_length"],
                output_length=config["output_length"],
                num_requests=config.get("num_requests", 1)
            )
            
        elif workload_type == "random":
            return RandomWorkload(
                batch_size_range=config["batch_size_range"],
                input_length_range=config["input_length_range"],
                output_length_range=config["output_length_range"],
                num_requests=config.get("num_requests", 10),
                seed=config.get("seed")
            )
            
        elif workload_type == "distribution":
            return DistributionWorkload(
                batch_size_dist=config["batch_size_dist"],
                input_length_dist=config["input_length_dist"],
                output_length_dist=config["output_length_dist"],
                num_requests=config.get("num_requests", 100),
                seed=config.get("seed")
            )
            
        elif workload_type == "realistic":
            return RealisticWorkload(
                concurrent_users=config["concurrent_users"],
                arrival_rate=config["arrival_rate"],
                session_duration=config["session_duration"],
                input_length_profile=config["input_length_profile"],
                output_length_profile=config["output_length_profile"],
                max_requests=config.get("max_requests", 1000),
                seed=config.get("seed")
            )
            
        else:
            raise ValueError(f"不支持的负载类型: {workload_type}") 