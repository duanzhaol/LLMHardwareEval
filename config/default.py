"""默认配置文件，定义LLM性能模拟器的默认参数"""

# GPU硬件配置
DEFAULT_GPU_CONFIGS = {
    "A100": {
        "name": "NVIDIA A100",
        "specs": {
            "fp16_tflops": 312,
            "fp32_tflops": 156,
            "memory_bandwidth": 1935,  # GB/s
            "vram_gb": 40,
            "interconnect_bandwidth": 600,  # GB/s (NVLink)
        },
        "cost_per_hour": 4.0  # 美元/小时
    },
    "H100": {
        "name": "NVIDIA H100",
        "specs": {
            "fp16_tflops": 756,
            "fp32_tflops": 378,
            "memory_bandwidth": 3350,  # GB/s
            "vram_gb": 80,
            "interconnect_bandwidth": 900,  # GB/s (NVLink)
        },
        "cost_per_hour": 8.0  # 美元/小时
    },
    "V100": {
        "name": "NVIDIA V100",
        "specs": {
            "fp16_tflops": 125,
            "fp32_tflops": 62.5,
            "memory_bandwidth": 900,  # GB/s
            "vram_gb": 32,
            "interconnect_bandwidth": 300,  # GB/s (NVLink)
        },
        "cost_per_hour": 2.5  # 美元/小时
    }
}

# CPU硬件配置
DEFAULT_CPU_CONFIGS = {
    "EPYC": {
        "name": "AMD EPYC 7763",
        "specs": {
            "cores": 64,
            "frequency_ghz": 3.5,
            "memory_bandwidth": 204.8,  # GB/s
            "ram_gb": 512,
            "network_bandwidth": 200,  # GB/s
        },
        "cost_per_hour": 2.0  # 美元/小时
    },
    "Xeon": {
        "name": "Intel Xeon Platinum 8380",
        "specs": {
            "cores": 40,
            "frequency_ghz": 3.4,
            "memory_bandwidth": 256,  # GB/s
            "ram_gb": 256,
            "network_bandwidth": 100,  # GB/s
        },
        "cost_per_hour": 1.5  # 美元/小时
    }
}

# 模型配置
DEFAULT_MODEL_CONFIGS = {
    "7B": {
        "name": "LLM-7B",
        "hidden_size": 4096,
        "num_layers": 32,
        "num_heads": 32,
        "intermediate_size": 16384,  # 4 * hidden_size
        "vocab_size": 32000,
        "max_seq_length": 4096,
        "head_dim": 128,  # hidden_size / num_heads
        "parallelization_factor": 1.0
    },
    "13B": {
        "name": "LLM-13B",
        "hidden_size": 5120,
        "num_layers": 40,
        "num_heads": 40,
        "intermediate_size": 20480,  # 4 * hidden_size
        "vocab_size": 32000,
        "max_seq_length": 4096,
        "head_dim": 128,  # hidden_size / num_heads
        "parallelization_factor": 1.0
    },
    "70B": {
        "name": "LLM-70B",
        "hidden_size": 8192,
        "num_layers": 80,
        "num_heads": 64,
        "intermediate_size": 32768,  # 4 * hidden_size
        "vocab_size": 32000,
        "max_seq_length": 4096,
        "head_dim": 128,  # hidden_size / num_heads
        "parallelization_factor": 1.0
    }
}

# 集群配置
DEFAULT_CLUSTER_CONFIGS = {
    "gpu_small": {
        "name": "Small GPU Cluster",
        "hardware_type": "gpu",
        "hardware_count": 4,
        "hardware_model": "A100",
        "network_bandwidth": 100,  # GB/s
        "allocation_strategy": "greedy"
    },
    "gpu_medium": {
        "name": "Medium GPU Cluster",
        "hardware_type": "gpu",
        "hardware_count": 8,
        "hardware_model": "A100",
        "network_bandwidth": 200,  # GB/s
        "allocation_strategy": "even"
    },
    "gpu_large": {
        "name": "Large GPU Cluster",
        "hardware_type": "gpu",
        "hardware_count": 16,
        "hardware_model": "H100",
        "network_bandwidth": 400,  # GB/s
        "allocation_strategy": "performance"
    },
    "cpu_small": {
        "name": "Small CPU Cluster",
        "hardware_type": "cpu",
        "hardware_count": 8,
        "hardware_model": "EPYC",
        "network_bandwidth": 50,  # GB/s
        "allocation_strategy": "greedy"
    },
    "cpu_large": {
        "name": "Large CPU Cluster",
        "hardware_type": "cpu",
        "hardware_count": 32,
        "hardware_model": "EPYC",
        "network_bandwidth": 100,  # GB/s
        "allocation_strategy": "even"
    }
}

# 模拟策略配置
DEFAULT_STRATEGY_CONFIGS = {
    "roofline": {
        "name": "Roofline Model"
    },
    "analytical": {
        "name": "Analytical Model",
        "hardware_efficiency": 0.7,
        "overhead": 1e-5
    }
}

# 负载配置
DEFAULT_WORKLOAD_CONFIGS = {
    "constant": {
        "name": "Constant Workload",
        "type": "constant",
        "batch_size": 1,
        "input_length": 1024,
        "output_length": 128,
        "num_requests": 100
    },
    "random": {
        "name": "Random Workload",
        "type": "random",
        "batch_size_range": (1, 4),
        "input_length_range": (512, 2048),
        "output_length_range": (64, 512),
        "num_requests": 100,
        "seed": 42
    },
    "realistic": {
        "name": "Realistic Workload",
        "type": "realistic",
        "concurrent_users": 100,
        "arrival_rate": 10,  # 用户/秒
        "session_duration": 300,  # 秒
        "input_length_profile": {"short": 0.3, "medium": 0.5, "long": 0.2},
        "output_length_profile": {"short": 0.2, "medium": 0.6, "long": 0.2},
        "max_requests": 500,
        "seed": 42
    }
} 