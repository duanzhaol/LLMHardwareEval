# LLM硬件性能模拟器

LLM硬件性能模拟器（LLMHardwareEval）是一个用于模拟和评估大型语言模型（LLM）在不同硬件环境下性能表现的工具。它可以模拟不同模型、硬件条件、并发度和请求长度下的性能指标，包括预填充（prefill）和解码（decode）的运行时间，从而估算吞吐量、首词延迟（TTFT）和每词延迟（TPOT）等指标。

## 功能特点

- 支持不同的LLM模型配置（如7B、13B、70B等）
- 支持多种硬件类型（GPU和CPU）和不同硬件规格
- 支持多种模拟策略（Roofline、解析模型、经验数据等）
- 支持灵活的负载模式（恒定、随机、分布式、现实场景等）
- 能够评估硬件资源分配策略
- 生成详细的性能报告和可视化图表
- 支持计算吞吐量/成本（throughput/$）比率

## 系统架构

```
LLMHardwareEval/
├── models/                 # 模型定义
├── operators/              # 算子定义
├── hardware/               # 硬件定义
├── simulation/             # 模拟策略
├── workloads/              # 负载定义
├── cluster/                # 集群定义
├── metrics/                # 性能指标计算
├── config/                 # 配置文件
├── utils/                  # 工具函数
└── main.py                 # 主程序入口
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行默认模拟

```bash
python main.py
```

### 使用自定义配置

```bash
python main.py --config your_config.json
```

## 配置说明

### 配置文件示例

```json
{
  "model": {
    "size": "7B",
    "custom_config": {
      "hidden_size": 4096,
      "num_layers": 32,
      "num_heads": 32,
      "intermediate_size": 16384,
      "vocab_size": 32000,
      "max_seq_length": 4096
    }
  },
  "clusters": {
    "gpu_cluster": "gpu_medium",
    "cpu_cluster": "cpu_small"
  },
  "strategies": {
    "roofline": null,
    "analytical": {
      "hardware_efficiency": 0.6,
      "overhead": 0.00002
    }
  },
  "workloads": {
    "light_load": {
      "type": "constant",
      "batch_size": 1,
      "input_length": 512,
      "output_length": 64,
      "num_requests": 50
    },
    "heavy_load": {
      "type": "realistic",
      "concurrent_users": 50,
      "arrival_rate": 5,
      "session_duration": 300,
      "input_length_profile": {"short": 0.2, "medium": 0.6, "long": 0.2},
      "output_length_profile": {"short": 0.3, "medium": 0.5, "long": 0.2},
      "max_requests": 200
    }
  },
  "output": {
    "save_results": true,
    "output_dir": "my_results",
    "plot_charts": true
  }
}
```

## 核心组件说明

### 模型（Model）

定义LLM模型的结构和参数，包括隐藏层大小、层数、注意力头数等。

### 算子（Operator）

定义模型中的各种操作算子，如矩阵乘法、自注意力、前馈网络等，每个算子都有其计算时间估算逻辑。

### 硬件（Hardware）

定义硬件设备的规格，包括计算能力、内存带宽、通信带宽等。目前支持GPU和CPU两种类型。

### 模拟策略（SimulationStrategy）

定义如何估算算子在特定硬件上的执行时间。支持的策略包括：

- Roofline模型：基于计算密度和内存带宽限制
- 解析模型：考虑硬件效率和额外开销
- 经验模型：基于实际测量数据

### 负载（Workload）

定义LLM服务的负载模式，包括批大小、输入长度、输出长度等。支持的负载类型包括：

- 恒定负载：所有请求具有相同的参数
- 随机负载：参数在指定范围内随机生成
- 分布式负载：参数按照指定分布生成
- 现实负载：模拟真实世界的LLM服务负载

### 集群（Cluster）

定义计算集群的配置和资源分配策略。支持的集群类型包括GPU集群和CPU集群。

### 性能指标（Metrics）

计算和聚合性能指标，如TTFT、TPOT、吞吐量、成本效率等。

## 扩展指南

### 添加新的模型

1. 继承`Model`基类
2. 实现`_build_operators`方法

### 添加新的算子

1. 继承`Operator`基类
2. 实现`compute_time`方法

### 添加新的硬件类型

1. 继承`Hardware`基类
2. 实现必要的方法，如`get_compute_capability`、`get_memory_bandwidth`等

### 添加新的模拟策略

1. 继承`SimulationStrategy`基类
2. 实现`estimate_execution_time`方法

## 样例输出

模拟运行后，系统会生成如下输出：

1. JSON格式的完整模拟结果
2. CSV格式的汇总结果
3. 可视化图表，包括吞吐量对比、延迟对比、成本效益对比等

## 许可证

MIT

## 致谢

感谢所有对该项目有所贡献的开发者和研究人员。 