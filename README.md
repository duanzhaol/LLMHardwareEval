# LLM硬件性能模拟器
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/duanzhaol/LLMHardwareEval)
LLM硬件性能模拟器（LLMHardwareEval）是一个用于模拟和评估大型语言模型（LLM）在不同硬件环境下性能表现的工具。它可以模拟不同模型、硬件条件、并发度和请求长度下的性能指标，包括预填充（prefill）和解码（decode）的运行时间，从而估算吞吐量、首词延迟（TTFT）和每词延迟（TPOT）等指标。

## 功能特点

- 支持多种具体的LLM模型（如Llama3-8B、Qwen-32B等）
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
    "name": "llama3",
    "size": "8B"
  },
  "clusters": {
    "gpu_cluster": "gpu_medium",
    "cpu_cluster": "cpu_small"
  },
  "strategies": {
    "roofline": null,
    "serial": null
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
    "output_dir": "llama3_results",
    "plot_charts": true
  }
}
```

## 核心组件说明

### 模型（Model）

定义LLM模型的结构和参数，包括隐藏层大小、层数、注意力头数等。

模型定义使用具体的模型架构和大小:
```json
{
  "model": {
    "name": "llama3",
    "size": "8B"
  }
}
```

目前支持的模型架构：
- Llama系列 (`llama3`)：8B, 70B
- Qwen系列 (`qwen`)：7B, 32B
- Baichuan系列 (`baichuan`)：7B, 13B

每种模型架构都有其特定的参数配置和算子实现，准确地反映了实际模型的性能特征。

### 算子（Operator）

定义模型中的各种操作算子，如矩阵乘法、自注意力、前馈网络等，每个算子都有其计算时间估算逻辑。

### 硬件（Hardware）

定义硬件设备的规格，包括计算能力、内存带宽、通信带宽等。目前支持GPU和CPU两种类型。

### 模拟策略（SimulationStrategy）

定义如何估算算子在特定硬件上的执行时间。支持的策略包括：

- Roofline模型：假设计算和访存可以完全重叠，执行时间是两者的最大值。适用于理想情况下的性能上限估计。
  - 计算时间 = max(计算时间, 内存访问时间)
  - 计算时间 = 计算量 / 计算能力
  - 内存访问时间 = 内存访问量 / 内存带宽

- 串行模型：假设计算和访存完全串行执行，执行时间是两者之和。适用于保守的性能估计。
  - 计算时间 = 计算时间 + 内存访问时间
  - 计算时间 = 计算量 / 计算能力
  - 内存访问时间 = 内存访问量 / 内存带宽

这两种策略提供了性能估计的上下界：
- Roofline模型代表理想情况下的性能上限
- 串行模型代表保守情况下的性能下限

实际性能通常在这两个界限之间，具体取决于硬件架构和算子实现。

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

1. 在`config/model_configs.py`中添加新模型的配置
2. 在`models/`目录下创建新模型类，继承合适的基类
3. 实现特定的`_build_operators`方法

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

## 支持的模型

目前支持以下模型架构：

1. LLaMA系列
   - 支持MHA、GQA和MQA注意力机制
   - 使用RMSNorm
   - 使用旋转位置编码(RoPE)
   - 使用SwiGLU激活函数

2. Qwen系列
   - 支持MHA、GQA和MQA注意力机制
   - 使用RMSNorm
   - 使用旋转位置编码(RoPE)
   - 使用SwiGLU激活函数
   - 更大的中间层尺寸

每个模型都支持以下特性：
- 可配置的模型参数（层数、隐藏层大小、注意力头数等）
- 支持不同的注意力机制（MHA、GQA、MQA）
- 支持批处理和序列长度变化
- 支持KV缓存优化

## 模型配置示例

```json
{
    "model": {
        "type": "llama",  // 或 "qwen"
        "name": "llama-2-7b",  // 或 "qwen-7b"
        "config": {
            "hidden_size": 4096,
            "num_layers": 32,
            "num_heads": 32,
            "head_dim": 128,
            "intermediate_size": 11008,
            "vocab_size": 32000,  // LLaMA的词表大小
            // 或 "vocab_size": 151936,  // Qwen的词表大小
            "attention_type": "mha"  // 可选: "mha", "gqa", "mqa"
        }
    },
    "hardware": {
        // ... 硬件配置 ...
    },
    "simulation": {
        // ... 模拟配置 ...
    }
}
```

## 模型架构

本模拟器支持多种Transformer模型架构，包括：

### 注意力机制

- **Multi-Head Attention (MHA)**
  - 标准的多头注意力机制
  - 每个头独立计算注意力
  - 计算量：2 * B * S * S * H * D，其中B是批次大小，S是序列长度，H是头数，D是头维度
  - 内存访问：Q,K,V矩阵 + 注意力权重 + 输出 + 残差连接

- **Grouped Query Attention (GQA)**
  - 将查询头分组，每组共享相同的键值头
  - 减少内存使用和计算量
  - 计算量：2 * B * S * S * (H/K) * D，其中K是KV头数
  - 内存访问：Q矩阵 + K,V矩阵(共享) + 注意力权重 + 输出 + 残差连接

- **Multi-Query Attention (MQA)**
  - GQA的特例，只有一个KV头
  - 进一步减少内存使用和计算量
  - 计算量：2 * B * S * S * H * D
  - 内存访问：Q矩阵 + K,V矩阵(单头) + 注意力权重 + 输出 + 残差连接

### Transformer层结构

每个Transformer模型包含以下组件：

1. **词嵌入层（Embedding）**
   - 将输入token转换为向量表示
   - 主要是查表操作，计算量很小
   - 计算量：B * S（每个token一次查表）
   - 内存访问：词嵌入表（权重） + 输出
   - 注意：实际运行时，词嵌入表会被缓存，减少内存访问

2. **Transformer层**
   - **输入层归一化（LayerNorm）**
     - 对输入进行归一化处理
     - 计算量：4 * B * S * H
     - 内存访问：输入 + 统计量 + 输出

   - **自注意力层（带残差连接）**
     - 计算注意力权重和加权求和
     - 包含残差连接（输入直接加到输出）
     - 计算量：2 * B * S * S * H * D
     - 内存访问：Q,K,V矩阵 + 注意力权重 + 输出 + 残差连接

   - **层归一化（LayerNorm）**
     - 对注意力输出进行归一化
     - 计算量：4 * B * S * H
     - 内存访问：输入 + 统计量 + 输出

   - **前馈网络（带残差连接）**
     - 两层线性变换，中间有激活函数
     - 包含残差连接（输入直接加到输出）
     - 计算量：2 * B * S * H * (4H) + 2 * B * S * (4H) * H
     - 内存访问：输入 + 中间结果 + 输出 + 残差连接

残差连接的影响：
- 计算量：残差连接的计算量很小，主要是加法操作
- 内存访问：每个残差连接需要额外的读写操作（2 * B * S * H）
- 性能影响：残差连接主要影响内存访问，对计算量的影响较小

### 配置示例

```json
{
    "model": {
        "name": "llama",
        "attention_type": "gqa",
        "num_heads": 32,
        "num_kv_heads": 4,
        "head_dim": 128,
        "hidden_size": 4096,
        "intermediate_size": 16384,
        "num_layers": 32
    }
}
``` 