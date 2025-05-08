from typing import Dict, Any, List
from cluster.base import Cluster
from hardware.base import Hardware
from hardware.cpu import CPU

class CPUCluster(Cluster):
    """CPU集群类，管理CPU硬件资源"""
    
    def __init__(self, name: str, cpu_list: List[CPU], 
                network_bandwidth: float, 
                specs: Dict[str, Any] = None,
                allocation_strategy: str = "greedy"):
        """
        初始化CPU集群
        
        Args:
            name: 集群名称
            cpu_list: CPU列表
            network_bandwidth: 集群内网络带宽（GB/s）
            specs: 其他集群规格
            allocation_strategy: 资源分配策略，可选：
                - "greedy": 贪婪分配，尽可能少用CPU
                - "even": 均匀分配，尽可能均衡负载
                - "performance": 性能优先，尽可能使用高性能CPU
        """
        super().__init__(name, cpu_list, network_bandwidth, specs)
        self.allocation_strategy = allocation_strategy
        
        # 资源状态跟踪
        self.available_cpus = list(cpu_list)  # 可用CPU列表
        self.allocated_cpus = {}  # 已分配CPU字典 {task_id: [cpu1, cpu2, ...]}
        
        # 为了方便资源分配，按FLOPS排序CPU
        self.available_cpus.sort(key=lambda cpu: cpu.get_compute_capability("fp32"), reverse=True)
    
    def allocate_resources(self, model, batch_size: int, 
                         seq_length: int) -> List[Hardware]:
        """
        为给定模型和请求参数分配CPU资源
        
        Args:
            model: 模型对象
            batch_size: 批处理大小
            seq_length: 序列长度
            
        Returns:
            分配的CPU列表
        """
        # 估算模型所需的计算能力和内存
        required_memory = self._estimate_memory_requirement(model, batch_size, seq_length)
        required_compute = self._estimate_compute_requirement(model, batch_size, seq_length)
        
        # 根据不同的分配策略选择合适的CPU
        if self.allocation_strategy == "greedy":
            return self._greedy_allocation(required_memory, required_compute)
        elif self.allocation_strategy == "even":
            return self._even_allocation(required_memory)
        elif self.allocation_strategy == "performance":
            return self._performance_allocation(required_compute)
        else:
            raise ValueError(f"不支持的分配策略: {self.allocation_strategy}")
    
    def _estimate_memory_requirement(self, model, batch_size: int, seq_length: int) -> float:
        """估算模型运行所需的内存（GB）"""
        # 这里简化估算，实际应考虑模型参数大小、激活值大小等
        
        # 模型参数大小（假设FP32格式）
        param_bytes = 0
        if hasattr(model, "config") and "hidden_size" in model.config:
            hidden_size = model.config["hidden_size"]
            num_layers = model.config.get("num_layers", 12)
            vocab_size = model.config.get("vocab_size", 50000)
            
            # 简化的参数量计算
            param_count = num_layers * 12 * hidden_size * hidden_size + vocab_size * hidden_size
            param_bytes = param_count * 4  # FP32每个参数4字节
        else:
            # 默认值，如果无法从模型中获取
            param_bytes = 4e9  # 假设4GB参数
        
        # 激活值和中间结果大小（与batch_size和seq_length相关）
        activation_bytes = batch_size * seq_length * 4 * 2e6  # 假设每token激活值2MB
        
        # 总内存需求（字节转GB）
        total_memory_gb = (param_bytes + activation_bytes) / 1e9
        
        # 增加缓冲区
        return total_memory_gb * 1.5  # 增加50%缓冲（CPU通常需要更多缓冲）
    
    def _estimate_compute_requirement(self, model, batch_size: int, seq_length: int) -> float:
        """估算模型运行所需的计算能力（GFLOPS）"""
        # 简化估算，基于模型大小和输入规模
        
        flops_per_token = 0
        if hasattr(model, "config") and "hidden_size" in model.config:
            hidden_size = model.config["hidden_size"]
            num_layers = model.config.get("num_layers", 12)
            
            # 简化的计算量估算（每token每层）
            flops_per_token_per_layer = 24 * hidden_size * hidden_size  # 24个hidden_size^2级别的操作
            flops_per_token = flops_per_token_per_layer * num_layers
        else:
            # 默认值
            flops_per_token = 1e9  # 1GFLOPS每token
        
        # 总计算量
        total_flops = batch_size * seq_length * flops_per_token
        
        # 转换为GFLOPS
        return total_flops / 1e9
    
    def _greedy_allocation(self, required_memory: float, required_compute: float) -> List[CPU]:
        """贪婪分配策略，尽可能使用少量CPU"""
        allocated = []
        remaining_memory = required_memory
        remaining_compute = required_compute
        
        # 按内存容量降序排序可用CPU
        sorted_cpus = sorted(self.available_cpus, 
                            key=lambda cpu: cpu.get_ram_capacity(), 
                            reverse=True)
        
        for cpu in sorted_cpus:
            if remaining_memory <= 0 and remaining_compute <= 0:
                break
                
            # 检查CPU是否有足够内存和计算能力
            cpu_memory = cpu.get_ram_capacity()
            cpu_compute = cpu.get_compute_capability("fp32")
            
            if cpu_memory > 0 and cpu_compute > 0:
                allocated.append(cpu)
                remaining_memory -= cpu_memory
                remaining_compute -= cpu_compute
        
        # 如果所有可用CPU都不足以满足需求
        if remaining_memory > 0 or remaining_compute > 0:
            return []
        
        return allocated
    
    def _even_allocation(self, required_memory: float) -> List[CPU]:
        """均匀分配策略，尽可能均衡负载"""
        # 计算每个CPU平均需要的内存
        num_cpus = len(self.available_cpus)
        if num_cpus == 0:
            return []
            
        memory_per_cpu = required_memory / num_cpus
        
        # 检查每个CPU是否有足够内存
        for cpu in self.available_cpus:
            if cpu.get_ram_capacity() < memory_per_cpu:
                # 如果任一CPU内存不足，回退到贪婪分配
                return self._greedy_allocation(required_memory, 0)
        
        # 所有CPU都有足够内存，返回全部
        return self.available_cpus[:num_cpus]
    
    def _performance_allocation(self, required_compute: float) -> List[CPU]:
        """性能优先分配策略，尽可能使用高性能CPU"""
        allocated = []
        remaining_compute = required_compute
        
        # 按计算能力降序排序可用CPU
        sorted_cpus = sorted(self.available_cpus, 
                           key=lambda cpu: cpu.get_compute_capability("fp32"), 
                           reverse=True)
        
        for cpu in sorted_cpus:
            if remaining_compute <= 0:
                break
                
            # 检查CPU的计算能力
            cpu_compute = cpu.get_compute_capability("fp32")
            if cpu_compute > 0:
                allocated.append(cpu)
                remaining_compute -= cpu_compute
        
        # 如果所有可用CPU都不足以满足需求
        if remaining_compute > 0:
            return []
        
        return allocated
    
    def release_resources(self, task_id: str):
        """释放指定任务占用的资源"""
        if task_id in self.allocated_cpus:
            released_cpus = self.allocated_cpus.pop(task_id)
            self.available_cpus.extend(released_cpus)
            
            # 重新排序可用CPU
            self.available_cpus.sort(key=lambda cpu: cpu.get_compute_capability("fp32"), reverse=True) 