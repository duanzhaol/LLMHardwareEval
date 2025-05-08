from typing import Dict, Any, List
from cluster.base import Cluster
from hardware.base import Hardware
from hardware.gpu import GPU

class GPUCluster(Cluster):
    """GPU集群类，管理GPU硬件资源"""
    
    def __init__(self, name: str, gpu_list: List[GPU], 
                network_bandwidth: float, 
                specs: Dict[str, Any] = None,
                allocation_strategy: str = "greedy"):
        """
        初始化GPU集群
        
        Args:
            name: 集群名称
            gpu_list: GPU列表
            network_bandwidth: 集群内网络带宽（GB/s）
            specs: 其他集群规格
            allocation_strategy: 资源分配策略，可选：
                - "greedy": 贪婪分配，尽可能少用GPU
                - "even": 均匀分配，尽可能均衡负载
                - "performance": 性能优先，尽可能使用高性能GPU
        """
        super().__init__(name, gpu_list, network_bandwidth, specs)
        self.allocation_strategy = allocation_strategy
        
        # 资源状态跟踪
        self.available_gpus = list(gpu_list)  # 可用GPU列表
        self.allocated_gpus = {}  # 已分配GPU字典 {task_id: [gpu1, gpu2, ...]}
        
        # 为了方便资源分配，按FLOPS排序GPU
        self.available_gpus.sort(key=lambda gpu: gpu.get_compute_capability("fp16"), reverse=True)
    
    def allocate_resources(self, model, batch_size: int, 
                         seq_length: int) -> List[Hardware]:
        """
        为给定模型和请求参数分配GPU资源
        
        Args:
            model: 模型对象
            batch_size: 批处理大小
            seq_length: 序列长度
            
        Returns:
            分配的GPU列表
        """
        # 估算模型所需的计算能力和内存
        required_memory = self._estimate_memory_requirement(model, batch_size, seq_length)
        
        # 根据不同的分配策略选择合适的GPU
        if self.allocation_strategy == "greedy":
            return self._greedy_allocation(required_memory)
        elif self.allocation_strategy == "even":
            return self._even_allocation(required_memory)
        elif self.allocation_strategy == "performance":
            return self._performance_allocation(required_memory)
        else:
            raise ValueError(f"不支持的分配策略: {self.allocation_strategy}")
    
    def _estimate_memory_requirement(self, model, batch_size: int, seq_length: int) -> float:
        """估算模型运行所需的显存（GB）"""
        # 这里简化估算，实际应考虑模型参数大小、激活值大小、KV缓存等
        
        # 模型参数大小（假设FP16格式）
        param_bytes = 0
        if hasattr(model, "config") and "hidden_size" in model.config:
            hidden_size = model.config["hidden_size"]
            num_layers = model.config.get("num_layers", 12)
            vocab_size = model.config.get("vocab_size", 50000)
            
            # 简化的参数量计算（实际更复杂）
            # 每个Transformer层参数量：12 * hidden_size^2
            # 词嵌入参数量：vocab_size * hidden_size
            param_count = num_layers * 12 * hidden_size * hidden_size + vocab_size * hidden_size
            param_bytes = param_count * 2  # FP16每个参数2字节
        else:
            # 默认值，如果无法从模型中获取
            param_bytes = 2e9  # 假设2GB参数
        
        # 激活值和KV缓存大小（与batch_size和seq_length相关）
        activation_bytes = batch_size * seq_length * 2 * 2e6  # 假设每token激活值2MB
        
        # 总显存需求（字节转GB）
        total_memory_gb = (param_bytes + activation_bytes) / 1e9
        
        # 增加缓冲区
        return total_memory_gb * 1.2  # 增加20%缓冲
    
    def _greedy_allocation(self, required_memory: float) -> List[GPU]:
        """贪婪分配策略，尽可能使用少量GPU"""
        allocated = []
        remaining_memory = required_memory
        
        # 按显存容量降序排序可用GPU
        sorted_gpus = sorted(self.available_gpus, 
                            key=lambda gpu: gpu.get_vram_capacity(), 
                            reverse=True)
        
        for gpu in sorted_gpus:
            if remaining_memory <= 0:
                break
                
            # 检查GPU是否有足够显存
            gpu_memory = gpu.get_vram_capacity()
            if gpu_memory > 0:
                allocated.append(gpu)
                remaining_memory -= gpu_memory
        
        # 如果所有可用GPU都不足以满足需求
        if remaining_memory > 0:
            # 可以选择抛出异常或返回空列表
            return []
        
        return allocated
    
    def _even_allocation(self, required_memory: float) -> List[GPU]:
        """均匀分配策略，尽可能均衡负载"""
        # 计算每个GPU平均需要的显存
        num_gpus = len(self.available_gpus)
        if num_gpus == 0:
            return []
            
        memory_per_gpu = required_memory / num_gpus
        
        # 检查每个GPU是否有足够显存
        for gpu in self.available_gpus:
            if gpu.get_vram_capacity() < memory_per_gpu:
                # 如果任一GPU显存不足，回退到贪婪分配
                return self._greedy_allocation(required_memory)
        
        # 所有GPU都有足够显存，返回全部
        return self.available_gpus[:num_gpus]
    
    def _performance_allocation(self, required_memory: float) -> List[GPU]:
        """性能优先分配策略，尽可能使用高性能GPU"""
        allocated = []
        remaining_memory = required_memory
        
        # 按计算能力降序排序可用GPU
        sorted_gpus = sorted(self.available_gpus, 
                            key=lambda gpu: gpu.get_compute_capability("fp16"), 
                            reverse=True)
        
        for gpu in sorted_gpus:
            if remaining_memory <= 0:
                break
                
            # 检查GPU是否有足够显存
            gpu_memory = gpu.get_vram_capacity()
            if gpu_memory > 0:
                allocated.append(gpu)
                remaining_memory -= gpu_memory
        
        # 如果所有可用GPU都不足以满足需求
        if remaining_memory > 0:
            return []
        
        return allocated
    
    def release_resources(self, task_id: str):
        """释放指定任务占用的资源"""
        if task_id in self.allocated_gpus:
            released_gpus = self.allocated_gpus.pop(task_id)
            self.available_gpus.extend(released_gpus)
            
            # 重新排序可用GPU
            self.available_gpus.sort(key=lambda gpu: gpu.get_compute_capability("fp16"), reverse=True) 