from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple

class Workload(ABC):
    """负载基类，定义LLM服务的负载模式"""
    
    def __init__(self, name: str, params: Dict[str, Any]):
        """
        初始化负载
        
        Args:
            name: 负载名称
            params: 负载参数
        """
        self.name = name
        self.params = params
    
    @abstractmethod
    def generate_requests(self) -> List[Dict[str, Any]]:
        """
        生成请求列表
        
        Returns:
            请求列表，每个请求包含如批大小、输入长度、输出长度等信息
        """
        pass
    
    def __str__(self) -> str:
        return f"{self.name} Workload with params: {self.params}" 