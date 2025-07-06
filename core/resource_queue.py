#!/usr/bin/env python3
"""
简化的资源队列实现
用于新架构中的FIFO调度
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from core.enums import TaskPriority, ResourceType


@dataclass
class QueuedTask:
    """队列中的任务"""
    task_id: str
    priority: TaskPriority
    ready_time: float
    enqueue_time: float  # 入队时间，用于FIFO排序
    estimated_duration: float
    
    def __lt__(self, other):
        """比较函数：优先级高的先执行，同优先级按入队时间"""
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.enqueue_time < other.enqueue_time


@dataclass
class ResourceQueue:
    """单个资源的任务队列"""
    resource_id: str
    resource_type: ResourceType
    bandwidth: float
    
    # 按优先级组织的队列
    priority_queues: Dict[TaskPriority, deque] = field(default_factory=dict)
    
    # 资源状态
    current_time: float = 0.0
    busy_until: float = 0.0
    current_task: Optional[str] = None
    
    # 统计信息
    total_tasks_executed: int = 0
    total_busy_time: float = 0.0
    
    def __post_init__(self):
        """初始化优先级队列"""
        if not self.priority_queues:
            self.priority_queues = {
                priority: deque() for priority in TaskPriority
            }
    
    def enqueue(self, task_id: str, priority: TaskPriority, 
                ready_time: float, estimated_duration: float) -> bool:
        """将任务加入队列"""
        queued_task = QueuedTask(
            task_id=task_id,
            priority=priority,
            ready_time=ready_time,
            enqueue_time=self.current_time,  # 记录入队时间
            estimated_duration=estimated_duration
        )
        
        self.priority_queues[priority].append(queued_task)
        return True
    
    def get_next_task(self) -> Optional[QueuedTask]:
        """获取下一个可执行的任务（不移除）"""
        # 从高优先级到低优先级检查
        for priority in TaskPriority:
            queue = self.priority_queues[priority]
            
            # 找到第一个就绪的任务
            for task in queue:
                if task.ready_time <= self.current_time:
                    return task
        
        return None
    
    def dequeue_task(self, task_id: str, priority: TaskPriority) -> Optional[QueuedTask]:
        """从队列中移除指定任务"""
        queue = self.priority_queues[priority]
        
        for i, task in enumerate(queue):
            if task.task_id == task_id:
                # 使用 del 而不是 remove 以保持 deque 的效率
                del queue[i]
                return task
        
        return None
    
    def execute_task(self, task: QueuedTask, start_time: float) -> float:
        """执行任务并更新资源状态"""
        self.current_task = task.task_id
        self.busy_until = start_time + task.estimated_duration
        self.total_tasks_executed += 1
        self.total_busy_time += task.estimated_duration
        
        return self.busy_until
    
    def advance_time(self, new_time: float):
        """推进时间"""
        if new_time > self.current_time:
            self.current_time = new_time
            
            # 如果当前任务已完成，清除
            if self.busy_until <= new_time:
                self.current_task = None
    
    def is_busy(self) -> bool:
        """检查资源是否忙碌"""
        return self.current_time < self.busy_until
    
    def get_utilization(self, time_window: float) -> float:
        """计算资源利用率"""
        if time_window <= 0:
            return 0.0
        return min(self.total_busy_time / time_window * 100, 100.0)
    
    def get_queue_length(self) -> Dict[TaskPriority, int]:
        """获取各优先级队列长度"""
        return {
            priority: len(queue) 
            for priority, queue in self.priority_queues.items()
        }
    
    def get_next_available_time(self) -> float:
        """获取资源下次可用时间"""
        return max(self.busy_until, self.current_time)
    
    def clear(self):
        """清空队列和重置状态"""
        for queue in self.priority_queues.values():
            queue.clear()
        
        self.current_time = 0.0
        self.busy_until = 0.0
        self.current_task = None
        self.total_tasks_executed = 0
        self.total_busy_time = 0.0


class ResourceQueueManager:
    """资源队列管理器"""
    
    def __init__(self):
        self.resource_queues: Dict[str, ResourceQueue] = {}
    
    def add_resource(self, resource_id: str, resource_type: ResourceType, 
                     bandwidth: float) -> ResourceQueue:
        """添加资源"""
        queue = ResourceQueue(
            resource_id=resource_id,
            resource_type=resource_type,
            bandwidth=bandwidth
        )
        self.resource_queues[resource_id] = queue
        return queue
    
    def get_queue(self, resource_id: str) -> Optional[ResourceQueue]:
        """获取指定资源的队列"""
        return self.resource_queues.get(resource_id)
    
    def get_queues_by_type(self, resource_type: ResourceType) -> List[ResourceQueue]:
        """获取指定类型的所有资源队列"""
        return [
            queue for queue in self.resource_queues.values()
            if queue.resource_type == resource_type
        ]
    
    def find_best_queue(self, resource_type: ResourceType) -> Optional[ResourceQueue]:
        """找到指定类型中最空闲的资源"""
        queues = self.get_queues_by_type(resource_type)
        if not queues:
            return None
        
        # 选择最早可用的资源
        return min(queues, key=lambda q: q.get_next_available_time())
    
    def advance_all_queues(self, new_time: float):
        """推进所有队列的时间"""
        for queue in self.resource_queues.values():
            queue.advance_time(new_time)
    
    def get_global_stats(self) -> Dict:
        """获取全局统计信息"""
        total_executed = sum(q.total_tasks_executed for q in self.resource_queues.values())
        
        utilization_by_type = {}
        for res_type in ResourceType:
            queues = self.get_queues_by_type(res_type)
            if queues:
                total_busy = sum(q.total_busy_time for q in queues)
                total_capacity = sum(q.current_time for q in queues)
                utilization_by_type[res_type.value] = (
                    total_busy / total_capacity * 100 if total_capacity > 0 else 0
                )
        
        return {
            'total_tasks_executed': total_executed,
            'resource_utilization': utilization_by_type,
            'queue_lengths': {
                queue.resource_id: queue.get_queue_length()
                for queue in self.resource_queues.values()
            }
        }
    
    def reset_all(self):
        """重置所有队列"""
        for queue in self.resource_queues.values():
            queue.clear()
