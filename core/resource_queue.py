#!/usr/bin/env python3
"""
简化的资源队列实现
用于新架构中的FIFO调度
支持固定带宽和动态带宽两种模式
"""

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
from core.enums import TaskPriority, ResourceType
from core.models import SubSegment


@dataclass
class QueuedTask:
    """队列中的任务"""
    task_id: str
    priority: TaskPriority
    ready_time: float
    enqueue_time: float  # 入队时间，用于FIFO排序
    
    # 新增：子段信息
    sub_segments: List[SubSegment] = field(default_factory=list)
    current_segment_index: int = 0  # 当前要执行的子段索引
    
    def __lt__(self, other):
        """比较函数：优先级高的先执行，同优先级按入队时间"""
        if self.priority != other.priority:
            return self.priority.value < other.priority.value
        return self.enqueue_time < other.enqueue_time
    
    def get_current_segment(self) -> Optional[SubSegment]:
        """获取当前要执行的子段"""
        if 0 <= self.current_segment_index < len(self.sub_segments):
            return self.sub_segments[self.current_segment_index]
        return None
    
    def has_remaining_segments(self) -> bool:
        """检查是否还有剩余的子段"""
        return self.current_segment_index < len(self.sub_segments)


@dataclass
class ResourceQueue:
    """单个资源的任务队列"""
    resource_id: str
    resource_type: ResourceType
    bandwidth: float  # 固定带宽模式下使用
    
    # 按优先级组织的队列
    priority_queues: Dict[TaskPriority, deque] = field(default_factory=dict)
    
    # 资源状态
    current_time: float = 0.0
    busy_until: float = 0.0
    current_task: Optional[str] = None
    
    # 统计信息
    total_tasks_executed: int = 0
    total_busy_time: float = 0.0
    
    # 动态带宽支持
    bandwidth_manager: Optional['BandwidthManager'] = None  # 可选的带宽管理器
    last_used_bandwidth: float = 0.0  # 最近使用的实际带宽
    
    def __post_init__(self):
        """初始化优先级队列"""
        if not self.priority_queues:
            self.priority_queues = {
                priority: deque() for priority in TaskPriority
            }
    
    def enqueue(self, task_id: str, priority: TaskPriority, 
                ready_time: float, sub_segments: List[SubSegment]) -> bool:
        """将任务加入队列"""
        queued_task = QueuedTask(
            task_id=task_id,
            priority=priority,
            ready_time=ready_time,
            enqueue_time=self.current_time,  # 记录入队时间
            sub_segments=sub_segments
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
    
    def execute_task(self, task: QueuedTask, start_time: float, bandwidth: Optional[float] = None) -> float:
        """执行任务的当前子段并更新资源状态
        
        Args:
            task: 要执行的任务
            start_time: 开始时间
            bandwidth: 使用的带宽（如果为None，则使用动态带宽或默认带宽）
            
        Returns:
            任务结束时间
        """
        current_segment = task.get_current_segment()
        if not current_segment:
            return start_time
        
        # 确定使用的带宽
        if bandwidth is None:
            if self.bandwidth_manager:
                # 使用动态带宽
                return self._execute_with_dynamic_bandwidth(task, start_time, current_segment)
            else:
                # 使用默认固定带宽
                bandwidth = self.bandwidth
        
        # 计算子段执行时间
        duration = current_segment.get_duration(bandwidth)
        
        self.current_task = f"{task.task_id}_{current_segment.sub_id}"
        self.busy_until = start_time + duration
        self.total_tasks_executed += 1
        self.total_busy_time += duration
        self.last_used_bandwidth = bandwidth
        
        # 移动到下一个子段
        task.current_segment_index += 1
        
        return self.busy_until
    
    def _execute_with_dynamic_bandwidth(self, task: QueuedTask, start_time: float, 
                                      segment: SubSegment) -> float:
        """使用动态带宽执行任务"""
        if not self.bandwidth_manager:
            # 降级到固定带宽
            return self.execute_task(task, start_time, self.bandwidth)
        
        # 获取当前可用带宽
        current_bandwidth = self.bandwidth_manager.get_available_bandwidth(
            self.resource_type, start_time, exclude_resource=self.resource_id
        )
        
        # 计算执行时间
        duration = segment.get_duration(current_bandwidth)
        end_time = start_time + duration
        
        # 向带宽管理器注册这次使用
        actual_bandwidth = self.bandwidth_manager.allocate_bandwidth(
            self.resource_id,
            self.resource_type,
            f"{task.task_id}_{segment.sub_id}",
            start_time,
            end_time
        )
        
        # 如果实际分配的带宽不同，重新计算时间
        if abs(actual_bandwidth - current_bandwidth) > 0.1:
            duration = segment.get_duration(actual_bandwidth)
            end_time = start_time + duration
        
        # 更新状态
        self.current_task = f"{task.task_id}_{segment.sub_id}"
        self.busy_until = end_time
        self.total_tasks_executed += 1
        self.total_busy_time += duration
        self.last_used_bandwidth = actual_bandwidth
        
        # 移动到下一个子段
        task.current_segment_index += 1
        
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
    
    def __init__(self, bandwidth_manager: Optional['BandwidthManager'] = None):
        self.resource_queues: Dict[str, ResourceQueue] = {}
        self.bandwidth_manager = bandwidth_manager
    
    def add_resource(self, resource_id: str, resource_type: ResourceType, 
                     bandwidth: float = 0.0) -> ResourceQueue:
        """添加资源
        
        Args:
            resource_id: 资源ID
            resource_type: 资源类型
            bandwidth: 固定带宽值（如果使用动态带宽管理器，可以设为0）
        """
        queue = ResourceQueue(
            resource_id=resource_id,
            resource_type=resource_type,
            bandwidth=bandwidth,
            bandwidth_manager=self.bandwidth_manager
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
