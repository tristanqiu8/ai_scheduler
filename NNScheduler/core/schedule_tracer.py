#!/usr/bin/env python3
"""
调度追踪器 - 用于追踪和可视化整个系统的任务执行时间线
支持生成甘特图和Chrome Tracing格式
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import json
import time
from collections import defaultdict

from .enums import ResourceType, TaskPriority
from .resource_queue import ResourceQueueManager


class EventType(Enum):
    """事件类型"""
    TASK_ENQUEUE = "enqueue"
    TASK_START = "start"
    TASK_END = "end"
    TASK_PREEMPT = "preempt"
    TASK_RESUME = "resume"
    RESOURCE_IDLE = "idle"
    RESOURCE_BUSY = "busy"


@dataclass
class TraceEvent:
    """追踪事件"""
    timestamp: float
    event_type: EventType
    task_id: str
    resource_id: str
    resource_type: ResourceType
    priority: TaskPriority
    
    # 额外信息
    segment_id: Optional[str] = None
    bandwidth: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskExecution:
    """任务执行记录"""
    task_id: str
    resource_id: str
    resource_type: ResourceType
    priority: TaskPriority
    start_time: float
    end_time: float
    bandwidth: float
    segment_id: Optional[str] = None
    
    @property
    def duration(self) -> float:
        return self.end_time - self.start_time


class ScheduleTracer:
    """调度追踪器 - 记录和分析整个系统的执行时间线"""
    
    def __init__(self, queue_manager: ResourceQueueManager):
        self.queue_manager = queue_manager
        self.events: List[TraceEvent] = []
        self.executions: List[TaskExecution] = []
        
        # 任务相关信息
        self.task_info: Dict[str, Dict[str, Any]] = {}
        
        # 资源利用率追踪
        self.resource_busy_time: Dict[str, float] = defaultdict(float)
        self.resource_last_busy: Dict[str, float] = defaultdict(float)
        
        # 时间范围
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
    
    def record_enqueue(self, task_id: str, resource_id: str, priority: TaskPriority,
                      ready_time: float, segments: List[Any]):
        """记录任务入队事件"""
        event = TraceEvent(
            timestamp=ready_time,
            event_type=EventType.TASK_ENQUEUE,
            task_id=task_id,
            resource_id=resource_id,
            resource_type=self._get_resource_type(resource_id),
            priority=priority,
            metadata={"segment_count": len(segments)}
        )
        self.events.append(event)
        
        # 保存任务信息
        self.task_info[task_id] = {
            "priority": priority,
            "ready_time": ready_time,
            "segments": segments,
            "resource_id": resource_id
        }
    
    def record_execution(self, task_id: str, resource_id: str, 
                        start_time: float, end_time: float,
                        bandwidth: float, segment_id: Optional[str] = None):
        """记录任务执行"""
        # 确保任务信息存在
        if task_id not in self.task_info:
            # 如果任务信息不存在，创建一个默认的
            self.task_info[task_id] = {
                "priority": TaskPriority.NORMAL,
                "ready_time": start_time,
                "segments": [],
                "resource_id": resource_id
            }
        
        resource_type = self._get_resource_type(resource_id)
        priority = self.task_info[task_id].get("priority", TaskPriority.NORMAL)
        
        # 记录开始事件
        self.events.append(TraceEvent(
            timestamp=start_time,
            event_type=EventType.TASK_START,
            task_id=task_id,
            resource_id=resource_id,
            resource_type=resource_type,
            priority=priority,
            segment_id=segment_id,
            bandwidth=bandwidth,
            duration=end_time - start_time
        ))
        
        # 记录结束事件
        self.events.append(TraceEvent(
            timestamp=end_time,
            event_type=EventType.TASK_END,
            task_id=task_id,
            resource_id=resource_id,
            resource_type=resource_type,
            priority=priority,
            segment_id=segment_id
        ))
        
        # 记录执行
        execution = TaskExecution(
            task_id=task_id,
            resource_id=resource_id,
            resource_type=resource_type,
            priority=priority,
            start_time=start_time,
            end_time=end_time,
            bandwidth=bandwidth,
            segment_id=segment_id
        )
        self.executions.append(execution)
        
        # 更新资源利用率
        self.resource_busy_time[resource_id] += (end_time - start_time)
        self.resource_last_busy[resource_id] = end_time
        
        # 更新时间范围 - 确保值被正确设置
        if self.start_time is None:
            self.start_time = start_time
        else:
            self.start_time = min(self.start_time, start_time)
            
        if self.end_time is None:
            self.end_time = end_time
        else:
            self.end_time = max(self.end_time, end_time)
    
    def _get_resource_type(self, resource_id: str) -> ResourceType:
        """获取资源类型"""
        queue = self.queue_manager.get_queue(resource_id)
        return queue.resource_type if queue else ResourceType.NPU
    
    def get_timeline(self) -> Dict[str, List[TaskExecution]]:
        """获取按资源分组的执行时间线"""
        timeline = defaultdict(list)
        for execution in self.executions:
            timeline[execution.resource_id].append(execution)
        
        # 按时间排序
        for resource_id in timeline:
            timeline[resource_id].sort(key=lambda x: x.start_time)
        
        return dict(timeline)
    
    def get_task_timeline(self, task_id: str) -> List[TaskExecution]:
        """获取特定任务的执行时间线"""
        return [e for e in self.executions if e.task_id == task_id]
    
    def get_resource_utilization(self, time_window: float = None) -> Dict[str, float]:
        """
        计算资源利用率
        
        Args:
            time_window: 时间窗口，如果为None则使用实际执行时间跨度
            
        Returns:
            资源利用率字典
        """
        if not self.executions and not self.queue_manager:
            return {}
        
        # 确定时间基准
        if time_window is None:
            # 使用实际执行时间跨度
            if self.executions:
                actual_start = min(e.start_time for e in self.executions)
                actual_end = max(e.end_time for e in self.executions)
                total_time = actual_end - actual_start
            else:
                total_time = 0
        else:
            # 使用指定的时间窗口
            total_time = time_window
        
        if total_time <= 0:
            return {}
        
        utilization = {}
        # 包括所有资源（即使没有执行任务的）
        all_resources = set()
        if self.queue_manager:
            all_resources.update(self.queue_manager.resource_queues.keys())
        all_resources.update(self.resource_busy_time.keys())
        
        for resource_id in all_resources:
            busy_time = self.resource_busy_time.get(resource_id, 0.0)
            utilization[resource_id] = (busy_time / total_time) * 100
        
        return utilization
    
    def get_statistics(self, time_window: float = None) -> Dict[str, Any]:
        """
        获取调度统计信息
        
        Args:
            time_window: 时间窗口，用于计算资源利用率
        """
        # 计算实际时间跨度
        if self.executions:
            actual_start = min(e.start_time for e in self.executions)
            actual_end = max(e.end_time for e in self.executions)
            time_span = actual_end - actual_start
        else:
            time_span = 0
            
        stats = {
            "total_tasks": len(self.task_info),
            "total_executions": len(self.executions),
            "time_span": time_span,
            "resource_utilization": self.get_resource_utilization(time_window),
            "tasks_by_priority": defaultdict(int),
            "average_wait_time": 0,
            "average_execution_time": 0
        }
        
        # 按优先级统计任务数
        for task_info in self.task_info.values():
            stats["tasks_by_priority"][task_info["priority"].name] += 1
        
        # 计算平均等待时间和执行时间
        wait_times = []
        exec_times = []
        
        for execution in self.executions:
            exec_times.append(execution.duration)
            
            # 等待时间 = 开始时间 - 就绪时间
            task_info = self.task_info.get(execution.task_id)
            if task_info:
                wait_time = execution.start_time - task_info["ready_time"]
                wait_times.append(wait_time)
        
        if wait_times:
            stats["average_wait_time"] = sum(wait_times) / len(wait_times)
        if exec_times:
            stats["average_execution_time"] = sum(exec_times) / len(exec_times)
        
        return stats
