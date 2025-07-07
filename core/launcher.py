#!/usr/bin/env python3
"""
任务发射器 - 负责管理任务的发射时机和策略
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from .enums import TaskPriority, ResourceType
from .task import NNTask
from .resource_queue import ResourceQueueManager
from .schedule_tracer import ScheduleTracer
from .models import SubSegment


@dataclass
class TaskLaunchConfig:
    """任务发射配置"""
    task_id: str
    priority: TaskPriority
    fps_requirement: float
    dependencies: List[str] = field(default_factory=list)
    min_interval: float = field(init=False)  # 自动计算，不需要初始化时提供
    
    def __post_init__(self):
        # 根据FPS计算最小间隔
        if self.fps_requirement > 0:
            self.min_interval = 1000.0 / self.fps_requirement  # ms
        else:
            self.min_interval = float('inf')


@dataclass
class LaunchEvent:
    """发射事件"""
    time: float
    task_id: str
    instance_id: int  # 任务实例编号
    
    def __lt__(self, other):
        return self.time < other.time


@dataclass
class LaunchPlan:
    """发射计划"""
    events: List[LaunchEvent] = field(default_factory=list)
    task_schedules: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    def add_launch(self, task_id: str, time: float, instance_id: int):
        """添加发射事件"""
        event = LaunchEvent(time, task_id, instance_id)
        self.events.append(event)
        self.task_schedules[task_id].append(time)
        
    def sort_events(self):
        """按时间排序事件"""
        self.events.sort(key=lambda e: e.time)


class TaskLauncher:
    """任务发射器"""
    
    def __init__(self, queue_manager: ResourceQueueManager, tracer: Optional[ScheduleTracer] = None):
        self.queue_manager = queue_manager
        self.tracer = tracer
        
        # 任务配置
        self.task_configs: Dict[str, TaskLaunchConfig] = {}
        self.tasks: Dict[str, NNTask] = {}
        
        # 运行时状态
        self.task_last_launch: Dict[str, float] = {}  # 上次发射时间
        self.task_instance_count: Dict[str, int] = defaultdict(int)  # 实例计数
        self.pending_launches: List[LaunchEvent] = []  # 待发射队列（最小堆）
        
        # 依赖管理
        self.dependency_graph: Dict[str, Set[str]] = defaultdict(set)  # task -> dependents
        self.task_completions: Dict[Tuple[str, int], float] = {}  # (task_id, instance) -> completion_time
        
    def register_task(self, task: NNTask):
        """注册任务"""
        config = TaskLaunchConfig(
            task_id=task.task_id,
            priority=task.priority,
            fps_requirement=task.fps_requirement,
            dependencies=task.dependencies
        )
        
        self.task_configs[task.task_id] = config
        self.tasks[task.task_id] = task
        
        # 构建依赖图
        for dep_id in task.dependencies:
            self.dependency_graph[dep_id].add(task.task_id)
            
    def create_launch_plan(self, time_window: float, strategy: str = "eager") -> LaunchPlan:
        """创建发射计划
        
        Args:
            time_window: 时间窗口(ms)
            strategy: 发射策略 - "eager"(尽早), "lazy"(延迟), "balanced"(均衡)
            
        Returns:
            发射计划
        """
        plan = LaunchPlan()
        
        if strategy == "eager":
            plan = self._create_eager_plan(time_window)
        elif strategy == "lazy":
            plan = self._create_lazy_plan(time_window)
        else:  # balanced
            plan = self._create_balanced_plan(time_window)
            
        plan.sort_events()
        return plan
        
    def _create_eager_plan(self, time_window: float) -> LaunchPlan:
        """创建急切发射计划 - 尽早发射所有任务"""
        plan = LaunchPlan()
        
        for task_id, config in self.task_configs.items():
            if config.min_interval >= time_window:
                # 只发射一次
                if self._can_launch_at(task_id, 0.0, 0):
                    plan.add_launch(task_id, 0.0, 0)
            else:
                # 周期性发射
                time = 0.0
                instance = 0
                while time < time_window:
                    if self._can_launch_at(task_id, time, instance):
                        plan.add_launch(task_id, time, instance)
                        instance += 1
                    time += config.min_interval
                    
        return plan
        
    def _create_lazy_plan(self, time_window: float) -> LaunchPlan:
        """创建延迟发射计划 - 尽可能晚地发射"""
        plan = LaunchPlan()
        
        for task_id, config in self.task_configs.items():
            task = self.tasks[task_id]
            
            # 估算任务执行时间
            estimated_duration = self._estimate_task_duration(task)
            
            if config.min_interval >= time_window:
                # 只发射一次，尽量晚
                launch_time = max(0, time_window - estimated_duration - 10)  # 留10ms余量
                if self._can_launch_at(task_id, launch_time, 0):
                    plan.add_launch(task_id, launch_time, 0)
            else:
                # 周期性发射，从后往前规划
                instance = int(time_window / config.min_interval)
                time = instance * config.min_interval
                
                while instance >= 0 and time >= 0:
                    if time + estimated_duration <= time_window:
                        if self._can_launch_at(task_id, time, instance):
                            plan.add_launch(task_id, time, instance)
                    instance -= 1
                    time -= config.min_interval
                    
        return plan
        
    def _create_balanced_plan(self, time_window: float) -> LaunchPlan:
        """创建均衡发射计划 - 平衡负载"""
        plan = LaunchPlan()
        
        # 按优先级分组
        priority_groups = defaultdict(list)
        for task_id, config in self.task_configs.items():
            priority_groups[config.priority].append(task_id)
            
        # 为每个优先级组创建交错的发射时间
        for priority in TaskPriority:
            tasks = priority_groups[priority]
            if not tasks:
                continue
                
            # 计算组内偏移
            offset_step = 5.0  # 5ms偏移
            
            for i, task_id in enumerate(tasks):
                config = self.task_configs[task_id]
                base_offset = i * offset_step
                
                if config.min_interval >= time_window:
                    # 单次发射
                    launch_time = base_offset
                    if self._can_launch_at(task_id, launch_time, 0):
                        plan.add_launch(task_id, launch_time, 0)
                else:
                    # 周期性发射
                    time = base_offset
                    instance = 0
                    while time < time_window:
                        if self._can_launch_at(task_id, time, instance):
                            plan.add_launch(task_id, time, instance)
                            instance += 1
                        time += config.min_interval
                        
        return plan
        
    def execute_plan(self, plan: LaunchPlan, max_time: float):
        """执行发射计划
        
        Args:
            plan: 发射计划
            max_time: 最大执行时间
        """
        current_time = 0.0
        event_index = 0
        
        while current_time < max_time and event_index < len(plan.events):
            # 处理当前时间的所有发射事件
            while event_index < len(plan.events) and plan.events[event_index].time <= current_time:
                event = plan.events[event_index]
                self._launch_task(event.task_id, event.instance_id, current_time)
                event_index += 1
                
            # 时间推进
            current_time += 0.1  # 0.1ms步进
            
    def _launch_task(self, task_id: str, instance_id: int, current_time: float):
        """发射任务实例"""
        task = self.tasks[task_id]
        config = self.task_configs[task_id]
        
        # 检查依赖
        if not self._check_dependencies(task_id, instance_id):
            # 依赖未满足，延迟发射
            heapq.heappush(self.pending_launches, 
                          LaunchEvent(current_time + 1.0, task_id, instance_id))
            return
            
        # 为任务段分配资源队列
        resource_assignments = {}
        for segment in task.segments:
            queue = self.queue_manager.find_best_queue(segment.resource_type)
            if queue:
                resource_assignments[segment.resource_type] = queue.resource_id
                
        # 获取任务的子段（如果有分段的话）
        sub_segments = task.apply_segmentation()
        if not sub_segments:
            # 如果没有分段，使用原始段
            sub_segments = []
            for seg in task.segments:
                # 将 ResourceSegment 转换为 SubSegment
                sub_seg = SubSegment(
                    sub_id=seg.segment_id,
                    resource_type=seg.resource_type,
                    duration_table=seg.duration_table,
                    cut_overhead=0.0,
                    original_segment_id=seg.segment_id
                )
                sub_segments.append(sub_seg)
        
        # 将任务子段加入对应的资源队列
        for sub_seg in sub_segments:
            if sub_seg.resource_type in resource_assignments:
                queue_id = resource_assignments[sub_seg.resource_type]
                queue = self.queue_manager.get_queue(queue_id)
                if queue:
                    queue.enqueue(
                        f"{task_id}#{instance_id}",  # 带实例号的任务ID
                        config.priority,
                        current_time,
                        [sub_seg]  # 传入子段列表
                    )
                    
        # 记录发射
        self.task_last_launch[task_id] = current_time
        self.task_instance_count[task_id] = instance_id + 1
        
        if self.tracer:
            # 记录到追踪器
            for res_id in resource_assignments.values():
                self.tracer.record_enqueue(
                    f"{task_id}#{instance_id}",
                    res_id,
                    config.priority,
                    current_time,
                    task.segments
                )
                
    def _can_launch_at(self, task_id: str, time: float, instance_id: int) -> bool:
        """检查是否可以在指定时间发射"""
        config = self.task_configs[task_id]
        
        # 检查最小间隔
        if task_id in self.task_last_launch:
            if time - self.task_last_launch[task_id] < config.min_interval:
                return False
                
        # 检查依赖（假设依赖的前一个实例）
        if config.dependencies and instance_id > 0:
            for dep_id in config.dependencies:
                dep_key = (dep_id, instance_id - 1)
                if dep_key not in self.task_completions:
                    return False
                if self.task_completions[dep_key] > time:
                    return False
                    
        return True
        
    def _check_dependencies(self, task_id: str, instance_id: int) -> bool:
        """检查任务依赖是否满足"""
        config = self.task_configs[task_id]
        
        for dep_id in config.dependencies:
            # 检查相同实例号的依赖任务是否完成
            dep_key = (dep_id, instance_id)
            if dep_key not in self.task_completions:
                return False
                
        return True
        
    def _estimate_task_duration(self, task: NNTask) -> float:
        """估算任务执行时间"""
        # 使用默认带宽估算
        bandwidth_map = {
            ResourceType.NPU: 60.0,
            ResourceType.DSP: 40.0
        }
        return task.estimate_duration(bandwidth_map)
        
    def notify_task_completion(self, task_id: str, instance_id: int, completion_time: float):
        """通知任务完成"""
        key = (task_id.split('#')[0], instance_id)  # 去掉实例号后缀
        self.task_completions[key] = completion_time
        
        # 检查是否有等待该任务的发射
        new_pending = []
        while self.pending_launches:
            event = heapq.heappop(self.pending_launches)
            if self._check_dependencies(event.task_id, event.instance_id):
                # 依赖满足，立即发射
                self._launch_task(event.task_id, event.instance_id, completion_time)
            else:
                new_pending.append(event)
                
        # 重建待发射队列
        self.pending_launches = new_pending
        heapq.heapify(self.pending_launches)
