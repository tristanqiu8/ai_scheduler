#!/usr/bin/env python3
"""
修复后的调度执行器 - 正确实现段级模式
段级模式的核心思想：
1. 同一任务的段必须按顺序执行（保持依赖关系）
2. 不同任务的段可以交错执行（提高资源利用率）
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from core.models import SubSegment
from core.enums import ResourceType, TaskPriority
from core.resource_queue import ResourceQueueManager, QueuedTask
from core.schedule_tracer import ScheduleTracer
from core.launcher import LaunchPlan, LaunchEvent
from core.task import NNTask


@dataclass
class TaskInstance:
    """任务实例的执行状态"""
    task_id: str
    instance_id: int
    segments: List[SubSegment]
    current_segment_index: int = 0
    completed_segments: List[int] = field(default_factory=list)
    priority: TaskPriority = TaskPriority.NORMAL
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    # 新增：记录每个段的状态
    segment_enqueued: List[bool] = field(default_factory=list)
    
    def __post_init__(self):
        # 初始化段状态
        self.segment_enqueued = [False] * len(self.segments)
    
    @property
    def instance_key(self) -> Tuple[str, int]:
        return (self.task_id, self.instance_id)
    
    @property
    def is_completed(self) -> bool:
        return self.current_segment_index >= len(self.segments)
    
    def get_current_segment(self) -> Optional[SubSegment]:
        """获取当前待执行的段"""
        if not self.is_completed:
            return self.segments[self.current_segment_index]
        return None
    
    def can_enqueue_segment(self, segment_index: int) -> bool:
        """检查指定段是否可以入队"""
        if segment_index >= len(self.segments):
            return False
        
        # 已经入队的不能重复入队
        if self.segment_enqueued[segment_index]:
            return False
        
        # 第一个段总是可以入队
        if segment_index == 0:
            return True
        
        # 后续段需要前一个段完成
        return (segment_index - 1) in self.completed_segments


@dataclass
class SegmentCompletion:
    """段完成信息"""
    task_id: str
    instance_id: int
    segment_index: int
    completion_time: float
    resource_id: str
    
    def __lt__(self, other):
        """比较方法，用于堆排序"""
        return self.completion_time < other.completion_time


class ScheduleExecutor:
    """调度执行器
    
    支持两种模式：
    1. 传统模式：任务按段顺序执行，同一时刻只有当前段在执行
    2. 段级模式：任务的段可以提前入队，但仍保持顺序依赖
    """
    
    def __init__(self, queue_manager: ResourceQueueManager, 
                 tracer: ScheduleTracer,
                 tasks: Dict[str, NNTask]):
        self.queue_manager = queue_manager
        self.tracer = tracer
        self.tasks = tasks
        
        # 执行状态
        self.task_instances: Dict[Tuple[str, int], TaskInstance] = {}
        self.segment_completions: List[SegmentCompletion] = []
        
        # 时间管理
        self.current_time = 0.0
        self.next_events: List[Tuple[float, str, any]] = []
        
        # 配置选项
        self.segment_mode = False  # 默认为传统模式
        
    def execute_plan(self, launch_plan: LaunchPlan, max_time: float, 
                    segment_mode: Optional[bool] = None) -> Dict:
        """执行发射计划"""
        # 如果指定了segment_mode，使用指定值；否则使用实例属性
        use_segment_mode = segment_mode if segment_mode is not None else self.segment_mode
        
        print(f"\n{'='*80}")
        print(f"开始执行调度 (max_time={max_time}ms, mode={'段级' if use_segment_mode else '传统'})")
        print(f"{'='*80}\n")
        
        self._reset_state()
        
        # 将发射事件加入队列
        for event in launch_plan.events:
            if event.time <= max_time:
                heapq.heappush(self.next_events, (event.time, "launch", event))
        
        # 主执行循环
        while self.current_time < max_time and (self.next_events or self._has_active_tasks()):
            # 处理事件
            self._process_events(use_segment_mode)
            
            # 尝试调度任务段
            if use_segment_mode:
                self._schedule_segments_smart()
            else:
                self._schedule_segments_traditional()
            
            # 执行队列中的就绪任务
            self._execute_ready_tasks()
            
            # 推进时间
            self._advance_time(max_time)
        
        return self._get_execution_stats()
    
    def _reset_state(self):
        """重置执行器状态"""
        self.task_instances.clear()
        self.segment_completions.clear()
        self.current_time = 0.0
        self.next_events.clear()
        
        # 重置资源队列
        for queue in self.queue_manager.resource_queues.values():
            queue.clear()
    
    def _process_events(self, use_segment_mode: bool):
        """处理当前时间的所有事件"""
        while self.next_events and self.next_events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.next_events)
            
            if event_type == "launch":
                self._handle_launch_event(event_data)
            elif event_type == "completion":
                self._handle_completion_event(event_data)
    
    def _handle_launch_event(self, event: LaunchEvent):
        """处理任务发射事件"""
        if event.task_id not in self.tasks:
            print(f"警告：任务 {event.task_id} 未找到")
            return
        
        task = self.tasks[event.task_id]
        
        # 准备段
        segments = self._prepare_segments(task)
        
        # 创建任务实例
        instance = TaskInstance(
            task_id=event.task_id,
            instance_id=event.instance_id,
            priority=task.priority,
            segments=segments,
            start_time=self.current_time
        )
        
        self.task_instances[(event.task_id, event.instance_id)] = instance
        
        print(f"{self.current_time:>8.1f}ms: [LAUNCH] {event.task_id}#{event.instance_id} "
              f"({len(segments)} segments, priority={task.priority.name})")
    
    def _prepare_segments(self, task: NNTask) -> List[SubSegment]:
        """准备任务的执行段"""
        # 尝试获取已分段的子段
        segments = task.apply_segmentation()
        
        if not segments:
            # 如果没有子段，将原始段转换为子段
            segments = []
            for seg in task.segments:
                sub_seg = SubSegment(
                    sub_id=seg.segment_id,
                    resource_type=seg.resource_type,
                    duration_table=seg.duration_table,
                    cut_overhead=0.0,
                    original_segment_id=seg.segment_id
                )
                segments.append(sub_seg)
        
        return segments
    
    def _schedule_segments_traditional(self):
        """传统模式：每个任务只调度当前段"""
        for instance in self.task_instances.values():
            if instance.is_completed:
                continue
            
            # 只尝试调度当前段
            if instance.can_enqueue_segment(instance.current_segment_index):
                self._try_enqueue_segment(instance, instance.current_segment_index)
    
    def _schedule_segments_smart(self):
        """段级模式：智能调度，保持段间依赖但允许提前入队"""
        for instance in self.task_instances.values():
            if instance.is_completed:
                continue
            
            # 尝试调度所有可以入队的段
            for seg_idx in range(len(instance.segments)):
                if instance.can_enqueue_segment(seg_idx):
                    self._try_enqueue_segment(instance, seg_idx)
    
    def _try_enqueue_segment(self, instance: TaskInstance, segment_index: int):
        """尝试将指定段加入队列"""
        if segment_index >= len(instance.segments):
            return
        
        segment = instance.segments[segment_index]
        
        # 寻找可用资源
        queue = self.queue_manager.find_best_queue(segment.resource_type)
        if not queue:
            return
        
        segment_id = f"{instance.task_id}#{instance.instance_id}_seg{segment_index}"
        
        # 加入队列
        queue.enqueue(
            segment_id,
            instance.priority,
            self.current_time,
            [segment]
        )
        
        # 标记已入队
        instance.segment_enqueued[segment_index] = True
        
        # 记录入队
        self.tracer.record_enqueue(
            segment_id,
            queue.resource_id,
            instance.priority,
            self.current_time,
            [segment]
        )
        
        print(f"{self.current_time:>8.1f}ms: [ENQUEUE] {segment_id} to {queue.resource_id}")
    
    def _execute_ready_tasks(self):
        """在所有资源上执行就绪的任务"""
        for resource_id, queue in self.queue_manager.resource_queues.items():
            queue.advance_time(self.current_time)
            
            if not queue.is_busy():
                next_task = queue.get_next_task()
                
                if next_task and next_task.ready_time <= self.current_time:
                    self._execute_task(queue, next_task)
    
    def _execute_task(self, queue: 'ResourceQueue', queued_task: QueuedTask):
        """执行任务"""
        segment = queued_task.get_current_segment()
        if not segment:
            return
        
        # 执行任务
        end_time = queue.execute_task(queued_task, self.current_time)
        
        # 记录执行
        self.tracer.record_execution(
            queued_task.task_id,
            queue.resource_id,
            self.current_time,
            end_time,
            queue.bandwidth,
            segment_id=queued_task.task_id.split('_')[-1]
        )
        
        print(f"{self.current_time:>8.1f}ms: [EXECUTE] {queued_task.task_id} on {queue.resource_id} "
              f"(duration={end_time - self.current_time:.1f}ms, priority={queued_task.priority.name})")
        
        # 从队列中移除
        queue.dequeue_task(queued_task.task_id, queued_task.priority)
        
        # 添加完成事件
        parts = queued_task.task_id.split('#')
        if len(parts) == 2:
            base_task_id = parts[0]
            instance_parts = parts[1].split('_seg')
            if len(instance_parts) == 2:
                instance_id = int(instance_parts[0])
                segment_index = int(instance_parts[1])
                
                completion = SegmentCompletion(
                    task_id=base_task_id,
                    instance_id=instance_id,
                    segment_index=segment_index,
                    completion_time=end_time,
                    resource_id=queue.resource_id
                )
                heapq.heappush(self.next_events, (end_time, "completion", completion))
    
    def _handle_completion_event(self, event_data):
        """处理段完成事件"""
        completion = event_data
        
        # 更新任务实例状态
        instance = self.task_instances.get((completion.task_id, completion.instance_id))
        if instance:
            # 标记段完成
            instance.completed_segments.append(completion.segment_index)
            
            # 更新当前段索引
            if completion.segment_index == instance.current_segment_index:
                instance.current_segment_index += 1
            
            print(f"{self.current_time:>8.1f}ms: [COMPLETE] {completion.task_id}#{completion.instance_id}_seg{completion.segment_index}")
            
            # 检查任务是否完成
            if instance.is_completed:
                instance.completion_time = self.current_time
                print(f"{self.current_time:>8.1f}ms: [TASK_COMPLETE] {completion.task_id}#{completion.instance_id}")
        
        # 记录完成
        self.segment_completions.append(completion)
    
    def _has_active_tasks(self) -> bool:
        """检查是否还有活跃的任务"""
        # 检查是否有未完成的任务实例
        for instance in self.task_instances.values():
            if not instance.is_completed:
                return True
        
        # 检查资源队列中是否还有任务
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy():
                return True
            # 检查队列中是否有等待的任务
            for priority_queue in queue.priority_queues.values():
                if len(priority_queue) > 0:
                    return True
        
        return False
    
    def _advance_time(self, max_time: float):
        """推进仿真时间"""
        next_time = max_time
        
        # 找到下一个事件时间
        if self.next_events:
            next_time = min(next_time, self.next_events[0][0])
        
        # 找到下一个资源释放时间
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy() and queue.busy_until > self.current_time:
                next_time = min(next_time, queue.busy_until)
        
        # 如果没有找到任何事件，小步推进
        if next_time == max_time and self._has_active_tasks():
            next_time = self.current_time + 0.1
        
        # 推进时间
        if next_time > self.current_time:
            self.current_time = next_time
            
            # 更新所有资源队列的时间
            for queue in self.queue_manager.resource_queues.values():
                queue.advance_time(self.current_time)
    
    def _get_execution_stats(self) -> Dict:
        """获取执行统计信息"""
        completed_instances = sum(1 for instance in self.task_instances.values() 
                                if instance.is_completed)
        
        total_segments = sum(len(instance.segments) 
                           for instance in self.task_instances.values())
        
        executed_segments = len(self.segment_completions)
        
        return {
            'total_instances': len(self.task_instances),
            'completed_instances': completed_instances,
            'total_segments': total_segments,
            'total_segments_executed': executed_segments,
            'simulation_time': self.current_time
        }


def create_executor(queue_manager: ResourceQueueManager, 
                   tracer: ScheduleTracer,
                   tasks: Dict[str, NNTask],
                   mode: str = "traditional") -> ScheduleExecutor:
    """工厂函数：创建执行器"""
    executor = ScheduleExecutor(queue_manager, tracer, tasks)
    
    if mode == "segment_aware":
        executor.segment_mode = True
    
    return executor