#!/usr/bin/env python3
"""
调度执行器 - 负责执行发射计划并处理资源调度
重点实现：当任务A在DSP上执行时，同优先级的任务B可以使用空闲的NPU
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from .models import SubSegment
from .enums import ResourceType, TaskPriority
from .resource_queue import ResourceQueueManager, QueuedTask
from .schedule_tracer import ScheduleTracer
from .launcher import LaunchPlan, LaunchEvent
from .task import NNTask


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
    
    核心功能：
    1. 执行发射计划
    2. 管理任务实例状态
    3. 处理段间依赖
    4. 实现资源竞争和优先级调度
    5. 确保资源最大化利用（重点：NPU/DSP交替时的空隙利用）
    """
    
    def __init__(self, queue_manager: ResourceQueueManager, 
                 tracer: ScheduleTracer,
                 tasks: Dict[str, NNTask]):
        self.queue_manager = queue_manager
        self.tracer = tracer
        self.tasks = tasks  # task_id -> NNTask
        
        # 执行状态
        self.task_instances: Dict[Tuple[str, int], TaskInstance] = {}
        self.segment_completions: List[SegmentCompletion] = []
        
        # 时间管理
        self.current_time = 0.0
        self.next_events: List[Tuple[float, str, any]] = []  # (time, event_type, data)
        
    def execute_plan(self, launch_plan: LaunchPlan, max_time: float) -> Dict:
        """执行发射计划
        
        Args:
            launch_plan: 任务发射计划
            max_time: 最大仿真时间
            
        Returns:
            执行统计信息
        """
        print(f"\n{'='*80}")
        print(f"开始执行调度 (max_time={max_time}ms)")
        print(f"{'='*80}\n")
        
        # 重置状态
        self._reset_state()
        
        # 将发射事件加入事件队列
        for event in launch_plan.events:
            heapq.heappush(self.next_events, (event.time, "launch", event))
        
        # 主执行循环
        while self.current_time < max_time and (self.next_events or self._has_active_tasks()):
            # 1. 处理当前时间的所有事件
            self._process_events()
            
            # 2. 尝试调度所有可用资源上的任务
            self._schedule_ready_segments()
            
            # 3. 推进时间到下一个事件
            self._advance_time(max_time)
        
        # 返回执行统计
        return self._get_execution_stats()
    
    def _reset_state(self):
        """重置执行器状态"""
        self.task_instances.clear()
        self.segment_completions.clear()
        self.current_time = 0.0
        self.next_events.clear()
        
    def _process_events(self):
        """处理当前时间的所有事件"""
        while self.next_events and self.next_events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.next_events)
            
            if event_type == "launch":
                self._handle_launch_event(event_data)
            elif event_type == "completion":
                self._handle_completion_event(event_data)
    
    def _handle_launch_event(self, event: LaunchEvent):
        """处理任务发射事件"""
        task = self.tasks[event.task_id]
        
        # 创建任务实例
        instance = TaskInstance(
            task_id=event.task_id,
            instance_id=event.instance_id,
            segments=self._prepare_segments(task),
            priority=task.priority,
            start_time=self.current_time
        )
        
        self.task_instances[instance.instance_key] = instance
        
        print(f"{self.current_time:>8.1f}ms: [LAUNCH] {event.task_id}#{event.instance_id} "
              f"({len(instance.segments)} segments, priority={task.priority.name})")
        
        # 立即尝试调度第一个段
        self._try_schedule_segment(instance)
    
    def _handle_completion_event(self, completion: SegmentCompletion):
        """处理段完成事件"""
        instance_key = (completion.task_id, completion.instance_id)
        instance = self.task_instances.get(instance_key)
        
        if not instance:
            return
        
        # 记录段完成
        instance.completed_segments.append(completion.segment_index)
        instance.current_segment_index += 1
        
        # 释放资源
        queue = self.queue_manager.get_queue(completion.resource_id)
        if queue:
            queue.busy_until = queue.current_time  # 标记资源为空闲
            queue.current_task = None
        
        print(f"{self.current_time:>8.1f}ms: [COMPLETE] {completion.task_id}#{completion.instance_id}"
              f"_seg{completion.segment_index} on {completion.resource_id}")
        
        # 如果任务未完成，尝试调度下一个段
        if not instance.is_completed:
            self._try_schedule_segment(instance)
        else:
            instance.completion_time = self.current_time
            print(f"{self.current_time:>8.1f}ms: [FINISHED] {completion.task_id}#{completion.instance_id} "
                  f"(total time: {instance.completion_time - instance.start_time:.1f}ms)")
    
    def _try_schedule_segment(self, instance: TaskInstance):
        """尝试调度任务的当前段
        
        这是核心调度逻辑，确保：
        1. 段按顺序执行（依赖关系）
        2. 资源空闲时立即分配
        3. 同优先级任务可以充分利用空闲资源
        """
        segment = instance.get_current_segment()
        if not segment:
            return
        
        # 检查前序段是否完成（段间依赖）
        if instance.current_segment_index > 0:
            prev_completion = self._find_segment_completion(
                instance.task_id, 
                instance.instance_id, 
                instance.current_segment_index - 1
            )
            if not prev_completion or prev_completion.completion_time > self.current_time:
                # 前序段未完成，等待
                return
        
        # 寻找可用资源
        best_queue = self.queue_manager.find_best_queue(segment.resource_type)
        if not best_queue:
            return
        
        # 加入队列
        best_queue.enqueue(
            f"{instance.task_id}#{instance.instance_id}_seg{instance.current_segment_index}",
            instance.priority,
            self.current_time,
            [segment]
        )
        
        # 记录入队 - 注意参数顺序
        self.tracer.record_enqueue(
            f"{instance.task_id}#{instance.instance_id}_seg{instance.current_segment_index}",
            best_queue.resource_id,
            instance.priority,
            self.current_time,
            [segment]  # segments 列表
        )
    
    def _schedule_ready_segments(self):
        """调度所有资源上的就绪任务
        
        关键：确保所有空闲资源都被充分利用
        """
        scheduled_any = True
        
        while scheduled_any:
            scheduled_any = False
            
            # 检查每个资源队列
            for resource_id, queue in self.queue_manager.resource_queues.items():
                # 更新队列状态
                queue.advance_time(self.current_time)
                
                # 如果资源空闲，尝试调度下一个任务
                if not queue.is_busy():
                    next_task = queue.get_next_task()
                    
                    if next_task and next_task.ready_time <= self.current_time:
                        # 执行任务
                        self._execute_segment(next_task, queue)
                        scheduled_any = True
    
    def _execute_segment(self, queued_task: QueuedTask, queue):
        """执行任务段"""
        segment = queued_task.sub_segments[0]
        duration = segment.get_duration(queue.bandwidth)
        end_time = self.current_time + duration
        
        # 记录执行
        self.tracer.record_execution(
            queued_task.task_id,
            queue.resource_id,
            self.current_time,
            end_time,
            queue.bandwidth,
            segment.sub_id
        )
        
        # 更新队列状态
        queue.busy_until = end_time
        queue.current_task = queued_task.task_id
        queue.total_tasks_executed += 1
        queue.total_busy_time += duration
        
        # 从队列移除
        queue.dequeue_task(queued_task.task_id, queued_task.priority)
        
        # 解析任务信息
        parts = queued_task.task_id.split('_seg')
        task_instance_part = parts[0]  # e.g., "T1#0"
        segment_index = int(parts[1]) if len(parts) > 1 else 0
        
        task_id_parts = task_instance_part.split('#')
        task_id = task_id_parts[0]
        instance_id = int(task_id_parts[1]) if len(task_id_parts) > 1 else 0
        
        print(f"{self.current_time:>8.1f}ms: [EXECUTE] {queued_task.task_id} on {queue.resource_id} "
              f"(duration={duration:.1f}ms, priority={queued_task.priority.name})")
        
        # 添加完成事件
        completion = SegmentCompletion(
            task_id=task_id,
            instance_id=instance_id,
            segment_index=segment_index,
            completion_time=end_time,
            resource_id=queue.resource_id
        )
        
        heapq.heappush(self.next_events, (end_time, "completion", completion))
        self.segment_completions.append(completion)
    
    def _advance_time(self, max_time: float):
        """推进仿真时间"""
        next_time = max_time
        
        # 找到下一个事件时间
        if self.next_events:
            next_time = min(next_time, self.next_events[0][0])
        
        # 找到下一个资源释放时间
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy() and queue.busy_until < next_time:
                next_time = queue.busy_until
        
        # 时间推进
        if next_time > self.current_time:
            self.current_time = min(next_time, max_time)
    
    def _prepare_segments(self, task: NNTask) -> List[SubSegment]:
        """准备任务的执行段"""
        # 应用分段策略
        sub_segments = task.apply_segmentation()
        
        if not sub_segments:
            # 如果没有分段，转换原始段
            sub_segments = []
            for i, seg in enumerate(task.segments):
                sub_seg = SubSegment(
                    sub_id=f"{seg.segment_id}_{i}",
                    resource_type=seg.resource_type,
                    duration_table=seg.duration_table,
                    cut_overhead=0.0,
                    original_segment_id=seg.segment_id
                )
                sub_segments.append(sub_seg)
        
        return sub_segments
    
    def _find_segment_completion(self, task_id: str, instance_id: int, 
                                segment_index: int) -> Optional[SegmentCompletion]:
        """查找段完成记录"""
        for completion in self.segment_completions:
            if (completion.task_id == task_id and 
                completion.instance_id == instance_id and 
                completion.segment_index == segment_index):
                return completion
        return None
    
    def _has_active_tasks(self) -> bool:
        """检查是否还有活跃任务"""
        # 检查未完成的任务实例
        for instance in self.task_instances.values():
            if not instance.is_completed:
                return True
        
        # 检查队列中的任务
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy():
                return True
            
            # 检查是否有待处理的任务
            for priority_queue in queue.priority_queues.values():
                if priority_queue:
                    return True
        
        return False
    
    def _get_execution_stats(self) -> Dict:
        """获取执行统计信息"""
        stats = {
            'total_instances': len(self.task_instances),
            'completed_instances': sum(1 for inst in self.task_instances.values() if inst.is_completed),
            'total_segments_executed': len(self.segment_completions),
            'simulation_time': self.current_time,
        }
        
        # 任务完成时间统计
        completion_times = {}
        for instance in self.task_instances.values():
            if instance.completion_time:
                task_id = instance.task_id
                if task_id not in completion_times:
                    completion_times[task_id] = []
                completion_times[task_id].append(
                    instance.completion_time - instance.start_time
                )
        
        stats['average_completion_times'] = {
            task_id: sum(times) / len(times)
            for task_id, times in completion_times.items()
        }
        
        return stats


def create_executor(queue_manager: ResourceQueueManager, 
                   tracer: ScheduleTracer,
                   tasks: Dict[str, NNTask], mode="default") -> ScheduleExecutor:
    """创建执行器实例的工厂函数"""
    if mode == "segment_aware":
        from .segment_aware_executor import SegmentAwareExecutor
        return SegmentAwareExecutor(queue_manager, tracer, tasks)
    else:
        return ScheduleExecutor(queue_manager, tracer, tasks)