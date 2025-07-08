#!/usr/bin/env python3
"""
段感知执行器 - 支持ACPU_Runtime模式和段级发射
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from core.models import SubSegment, ResourceSegment
from core.enums import ResourceType, TaskPriority
from core.resource_queue import ResourceQueueManager, QueuedTask
from core.schedule_tracer import ScheduleTracer
from core.launcher import LaunchPlan, LaunchEvent
from core.task import NNTask


@dataclass
class SegmentCompletion:
    """段完成信息"""
    task_id: str
    instance_id: int
    segment_index: int
    completion_time: float
    resource_id: str
    
    def __lt__(self, other):
        return self.completion_time < other.completion_time


@dataclass
class SegmentAwareTaskInstance:
    """支持混合模式的任务实例"""
    task_id: str
    instance_id: int
    priority: TaskPriority
    
    # 发射模式
    launch_mode: str = "whole"  # "whole" or "segment"
    
    # 段信息
    segments: List[SubSegment] = field(default_factory=list)
    segment_launched: List[bool] = field(default_factory=list)  # 段是否已发射
    segment_completed: List[bool] = field(default_factory=list)  # 段是否已完成
    
    # 执行状态
    launch_time: float = 0.0
    completion_time: Optional[float] = None
    
    @property
    def instance_key(self) -> Tuple[str, int]:
        return (self.task_id, self.instance_id)
    
    @property
    def is_completed(self) -> bool:
        return all(self.segment_completed)
    
    def get_next_segment_index(self) -> Optional[int]:
        """获取下一个未发射的段索引"""
        for i, launched in enumerate(self.segment_launched):
            if not launched:
                return i
        return None


class SegmentAwareExecutor:
    """段感知执行器
    
    支持两种模式：
    1. 整体发射（ACPU_Runtime）：任务的所有段一次性发射
    2. 段级发射：任务的段可以独立发射和调度
    """
    
    def __init__(self, queue_manager: ResourceQueueManager, 
                 tracer: ScheduleTracer,
                 tasks: Dict[str, NNTask]):
        self.queue_manager = queue_manager
        self.tracer = tracer
        self.tasks = tasks
        
        # 执行状态
        self.task_instances: Dict[Tuple[str, int], SegmentAwareTaskInstance] = {}
        self.segment_completions: List[SegmentCompletion] = []
        
        # 配置哪些任务使用段级发射
        self.segment_launch_tasks = {"T2", "T3", "T4", "T5", "T6", "T7", "T8"}  # 可配置
        
        # 时间管理
        self.current_time = 0.0
        self.next_events: List[Tuple[float, str, any]] = []
        
    def execute_plan(self, launch_plan: LaunchPlan, max_time: float) -> Dict:
        """执行发射计划"""
        print(f"\n{'='*80}")
        print(f"开始段感知调度 (max_time={max_time}ms)")
        print(f"{'='*80}\n")
        
        self._reset_state()
        
        # 将发射事件加入队列
        for event in launch_plan.events:
            heapq.heappush(self.next_events, (event.time, "launch", event))
        
        # 主执行循环
        while self.current_time < max_time and (self.next_events or self._has_active_tasks()):
            # 处理事件
            self._process_events()
            
            # 检查段级任务的下一段
            self._check_segment_tasks()
            
            # 调度资源
            self._schedule_resources()
            
            # 推进时间
            self._advance_time(max_time)
        
        return self._get_execution_stats()
    
    def _reset_state(self):
        """重置状态"""
        self.task_instances.clear()
        self.segment_completions.clear()
        self.current_time = 0.0
        self.next_events.clear()
        
    def _process_events(self):
        """处理事件"""
        while self.next_events and self.next_events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.next_events)
            
            if event_type == "launch":
                self._handle_launch_event(event_data)
            elif event_type == "completion":
                self._handle_completion_event(event_data)
    
    def _handle_launch_event(self, event: LaunchEvent):
        """处理任务发射"""
        task = self.tasks[event.task_id]
        
        # 确定发射模式
        launch_mode = "segment" if event.task_id in self.segment_launch_tasks else "whole"
        
        # 创建任务实例
        instance = SegmentAwareTaskInstance(
            task_id=event.task_id,
            instance_id=event.instance_id,
            priority=task.priority,
            launch_mode=launch_mode,
            segments=self._prepare_segments(task),
            launch_time=self.current_time
        )
        
        # 初始化段状态
        instance.segment_launched = [False] * len(instance.segments)
        instance.segment_completed = [False] * len(instance.segments)
        
        self.task_instances[instance.instance_key] = instance
        
        print(f"{self.current_time:>8.1f}ms: [LAUNCH] {event.task_id}#{event.instance_id} "
              f"({len(instance.segments)} segments, mode={launch_mode}, priority={task.priority.name})")
        
        if launch_mode == "whole":
            # 整体发射模式：立即发射所有段
            self._launch_all_segments(instance)
        else:
            # 段级发射模式：只发射第一个段
            self._launch_segment(instance, 0)
    
    def _launch_all_segments(self, instance: SegmentAwareTaskInstance):
        """整体发射所有段（ACPU_Runtime模式）"""
        for i in range(len(instance.segments)):
            self._launch_segment(instance, i)
    
    def _launch_segment(self, instance: SegmentAwareTaskInstance, segment_index: int):
        """发射指定段"""
        if segment_index >= len(instance.segments):
            return
            
        if instance.segment_launched[segment_index]:
            return  # 已发射
        
        segment = instance.segments[segment_index]
        
        # 检查前序依赖（只在同资源类型时需要）
        if segment_index > 0:
            prev_segment = instance.segments[segment_index - 1]
            if (segment.resource_type == prev_segment.resource_type and 
                not instance.segment_completed[segment_index - 1]):
                return  # 等待前序段完成
        
        # 找最佳队列
        best_queue = self.queue_manager.find_best_queue(segment.resource_type)
        if not best_queue:
            return
        
        # 段ID
        segment_id = f"{instance.task_id}#{instance.instance_id}_seg{segment_index}"
        
        # 加入队列
        best_queue.enqueue(
            segment_id,
            instance.priority,
            self.current_time,
            [segment]
        )
        
        instance.segment_launched[segment_index] = True
        
        # 记录
        self.tracer.record_enqueue(
            segment_id,
            best_queue.resource_id,
            instance.priority,
            self.current_time,
            [segment]
        )
        
        print(f"{self.current_time:>8.1f}ms: [ENQUEUE] {segment_id} to {best_queue.resource_id}")
    
    def _check_segment_tasks(self):
        """检查段级任务的下一段是否可发射"""
        for instance in self.task_instances.values():
            if instance.launch_mode != "segment":
                continue
                
            # 找下一个未发射的段
            next_idx = instance.get_next_segment_index()
            if next_idx is not None:
                self._launch_segment(instance, next_idx)
    
    def _schedule_resources(self):
        """调度资源"""
        scheduled_any = True
        
        while scheduled_any:
            scheduled_any = False
            
            for resource_id, queue in self.queue_manager.resource_queues.items():
                queue.advance_time(self.current_time)
                
                if not queue.is_busy():
                    next_task = queue.get_next_task()
                    
                    if next_task and next_task.ready_time <= self.current_time:
                        self._execute_segment_on_resource(next_task, queue)
                        scheduled_any = True
    
    def _execute_segment_on_resource(self, queued_task: QueuedTask, queue):
        """在资源上执行段"""
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
        
        # 更新队列
        queue.busy_until = end_time
        queue.current_task = queued_task.task_id
        queue.total_tasks_executed += 1
        queue.total_busy_time += duration
        queue.dequeue_task(queued_task.task_id, queued_task.priority)
        
        print(f"{self.current_time:>8.1f}ms: [EXECUTE] {queued_task.task_id} on {queue.resource_id} "
              f"(duration={duration:.1f}ms, priority={queued_task.priority.name})")
        
        # 解析段信息
        parts = queued_task.task_id.split('_seg')
        if len(parts) == 2:
            task_instance_part = parts[0]
            segment_index = int(parts[1])
            
            task_parts = task_instance_part.split('#')
            task_id = task_parts[0]
            instance_id = int(task_parts[1]) if len(task_parts) > 1 else 0
            
            # 添加完成事件
            completion = SegmentCompletion(
                task_id=task_id,
                instance_id=instance_id,
                segment_index=segment_index,
                completion_time=end_time,
                resource_id=queue.resource_id
            )
            
            heapq.heappush(self.next_events, (end_time, "completion", completion))
    
    def _handle_completion_event(self, completion: SegmentCompletion):
        """处理段完成"""
        instance_key = (completion.task_id, completion.instance_id)
        instance = self.task_instances.get(instance_key)
        
        if not instance:
            return
        
        # 标记段完成
        instance.segment_completed[completion.segment_index] = True
        
        # 释放资源
        queue = self.queue_manager.get_queue(completion.resource_id)
        if queue:
            queue.busy_until = self.current_time
            queue.current_task = None
        
        print(f"{self.current_time:>8.1f}ms: [COMPLETE] {completion.task_id}#{completion.instance_id}"
              f"_seg{completion.segment_index} on {completion.resource_id}")
        
        # 检查任务是否完成
        if instance.is_completed:
            instance.completion_time = self.current_time
            total_time = instance.completion_time - instance.launch_time
            print(f"{self.current_time:>8.1f}ms: [FINISHED] {completion.task_id}#{completion.instance_id} "
                  f"(total time: {total_time:.1f}ms)")
    
    def _has_active_tasks(self) -> bool:
        """是否还有活动任务"""
        return any(not instance.is_completed for instance in self.task_instances.values())
    
    def _advance_time(self, max_time: float):
        """推进时间"""
        next_time = max_time
        
        if self.next_events:
            next_time = min(next_time, self.next_events[0][0])
        
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy() and queue.busy_until < next_time:
                next_time = queue.busy_until
        
        if next_time > self.current_time:
            self.current_time = min(next_time, max_time)
    
    def _prepare_segments(self, task: NNTask) -> List[SubSegment]:
        """准备任务段"""
        sub_segments = task.apply_segmentation()
        
        if not sub_segments:
            # 转换原始段为子段格式
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
    
    def _get_execution_stats(self) -> Dict:
        """获取执行统计"""
        total_instances = len(self.task_instances)
        completed_instances = sum(1 for inst in self.task_instances.values() if inst.is_completed)
        
        total_segments = sum(len(inst.segments) for inst in self.task_instances.values())
        completed_segments = sum(sum(inst.segment_completed) for inst in self.task_instances.values())
        
        return {
            'total_instances': total_instances,
            'completed_instances': completed_instances,
            'total_segments': total_segments,
            'completed_segments': completed_segments,
            'total_executions': len(self.tracer.executions),
            'current_time': self.current_time
        }


@dataclass
class SegmentCompletion:
    """段完成信息"""
    task_id: str
    instance_id: int
    segment_index: int
    completion_time: float
    resource_id: str
    
    def __lt__(self, other):
        return self.completion_time < other.completion_time


@dataclass
class SegmentCompletion:
    """段完成信息"""
    task_id: str
    instance_id: int
    segment_index: int
    completion_time: float
    resource_id: str
    
    def __lt__(self, other):
        return self.completion_time < other.completion_time
