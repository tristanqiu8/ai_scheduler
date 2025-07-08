#!/usr/bin/env python3
"""
调度执行器 - 负责执行发射计划
现在内置段感知能力，向后兼容原有接口
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
class TaskInstance:
    """任务实例"""
    task_id: str
    instance_id: int
    priority: TaskPriority
    
    # 段信息
    segments: List[SubSegment] = field(default_factory=list)
    current_segment_index: int = 0
    
    # 时间信息
    start_time: float = 0.0
    completion_time: Optional[float] = None
    
    @property
    def is_completed(self) -> bool:
        return self.current_segment_index >= len(self.segments)
    
    def get_current_segment(self) -> Optional[SubSegment]:
        """获取当前段"""
        if 0 <= self.current_segment_index < len(self.segments):
            return self.segments[self.current_segment_index]
        return None


class ScheduleExecutor:
    """调度执行器
    
    支持两种模式：
    1. 传统模式：任务按段顺序执行（默认）
    2. 段级模式：所有段同时入队（通过segment_mode参数控制）
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
        
        # 配置选项（可通过属性设置）
        self.segment_mode = False  # 默认为传统模式，保持向后兼容
        
    def execute_plan(self, launch_plan: LaunchPlan, max_time: float, 
                    segment_mode: Optional[bool] = None) -> Dict:
        """执行发射计划
        
        Args:
            launch_plan: 发射计划
            max_time: 最大仿真时间
            segment_mode: 是否使用段级模式（None表示使用实例默认值）
        """
        # 如果指定了segment_mode，使用指定值；否则使用实例属性
        use_segment_mode = segment_mode if segment_mode is not None else self.segment_mode
        
        print(f"\n{'='*80}")
        print(f"开始执行调度 (max_time={max_time}ms)")
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
            
            # 调度就绪的段
            self._schedule_ready_segments()
            
            # 检查完成的段
            self._check_completions()
            
            # 推进时间
            self._advance_time(max_time)
        
        return self._get_execution_stats()
    
    def _reset_state(self):
        """重置执行器状态"""
        self.task_instances.clear()
        self.segment_completions.clear()
        self.current_time = 0.0
        self.next_events.clear()
        
    def _process_events(self, use_segment_mode: bool):
        """处理当前时间的所有事件"""
        while self.next_events and self.next_events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.next_events)
            
            if event_type == "launch":
                self._handle_launch_event(event_data, use_segment_mode)
            elif event_type == "completion":
                self._handle_completion_event(event_data)
    
    def _handle_launch_event(self, event: LaunchEvent, use_segment_mode: bool):
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
        
        if use_segment_mode:
            # 段级模式：所有段同时入队
            for i, segment in enumerate(segments):
                self._enqueue_segment_immediately(instance, i, segment)
        # 否则使用传统的逐段执行（通过_schedule_ready_segments处理）
    
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
    
    def _enqueue_segment_immediately(self, instance: TaskInstance, 
                                   index: int, segment: SubSegment):
        """立即将段加入队列（段级模式）"""
        queue = self.queue_manager.find_best_queue(segment.resource_type)
        if not queue:
            return
        
        segment_id = f"{instance.task_id}#{instance.instance_id}_seg{index}"
        
        queue.enqueue(
            segment_id,
            instance.priority,
            self.current_time,
            [segment]
        )
        
        self.tracer.record_enqueue(
            segment_id,
            queue.resource_id,
            instance.priority,
            self.current_time,
            [segment]
        )
        
        print(f"{self.current_time:>8.1f}ms: [ENQUEUE] {segment_id} to {queue.resource_id}")
    
    def _schedule_ready_segments(self):
        """调度就绪的段（传统模式）"""
        # 检查每个任务实例，看是否有段可以执行
        for instance in self.task_instances.values():
            if instance.is_completed:
                continue
            
            # 传统模式下，只有当前段可以执行
            self._try_schedule_segment(instance)
        
        # 执行资源上的就绪任务
        self._execute_ready_tasks()
    
    def _try_schedule_segment(self, instance: TaskInstance):
        """尝试调度任务的当前段（传统模式）"""
        segment = instance.get_current_segment()
        if not segment:
            return
        
        # 检查前序段是否完成
        if instance.current_segment_index > 0:
            prev_completion = self._find_segment_completion(
                instance.task_id, 
                instance.instance_id, 
                instance.current_segment_index - 1
            )
            if not prev_completion or prev_completion.completion_time > self.current_time:
                return
        
        # 寻找可用资源
        best_queue = self.queue_manager.find_best_queue(segment.resource_type)
        if not best_queue:
            return
        
        # 检查是否已经在队列中
        segment_id = f"{instance.task_id}#{instance.instance_id}_seg{instance.current_segment_index}"
        
        # 检查是否已经入队
        already_queued = False
        for priority_queue in best_queue.priority_queues.values():
            if any(task.task_id == segment_id for task in priority_queue):
                already_queued = True
                break
        
        if not already_queued and best_queue.current_task != segment_id:
            # 加入队列
            best_queue.enqueue(
                segment_id,
                instance.priority,
                self.current_time,
                [segment]
            )
            
            self.tracer.record_enqueue(
                segment_id,
                best_queue.resource_id,
                instance.priority,
                self.current_time,
                [segment]
            )
    
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
        heapq.heappush(self.next_events, (end_time, "completion", {
            'task_id': queued_task.task_id,
            'resource_id': queue.resource_id,
            'end_time': end_time
        }))
    
    def _handle_completion_event(self, event_data: Dict):
        """处理段完成事件"""
        task_id = event_data['task_id']
        
        # 解析任务ID
        parts = task_id.split('#')
        if len(parts) != 2:
            return
        
        base_task_id = parts[0]
        instance_info = parts[1].split('_')
        if len(instance_info) < 2:
            return
        
        instance_id = int(instance_info[0])
        seg_idx = int(instance_info[1].replace('seg', ''))
        
        # 记录完成
        completion = SegmentCompletion(
            task_id=base_task_id,
            instance_id=instance_id,
            segment_index=seg_idx,
            completion_time=event_data['end_time'],
            resource_id=event_data['resource_id']
        )
        
        self.segment_completions.append(completion)
        
        print(f"{event_data['end_time']:>8.1f}ms: [COMPLETE] {task_id} on {event_data['resource_id']}")
        
        # 更新任务实例
        instance_key = (base_task_id, instance_id)
        if instance_key in self.task_instances:
            instance = self.task_instances[instance_key]
            if seg_idx == instance.current_segment_index:
                instance.current_segment_index += 1
                
                if instance.is_completed:
                    instance.completion_time = event_data['end_time']
                    total_time = instance.completion_time - instance.start_time
                    print(f"{event_data['end_time']:>8.1f}ms: [FINISHED] {base_task_id}#{instance_id} "
                          f"(total time: {total_time:.1f}ms)")
    
    def _check_completions(self):
        """检查并处理完成的段"""
        # 这个方法现在主要由事件驱动，这里保留以备扩展
        pass
    
    def _advance_time(self, max_time: float):
        """推进仿真时间到下一个事件"""
        next_time = max_time
        
        # 考虑事件队列中的时间
        if self.next_events:
            next_time = min(next_time, self.next_events[0][0])
        
        # 考虑资源的忙碌时间
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy() and queue.busy_until < next_time:
                next_time = queue.busy_until
        
        # 更新时间
        if next_time > self.current_time:
            self.current_time = next_time
    
    def _has_active_tasks(self) -> bool:
        """检查是否还有活跃的任务"""
        # 检查未完成的实例
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
    
    def _find_segment_completion(self, task_id: str, instance_id: int, 
                                segment_index: int) -> Optional[SegmentCompletion]:
        """查找特定段的完成记录"""
        for completion in self.segment_completions:
            if (completion.task_id == task_id and 
                completion.instance_id == instance_id and
                completion.segment_index == segment_index):
                return completion
        return None
    
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
                   tasks: Dict[str, NNTask], 
                   mode="default") -> ScheduleExecutor:
    """创建执行器实例的工厂函数
    
    Args:
        mode: "default" 或 "segment_aware"
    """
    executor = ScheduleExecutor(queue_manager, tracer, tasks)
    if mode == "segment_aware":
        executor.segment_mode = True
    return executor
