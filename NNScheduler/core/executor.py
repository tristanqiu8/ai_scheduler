#!/usr/bin/env python3
"""
修复后的调度执行器 - 解决重复执行和冗余打印问题
"""

from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import heapq
import random

from NNScheduler.core.models import SubSegment, ResourceBinding
from NNScheduler.core.enums import ResourceType, TaskPriority, RuntimeType
from NNScheduler.core.resource_queue import ResourceQueueManager, QueuedTask
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import LaunchPlan, LaunchEvent
from NNScheduler.core.task import NNTask

# 全局日志开关
ENABLE_EXECUTION_LOG = False


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
    dependencies_ready: bool = True

    # 新增：记录每个段的状态
    segment_enqueued: List[bool] = field(default_factory=list)
    segment_in_queue: Set[int] = field(default_factory=set)  # 当前在队列中的段

    def __post_init__(self):
        # 初始化段状态
        self.segment_enqueued = [False] * len(self.segments)
    
    @property
    def instance_key(self) -> Tuple[str, int]:
        return (self.task_id, self.instance_id)
    
    @property
    def is_completed(self) -> bool:
        return len(self.completed_segments) >= len(self.segments)
    
    def get_current_segment(self) -> Optional[SubSegment]:
        """获取当前待执行的段"""
        if self.current_segment_index < len(self.segments):
            return self.segments[self.current_segment_index]
        return None
    
    def can_enqueue_segment(self, segment_index: int) -> bool:
        """检查指定段是否可以入队"""
        if segment_index >= len(self.segments):
            return False

        # 已经入队或在队列中的不能重复入队
        if self.segment_enqueued[segment_index] or segment_index in self.segment_in_queue:
            return False

        # 第一个段总是可以入队
        if segment_index == 0:
            if not self.dependencies_ready:
                return False
            return True

        # 后续段需要前一个段完成
        return (segment_index - 1) in self.completed_segments
    
    def mark_segment_enqueued(self, segment_index: int):
        """标记段已入队"""
        self.segment_enqueued[segment_index] = True
        self.segment_in_queue.add(segment_index)
    
    def mark_segment_completed(self, segment_index: int):
        """标记段已完成"""
        if segment_index not in self.completed_segments:
            self.completed_segments.append(segment_index)
        self.segment_in_queue.discard(segment_index)
        if segment_index == self.current_segment_index:
            self.current_segment_index += 1


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
    """调度执行器 - 修复版本"""
    
    def __init__(self, queue_manager: ResourceQueueManager, 
                 tracer: ScheduleTracer,
                 tasks: Dict[str, NNTask],
                 random_slack_enabled: bool = False,
                 random_slack_std: float = 0.0,
                 random_slack_seed: Optional[int] = None,
                 launch_strategy: str = "balanced"):
        self.queue_manager = queue_manager
        self.tracer = tracer
        self.tasks = tasks

        self.launch_strategy = (launch_strategy or "balanced").lower()
        if self.launch_strategy not in {"eager", "lazy", "balanced", "sync", "fixed"}:
            self.launch_strategy = "balanced"

        self.random_slack_std = max(0.0, float(random_slack_std))
        self.random_slack_enabled = bool(random_slack_enabled and self.random_slack_std > 0.0)
        if self.launch_strategy == "sync":
            self.random_slack_enabled = False

        if self.random_slack_enabled:
            self._slack_rng = random.Random(random_slack_seed)
        else:
            self._slack_rng = None

        # 执行状态
        self.task_instances: Dict[Tuple[str, int], TaskInstance] = {}
        self.segment_completions: List[SegmentCompletion] = []
        self.completed_instances: Dict[Tuple[str, int], float] = {}
        self.pending_dependency_instances: Set[Tuple[str, int]] = set()

        # 时间管理
        self.current_time = 0.0
        self.next_events: List[Tuple[float, str, any]] = []
        
        # 配置选项
        self.segment_mode = False
        
        # 新增：跟踪已处理的段，避免重复
        self.processed_segments: Set[str] = set()
        self.resource_bindings: Dict[Tuple[str, int], ResourceBinding] = {}
        self.bound_resource_usage: Dict[str, Tuple[str, int]] = {}
        
    def execute_plan(self, launch_plan: LaunchPlan, max_time: float, 
                    segment_mode: Optional[bool] = None) -> Dict:
        """执行发射计划"""
        use_segment_mode = segment_mode if segment_mode is not None else self.segment_mode
        
        # 只在方法开始时打印一次
        if self.current_time == 0.0:
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
        self.completed_instances.clear()
        self.pending_dependency_instances.clear()
        self.current_time = 0.0
        self.next_events.clear()
        self.processed_segments.clear()
        self.resource_bindings.clear()
        self.bound_resource_usage.clear()

    def _get_or_create_binding(self, instance: TaskInstance) -> ResourceBinding:
        """Get or create a binding record for a DSP runtime task instance."""
        key = instance.instance_key
        binding = self.resource_bindings.get(key)
        if binding is None:
            binding = ResourceBinding(
                task_id=instance.task_id,
                bound_resources=set(),
                binding_start=self.current_time,
                binding_end=0.0
            )
            self.resource_bindings[key] = binding
        return binding

    @staticmethod
    def _extract_resource_index(resource_id: str) -> Optional[int]:
        """Extract numeric suffix index from resource identifier."""
        parts = resource_id.split("_")
        if not parts:
            return None
        suffix = parts[-1]
        return int(suffix) if suffix.isdigit() else None

    def _select_unbound_queue(self, resource_type: ResourceType, binding_key: Tuple[str, int],
                              preferred_indices: Optional[Set[int]] = None) -> Optional['ResourceQueue']:
        """Select a queue not bound to other DSP runtime instances, preferring specific indices."""
        queues = self.queue_manager.get_queues_by_type(resource_type)
        if not queues:
            return None

        def sort_key(queue):
            index = self._extract_resource_index(queue.resource_id)
            preferred_rank = 1
            if preferred_indices:
                preferred_rank = 0 if index in preferred_indices else 1
            return (preferred_rank, queue.get_next_available_time())

        for queue in sorted(queues, key=sort_key):
            owner = self.bound_resource_usage.get(queue.resource_id)
            if owner is None or owner == binding_key:
                return queue
        return None

    def _get_queue_for_segment(self, instance: TaskInstance, segment: SubSegment) -> Optional['ResourceQueue']:
        """Determine the queue a segment should be enqueued to, respecting DSP bindings."""
        task_def = self.tasks.get(instance.task_id)
        if not task_def or task_def.runtime_type != RuntimeType.DSP_RUNTIME:
            return self._select_unbound_queue(segment.resource_type, instance.instance_key)

        binding = self._get_or_create_binding(instance)
        binding_key = instance.instance_key
        resource_type = segment.resource_type

        existing_id = binding.resource_map.get(resource_type)
        if existing_id:
            queue = self.queue_manager.get_queue(existing_id)
            if queue is not None:
                return queue
            binding.bound_resources.discard(existing_id)
            binding.resource_map.pop(resource_type, None)

        preferred_indices: Optional[Set[int]] = None
        complement_type = ResourceType.DSP if resource_type == ResourceType.NPU else ResourceType.NPU
        complement_id = binding.resource_map.get(complement_type)
        if complement_id:
            idx = self._extract_resource_index(complement_id)
            if idx is not None:
                preferred_indices = {idx}

        queue = self._select_unbound_queue(resource_type, binding_key, preferred_indices)
        if queue is None:
            return None

        binding.bound_resources.add(queue.resource_id)
        binding.resource_map[resource_type] = queue.resource_id
        self.bound_resource_usage[queue.resource_id] = binding_key

        if complement_type not in binding.resource_map:
            complement_pref: Optional[Set[int]] = None
            idx = self._extract_resource_index(queue.resource_id)
            if idx is not None:
                complement_pref = {idx}
            complement_queue = self._select_unbound_queue(complement_type, binding_key, complement_pref)
            if complement_queue is not None:
                binding.bound_resources.add(complement_queue.resource_id)
                binding.resource_map[complement_type] = complement_queue.resource_id
                self.bound_resource_usage[complement_queue.resource_id] = binding_key

        return queue

    def _reserve_bound_resource(self, queue: 'ResourceQueue', binding_key: Tuple[str, int], end_time: float):
        """Reserve a bound resource until the specified end time."""
        if queue is None:
            return

        previous_busy = queue.busy_until
        if end_time > previous_busy:
            start_time = max(self.current_time, previous_busy)
            if end_time > start_time:
                queue.total_busy_time += end_time - start_time
            queue.busy_until = end_time

        queue.current_task = f"{binding_key[0]}#{binding_key[1]}_BIND"
        self.bound_resource_usage[queue.resource_id] = binding_key

    def _dependencies_satisfied(self, task: NNTask, instance_id: int) -> bool:
        """检查给定任务实例的依赖是否完成"""
        if not task.dependencies:
            return True

        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            dep_instance = self._map_dependency_instance(task, dep_task, instance_id)
            if (dep_id, dep_instance) not in self.completed_instances:
                return False
        return True

    def _map_dependency_instance(self, task: NNTask, dep_task: Optional[NNTask], instance_id: int) -> int:
        """根据帧率映射依赖实例编号"""
        if dep_task is None:
            return instance_id

        dep_fps = getattr(dep_task, "fps_requirement", 0.0) or 0.0
        task_fps = getattr(task, "fps_requirement", 0.0) or 0.0

        if dep_fps <= 0 or task_fps <= 0:
            return instance_id

        if dep_fps < task_fps:
            ratio = task_fps / dep_fps
            if ratio > 0:
                return int(instance_id / ratio)
        return instance_id

    def _register_instance_completion(self, instance: TaskInstance):
        """记录任务实例完成并尝试释放依赖"""
        completion_time = instance.completion_time if instance.completion_time is not None else self.current_time
        key = (instance.task_id, instance.instance_id)
        self.completed_instances[key] = completion_time

        binding = self.resource_bindings.pop(key, None)
        if binding is not None:
            binding.binding_end = completion_time
            for res_id in list(binding.bound_resources):
                if self.bound_resource_usage.get(res_id) == key:
                    del self.bound_resource_usage[res_id]

    def _activate_pending_instances(self):
        """检查等待依赖的实例，依赖满足后立即激活"""
        if not self.pending_dependency_instances:
            return

        pending_keys = list(self.pending_dependency_instances)
        for key in pending_keys:
            instance = self.task_instances.get(key)
            if not instance:
                self.pending_dependency_instances.discard(key)
                continue

            task = self.tasks.get(instance.task_id)
            if not task:
                self.pending_dependency_instances.discard(key)
                continue

            if self._dependencies_satisfied(task, instance.instance_id):
                instance.dependencies_ready = True
                self.pending_dependency_instances.discard(key)
    
    def _process_events(self, use_segment_mode: bool):
        """处理当前时间的所有事件"""
        while self.next_events and self.next_events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.next_events)
            
            if event_type == "launch":
                self._handle_launch_event(event_data)
            elif event_type == "completion":
                self._handle_completion_event(event_data, use_segment_mode)
    
    def _handle_launch_event(self, event: LaunchEvent):
        """处理任务发射事件"""
        if event.task_id not in self.tasks:
            return

        task = self.tasks[event.task_id]

        # 应用分段策略
        segments = self._prepare_segments(task)

        # 创建任务实例
        instance = TaskInstance(
            task_id=event.task_id,
            instance_id=event.instance_id,
            segments=segments,
            priority=task.priority,
            start_time=self.current_time
        )

        # 检查依赖是否满足
        if task.dependencies:
            if self._dependencies_satisfied(task, event.instance_id):
                instance.dependencies_ready = True
            else:
                instance.dependencies_ready = False
                self.pending_dependency_instances.add(instance.instance_key)
        else:
            instance.dependencies_ready = True

        self.task_instances[instance.instance_key] = instance

        if ENABLE_EXECUTION_LOG:
            print(f"{self.current_time:>8.1f}ms: [LAUNCH] {event.task_id}#{event.instance_id} "
                  f"({len(segments)} segments, priority={task.priority.name})")
    
    def _handle_completion_event(self, completion: SegmentCompletion, use_segment_mode: bool):
        """处理段完成事件"""
        instance_key = (completion.task_id, completion.instance_id)
        if instance_key not in self.task_instances:
            return
        
        instance = self.task_instances[instance_key]
        
        # 标记段完成
        instance.mark_segment_completed(completion.segment_index)
        
        if ENABLE_EXECUTION_LOG:
            print(f"{completion.completion_time:>8.1f}ms: [COMPLETE] "
                  f"{completion.task_id}#{completion.instance_id}_seg{completion.segment_index}")
        
        # 检查任务是否全部完成
        if instance.is_completed and instance.completion_time is None:
            instance.completion_time = completion.completion_time
            if ENABLE_EXECUTION_LOG:
                print(f"{completion.completion_time:>8.1f}ms: [TASK_COMPLETE] "
                      f"{completion.task_id}#{completion.instance_id}")

            # 记录完成并尝试释放依赖
            self._register_instance_completion(instance)
            self._activate_pending_instances()
    
    def _schedule_segments_traditional(self):
        """传统模式调度：按顺序执行段"""
        for instance in self.task_instances.values():
            if instance.is_completed:
                continue
            
            # 只调度当前段
            if instance.can_enqueue_segment(instance.current_segment_index):
                self._try_enqueue_segment(instance, instance.current_segment_index)
    
    def _schedule_segments_smart(self):
        """段级模式调度：智能调度所有可用段"""
        for instance in self.task_instances.values():
            if instance.is_completed:
                continue
            
            # 尝试调度所有可以入队的段（但避免重复）
            for seg_idx in range(len(instance.segments)):
                if instance.can_enqueue_segment(seg_idx):
                    self._try_enqueue_segment(instance, seg_idx)
    
    def _try_enqueue_segment(self, instance: TaskInstance, segment_index: int):
        """尝试将指定段加入队列"""
        if segment_index >= len(instance.segments):
            return
        
        segment = instance.segments[segment_index]
        segment_id = f"{instance.task_id}#{instance.instance_id}_seg{segment_index}"
        
        # 避免重复处理
        if segment_id in self.processed_segments:
            return
        
        # 寻找可用资源
        queue = self._get_queue_for_segment(instance, segment)
        if not queue:
            return
        
        task_def = self.tasks.get(instance.task_id)
        task_name = task_def.name if task_def else None

        base_ready_time = self.current_time
        ready_time = base_ready_time
        jitter_ms = 0.0
        if segment_index == 0 and self.random_slack_enabled and self._slack_rng:
            jitter = self._slack_rng.gauss(0.0, self.random_slack_std)
            ready_time = max(0.0, base_ready_time + jitter)
            jitter_ms = ready_time - base_ready_time

        # 加入队列
        queue.enqueue(
            segment_id,
            instance.priority,
            ready_time,
            [segment]
        )
        
        # 标记已入队
        instance.mark_segment_enqueued(segment_index)
        self.processed_segments.add(segment_id)
        
        # 记录入队
        self.tracer.record_enqueue(
            segment_id,
            queue.resource_id,
            instance.priority,
            ready_time,
            [segment],
            original_task_id=instance.task_id,
            task_name=task_name,
            instance_id=instance.instance_id,
            segment_index=segment_index,
            jitter_ms=jitter_ms if segment_index == 0 else None
        )
        
        if ENABLE_EXECUTION_LOG:
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
        
        full_task_id = queued_task.task_id
        task_instance_part = full_task_id
        segment_index = 0
        if "_seg" in full_task_id:
            prefix, suffix = full_task_id.split("_seg", 1)
            task_instance_part = prefix
            digits = ""
            for ch in suffix:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                segment_index = int(digits)
        
        task_id = task_instance_part
        instance_id = 0
        if "#" in task_instance_part:
            parts = task_instance_part.split("#", 1)
            task_id = parts[0]
            digits = ""
            for ch in parts[1]:
                if ch.isdigit():
                    digits += ch
                else:
                    break
            if digits:
                instance_id = int(digits)
        
        task_def = self.tasks.get(task_id)
        task_name = task_def.name if task_def else None

        binding_key = (task_id, instance_id)
        binding = self.resource_bindings.get(binding_key)
        complement_queue = None
        complement_type: Optional[ResourceType] = None
        if task_def and task_def.runtime_type == RuntimeType.DSP_RUNTIME and binding:
            complement_type = ResourceType.DSP if segment.resource_type == ResourceType.NPU else ResourceType.NPU
            complement_id = binding.resource_map.get(complement_type)
            if complement_id:
                complement_queue = self.queue_manager.get_queue(complement_id)
            if complement_queue and complement_queue.resource_id != queue.resource_id and complement_queue.is_busy():
                return
        
        # 执行任务
        end_time = queue.execute_task(queued_task, self.current_time)

        if task_def and task_def.runtime_type == RuntimeType.DSP_RUNTIME and binding and complement_type is not None:
            complement_id = binding.resource_map.get(complement_type)
            if complement_id:
                bound_queue = self.queue_manager.get_queue(complement_id)
                if bound_queue and bound_queue.resource_id != queue.resource_id:
                    self._reserve_bound_resource(bound_queue, binding_key, end_time)
        
        # 记录执行
        self.tracer.record_execution(
            queued_task.task_id,
            queue.resource_id,
            self.current_time,
            end_time,
            queue.bandwidth,
            segment.sub_id,
            original_task_id=task_id,
            task_name=task_name,
            instance_id=instance_id,
            segment_index=segment_index
        )
        
        if ENABLE_EXECUTION_LOG:
            print(f"{self.current_time:>8.1f}ms: [EXECUTE] {queued_task.task_id} on {queue.resource_id} "
                  f"(duration={end_time - self.current_time:.1f}ms, priority={queued_task.priority.name})")
        
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
        
        # 从队列移除
        queue.dequeue_task(queued_task.task_id, queued_task.priority)
    
    def _prepare_segments(self, task: NNTask) -> List[SubSegment]:
        """准备任务的执行段"""
        # 应用分段策略
        sub_segments = task.apply_segmentation()
        if sub_segments:
            return sub_segments
        else:
            return task.segments
    
    def _advance_time(self, max_time: float):
        """推进仿真时间"""
        # 找下一个事件时间
        next_time = max_time
        
        if self.next_events:
            next_time = min(next_time, self.next_events[0][0])
        
        # 检查资源的下次可用时间
        for queue in self.queue_manager.resource_queues.values():
            if queue.is_busy():
                next_time = min(next_time, queue.busy_until)
            # 考虑等待中的任务最早就绪时间（处理ready_time扰动）
            next_ready: Optional[float] = None
            for priority_queue in queue.priority_queues.values():
                for queued_task in priority_queue:
                    if queued_task.ready_time > self.current_time:
                        if next_ready is None or queued_task.ready_time < next_ready:
                            next_ready = queued_task.ready_time
            if next_ready is not None:
                next_time = min(next_time, next_ready)
        
        # 推进时间
        if next_time > self.current_time:
            self.current_time = min(next_time, max_time)
    
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
    executor = ScheduleExecutor(queue_manager, tracer, tasks)
    if mode == "segment_aware":
        executor.segment_mode = True
    return executor


def set_execution_log_enabled(enabled: bool):
    """设置执行日志开关
    
    Args:
        enabled: True 启用日志，False 禁用日志（默认）
    """
    global ENABLE_EXECUTION_LOG
    ENABLE_EXECUTION_LOG = enabled
