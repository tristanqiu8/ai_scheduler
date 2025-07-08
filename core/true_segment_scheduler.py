#!/usr/bin/env python3
"""
真正的段级调度器 - 解决当前实现的问题
核心改进：
1. 段按需发射，不是一次性全部入队
2. 智能资源分配，不再严格FIFO
3. 支持跨任务的段并行
"""

from typing import Dict, List, Set, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import heapq

from core.models import SubSegment
from core.enums import ResourceType, TaskPriority
from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import LaunchPlan, LaunchEvent
from core.task import NNTask


@dataclass
class SmartSegmentInstance:
    """智能段实例"""
    task_id: str
    instance_id: int
    segment_index: int
    segment: SubSegment
    priority: TaskPriority
    
    # 状态
    state: str = "pending"  # pending -> ready -> running -> completed
    
    # 依赖管理
    depends_on: Optional[Tuple[str, int, int]] = None  # 前序段
    
    # 执行信息
    earliest_start_time: float = 0.0  # 最早可开始时间
    actual_start_time: Optional[float] = None
    completion_time: Optional[float] = None
    resource_id: Optional[str] = None
    
    @property
    def key(self) -> Tuple[str, int, int]:
        return (self.task_id, self.instance_id, self.segment_index)
    
    @property
    def full_id(self) -> str:
        return f"{self.task_id}#{self.instance_id}_seg{self.segment_index}"


class TrueSegmentScheduler:
    """真正的段级调度器
    
    核心特性：
    1. JIT（Just-In-Time）段发射
    2. 资源感知的智能调度
    3. 最小化资源空闲时间
    """
    
    def __init__(self, queue_manager: ResourceQueueManager, 
                 tracer: ScheduleTracer,
                 tasks: Dict[str, NNTask]):
        self.queue_manager = queue_manager
        self.tracer = tracer
        self.tasks = tasks
        
        # 段管理
        self.all_segments: Dict[Tuple[str, int, int], SmartSegmentInstance] = {}
        self.pending_segments: Set[Tuple[str, int, int]] = set()
        self.ready_segments: List[SmartSegmentInstance] = []
        
        # 资源状态追踪
        self.resource_next_free: Dict[str, float] = {}
        for res_id in queue_manager.resource_queues:
            self.resource_next_free[res_id] = 0.0
        
        # 任务实例追踪
        self.task_instances: Dict[Tuple[str, int], float] = {}  # 发射时间
        
        # 时间和事件
        self.current_time = 0.0
        self.events: List[Tuple[float, str, any]] = []
        
    def execute_plan(self, launch_plan: LaunchPlan, max_time: float) -> Dict:
        """执行计划"""
        print(f"\n{'='*80}")
        print(f"真正的段级调度执行 (max_time={max_time}ms)")
        print(f"{'='*80}\n")
        
        self._reset_state()
        
        # 添加任务发射事件
        for event in launch_plan.events:
            heapq.heappush(self.events, (event.time, "launch_task", event))
        
        # 主循环
        while self.current_time < max_time and self._has_work():
            # 处理事件
            self._process_events()
            
            # 智能调度
            self._smart_schedule()
            
            # 时间推进
            self._advance_time(max_time)
        
        return self._get_stats()
    
    def _reset_state(self):
        """重置状态"""
        self.all_segments.clear()
        self.pending_segments.clear()
        self.ready_segments.clear()
        self.task_instances.clear()
        self.current_time = 0.0
        self.events.clear()
        for res_id in self.resource_next_free:
            self.resource_next_free[res_id] = 0.0
    
    def _process_events(self):
        """处理事件"""
        while self.events and self.events[0][0] <= self.current_time:
            event_time, event_type, event_data = heapq.heappop(self.events)
            
            if event_type == "launch_task":
                self._handle_task_launch(event_data)
            elif event_type == "segment_complete":
                self._handle_segment_completion(event_data)
    
    def _handle_task_launch(self, event: LaunchEvent):
        """处理任务发射 - 只创建段，不立即入队"""
        task = self.tasks[event.task_id]
        self.task_instances[(event.task_id, event.instance_id)] = self.current_time
        
        print(f"{self.current_time:>8.1f}ms: [LAUNCH TASK] {event.task_id}#{event.instance_id}")
        
        # 准备段
        segments = self._prepare_segments(task)
        
        # 创建段实例
        for i, segment in enumerate(segments):
            seg_instance = SmartSegmentInstance(
                task_id=event.task_id,
                instance_id=event.instance_id,
                segment_index=i,
                segment=segment,
                priority=task.priority,
                earliest_start_time=self.current_time
            )
            
            # 设置依赖
            if i > 0:
                seg_instance.depends_on = (event.task_id, event.instance_id, i-1)
            else:
                seg_instance.state = "ready"
                self.ready_segments.append(seg_instance)
            
            self.all_segments[seg_instance.key] = seg_instance
            if seg_instance.state == "pending":
                self.pending_segments.add(seg_instance.key)
        
        print(f"          创建 {len(segments)} 个段，第一段就绪")
    
    def _smart_schedule(self):
        """智能调度 - 核心算法"""
        # 1. 更新资源状态
        for res_id, queue in self.queue_manager.resource_queues.items():
            queue.advance_time(self.current_time)
            if not queue.is_busy():
                self.resource_next_free[res_id] = self.current_time
        
        # 2. 检查pending段是否可以变为ready
        newly_ready = []
        for seg_key in list(self.pending_segments):
            seg = self.all_segments[seg_key]
            if seg.depends_on:
                dep_seg = self.all_segments.get(seg.depends_on)
                if dep_seg and dep_seg.state == "completed":
                    seg.state = "ready"
                    seg.earliest_start_time = max(self.current_time, dep_seg.completion_time)
                    newly_ready.append(seg)
                    self.pending_segments.remove(seg_key)
        
        self.ready_segments.extend(newly_ready)
        
        # 3. 为每个空闲资源选择最佳段
        scheduled = set()
        
        for res_id, queue in self.queue_manager.resource_queues.items():
            if queue.is_busy():
                continue
            
            # 找到该资源类型的所有就绪段
            candidates = [seg for seg in self.ready_segments 
                         if seg.segment.resource_type == queue.resource_type 
                         and seg.key not in scheduled]
            
            if not candidates:
                continue
            
            # 选择最佳段（考虑优先级和等待时间）
            best_seg = self._select_best_segment(candidates)
            if best_seg:
                self._execute_segment(best_seg, queue)
                scheduled.add(best_seg.key)
        
        # 移除已调度的段
        self.ready_segments = [seg for seg in self.ready_segments if seg.key not in scheduled]
    
    def _select_best_segment(self, candidates: List[SmartSegmentInstance]) -> Optional[SmartSegmentInstance]:
        """选择最佳段执行
        
        改进策略：
        1. 优先级最高
        2. 执行时间短的优先（SJF - Shortest Job First）
        3. 可以释放其他资源的段优先
        4. 等待时间最长
        """
        if not candidates:
            return None
        
        def score(seg):
            priority_score = {
                TaskPriority.CRITICAL: 10000,
                TaskPriority.HIGH: 1000,
                TaskPriority.NORMAL: 100,
                TaskPriority.LOW: 10
            }[seg.priority]
            
            # 获取段的预估执行时间
            bandwidth = 60.0 if seg.segment.resource_type == ResourceType.NPU else 40.0
            duration = seg.segment.get_duration(bandwidth)
            
            # 短任务优先（SJF）- 反转时间作为分数
            sjf_score = 100 / (duration + 1)  # 避免除零
            
            # 检查是否能释放其他资源
            task_key = (seg.task_id, seg.instance_id)
            next_seg_idx = seg.segment_index + 1
            release_score = 0
            
            # 如果这个段完成后，下一个段使用不同资源，给予加分
            for s in self.all_segments.values():
                if (s.task_id, s.instance_id) == task_key and s.segment_index == next_seg_idx:
                    if s.segment.resource_type != seg.segment.resource_type:
                        release_score = 50  # 可以释放当前资源给其他任务
                    break
            
            wait_time = self.current_time - seg.earliest_start_time
            
            # 综合评分
            total_score = priority_score + sjf_score * 10 + release_score + wait_time
            
            return total_score
        
        return max(candidates, key=score)
    
    def _execute_segment(self, seg: SmartSegmentInstance, queue):
        """执行段"""
        duration = seg.segment.get_duration(queue.bandwidth)
        end_time = self.current_time + duration
        
        seg.state = "running"
        seg.actual_start_time = self.current_time
        seg.resource_id = queue.resource_id
        
        # 记录执行
        self.tracer.record_execution(
            seg.full_id,
            queue.resource_id,
            self.current_time,
            end_time,
            queue.bandwidth,
            seg.segment.sub_id
        )
        
        # 更新队列
        queue.busy_until = end_time
        queue.current_task = seg.full_id
        self.resource_next_free[queue.resource_id] = end_time
        
        print(f"{self.current_time:>8.1f}ms: [EXECUTE] {seg.full_id} on {queue.resource_id} "
              f"(duration={duration:.1f}ms, priority={seg.priority.name})")
        
        # 添加完成事件
        heapq.heappush(self.events, (end_time, "segment_complete", seg.key))
    
    def _handle_segment_completion(self, seg_key: Tuple[str, int, int]):
        """处理段完成"""
        seg = self.all_segments.get(seg_key)
        if not seg:
            return
        
        seg.state = "completed"
        seg.completion_time = self.current_time
        
        # 释放资源
        if seg.resource_id:
            queue = self.queue_manager.get_queue(seg.resource_id)
            if queue:
                queue.busy_until = self.current_time
                queue.current_task = None
        
        print(f"{self.current_time:>8.1f}ms: [COMPLETE] {seg.full_id}")
        
        # 检查任务是否完成
        task_key = (seg.task_id, seg.instance_id)
        task_segments = [s for s in self.all_segments.values() 
                        if (s.task_id, s.instance_id) == task_key]
        
        if all(s.state == "completed" for s in task_segments):
            launch_time = self.task_instances.get(task_key, 0)
            total_time = self.current_time - launch_time
            print(f"{self.current_time:>8.1f}ms: [FINISHED] {seg.task_id}#{seg.instance_id} "
                  f"(total: {total_time:.1f}ms)")
    
    def _has_work(self) -> bool:
        """是否还有工作"""
        return (bool(self.events) or 
                bool(self.ready_segments) or
                bool(self.pending_segments) or
                any(s.state == "running" for s in self.all_segments.values()))
    
    def _advance_time(self, max_time: float):
        """时间推进"""
        next_time = max_time
        
        # 下一个事件
        if self.events:
            next_time = min(next_time, self.events[0][0])
        
        # 下一个资源释放
        for res_id, free_time in self.resource_next_free.items():
            if free_time > self.current_time:
                next_time = min(next_time, free_time)
        
        # 推进
        if next_time > self.current_time:
            self.current_time = min(next_time, max_time)
    
    def _prepare_segments(self, task: NNTask) -> List[SubSegment]:
        """准备段"""
        sub_segments = task.apply_segmentation()
        
        if not sub_segments:
            sub_segments = []
            for i, seg in enumerate(task.segments):
                sub_seg = SubSegment(
                    sub_id=f"{seg.segment_id}",
                    resource_type=seg.resource_type,
                    duration_table=seg.duration_table,
                    cut_overhead=0.0,
                    original_segment_id=seg.segment_id
                )
                sub_segments.append(sub_seg)
        
        return sub_segments
    
    def _get_stats(self) -> Dict:
        """获取统计"""
        completed_tasks = set()
        total_segments = len(self.all_segments)
        completed_segments = sum(1 for s in self.all_segments.values() if s.state == "completed")
        
        for seg in self.all_segments.values():
            if seg.state == "completed":
                task_key = (seg.task_id, seg.instance_id)
                task_segs = [s for s in self.all_segments.values() 
                            if (s.task_id, s.instance_id) == task_key]
                if all(s.state == "completed" for s in task_segs):
                    completed_tasks.add(task_key)
        
        # 计算资源利用率
        total_busy_time = defaultdict(float)
        for exec in self.tracer.executions:
            total_busy_time[exec.resource_id] += (exec.end_time - exec.start_time)
        
        utilization = {}
        for res_id in self.queue_manager.resource_queues:
            if self.current_time > 0:
                utilization[res_id] = (total_busy_time[res_id] / self.current_time) * 100
            else:
                utilization[res_id] = 0.0
        
        return {
            'total_instances': len(self.task_instances),
            'completed_instances': len(completed_tasks),
            'total_segments': total_segments,
            'completed_segments': completed_segments,
            'current_time': self.current_time,
            'resource_utilization': utilization
        }
