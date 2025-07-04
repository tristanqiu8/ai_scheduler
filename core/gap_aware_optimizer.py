#!/usr/bin/env python3
"""
空隙感知优化器 - 贪心算法版本
基于无冲突的原始调度，逐个将NPU段插入前面的空隙
"""

from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass
import copy

from .enums import ResourceType, TaskPriority
from .task import NNTask
from .models import TaskScheduleInfo, SubSegment
from .scheduler import MultiResourceScheduler


@dataclass
class ResourceGap:
    """资源空隙定义"""
    resource_id: str
    resource_type: ResourceType
    start_time: float
    end_time: float
    duration: float


@dataclass 
class SegmentCandidate:
    """待优化的段候选"""
    task_id: str
    task: NNTask
    segment: SubSegment
    original_start: float
    original_end: float
    event_index: int  # 原始调度中的事件索引


class GapAwareOptimizer:
    """空隙感知优化器 - 贪心算法实现"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        self.original_schedule = None  # 保存原始调度
        
    def optimize_schedule(self, time_window: float = 100.0) -> List[TaskScheduleInfo]:
        """使用贪心算法优化调度"""
        print("\n🔍 执行贪心空隙优化...")
        
        # 1. 保存原始调度作为基准
        self.original_schedule = copy.deepcopy(self.scheduler.schedule_history)
        print(f"  原始调度事件数: {len(self.original_schedule)}")
        
        # 2. 收集所有可移动的NPU段（按时间顺序）
        npu_segments = self._collect_npu_segments()
        print(f"  找到 {len(npu_segments)} 个NPU段")
        
        # 3. 初始化工作调度（复制原始调度）
        working_schedule = copy.deepcopy(self.original_schedule)
        
        # 4. 贪心算法：逐个尝试将段插入更早的位置
        optimized_count = 0
        for candidate in npu_segments:
            print(f"\n  处理: {candidate.task_id}.{candidate.segment.sub_id} "
                  f"@ {candidate.original_start:.1f}ms")
            
            # 寻找可用的空隙
            best_gap = self._find_best_gap_for_segment(
                candidate, 
                working_schedule, 
                time_window
            )
            
            if best_gap and best_gap['new_start'] < candidate.original_start - 0.01:
                # 执行移动
                print(f"    ✓ 移动到 {best_gap['new_start']:.1f}ms (提前 "
                      f"{candidate.original_start - best_gap['new_start']:.1f}ms)")
                
                working_schedule = self._move_segment(
                    working_schedule,
                    candidate,
                    best_gap
                )
                optimized_count += 1
                
                # 级联移动：将后续的所有事件向前压缩
                working_schedule = self._cascade_compress(working_schedule, time_window)
            else:
                print(f"    ✗ 无法优化")
        
        # 5. 更新调度器的调度历史
        self.scheduler.schedule_history = working_schedule
        print(f"\n✅ 优化完成，共优化 {optimized_count} 个段")
        
        return working_schedule
    
    def _collect_npu_segments(self) -> List[SegmentCandidate]:
        """收集所有NPU段，按时间顺序"""
        candidates = []
        seen = set()  # 避免重复
        
        for event_idx, event in enumerate(self.original_schedule):
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
            
            # 从子段调度中提取NPU段
            if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                for sub_seg_id, start, end in event.sub_segment_schedule:
                    # 创建唯一标识符
                    segment_key = (event.task_id, sub_seg_id, start)
                    if segment_key in seen:
                        continue
                    seen.add(segment_key)
                    
                    # 找到对应的段对象
                    for seg in task.get_sub_segments_for_scheduling():
                        if (seg.sub_id == sub_seg_id and 
                            seg.resource_type == ResourceType.NPU):
                            candidates.append(SegmentCandidate(
                                task_id=task.task_id,
                                task=task,
                                segment=seg,
                                original_start=start,
                                original_end=end,
                                event_index=event_idx
                            ))
                            break
        
        # 按开始时间排序
        candidates.sort(key=lambda c: c.original_start)
        
        print("\n  [DEBUG] 收集到的NPU段:")
        for c in candidates:
            print(f"    {c.task_id}.{c.segment.sub_id} @ {c.original_start:.1f}-{c.original_end:.1f}ms")
        
        return candidates
    
    def _find_best_gap_for_segment(self, candidate: SegmentCandidate, 
                                   schedule: List[TaskScheduleInfo],
                                   time_window: float) -> Optional[Dict]:
        """为段找到最佳空隙"""
        # 获取段的执行时长
        resource = next((r for r in self.scheduler.resources[ResourceType.NPU] 
                        if r.unit_id == "NPU_0"), None)
        if not resource:
            print(f"    [DEBUG] 找不到NPU_0资源")
            return None
        
        duration = candidate.segment.get_duration(resource.bandwidth)
        print(f"    [DEBUG] 段时长: {duration}ms")
        
        # 构建NPU占用时间线（排除当前段）
        npu_timeline = []
        for event in schedule:
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    # 跳过当前段本身
                    if (event.task_id == candidate.task_id and 
                        sub_id == candidate.segment.sub_id and
                        abs(start - candidate.original_start) < 0.01):
                        continue
                    
                    # 检查是否是NPU事件
                    # 问题可能在这里！需要检查具体的资源类型
                    task = self.scheduler.tasks.get(event.task_id)
                    if task:
                        # 找到对应的子段
                        for seg in task.get_sub_segments_for_scheduling():
                            if seg.sub_id == sub_id and seg.resource_type == ResourceType.NPU:
                                npu_timeline.append((start, end))
                                print(f"    [DEBUG] NPU占用: {start:.1f}-{end:.1f}ms ({event.task_id})")
                                break
        
        # 排序时间线
        npu_timeline.sort()
        print(f"    [DEBUG] NPU时间线: {npu_timeline}")
        
        # 寻找空隙
        earliest_start = 0.0
        print(f"    [DEBUG] 寻找空隙...")
        
        for i, (occ_start, occ_end) in enumerate(npu_timeline):
            print(f"    [DEBUG] 检查空隙: {earliest_start:.1f}-{occ_start:.1f}ms")
            
            # 检查当前位置是否能容纳段
            if earliest_start + duration <= occ_start:
                print(f"    [DEBUG] 空隙足够大！")
                # 检查顺序约束
                if self._check_segment_order_constraint(
                    candidate, earliest_start, schedule
                ):
                    print(f"    [DEBUG] 顺序约束通过！")
                    return {
                        'new_start': earliest_start,
                        'new_end': earliest_start + duration,
                        'gap_before_event': i
                    }
                else:
                    print(f"    [DEBUG] 顺序约束失败")
            
            # 更新下一个可能的开始时间
            earliest_start = max(earliest_start, occ_end)
        
        # 检查最后的空间
        print(f"    [DEBUG] 检查最后空隙: {earliest_start:.1f}ms之后")
        if (earliest_start + duration <= time_window and
            earliest_start < candidate.original_start - 0.01):
            if self._check_segment_order_constraint(
                candidate, earliest_start, schedule
            ):
                return {
                    'new_start': earliest_start,
                    'new_end': earliest_start + duration,
                    'gap_before_event': len(npu_timeline)
                }
        
        print(f"    [DEBUG] 未找到合适空隙")
        return None
    
    def _check_segment_order_constraint(self, candidate: SegmentCandidate,
                                       new_start: float,
                                       schedule: List[TaskScheduleInfo]) -> bool:
        """检查段顺序约束（同一任务内的段必须保持顺序）"""
        # 找到同任务的其他段
        for event in schedule:
            if event.task_id != candidate.task_id:
                continue
            
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    # 获取段编号进行比较
                    if sub_id == candidate.segment.sub_id:
                        continue
                    
                    # 提取段索引 (main_0, main_1, main_2)
                    try:
                        candidate_idx = int(candidate.segment.sub_id.split('_')[-1])
                        other_idx = int(sub_id.split('_')[-1])
                        
                        # 如果是前面的段，新位置必须在它之后
                        if other_idx < candidate_idx and end > new_start:
                            return False
                        
                        # 如果是后面的段，新位置必须在它之前
                        if other_idx > candidate_idx and start < new_start + 0.01:
                            return False
                    except:
                        pass
        
        return True
    
    def _move_segment(self, schedule: List[TaskScheduleInfo], 
                     candidate: SegmentCandidate,
                     gap_info: Dict) -> List[TaskScheduleInfo]:
        """移动段到新位置"""
        new_schedule = []
        
        for event in schedule:
            if event.task_id == candidate.task_id:
                # 这是包含要移动段的事件
                new_sub_schedules = []
                
                for sub_id, start, end in event.sub_segment_schedule:
                    if sub_id == candidate.segment.sub_id:
                        # 这是要移动的段，使用新时间
                        new_sub_schedules.append((
                            sub_id,
                            gap_info['new_start'],
                            gap_info['new_end']
                        ))
                    else:
                        # 其他段保持不变（暂时）
                        new_sub_schedules.append((sub_id, start, end))
                
                # 创建新事件
                new_event = TaskScheduleInfo(
                    task_id=event.task_id,
                    start_time=min(s[1] for s in new_sub_schedules),
                    end_time=max(s[2] for s in new_sub_schedules),
                    assigned_resources=event.assigned_resources,
                    actual_latency=event.actual_latency,
                    runtime_type=event.runtime_type,
                    sub_segment_schedule=new_sub_schedules
                )
                new_schedule.append(new_event)
            else:
                # 其他事件保持不变
                new_schedule.append(event)
        
        return new_schedule
    
    def _cascade_compress(self, schedule: List[TaskScheduleInfo], 
                         time_window: float) -> List[TaskScheduleInfo]:
        """级联压缩：将所有事件尽可能向前移动"""
        # 按开始时间排序
        sorted_schedule = sorted(schedule, key=lambda e: e.start_time)
        compressed = []
        
        # 跟踪每个资源的最早可用时间
        resource_available = defaultdict(float)
        
        for event in sorted_schedule:
            # 计算这个事件的最早开始时间
            earliest_start = 0.0
            
            # 考虑资源约束
            for res_type, res_id in event.assigned_resources.items():
                earliest_start = max(earliest_start, resource_available[res_id])
            
            # 考虑任务依赖
            task = self.scheduler.tasks[event.task_id]
            for dep_task_id in task.dependencies:
                # 找依赖任务的最后执行时间
                for prev_event in compressed:
                    if prev_event.task_id == dep_task_id:
                        earliest_start = max(earliest_start, prev_event.end_time)
            
            # 考虑同任务的前序事件
            for prev_event in compressed:
                if prev_event.task_id == event.task_id:
                    earliest_start = max(earliest_start, 
                                       prev_event.end_time + task.min_interval_ms)
            
            # 移动事件
            if earliest_start < event.start_time - 0.01:
                # 需要向前移动
                time_shift = earliest_start - event.start_time
                
                # 创建新事件
                new_sub_schedules = []
                if hasattr(event, 'sub_segment_schedule'):
                    for sub_id, start, end in event.sub_segment_schedule:
                        new_sub_schedules.append((
                            sub_id,
                            start + time_shift,
                            end + time_shift
                        ))
                
                new_event = TaskScheduleInfo(
                    task_id=event.task_id,
                    start_time=event.start_time + time_shift,
                    end_time=event.end_time + time_shift,
                    assigned_resources=event.assigned_resources,
                    actual_latency=event.actual_latency,
                    runtime_type=event.runtime_type,
                    sub_segment_schedule=new_sub_schedules
                )
                compressed.append(new_event)
                
                # 更新资源可用时间
                for res_type, res_id in new_event.assigned_resources.items():
                    resource_available[res_id] = new_event.end_time
            else:
                # 保持原位置
                compressed.append(event)
                
                # 更新资源可用时间
                for res_type, res_id in event.assigned_resources.items():
                    resource_available[res_id] = event.end_time
        
        return compressed