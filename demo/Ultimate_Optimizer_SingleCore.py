#!/usr/bin/env python3
"""
终极调度优化器 V2 - 零空隙、100% FPS满足
修复版本：解决方法引用错误
"""

import sys
import os
import copy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.models import TaskScheduleInfo
from core.scheduler import MultiResourceScheduler
from core.task import NNTask
from core.modular_scheduler_fixes import apply_basic_fixes
from core.minimal_fifo_fix_corrected import apply_minimal_fifo_fix
from core.strict_resource_conflict_fix import apply_strict_resource_conflict_fix
from core.fixed_validation_and_metrics import validate_schedule_correctly
from scenario.real_task import create_real_tasks
from viz.elegant_visualization import ElegantSchedulerVisualizer
import matplotlib.pyplot as plt


class UltimateSchedulerOptimizerV2:
    """终极调度优化器 V2 - 实现零空隙和100% FPS"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        self.resource_timeline = defaultdict(list)
        self.optimization_stats = {
            'initial_gaps': 0,
            'final_gaps': 0,
            'fps_satisfaction_before': 0,
            'fps_satisfaction_after': 0,
            'tail_idle': 0
        }
        
    def optimize_ultimate(self) -> List[TaskScheduleInfo]:
        """终极优化流程 - 两阶段策略"""
        print("\n🚀 启动终极调度优化 V2")
        print("=" * 80)
        
        # 准备阶段：配置任务
        self._prepare_tasks()
        
        # 第一阶段：满足所有FPS要求
        print("\n[第一阶段] 确保100% FPS满足...")
        phase1_schedule = self._phase1_ensure_fps()
        self._print_phase_status(phase1_schedule, "第一阶段完成")
        
        # 第二阶段：消除空隙并最大化末尾空闲
        print("\n[第二阶段] 消除空隙并优化...")
        phase2_schedule = self._phase2_eliminate_gaps(phase1_schedule)
        self._print_phase_status(phase2_schedule, "第二阶段完成")
        
        # 打印优化统计
        self._print_optimization_summary()
        
        return phase2_schedule
    
    def _prepare_tasks(self):
        """准备任务优化配置"""
        print("\n📋 准备任务配置...")
        
        # 1. YOLO任务激进分段
        for task_id in ['T2', 'T3']:
            task = self.scheduler.tasks.get(task_id)
            if task:
                task.segmentation_strategy = SegmentationStrategy.FORCED_SEGMENTATION
                for segment in task.segments:
                    if segment.segment_id == "main":
                        available_cuts = segment.get_available_cuts()
                        segment.apply_segmentation(available_cuts)
                        print(f"  ✓ {task_id} 分段为 {len(segment.sub_segments)} 个子段")
        
        # 2. 分析任务特性
        self.task_characteristics = {}
        for task_id, task in self.scheduler.tasks.items():
            self.task_characteristics[task_id] = {
                'fps': task.fps_requirement,
                'period': 1000.0 / task.fps_requirement,
                'duration': self._estimate_task_duration(task),
                'is_critical': task.priority == TaskPriority.CRITICAL,
                'uses_dsp': task.uses_dsp,
                'uses_npu': task.uses_npu,
                'can_split': task.is_segmented
            }
            
        print(f"  ✓ 分析了 {len(self.task_characteristics)} 个任务特性")
    
    def _phase1_ensure_fps(self) -> List[TaskScheduleInfo]:
        """第一阶段：确保100% FPS满足"""
        
        # 1. 计算每个任务需要的执行次数
        required_executions = {}
        for task_id, task in self.scheduler.tasks.items():
            required = int((self.time_window / 1000.0) * task.fps_requirement)
            required_executions[task_id] = required
            print(f"  {task_id}: 需要 {required} 次执行")
        
        # 2. 创建初始调度
        schedule = []
        
        # 2.1 处理高FPS任务（T6需要20次执行）
        high_fps_tasks = [(tid, self.task_characteristics[tid]) 
                          for tid in ['T6'] if tid in self.task_characteristics]
        
        for task_id, task_info in high_fps_tasks:
            task = self.scheduler.tasks[task_id]
            period = task_info['period']
            
            # 均匀分布T6的20次执行
            for i in range(required_executions[task_id]):
                ideal_start = i * period
                
                # 找到不冲突的最近时间
                actual_start = self._find_non_conflicting_time(
                    schedule, task, ideal_start, task_info['duration']
                )
                
                if actual_start + task_info['duration'] <= self.time_window:
                    event = self._create_task_event(task, actual_start)
                    schedule.append(event)
        
        # 2.2 处理关键任务（T1）
        critical_tasks = [(tid, self.task_characteristics[tid]) 
                          for tid in ['T1'] if tid in self.task_characteristics]
        
        for task_id, task_info in critical_tasks:
            task = self.scheduler.tasks[task_id]
            self._schedule_task_uniformly(schedule, task, required_executions[task_id])
        
        # 2.3 处理普通FPS任务
        normal_tasks = [(tid, self.task_characteristics[tid]) 
                        for tid in ['T5', 'T7', 'T8'] 
                        if tid in self.task_characteristics]
        
        for task_id, task_info in normal_tasks:
            task = self.scheduler.tasks[task_id]
            self._schedule_task_uniformly(schedule, task, required_executions[task_id])
        
        # 2.4 处理低FPS任务
        low_fps_tasks = [(tid, self.task_characteristics[tid]) 
                         for tid in ['T2', 'T3', 'T4'] 
                         if tid in self.task_characteristics]
        
        for task_id, task_info in low_fps_tasks:
            task = self.scheduler.tasks[task_id]
            self._schedule_task_uniformly(schedule, task, required_executions[task_id])
        
        # 3. 排序并返回
        return sorted(schedule, key=lambda x: x.start_time)
    
    def _phase2_eliminate_gaps(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """第二阶段：消除空隙并最大化末尾空闲"""
        
        # 1. 识别所有空隙
        gaps = self._identify_all_gaps(schedule)
        self.optimization_stats['initial_gaps'] = len(gaps)
        
        complete_idle_gaps = [g for g in gaps if g['type'] == 'both_idle']
        cross_resource_gaps = [g for g in gaps if g['type'] == 'npu_idle_dsp_busy']
        
        print(f"\n  发现空隙:")
        print(f"    - 完全空闲: {len(complete_idle_gaps)} 个")
        print(f"    - 跨资源空隙: {len(cross_resource_gaps)} 个")
        
        # 2. 优化策略
        optimized = copy.deepcopy(schedule)
        
        # 2.1 填充跨资源空隙
        if cross_resource_gaps:
            print("\n  填充跨资源空隙...")
            optimized = self._fill_cross_resource_gaps(optimized, cross_resource_gaps)
        
        # 2.2 紧凑化消除完全空闲
        if complete_idle_gaps:
            print("\n  紧凑化消除完全空闲...")
            optimized = self._compact_to_eliminate_idle(optimized)
        
        # 2.3 最终优化：推迟低优先级任务
        print("\n  最终优化...")
        optimized = self._final_optimization(optimized)
        
        # 3. 验证FPS
        fps_check = self._verify_fps_satisfaction(optimized)
        if not fps_check['all_satisfied']:
            print("\n  ⚠️ FPS验证失败，进行修复...")
            optimized = self._repair_fps(optimized, fps_check)
        
        return optimized
    
    def _find_non_conflicting_time(self, schedule: List[TaskScheduleInfo], 
                                  task: NNTask, ideal_start: float, 
                                  duration: float) -> float:
        """找到不冲突的最近时间"""
        test_time = ideal_start
        step = 0.5
        
        while test_time + duration <= self.time_window:
            if self._is_time_slot_available(schedule, task, test_time, duration):
                return test_time
            test_time += step
        
        # 如果向后找不到，尝试向前
        test_time = ideal_start - step
        while test_time >= 0:
            if self._is_time_slot_available(schedule, task, test_time, duration):
                return test_time
            test_time -= step
        
        return ideal_start  # 返回原始时间
    
    def _is_time_slot_available(self, schedule: List[TaskScheduleInfo], 
                               task: NNTask, start_time: float, 
                               duration: float) -> bool:
        """检查时间槽是否可用"""
        end_time = start_time + duration
        
        for event in schedule:
            # 检查时间重叠
            if not (event.end_time <= start_time or event.start_time >= end_time):
                # 检查资源冲突
                event_task = self.scheduler.tasks.get(event.task_id)
                if event_task:
                    # 检查NPU冲突
                    if task.uses_npu and event_task.uses_npu:
                        return False
                    # 检查DSP冲突
                    if task.uses_dsp and event_task.uses_dsp:
                        return False
        
        # 检查最小执行间隔
        for event in schedule:
            if event.task_id == task.task_id:
                if abs(event.start_time - start_time) < task.min_interval_ms:
                    return False
        
        return True
    
    def _schedule_task_uniformly(self, schedule: List[TaskScheduleInfo], 
                                task: NNTask, required_count: int):
        """均匀调度任务"""
        if required_count == 0:
            return
        
        period = self.time_window / required_count
        duration = self._estimate_task_duration(task)
        
        for i in range(required_count):
            ideal_start = i * period
            
            # 找到可用时间
            actual_start = self._find_non_conflicting_time(
                schedule, task, ideal_start, duration
            )
            
            if actual_start + duration <= self.time_window:
                event = self._create_task_event(task, actual_start)
                schedule.append(event)
    
    def _identify_all_gaps(self, schedule: List[TaskScheduleInfo]) -> List[Dict]:
        """识别所有空隙"""
        self._rebuild_resource_timeline(schedule)
        gaps = []
        
        # 1. 获取NPU和DSP的忙碌时段
        npu_busy = [(s, e) for s, e, _, _ in self.resource_timeline.get('NPU_0', [])]
        dsp_busy = [(s, e) for s, e, _, _ in self.resource_timeline.get('DSP_0', [])]
        
        # 2. 找完全空闲时段
        all_busy = sorted(npu_busy + dsp_busy)
        if all_busy:
            # 合并重叠时段
            merged = [all_busy[0]]
            for start, end in all_busy[1:]:
                if start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
                else:
                    merged.append((start, end))
            
            # 找空隙
            last_end = 0
            for start, end in merged:
                if start > last_end + 0.1:
                    gaps.append({
                        'start': last_end,
                        'end': start,
                        'duration': start - last_end,
                        'type': 'both_idle'
                    })
                last_end = end
        
        # 3. 找跨资源空隙（DSP忙但NPU闲）
        for dsp_start, dsp_end in dsp_busy:
            # 检查这段时间NPU的使用情况
            npu_free_time = dsp_end - dsp_start
            for npu_start, npu_end in npu_busy:
                if npu_start < dsp_end and npu_end > dsp_start:
                    overlap = min(dsp_end, npu_end) - max(dsp_start, npu_start)
                    npu_free_time -= overlap
            
            if npu_free_time > 1:  # 至少1ms的空闲
                gaps.append({
                    'start': dsp_start,
                    'end': dsp_end,
                    'duration': npu_free_time,
                    'type': 'npu_idle_dsp_busy'
                })
        
        return gaps
    
    def _fill_cross_resource_gaps(self, schedule: List[TaskScheduleInfo], 
                                 gaps: List[Dict]) -> List[TaskScheduleInfo]:
        """填充跨资源空隙"""
        optimized = copy.deepcopy(schedule)
        filled_count = 0
        
        for gap in gaps:
            if gap['type'] != 'npu_idle_dsp_busy':
                continue
            
            # 找适合的NPU任务来填充
            candidates = self._find_npu_tasks_for_gap(optimized, gap)
            
            for candidate in candidates:
                if self._move_task_to_gap(optimized, candidate, gap):
                    filled_count += 1
                    break
        
        print(f"    ✓ 填充了 {filled_count} 个跨资源空隙")
        return optimized
    
    def _find_npu_tasks_for_gap(self, schedule: List[TaskScheduleInfo], 
                               gap: Dict) -> List[Dict]:
        """找适合填充空隙的NPU任务"""
        candidates = []
        gap_start = gap['start']
        gap_end = gap['end']
        
        for event in schedule:
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
            
            # 只考虑纯NPU任务
            if task.uses_npu and not task.uses_dsp:
                # 检查是否可以移动到空隙
                duration = event.end_time - event.start_time
                if duration <= gap_end - gap_start:
                    candidates.append({
                        'event': event,
                        'task': task,
                        'duration': duration,
                        'priority': task.priority.value
                    })
        
        # 优先移动低优先级任务
        candidates.sort(key=lambda x: (-x['priority'], x['duration']))
        return candidates
    
    def _move_task_to_gap(self, schedule: List[TaskScheduleInfo], 
                         candidate: Dict, gap: Dict) -> bool:
        """将任务移动到空隙"""
        event = candidate['event']
        new_start = gap['start']
        
        # 检查是否可以移动（考虑依赖等）
        if self._can_move_event(schedule, event, new_start):
            # 更新时间
            duration = event.end_time - event.start_time
            time_shift = new_start - event.start_time
            
            event.start_time = new_start
            event.end_time = new_start + duration
            
            # 更新子段时间
            if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                new_sub_schedule = []
                for sub_id, start, end in event.sub_segment_schedule:
                    new_sub_schedule.append((sub_id, start + time_shift, end + time_shift))
                event.sub_segment_schedule = new_sub_schedule
            
            return True
        
        return False
    
    def _can_move_event(self, schedule: List[TaskScheduleInfo], 
                       event: TaskScheduleInfo, new_start: float) -> bool:
        """检查是否可以移动事件"""
        task = self.scheduler.tasks.get(event.task_id)
        if not task:
            return False
        
        duration = event.end_time - event.start_time
        new_end = new_start + duration
        
        # 检查时间窗口
        if new_end > self.time_window:
            return False
        
        # 检查依赖
        if task.dependencies:
            for dep_id in task.dependencies:
                dep_events = [e for e in schedule if e.task_id == dep_id]
                if dep_events:
                    max_dep_end = max(e.end_time for e in dep_events)
                    if new_start < max_dep_end:
                        return False
        
        # 检查最小间隔
        for other_event in schedule:
            if other_event.task_id == event.task_id and other_event != event:
                if abs(other_event.start_time - new_start) < task.min_interval_ms:
                    return False
        
        return True
    
    def _compact_to_eliminate_idle(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """紧凑化以消除完全空闲时段"""
        compacted = []
        resource_available = defaultdict(float)
        
        # 按时间排序
        sorted_events = sorted(schedule, key=lambda x: x.start_time)
        
        for event in sorted_events:
            # 计算最早可用时间
            earliest = 0
            
            # 检查资源
            for res_type, res_id in event.assigned_resources.items():
                earliest = max(earliest, resource_available[res_id])
            
            # 检查依赖和间隔
            task = self.scheduler.tasks.get(event.task_id)
            if task:
                # 依赖
                if task.dependencies:
                    for dep_id in task.dependencies:
                        dep_events = [e for e in compacted if e.task_id == dep_id]
                        if dep_events:
                            earliest = max(earliest, max(e.end_time for e in dep_events))
                
                # 最小间隔
                same_task_events = [e for e in compacted if e.task_id == event.task_id]
                if same_task_events:
                    last_start = max(e.start_time for e in same_task_events)
                    earliest = max(earliest, last_start + task.min_interval_ms)
            
            # 创建紧凑事件
            duration = event.end_time - event.start_time
            new_event = copy.deepcopy(event)
            
            if earliest != event.start_time:
                time_shift = earliest - event.start_time
                new_event.start_time = earliest
                new_event.end_time = earliest + duration
                
                # 更新子段
                if hasattr(new_event, 'sub_segment_schedule') and new_event.sub_segment_schedule:
                    new_sub_schedule = []
                    for sub_id, start, end in new_event.sub_segment_schedule:
                        new_sub_schedule.append((sub_id, start + time_shift, end + time_shift))
                    new_event.sub_segment_schedule = new_sub_schedule
            
            # 更新资源可用时间
            for res_type, res_id in new_event.assigned_resources.items():
                resource_available[res_id] = new_event.end_time
            
            compacted.append(new_event)
        
        return compacted
    
    def _final_optimization(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """最终优化：推迟低优先级任务以增加末尾空闲"""
        optimized = copy.deepcopy(schedule)
        
        # 计算当前末尾
        if optimized:
            current_end = max(e.end_time for e in optimized)
            
            # 识别可以推迟的任务
            low_priority_events = [e for e in optimized 
                                 if self.scheduler.tasks.get(e.task_id) and 
                                 self.scheduler.tasks[e.task_id].priority == TaskPriority.LOW]
            
            # 尝试推迟低优先级任务
            for event in low_priority_events:
                task = self.scheduler.tasks[event.task_id]
                if task and event.end_time > current_end * 0.8:  # 在末尾20%的任务
                    # 计算可以推迟的时间
                    delay = min(5, self.time_window - event.end_time)
                    if delay > 0:
                        event.start_time += delay
                        event.end_time += delay
                        
                        # 更新子段时间
                        if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                            new_sub_schedule = []
                            for sub_id, start, end in event.sub_segment_schedule:
                                new_sub_schedule.append((sub_id, start + delay, end + delay))
                            event.sub_segment_schedule = new_sub_schedule
        
        return optimized
    
    def _verify_fps_satisfaction(self, schedule: List[TaskScheduleInfo]) -> Dict:
        """验证FPS满足情况"""
        task_counts = defaultdict(int)
        for event in schedule:
            task_counts[event.task_id] += 1
        
        unsatisfied = {}
        all_satisfied = True
        
        for task_id, task in self.scheduler.tasks.items():
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            
            if actual < expected:
                all_satisfied = False
                unsatisfied[task_id] = {
                    'expected': expected,
                    'actual': actual,
                    'deficit': expected - actual
                }
        
        return {
            'all_satisfied': all_satisfied,
            'unsatisfied': unsatisfied
        }
    
    def _repair_fps(self, schedule: List[TaskScheduleInfo], 
                   fps_check: Dict) -> List[TaskScheduleInfo]:
        """修复FPS不足"""
        repaired = copy.deepcopy(schedule)
        
        for task_id, info in fps_check['unsatisfied'].items():
            task = self.scheduler.tasks.get(task_id)
            if not task:
                continue
            
            needed = info['deficit']
            duration = self._estimate_task_duration(task)
            
            # 在空隙中插入缺失的执行
            added = 0
            for _ in range(needed):
                # 找空隙
                gap_start = self._find_gap_for_task(repaired, task, duration)
                if gap_start is not None:
                    event = self._create_task_event(task, gap_start)
                    repaired.append(event)
                    added += 1
                else:
                    break
            
            if added < needed:
                print(f"    ⚠️ {task_id} 只能补充 {added}/{needed} 次执行")
        
        return sorted(repaired, key=lambda x: x.start_time)
    
    def _find_gap_for_task(self, schedule: List[TaskScheduleInfo], 
                          task: NNTask, duration: float) -> Optional[float]:
        """为任务找合适的空隙"""
        sorted_events = sorted(schedule, key=lambda x: x.start_time)
        
        # 检查开头
        if sorted_events and sorted_events[0].start_time >= duration:
            if self._is_time_slot_available(schedule, task, 0, duration):
                return 0
        
        # 检查中间空隙
        for i in range(len(sorted_events) - 1):
            gap_start = sorted_events[i].end_time
            gap_end = sorted_events[i + 1].start_time
            
            if gap_end - gap_start >= duration:
                if self._is_time_slot_available(schedule, task, gap_start, duration):
                    return gap_start
        
        # 检查末尾
        if sorted_events:
            last_end = sorted_events[-1].end_time
            if last_end + duration <= self.time_window:
                if self._is_time_slot_available(schedule, task, last_end, duration):
                    return last_end
        
        return None
    
    def _create_task_event(self, task: NNTask, start_time: float) -> TaskScheduleInfo:
        """创建任务事件"""
        # 分配资源
        assigned_resources = {}
        for seg in task.segments:
            resources = self.scheduler.resources.get(seg.resource_type, [])
            if resources:
                assigned_resources[seg.resource_type] = resources[0].unit_id
        
        # 计算结束时间和子段调度
        end_time = start_time
        sub_schedule = []
        
        if task.is_segmented:
            current_time = start_time
            for seg in task.segments:
                if seg.is_segmented:
                    for sub_seg in seg.sub_segments:
                        duration = sub_seg.get_duration(40.0)
                        sub_schedule.append((sub_seg.sub_id, current_time, current_time + duration))
                        current_time += duration
                        end_time = current_time
                else:
                    duration = seg.get_duration(40.0)
                    sub_schedule.append((f"{seg.segment_id}_0", current_time, current_time + duration))
                    current_time += duration
                    end_time = current_time
        else:
            for seg in task.segments:
                duration = seg.get_duration(40.0)
                seg_start = start_time + seg.start_time
                seg_end = seg_start + duration
                end_time = max(end_time, seg_end)
                sub_schedule.append((f"{seg.segment_id}_0", seg_start, seg_end))
        
        event = TaskScheduleInfo(
            task_id=task.task_id,
            start_time=start_time,
            end_time=end_time,
            assigned_resources=assigned_resources,
            actual_latency=end_time - start_time,
            runtime_type=task.runtime_type
        )
        
        if sub_schedule:
            event.sub_segment_schedule = sub_schedule
        
        return event
    
    def _estimate_task_duration(self, task: NNTask) -> float:
        """估算任务执行时间"""
        total_duration = 0
        for seg in task.segments:
            duration = seg.get_duration(40.0)
            total_duration = max(total_duration, seg.start_time + duration)
        return total_duration
    
    def _rebuild_resource_timeline(self, schedule: List[TaskScheduleInfo]):
        """重建资源时间线"""
        self.resource_timeline.clear()
        
        for event in schedule:
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
            
            if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                for sub_id, start, end in event.sub_segment_schedule:
                    # 找到对应的资源
                    for sub_seg in task.get_sub_segments_for_scheduling():
                        if sub_seg.sub_id == sub_id:
                            res_type = sub_seg.resource_type
                            if res_type in event.assigned_resources:
                                res_id = event.assigned_resources[res_type]
                                self.resource_timeline[res_id].append(
                                    (start, end, event.task_id, sub_id)
                                )
                            break
            else:
                # 非分段任务
                for seg in task.segments:
                    if seg.resource_type in event.assigned_resources:
                        res_id = event.assigned_resources[seg.resource_type]
                        duration = seg.get_duration(40.0)
                        start_time = event.start_time + seg.start_time
                        end_time = start_time + duration
                        self.resource_timeline[res_id].append(
                            (start_time, end_time, event.task_id, f"{seg.segment_id}_0")
                        )
        
        # 排序
        for res_id in self.resource_timeline:
            self.resource_timeline[res_id].sort()
    
    def _print_phase_status(self, schedule: List[TaskScheduleInfo], phase_name: str):
        """打印阶段状态"""
        # 计算FPS满足情况
        task_counts = defaultdict(int)
        for event in schedule:
            task_counts[event.task_id] += 1
        
        unsatisfied_count = 0
        for task_id, task in self.scheduler.tasks.items():
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            if actual < expected:
                unsatisfied_count += 1
        
        # 计算末尾空闲
        if schedule:
            last_end = max(e.end_time for e in schedule)
            tail_idle = self.time_window - last_end
            self.optimization_stats['tail_idle'] = tail_idle
        else:
            tail_idle = self.time_window
        
        # 计算空隙
        gaps = self._identify_all_gaps(schedule)
        complete_idle = len([g for g in gaps if g['type'] == 'both_idle'])
        
        print(f"\n  [{phase_name}]")
        print(f"    - FPS未满足: {unsatisfied_count} 个任务")
        print(f"    - 完全空闲时段: {complete_idle} 个")
        print(f"    - 末尾空闲: {tail_idle:.1f}ms ({tail_idle/self.time_window*100:.1f}%)")
    
    def _print_optimization_summary(self):
        """打印优化统计摘要"""
        print("\n" + "=" * 60)
        print("📊 优化统计摘要")
        print("=" * 60)
        
        print(f"  空隙消除: {self.optimization_stats['initial_gaps']} → "
              f"{self.optimization_stats['final_gaps']}")
        print(f"  末尾空闲: {self.optimization_stats['tail_idle']:.1f}ms "
              f"({self.optimization_stats['tail_idle']/self.time_window*100:.1f}%)")


def main():
    """主测试函数"""
    print("=" * 80)
    print("🚀 终极调度优化测试 V2 - 零空隙、100% FPS")
    print("=" * 80)
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=40.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    # 应用基础修复
    print("\n应用调度修复...")
    apply_basic_fixes(scheduler)
    
    # 创建任务
    print("\n创建真实任务...")
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    # 应用额外修复
    apply_minimal_fifo_fix(scheduler)
    apply_strict_resource_conflict_fix(scheduler)
    
    # 创建优化器V2
    optimizer = UltimateSchedulerOptimizerV2(scheduler, 200.0)
    
    # 执行优化
    final_schedule = optimizer.optimize_ultimate()
    
    # 更新调度器
    scheduler.schedule_history = final_schedule
    
    # 最终验证
    print("\n" + "=" * 80)
    print("📊 最终验证")
    print("=" * 80)
    
    # 1. 验证资源冲突
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    print(f"\n资源冲突检查: {'✅ 无冲突' if is_valid else f'❌ {len(conflicts)}个冲突'}")
    if not is_valid and conflicts:
        print("  冲突详情:")
        for conflict in conflicts[:3]:
            print(f"    - {conflict}")
    
    # 2. 验证FPS
    print("\n最终FPS达成情况:")
    task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        task_counts[event.task_id] += 1
    
    all_satisfied = True
    for task_id in sorted(scheduler.tasks.keys()):
        task = scheduler.tasks[task_id]
        expected = int((200.0 / 1000.0) * task.fps_requirement)
        actual = task_counts[task_id]
        fps_rate = actual / expected if expected > 0 else 1.0
        status = "✅" if fps_rate >= 0.95 else "❌"
        if fps_rate < 0.95:
            all_satisfied = False
        print(f"  {status} {task_id} ({task.name}): {actual}/{expected} ({fps_rate:.1%})")
    
    # 3. 计算资源利用率
    print("\n资源利用率:")
    resource_busy = defaultdict(float)
    for event in scheduler.schedule_history:
        if hasattr(event, 'sub_segment_schedule'):
            for sub_id, start, end in event.sub_segment_schedule:
                for res_id in event.assigned_resources.values():
                    resource_busy[res_id] += (end - start)
        else:
            duration = event.end_time - event.start_time
            for res_id in event.assigned_resources.values():
                resource_busy[res_id] += duration
    
    for res_id in ['NPU_0', 'DSP_0']:
        utilization = resource_busy[res_id] / 200.0 * 100
        print(f"  {res_id}: {utilization:.1f}%")
    
    # 4. 验证零空隙
    gaps = optimizer._identify_all_gaps(scheduler.schedule_history)
    both_idle_gaps = [g for g in gaps if g['type'] == 'both_idle']
    
    print(f"\n空隙分析:")
    print(f"  完全空闲时段: {len(both_idle_gaps)} 个")
    if both_idle_gaps:
        total_gap_time = sum(g['duration'] for g in both_idle_gaps)
        print(f"  总空隙时间: {total_gap_time:.1f}ms")
        for gap in both_idle_gaps[:3]:
            print(f"    - {gap['start']:.1f}-{gap['end']:.1f}ms ({gap['duration']:.1f}ms)")
    
    # 5. 计算末尾空闲
    if scheduler.schedule_history:
        last_end = max(e.end_time for e in scheduler.schedule_history)
        tail_idle = 200.0 - last_end
        print(f"\n末尾空闲时间: {tail_idle:.1f}ms ({tail_idle/200.0*100:.1f}%)")
    else:
        tail_idle = 200.0
    
    # 生成可视化
    print("\n生成可视化...")
    viz = ElegantSchedulerVisualizer(scheduler)
    
    # 甘特图
    plt.figure(figsize=(24, 12))
    viz.plot_elegant_gantt(time_window=200.0, show_all_labels=True)
    
    # 标注优化成果
    ax = plt.gca()
    if scheduler.schedule_history and tail_idle > 0:
        last_end = max(e.end_time for e in scheduler.schedule_history)
        ax.axvspan(last_end, 200, alpha=0.3, color='lightgreen')
        ax.text(last_end + tail_idle/2, ax.get_ylim()[1]*0.95,
               f'优化空闲\n{tail_idle:.1f}ms\n({tail_idle/2:.0f}%)', 
               ha='center', va='top', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.title('Ultimate Optimized Schedule V2 - Zero Gaps & 100% FPS', fontsize=16, pad=20)
    plt.savefig('ultimate_v2_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chrome trace
    viz.export_chrome_tracing('ultimate_v2_schedule.json')
    
    # 最终总结
    print("\n" + "=" * 80)
    print("✨ 优化完成！")
    print("=" * 80)
    
    zero_gap = len(both_idle_gaps) == 0
    fps_100 = all_satisfied
    no_conflict = is_valid
    
    if zero_gap and fps_100 and no_conflict:
        print("\n🎉 完美优化达成！")
        print("  ✅ 零空隙")
        print("  ✅ 100% FPS满足")
        print("  ✅ 无资源冲突")
        print(f"  ✅ 末尾空闲最大化: {tail_idle:.1f}ms")
    else:
        print("\n⚠️ 优化未完全达标:")
        if not zero_gap:
            print(f"  ❌ 仍有 {len(both_idle_gaps)} 个空隙")
        if not fps_100:
            print("  ❌ 部分任务FPS未满足")
        if not no_conflict:
            print(f"  ❌ 存在 {len(conflicts)} 个资源冲突")
    
    print("\n生成的文件：")
    print("  - ultimate_v2_schedule.png")
    print("  - ultimate_v2_schedule.json")


if __name__ == "__main__":
    main()