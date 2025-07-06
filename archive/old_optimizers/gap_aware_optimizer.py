#!/usr/bin/env python3
"""
调试版空隙优化器 - 一步步解决问题
"""

from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy

from .enums import ResourceType
from .models import TaskScheduleInfo
from .scheduler import MultiResourceScheduler


class GapAwareOptimizer:
    """调试版空隙优化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        self.original_schedule = None
        
    def optimize_schedule(self, time_window: float = 100.0) -> List[TaskScheduleInfo]:
        """优化调度"""
        print("\n🔍 执行调试版空隙优化...")
        
        # 1. 保存原始调度
        self.original_schedule = copy.deepcopy(self.scheduler.schedule_history)
        print(f"\n原始调度分析:")
        print(f"  事件总数: {len(self.original_schedule)}")
        
        # 打印每个事件的详细信息
        for idx, event in enumerate(self.original_schedule):
            print(f"\n  事件{idx}: 任务{event.task_id}")
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    # 判断资源类型
                    res_type = self._get_resource_type_for_segment(event.task_id, sub_id)
                    print(f"    {sub_id}: {start:.1f}-{end:.1f}ms ({res_type})")
        
        # 2. 找出NPU空隙
        print("\n分析NPU空隙:")
        npu_gaps = self._find_npu_gaps()
        
        # 3. 找出可移动的B段
        print("\n分析可移动的段:")
        movable_segments = self._find_movable_segments()
        
        # 4. 执行移动
        print("\n执行优化移动:")
        optimized = self._do_optimization(npu_gaps, movable_segments)
        
        return optimized
    
    def _get_resource_type_for_segment(self, task_id: str, sub_id: str) -> str:
        """获取段的资源类型"""
        task = self.scheduler.tasks.get(task_id)
        if task:
            for seg in task.get_sub_segments_for_scheduling():
                if seg.sub_id == sub_id:
                    return seg.resource_type.value
        return "unknown"
    
    def _find_npu_gaps(self) -> List[Tuple[float, float]]:
        """找出NPU空闲时段"""
        # 收集所有NPU占用时段
        npu_busy = []
        
        for event in self.original_schedule:
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    res_type = self._get_resource_type_for_segment(event.task_id, sub_id)
                    if res_type == "NPU":
                        npu_busy.append((start, end, event.task_id, sub_id))
        
        # 排序
        npu_busy.sort()
        print(f"  NPU占用时段:")
        for start, end, task_id, sub_id in npu_busy:
            print(f"    {start:.1f}-{end:.1f}ms: {task_id}.{sub_id}")
        
        # 找空隙
        gaps = []
        last_end = 0
        for start, end, _, _ in npu_busy:
            if start > last_end + 0.1:
                gaps.append((last_end, start))
            last_end = max(last_end, end)
        
        print(f"\n  NPU空隙:")
        for start, end in gaps:
            print(f"    {start:.1f}-{end:.1f}ms (持续{end-start:.1f}ms)")
        
        return gaps
    
    def _find_movable_segments(self) -> List[Dict]:
        """找出任务B的可移动段"""
        segments = []
        
        for event_idx, event in enumerate(self.original_schedule):
            if event.task_id == 'B' and hasattr(event, 'sub_segment_schedule'):
                for seg_idx, (sub_id, start, end) in enumerate(event.sub_segment_schedule):
                    res_type = self._get_resource_type_for_segment('B', sub_id)
                    if res_type == "NPU":
                        segments.append({
                            'event_idx': event_idx,
                            'seg_idx': seg_idx,
                            'sub_id': sub_id,
                            'start': start,
                            'end': end,
                            'duration': end - start
                        })
                        print(f"  B.{sub_id}: {start:.1f}-{end:.1f}ms (可移动)")
        
        return segments
    
    def _do_optimization(self, gaps: List[Tuple[float, float]], 
                        segments: List[Dict]) -> List[TaskScheduleInfo]:
        """执行优化"""
        # 创建调度副本
        new_schedule = copy.deepcopy(self.original_schedule)
        
        # 记录移动
        moves = {}  # (event_idx, seg_idx) -> new_start
        
        # 尝试将B的段插入空隙
        gap_idx = 0
        for seg in segments:
            if gap_idx >= len(gaps):
                break
                
            gap_start, gap_end = gaps[gap_idx]
            gap_size = gap_end - gap_start
            
            if seg['duration'] <= gap_size + 0.01:
                # 可以放入
                new_start = gap_start
                moves[(seg['event_idx'], seg['seg_idx'])] = new_start
                print(f"  ✓ 移动 B.{seg['sub_id']}: {seg['start']:.1f} -> {new_start:.1f}ms")
                
                # 更新空隙
                gap_start += seg['duration']
                if gap_end - gap_start < 1.0:
                    gap_idx += 1
                else:
                    gaps[gap_idx] = (gap_start, gap_end)
            else:
                print(f"  ✗ B.{seg['sub_id']} 太大，无法放入空隙")
        
        # 应用移动
        final_schedule = []
        for idx, event in enumerate(new_schedule):
            if any((idx, i) in moves for i in range(10)):  # 检查是否有移动
                # 需要修改这个事件
                new_sub_schedules = []
                
                if hasattr(event, 'sub_segment_schedule'):
                    # 计算时间偏移量
                    time_shift = None
                    for seg_idx, (sub_id, start, end) in enumerate(event.sub_segment_schedule):
                        key = (idx, seg_idx)
                        if key in moves:
                            # 有段被移动了，计算整体偏移
                            if time_shift is None:
                                time_shift = moves[key] - start
                            new_start = moves[key]
                            duration = end - start
                            new_sub_schedules.append((sub_id, new_start, new_start + duration))
                        else:
                            # 其他段也要跟随移动
                            if time_shift is not None and seg_idx > 0:
                                # 如果前面的段被移动了，后面的段也要跟着移动
                                duration = end - start
                                # 计算新的开始时间：前一个段的结束时间
                                if new_sub_schedules:
                                    new_start = new_sub_schedules[-1][2]  # 前一段的结束时间
                                else:
                                    new_start = start + time_shift
                                new_sub_schedules.append((sub_id, new_start, new_start + duration))
                            else:
                                new_sub_schedules.append((sub_id, start, end))
                
                # 创建新事件
                new_event = copy.deepcopy(event)
                new_event.sub_segment_schedule = new_sub_schedules
                new_event.start_time = min(s[1] for s in new_sub_schedules)
                new_event.end_time = max(s[2] for s in new_sub_schedules)
                final_schedule.append(new_event)
            else:
                final_schedule.append(event)
        
        # 验证结果
        print("\n优化后调度:")
        for idx, event in enumerate(final_schedule):
            print(f"\n  事件{idx}: 任务{event.task_id}")
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    res_type = self._get_resource_type_for_segment(event.task_id, sub_id)
                    print(f"    {sub_id}: {start:.1f}-{end:.1f}ms ({res_type})")
        
        return final_schedule
        
        # # 使用 debug_compactor 进一步优化
        # print("\n应用调度紧凑化...")
        # try:
        #     from .debug_compactor import DebugCompactor
        #     compactor = DebugCompactor(self.scheduler, 100.0)
        #     # 临时更新调度历史
        #     self.scheduler.schedule_history = final_schedule
        #     compacted_schedule, idle_time = compactor.simple_compact()
        #     print(f"  紧凑化后末尾空闲: {idle_time:.1f}ms")
        #     return compacted_schedule
        # except Exception as e:
        #     print(f"  紧凑化失败: {e}")
        #     return final_schedule