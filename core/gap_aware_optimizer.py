#!/usr/bin/env python3
"""
空隙感知优化器 - 作为调度后的优化步骤
在第一阶段调度完成后，识别并利用资源空隙
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
    
    def can_fit_segment(self, segment: SubSegment, bandwidth: float) -> bool:
        """检查段是否能放入此空隙"""
        segment_duration = segment.get_duration(bandwidth)
        return segment_duration <= self.duration


class GapAwareOptimizer:
    """空隙感知优化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        self.resource_gaps: Dict[ResourceType, List[ResourceGap]] = {}
        self.optimization_history = []
        
    def _print_gap_summary(self, gaps: Dict[ResourceType, List[ResourceGap]]):
        """打印空隙摘要"""
        total_gaps = sum(len(g) for g in gaps.values())
        print(f"  发现 {total_gaps} 个资源空隙")
        
        for res_type, gap_list in gaps.items():
            if gap_list:
                print(f"\n  {res_type.value} 空隙:")
                # 按持续时间排序，显示前5个最大的空隙
                sorted_gaps = sorted(gap_list, key=lambda g: g.duration, reverse=True)
                for gap in sorted_gaps[:5]:
                    print(f"    {gap.resource_id}: {gap.start_time:.1f}-{gap.end_time:.1f}ms "
                        f"(持续{gap.duration:.1f}ms)")
        
    def analyze_resource_timeline(self, time_window: float) -> Dict[str, List[Tuple[float, float]]]:
        """分析资源占用时间线"""
        resource_timeline = defaultdict(list)
    
        for event in self.scheduler.schedule_history:
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
                
            # 处理每个子段的资源占用
            if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                for sub_seg_id, start_time, end_time in event.sub_segment_schedule:
                    # 找到对应的资源类型
                    for seg in task.segments:
                        # 检查是否是这个段的子段
                        if sub_seg_id.startswith(seg.segment_id):
                            res_type = seg.resource_type
                            if res_type in event.assigned_resources:
                                resource_id = event.assigned_resources[res_type]
                                resource_timeline[resource_id].append((start_time, end_time, event.task_id))
                            break
        
        # 排序并合并重叠的时间段
        for resource_id in resource_timeline:
            timeline = resource_timeline[resource_id]
            timeline.sort(key=lambda x: x[0])
            
            # 合并重叠时段（可选）
            merged = []
            for start, end, task_id in timeline:
                if merged and start <= merged[-1][1]:
                    # 重叠，扩展前一个时段
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end), merged[-1][2])
                else:
                    merged.append((start, end, task_id))
            
            resource_timeline[resource_id] = merged
        
        return dict(resource_timeline)
    
    def find_resource_gaps(self, time_window: float) -> Dict[ResourceType, List[ResourceGap]]:
        """查找所有资源的空隙 - 改进版"""
        self.resource_gaps = {ResourceType.NPU: [], ResourceType.DSP: []}
        timeline = self.analyze_resource_timeline(time_window)
        
        # 分析每个资源的空隙
        for res_type, resources in self.scheduler.resources.items():
            for resource in resources:
                resource_id = resource.unit_id
                
                if resource_id not in timeline:
                    # 整个资源都是空闲的
                    gap = ResourceGap(
                        resource_id=resource_id,
                        resource_type=res_type,
                        start_time=0.0,
                        end_time=time_window,
                        duration=time_window
                    )
                    self.resource_gaps[res_type].append(gap)
                else:
                    # 查找占用之间的空隙
                    occupations = timeline[resource_id]
                    current_time = 0.0
                    
                    for start, end, task_id in occupations:
                        # 如果有空隙
                        if start > current_time + 0.01:
                            gap = ResourceGap(
                                resource_id=resource_id,
                                resource_type=res_type,
                                start_time=current_time,
                                end_time=start,
                                duration=start - current_time
                            )
                            self.resource_gaps[res_type].append(gap)
                        
                        current_time = end
                    
                    # 检查末尾空隙
                    if current_time < time_window - 0.01:
                        gap = ResourceGap(
                            resource_id=resource_id,
                            resource_type=res_type,
                            start_time=current_time,
                            end_time=time_window,
                            duration=time_window - current_time
                        )
                        self.resource_gaps[res_type].append(gap)
        
        return self.resource_gaps
    
    def find_insertable_segments(self) -> List[Dict]:
        """查找可以插入的任务段"""
        insertable = []
        
        # 统计每个任务的执行次数
        task_exec_counts = defaultdict(int)
        for event in self.scheduler.schedule_history:
            task_exec_counts[event.task_id] += 1
        
        # 检查每个任务
        for task_id, task in self.scheduler.tasks.items():
            # 跳过非分段任务
            if not task.is_segmented:
                continue
                
            # 检查FPS满足情况
            current_count = task_exec_counts[task_id]
            time_window = self.scheduler.schedule_history[-1].end_time if self.scheduler.schedule_history else 100.0
            expected_count = int((time_window / 1000.0) * task.fps_requirement)
            
            # 如果还有执行次数的空间
            if current_count < expected_count * 1.2:  # 留20%的余量
                sub_segments = task.get_sub_segments_for_scheduling()
                for sub_seg in sub_segments:
                    insertable.append({
                        'task_id': task_id,
                        'task': task,
                        'segment': sub_seg,
                        'priority': task.priority,
                        'remaining_executions': expected_count - current_count
                    })
        
        # 按优先级排序
        insertable.sort(key=lambda x: (x['priority'].value, x['remaining_executions']), reverse=True)
        
        return insertable
    
    def try_fit_consecutive_segments(self, segments: List[SubSegment], gap: ResourceGap, 
                                bandwidth: float) -> List[SubSegment]:
        """尝试将连续的段插入空隙，允许微小误差"""
        fitting_segments = []
        total_duration = 0.0
        tolerance = 0.01  # 允许0.01ms的误差
        
        for segment in segments:
            seg_duration = segment.get_duration(bandwidth)
            
            # 检查是否能放入（考虑容差）
            if total_duration + seg_duration <= gap.duration + tolerance:
                fitting_segments.append(segment)
                total_duration += seg_duration
            else:
                # 检查是否"几乎"能填满空隙
                if abs(total_duration + seg_duration - gap.duration) < tolerance:
                    fitting_segments.append(segment)
                    break
                else:
                    break
        
        return fitting_segments
    
    def find_optimal_segment_placements(self, segments: List[SubSegment], 
                                   gaps: List[ResourceGap], 
                                   original_event) -> List[Dict]:
        """为每个段找到最优放置位置"""
        placements = []
        used_gap_portions = {}  # 跟踪每个空隙已使用的部分
        
        # 获取资源带宽映射
        bandwidth_map = {}
        for res_type, resources in self.scheduler.resources.items():
            for res in resources:
                bandwidth_map[res.unit_id] = res.bandwidth
        
        # 按段的原始顺序处理
        for seg in segments:
            best_placement = None
            best_score = float('inf')
            
            for gap in gaps:
                if gap.resource_id not in bandwidth_map:
                    continue
                    
                bandwidth = bandwidth_map[gap.resource_id]
                seg_duration = seg.get_duration(bandwidth)
                
                # 计算在这个空隙中的可用起始时间
                gap_start = gap.start_time
                if gap.resource_id in used_gap_portions:
                    # 考虑已经放置的段
                    gap_start = max(gap_start, used_gap_portions[gap.resource_id])
                
                # 检查是否能放入
                if gap_start + seg_duration <= gap.end_time + 0.01:
                    # 计算评分（越早越好，但要考虑段之间的依赖）
                    score = gap_start
                    
                    # 如果这个段依赖于前面的段，确保顺序正确
                    if placements:
                        last_placement = placements[-1]
                        min_start = last_placement['end']
                        if gap_start < min_start:
                            continue  # 违反依赖顺序
                    
                    # 优先使用原调度之前的空隙
                    if gap_start < original_event.start_time:
                        score -= 100  # 奖励提前执行
                    
                    if score < best_score:
                        best_score = score
                        best_placement = {
                            'segment': seg,
                            'gap': gap,
                            'start': gap_start,
                            'end': gap_start + seg_duration,
                            'resource_id': gap.resource_id
                        }
            
            if best_placement:
                placements.append(best_placement)
                # 更新空隙使用情况
                res_id = best_placement['resource_id']
                used_gap_portions[res_id] = best_placement['end']
        
        return placements

    def _create_schedule_event(self, task: NNTask, resource_id: str, 
                          segment_placements: List[Dict]) -> TaskScheduleInfo:
        """创建单个调度事件"""
        sub_schedules = []
        start_time = segment_placements[0]['start']
        end_time = segment_placements[-1]['end']
        
        for placement in segment_placements:
            sub_schedules.append((
                placement['segment'].sub_id,
                placement['start'],
                placement['end']
            ))
        
        # 确定资源类型
        resource_type = None
        for res_type, resources in self.scheduler.resources.items():
            if any(r.unit_id == resource_id for r in resources):
                resource_type = res_type
                break
        
        return TaskScheduleInfo(
            task_id=task.task_id,
            start_time=start_time,
            end_time=end_time,
            assigned_resources={resource_type: resource_id},
            actual_latency=end_time - start_time,
            runtime_type=task.runtime_type,
            sub_segment_schedule=sub_schedules
        )

    def is_optimization_beneficial(self, placements: List[Dict], original_event) -> bool:
        """判断优化是否有益"""
        if not placements:
            return False
        
        # 如果能将任何段提前执行，就是有益的
        earliest_new_start = min(p['start'] for p in placements)
        return earliest_new_start < original_event.start_time - 0.01
    
    def update_gaps_after_placement(self, gaps: List[ResourceGap], placements: List[Dict]):
        """更新空隙状态"""
        gap_updates = {}
        
        # 收集每个空隙的使用情况
        for placement in placements:
            gap = placement['gap']
            gap_id = id(gap)
            
            if gap_id not in gap_updates:
                gap_updates[gap_id] = {
                    'gap': gap,
                    'used_portions': []
                }
            
            gap_updates[gap_id]['used_portions'].append((placement['start'], placement['end']))
        
        # 更新或移除空隙
        for gap_id, update in gap_updates.items():
            gap = update['gap']
            used_portions = sorted(update['used_portions'])
            
            # 简化处理：如果空隙被完全使用，移除它
            total_used = sum(end - start for start, end in used_portions)
            if total_used >= gap.duration - 0.01:
                if gap in gaps:
                    gaps.remove(gap)
            else:
                # 更新空隙的起始时间（简化处理）
                last_used_end = max(end for start, end in used_portions)
                gap.start_time = last_used_end
                gap.duration = gap.end_time - gap.start_time

    def create_optimized_schedules(self, task: NNTask, placements: List[Dict]) -> List[TaskScheduleInfo]:
        """根据段放置创建调度事件"""
        # 将连续使用相同资源的段组合成一个调度事件
        events = []
        current_event_segments = []
        current_resource = None
        
        for placement in placements:
            if current_resource != placement['resource_id']:
                # 开始新的事件
                if current_event_segments:
                    # 保存前一个事件
                    event = self._create_schedule_event(task, current_resource, current_event_segments)
                    events.append(event)
                
                current_resource = placement['resource_id']
                current_event_segments = [placement]
            else:
                # 继续当前事件
                current_event_segments.append(placement)
        
        # 保存最后一个事件
        if current_event_segments:
            event = self._create_schedule_event(task, current_resource, current_event_segments)
            events.append(event)
        
        return events
    
    def analyze_optimization_opportunities(self, time_window: float) -> List[Dict]:
        """分析哪些任务实例可以被优化"""
        opportunities = []
        
        # 统计每个任务的执行次数
        task_exec_counts = defaultdict(int)
        for event in self.scheduler.schedule_history:
            task_exec_counts[event.task_id] += 1
        
        # 检查每个调度事件
        for i, event in enumerate(self.scheduler.schedule_history):
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
                
            # 只考虑分段任务
            if not task.is_segmented:
                continue
            
            # 修改：考虑所有可以优化的任务实例
            # 不仅仅是开始时间大于15ms的
            print(f"\n[DEBUG] 检查事件 {i}: 任务{event.task_id} @ {event.start_time:.1f}ms")
            
            opportunities.append({
                'task': task,
                'event': event,
                'exec_count': task_exec_counts[task.task_id],
                'index': i
            })
        
        # 按开始时间排序，但优先处理较早的任务以便释放更多空间
        opportunities.sort(key=lambda x: x['event'].start_time)
        
        return opportunities
    
    def _extract_segments_from_event(self, task: NNTask, event: TaskScheduleInfo) -> List[Dict]:
        """从调度事件中提取段信息"""
        segments = []
        
        # 从sub_segment_schedule中提取
        if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
            for sub_seg_id, start, end in event.sub_segment_schedule:
                # 找到对应的段
                for seg in task.get_sub_segments_for_scheduling():
                    if seg.sub_id == sub_seg_id:
                        segments.append({
                            'segment': seg,
                            'original_start': start,
                            'original_end': end,
                            'duration': end - start
                        })
                        break
        
        return segments
    
    def _find_best_placement_for_segments(self, segments: List[Dict], 
                                     gaps: List[ResourceGap]) -> List[Dict]:
        """为一组段找到最佳放置方案"""
        if not segments or not gaps:
            return []
        
        # 构建资源带宽映射
        bandwidth_map = {}
        for res_type, resources in self.scheduler.resources.items():
            for resource in resources:
                bandwidth_map[resource.unit_id] = resource.bandwidth
        
        placement = []
        
        gaps_copy = copy.deepcopy(gaps)
        
        # 为了确保段的连续性，创建一个统一的时间线
        for seg_info in segments:
            segment = seg_info['segment']
            best_placement = None
            best_score = float('inf')
            
            # 计算这个段的最早可能开始时间
            min_start_time = 0.0
            if placement:
                # 必须在前一个段之后
                min_start_time = placement[-1]['end']
            
            # 遍历所有空隙找最优位置
            for gap in gaps_copy:
                if gap.resource_id not in bandwidth_map:
                    continue
                    
                bandwidth = bandwidth_map[gap.resource_id]
                duration = segment.get_duration(bandwidth)
                
                # 计算在这个空隙中的开始时间
                start_time = max(gap.start_time, min_start_time)
                
                # 检查是否能放入
                if start_time + duration <= gap.end_time + 0.01:
                    # 评分：优先选择最早的时间
                    score = start_time
                    
                    if score < best_score:
                        best_score = score
                        best_placement = {
                            'segment': segment,
                            'gap': gap,
                            'resource_id': gap.resource_id,
                            'start': start_time,
                            'end': start_time + duration,
                            'bandwidth': bandwidth
                        }
            
            if not best_placement:
                # 无法放置这个段
                return []
            
            placement.append(best_placement)
            
            # 更新空隙（临时，用于计算后续段的位置）
            gap = best_placement['gap']
            if best_placement['end'] >= gap.end_time - 0.01:
                # 空隙被完全使用
                gaps_copy.remove(gap)
            else:
                # 更新空隙的起始时间
                gap.start_time = best_placement['end']
                gap.duration = gap.end_time - gap.start_time
        
        return placement
    
    def _create_single_event(self, task: NNTask, segment_group: List[Dict]) -> TaskScheduleInfo:
        """创建单个调度事件"""
        sub_schedules = []
        for p in segment_group:
            sub_schedules.append((
                p['segment'].sub_id,
                p['start'],
                p['end']
            ))
        
        return TaskScheduleInfo(
            task_id=task.task_id,
            start_time=segment_group[0]['start'],
            end_time=segment_group[-1]['end'],
            assigned_resources={ResourceType.NPU: segment_group[0]['resource_id']},
            actual_latency=segment_group[-1]['end'] - segment_group[0]['start'],
            runtime_type=task.runtime_type,
            sub_segment_schedule=sub_schedules
        )
    
    def _update_gaps_after_placement(self, gaps: List[ResourceGap], placement: List[Dict]):
        """更新空隙列表"""
        # 这个方法已经在_find_best_placement_for_segments中实时更新了
        pass
    
    def _create_events_from_placement(self, task: NNTask, placement: List[Dict]) -> List[TaskScheduleInfo]:
        """根据放置方案创建调度事件"""
        if not placement:
            return []
        
        # 将连续的、使用相同资源的段组合成一个事件
        events = []
        current_group = [placement[0]]
        
        for p in placement[1:]:
            last_p = current_group[-1]
            # 检查是否可以合并（相同资源且时间连续）
            if (p['resource_id'] == last_p['resource_id'] and 
                abs(p['start'] - last_p['end']) < 0.01):
                current_group.append(p)
            else:
                # 创建事件并开始新组
                event = self._create_single_event(task, current_group)
                events.append(event)
                current_group = [p]
        
        # 创建最后一个事件
        if current_group:
            event = self._create_single_event(task, current_group)
            events.append(event)
        
        return events
    
    def _optimize_single_task_instance(self, task: NNTask, segments: List[Dict], 
                                  gaps: Dict[ResourceType, List[ResourceGap]], 
                                  original_event) -> Dict:
        """优化单个任务实例，保持所有段的完整性"""
        result = {
            'beneficial': False,
            'new_events': []
        }
        
        # 获取NPU段
        npu_segments = [s for s in segments if s['segment'].resource_type == ResourceType.NPU]
        if not npu_segments:
            return result
        
        # 尝试为所有段找到最优调度方案
        placement_plan = self._find_best_placement_for_segments(npu_segments, gaps[ResourceType.NPU])
        
        if not placement_plan:
            return result
        
        # 检查是否有改进
        earliest_new_time = min(p['start'] for p in placement_plan)
        if earliest_new_time >= original_event.start_time - 0.01:
            return result  # 没有改进
        
        # 创建新的调度事件
        result['new_events'] = self._create_events_from_placement(task, placement_plan)
        result['beneficial'] = True
        
        # 更新空隙（在副本上）
        self._update_gaps_after_placement(gaps[ResourceType.NPU], placement_plan)
        
        return result
    
    def optimize_schedule(self, time_window: float = 100.0) -> List[TaskScheduleInfo]:
        """执行空隙优化 - 动态更新空隙版本"""
        print("\n🔍 执行空隙感知优化...")
        
        # 1. 初始分析资源空隙
        gaps = self.find_resource_gaps(time_window)
        self._print_gap_summary(gaps)
        
        # 2. 分析优化机会
        optimization_candidates = self.analyze_optimization_opportunities(time_window)
        print(f"\n  找到 {len(optimization_candidates)} 个优化候选")
        
        # 3. 逐个任务实例进行优化
        optimizations = []
        
        for i, candidate in enumerate(optimization_candidates):
            task = candidate['task']
            original_event = candidate['event']
            
            print(f"\n  处理候选 {i}: {task.task_id} @ {original_event.start_time:.1f}-{original_event.end_time:.1f}ms")
            # 获取这个任务实例的所有段
            all_segments = self._extract_segments_from_event(task, original_event)
            
            if not all_segments:
                continue
            
            # 尝试优化这个完整的任务实例
            optimization_result = self._optimize_single_task_instance(
                task, 
                all_segments, 
                gaps,  # 使用实时更新的空隙
                original_event
            )
            
            if optimization_result['beneficial']:
                # 关键步骤：释放原来占用的时间段，添加到空隙列表
                self._release_original_time_slots(original_event, gaps)
                
                # 移除原调度
                if original_event in self.scheduler.schedule_history:
                    self.scheduler.schedule_history.remove(original_event)
                
                # 添加优化后的调度
                optimizations.extend(optimization_result['new_events'])
                
                print(f"  ✅ 优化任务 {task.task_id} 实例 (原始时间: {original_event.start_time:.1f}ms)")
                for event in optimization_result['new_events']:
                    print(f"     新调度: {event.start_time:.1f}-{event.end_time:.1f}ms")
        
        # 4. 更新调度历史
        if optimizations:
            self.scheduler.schedule_history.extend(optimizations)
            self.scheduler.schedule_history.sort(key=lambda x: x.start_time)
        
        print(f"\n  完成优化，生成 {len(optimizations)} 个新调度事件")
        return optimizations
    
    def _release_original_time_slots(self, original_event: TaskScheduleInfo, 
                                gaps: Dict[ResourceType, List[ResourceGap]]):
        """释放原事件占用的时间段，添加为新的空隙"""
        print(f"\n  [DEBUG] 释放任务 {original_event.task_id} 的原时间段:")
        
        # 处理每个资源类型
        for res_type, res_id in original_event.assigned_resources.items():
            if res_type not in gaps:
                continue
            
            # 获取原事件的时间段
            if hasattr(original_event, 'sub_segment_schedule') and original_event.sub_segment_schedule:
                # 分段任务：释放每个子段
                for sub_seg_id, start, end in original_event.sub_segment_schedule:
                    print(f"    释放 {res_id}: {start:.1f}-{end:.1f}ms")
                    
                    # 创建新的空隙
                    new_gap = ResourceGap(
                        resource_id=res_id,
                        resource_type=res_type,
                        start_time=start,
                        end_time=end,
                        duration=end - start
                    )
                    
                    # 将新空隙添加到列表并合并相邻空隙
                    self._add_and_merge_gap(gaps[res_type], new_gap)
            else:
                # 非分段任务
                print(f"    释放 {res_id}: {original_event.start_time:.1f}-{original_event.end_time:.1f}ms")
                
                new_gap = ResourceGap(
                    resource_id=res_id,
                    resource_type=res_type,
                    start_time=original_event.start_time,
                    end_time=original_event.end_time,
                    duration=original_event.end_time - original_event.start_time
                )
                
                self._add_and_merge_gap(gaps[res_type], new_gap)

    def _add_and_merge_gap(self, gap_list: List[ResourceGap], new_gap: ResourceGap):
        """添加新空隙并合并相邻的空隙"""
        # 先添加新空隙
        gap_list.append(new_gap)
        
        # 按资源ID分组
        gaps_by_resource = defaultdict(list)
        for gap in gap_list:
            gaps_by_resource[gap.resource_id].append(gap)
        
        # 对每个资源，合并相邻空隙
        merged_gaps = []
        for resource_id, resource_gaps in gaps_by_resource.items():
            # 按开始时间排序
            resource_gaps.sort(key=lambda g: g.start_time)
            
            # 合并相邻空隙
            current_gap = None
            for gap in resource_gaps:
                if current_gap is None:
                    current_gap = gap
                elif abs(current_gap.end_time - gap.start_time) < 0.01:
                    # 相邻，合并
                    current_gap.end_time = gap.end_time
                    current_gap.duration = current_gap.end_time - current_gap.start_time
                else:
                    # 不相邻，保存当前并开始新的
                    merged_gaps.append(current_gap)
                    current_gap = gap
            
            if current_gap:
                merged_gaps.append(current_gap)
        
        # 更新空隙列表
        gap_list.clear()
        gap_list.extend(merged_gaps)
