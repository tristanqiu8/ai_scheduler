#!/usr/bin/env python3
"""
空隙感知调度测试 - 基于真实任务场景（含依赖关系）
扩展自 test/gap_smoke_test.py，使用 scenario/real_task.py 中的真实任务
保持1个NPU和1个DSP的配置
"""

import sys
import os
import copy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

# 添加项目根目录到路径
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


class DependencyAwareGapScheduler:
    """考虑依赖关系的空隙感知调度器"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        self.dependency_graph = self._build_dependency_graph()
        
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """构建任务依赖图"""
        dep_graph = defaultdict(set)
        for task_id, task in self.scheduler.tasks.items():
            for dep in task.dependencies:
                dep_graph[dep].add(task_id)  # dep -> dependents
        return dict(dep_graph)
    
    def _get_task_dependencies(self, task_id: str) -> Set[str]:
        """获取任务的所有依赖（包括传递依赖）"""
        task = self.scheduler.tasks.get(task_id)
        if not task:
            return set()
        
        all_deps = set()
        to_process = list(task.dependencies)
        
        while to_process:
            dep = to_process.pop()
            if dep not in all_deps:
                all_deps.add(dep)
                dep_task = self.scheduler.tasks.get(dep)
                if dep_task:
                    to_process.extend(dep_task.dependencies)
        
        return all_deps
    
    def _get_dependent_tasks(self, task_id: str) -> Set[str]:
        """获取依赖于指定任务的所有任务（包括传递依赖）"""
        all_dependents = set()
        to_process = list(self.dependency_graph.get(task_id, []))
        
        while to_process:
            dep = to_process.pop()
            if dep not in all_dependents:
                all_dependents.add(dep)
                to_process.extend(self.dependency_graph.get(dep, []))
        
        return all_dependents
    
    def _can_move_task_to_time(self, task_id: str, new_start: float, 
                               schedule: List[TaskScheduleInfo]) -> bool:
        """检查任务是否可以移动到新的时间（考虑依赖关系）"""
        task = self.scheduler.tasks.get(task_id)
        if not task:
            return False
        
        # 构建任务执行时间映射
        task_times = {}
        for event in schedule:
            if event.task_id not in task_times:
                task_times[event.task_id] = []
            task_times[event.task_id].append((event.start_time, event.end_time))
        
        # 检查依赖约束
        for dep_id in task.dependencies:
            if dep_id in task_times:
                # 找到依赖任务的最晚结束时间
                dep_end_times = [end for _, end in task_times[dep_id]]
                if dep_end_times:
                    latest_dep_end = max(dep_end_times)
                    if new_start < latest_dep_end:
                        return False
        
        # 检查被依赖约束
        task_duration = self._estimate_task_duration(task)
        new_end = new_start + task_duration
        
        for dependent_id in self._get_dependent_tasks(task_id):
            if dependent_id in task_times:
                # 找到依赖任务的最早开始时间
                dep_start_times = [start for start, _ in task_times[dependent_id]]
                if dep_start_times:
                    earliest_dep_start = min(dep_start_times)
                    if new_end > earliest_dep_start:
                        return False
        
        return True
    
    def _estimate_task_duration(self, task: NNTask) -> float:
        """估算任务执行时间"""
        total_duration = 0
        for seg in task.segments:
            # 使用默认带宽估算
            duration = seg.get_duration(40.0)
            total_duration = max(total_duration, seg.start_time + duration)
        return total_duration
    
    def create_gap_aware_schedule(self, baseline_schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """创建考虑依赖关系的空隙感知调度"""
        print("\n🔍 创建依赖感知的空隙优化调度...")
        
        # 1. 分析资源占用情况
        resource_timeline = self._analyze_resource_timeline(baseline_schedule)
        
        # 2. 找出跨资源空隙
        cross_resource_gaps = self._find_cross_resource_gaps(resource_timeline)
        
        # 3. 识别可移动的任务段（考虑依赖关系）
        movable_segments = self._find_movable_segments_with_dependencies(
            baseline_schedule, cross_resource_gaps)
        
        # 4. 执行优化移动
        optimized_schedule = self._optimize_with_dependencies(
            baseline_schedule, cross_resource_gaps, movable_segments)
        
        return optimized_schedule
    
    def _analyze_resource_timeline(self, schedule: List[TaskScheduleInfo]) -> Dict[str, List[Tuple[float, float, str]]]:
        """分析各资源的占用时间线"""
        timeline = defaultdict(list)
        
        for event in schedule:
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
                
            # 处理分段任务
            if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                for sub_id, start, end in event.sub_segment_schedule:
                    # 找到对应的子段来确定资源类型
                    for sub_seg in task.get_sub_segments_for_scheduling():
                        if sub_seg.sub_id == sub_id:
                            res_type = sub_seg.resource_type
                            if res_type in event.assigned_resources:
                                res_id = event.assigned_resources[res_type]
                                timeline[res_id].append((start, end, event.task_id))
                            break
            else:
                # 处理非分段任务
                for seg in task.segments:
                    if seg.resource_type in event.assigned_resources:
                        res_id = event.assigned_resources[seg.resource_type]
                        resource = next((r for r in self.scheduler.resources[seg.resource_type] 
                                       if r.unit_id == res_id), None)
                        if resource:
                            duration = seg.get_duration(resource.bandwidth)
                            start_time = event.start_time + seg.start_time
                            end_time = start_time + duration
                            timeline[res_id].append((start_time, end_time, event.task_id))
        
        # 排序时间线
        for res_id in timeline:
            timeline[res_id].sort()
        
        return dict(timeline)
    
    def _find_cross_resource_gaps(self, timeline: Dict[str, List[Tuple[float, float, str]]]) -> List[Dict]:
        """找出跨资源空隙（一个资源忙碌时另一个资源空闲）"""
        gaps = []
        
        # 分离NPU和DSP资源
        npu_resources = [res for res in timeline.keys() if 'NPU' in res]
        dsp_resources = [res for res in timeline.keys() if 'DSP' in res]
        
        # 查找DSP忙碌时NPU的空隙
        for dsp_res in dsp_resources:
            for start, end, task_id in timeline.get(dsp_res, []):
                # 检查这段时间内NPU的空闲
                for npu_res in npu_resources:
                    npu_busy = timeline.get(npu_res, [])
                    
                    # 找出NPU在[start, end]期间的空闲时段
                    current = start
                    for npu_start, npu_end, _ in npu_busy:
                        if npu_start > current and npu_start < end:
                            # 找到空隙
                            gap_end = min(npu_start, end)
                            if gap_end - current > 0.1:  # 忽略过小的空隙
                                gaps.append({
                                    'type': 'NPU_gap_during_DSP',
                                    'start': current,
                                    'end': gap_end,
                                    'duration': gap_end - current,
                                    'npu_res': npu_res,
                                    'dsp_res': dsp_res,
                                    'dsp_task': task_id
                                })
                        current = max(current, npu_end)
                    
                    # 检查末尾是否还有空隙
                    if current < end:
                        gaps.append({
                            'type': 'NPU_gap_during_DSP',
                            'start': current,
                            'end': end,
                            'duration': end - current,
                            'npu_res': npu_res,
                            'dsp_res': dsp_res,
                            'dsp_task': task_id
                        })
        
        print(f"\n📊 发现 {len(gaps)} 个跨资源空隙:")
        for gap in gaps[:5]:  # 只显示前5个
            print(f"  {gap['start']:.1f}-{gap['end']:.1f}ms: "
                  f"NPU空闲 (DSP执行{gap['dsp_task']})")
        
        return gaps
    
    def _find_movable_segments_with_dependencies(self, schedule: List[TaskScheduleInfo], 
                                                gaps: List[Dict]) -> List[Dict]:
        """找出可移动的任务段（考虑依赖关系）"""
        movable = []
        
        # 统计每个任务的执行次数（用于处理周期性任务）
        task_execution_count = defaultdict(int)
        for event in schedule:
            task_execution_count[event.task_id] += 1
        
        for event_idx, event in enumerate(schedule):
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
            
            # 只考虑可以独立移动的任务（纯NPU任务或单段任务）
            if not hasattr(event, 'sub_segment_schedule'):
                continue
            
            # 检查是否是纯NPU任务
            is_pure_npu = True
            npu_segments = []
            
            for seg_idx, (sub_id, start, end) in enumerate(event.sub_segment_schedule):
                # 找到对应的子段来确定资源类型
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        if sub_seg.resource_type == ResourceType.NPU:
                            npu_segments.append({
                                'seg_idx': seg_idx,
                                'sub_id': sub_id,
                                'start': start,
                                'end': end,
                                'duration': end - start
                            })
                        else:
                            is_pure_npu = False
                        break
            
            # 只处理纯NPU任务或单个NPU段的任务
            if not npu_segments or (not is_pure_npu and len(event.sub_segment_schedule) > 1):
                continue
            
            # 检查依赖约束
            move_constraints = {
                'earliest_start': 0.0,
                'latest_end': float('inf')
            }
            
            # 检查前置依赖
            for dep_id in task.dependencies:
                dep_events = [e for e in schedule if e.task_id == dep_id]
                if dep_events:
                    # 对于周期性任务，需要匹配相应的执行实例
                    latest_dep_end = max(e.end_time for e in dep_events)
                    move_constraints['earliest_start'] = max(
                        move_constraints['earliest_start'], latest_dep_end)
            
            # 检查后续依赖
            for dependent_id in self._get_dependent_tasks(event.task_id):
                dep_events = [e for e in schedule if e.task_id == dependent_id]
                if dep_events:
                    earliest_dep_start = min(e.start_time for e in dep_events)
                    move_constraints['latest_end'] = min(
                        move_constraints['latest_end'], earliest_dep_start)
            
            # 评估每个空隙
            for gap in gaps:
                # 检查时间约束
                if gap['end'] <= move_constraints['earliest_start']:
                    continue
                if gap['start'] >= move_constraints['latest_end']:
                    continue
                
                # 对于纯NPU任务，尝试整体移动
                if is_pure_npu:
                    total_duration = sum(seg['duration'] for seg in npu_segments)
                    adjusted_gap_start = max(gap['start'], move_constraints['earliest_start'])
                    adjusted_gap_end = min(gap['end'], move_constraints['latest_end'])
                    
                    if total_duration <= adjusted_gap_end - adjusted_gap_start:
                        movable.append({
                            'event_idx': event_idx,
                            'event': event,
                            'task_id': event.task_id,
                            'segment': None,  # 整体移动
                            'is_whole_task': True,
                            'gap': gap,
                            'constraints': move_constraints,
                            'priority': task.priority.value,
                            'benefit': total_duration * (4 - task.priority.value)
                        })
                else:
                    # 单个NPU段，可以独立移动
                    for seg in npu_segments:
                        adjusted_gap_start = max(gap['start'], move_constraints['earliest_start'])
                        adjusted_gap_end = min(gap['end'], move_constraints['latest_end'])
                        
                        if seg['duration'] <= adjusted_gap_end - adjusted_gap_start:
                            movable.append({
                                'event_idx': event_idx,
                                'event': event,
                                'task_id': event.task_id,
                                'segment': seg,
                                'is_whole_task': False,
                                'gap': gap,
                                'constraints': move_constraints,
                                'priority': task.priority.value,
                                'benefit': seg['duration'] * (4 - task.priority.value)
                            })
        
        # 按效益排序
        movable.sort(key=lambda x: x['benefit'], reverse=True)
        
        print(f"\n📋 找到 {len(movable)} 个可移动的任务/段（考虑依赖）")
        for m in movable[:5]:
            if m['is_whole_task']:
                print(f"  {m['task_id']} (整体): 可移动到 {m['gap']['start']:.1f}-{m['gap']['end']:.1f}ms")
            else:
                print(f"  {m['task_id']}.{m['segment']['sub_id']}: "
                      f"可移动到 {m['gap']['start']:.1f}-{m['gap']['end']:.1f}ms")
        
        return movable
    
    def _optimize_with_dependencies(self, baseline_schedule: List[TaskScheduleInfo],
                                  gaps: List[Dict], movable_segments: List[Dict]) -> List[TaskScheduleInfo]:
        """执行考虑依赖关系的优化"""
        # 复制基线调度
        optimized = copy.deepcopy(baseline_schedule)
        
        # 记录已使用的空隙时间
        gap_usage = defaultdict(list)  # gap_idx -> [(start, end)]
        
        # 记录已移动的事件
        moved_events = set()
        
        # 尝试移动段
        total_moved = 0
        total_gap_utilized = 0.0
        
        for move_info in movable_segments:
            if move_info['event_idx'] in moved_events:
                continue
            
            event_idx = move_info['event_idx']
            segment = move_info['segment']
            gap = move_info['gap']
            gap_idx = gaps.index(gap)
            
            # 检查空隙是否还有足够空间
            available_start = gap['start']
            for used_start, used_end in gap_usage[gap_idx]:
                if used_end > available_start:
                    available_start = used_end
            
            # 计算需要的时长
            if move_info.get('is_whole_task'):
                # 整体任务移动，计算所有NPU段的总时长
                event = optimized[event_idx]
                task = self.scheduler.tasks.get(event.task_id)
                required_duration = 0
                if task and hasattr(event, 'sub_segment_schedule'):
                    for sub_id, start, end in event.sub_segment_schedule:
                        for sub_seg in task.get_sub_segments_for_scheduling():
                            if sub_seg.sub_id == sub_id and sub_seg.resource_type == ResourceType.NPU:
                                required_duration += (end - start)
                                break
            else:
                required_duration = segment['duration']
            
            if available_start + required_duration > gap['end']:
                continue
            
            # 检查依赖约束
            if available_start < move_info['constraints']['earliest_start']:
                available_start = move_info['constraints']['earliest_start']
            
            if available_start + required_duration > move_info['constraints']['latest_end']:
                continue
            
            # 创建临时调度来测试移动是否会造成冲突
            test_schedule = copy.deepcopy(optimized)
            test_event = test_schedule[event_idx]
            
            if move_info.get('is_whole_task'):
                # 整体移动纯NPU任务
                if hasattr(test_event, 'sub_segment_schedule'):
                    new_sub_schedule = []
                    time_shift = available_start - test_event.start_time
                    
                    for sub_id, start, end in test_event.sub_segment_schedule:
                        new_start = start + time_shift
                        new_end = end + time_shift
                        new_sub_schedule.append((sub_id, new_start, new_end))
                    
                    test_event.sub_segment_schedule = new_sub_schedule
                    test_event.start_time = min(s[1] for s in new_sub_schedule)
                    test_event.end_time = max(s[2] for s in new_sub_schedule)
            else:
                # 只移动NPU段到空隙，保持其他段的相对位置
                if hasattr(test_event, 'sub_segment_schedule'):
                    new_sub_schedule = []
                    
                    for sub_id, start, end in test_event.sub_segment_schedule:
                        if sub_id == segment['sub_id']:
                            # 这是要移动的NPU段
                            new_start = available_start
                            new_end = new_start + (end - start)
                            new_sub_schedule.append((sub_id, new_start, new_end))
                        else:
                            # 保持其他段不变
                            new_sub_schedule.append((sub_id, start, end))
                    
                    test_event.sub_segment_schedule = new_sub_schedule
                    test_event.start_time = min(s[1] for s in new_sub_schedule)
                    test_event.end_time = max(s[2] for s in new_sub_schedule)
                
                # 验证移动后是否有冲突
                moved_sub_id = segment['sub_id'] if segment else None
                if self._check_no_conflicts_after_move(test_schedule, event_idx, moved_sub_id):
                    # 无冲突，应用移动
                    optimized[event_idx] = test_event
                    
                    # 记录空隙使用
                    if move_info.get('is_whole_task'):
                        # 整体任务移动，记录所有段的时间
                        for sub_id, start, end in test_event.sub_segment_schedule:
                            gap_usage[gap_idx].append((start, end))
                    else:
                        gap_usage[gap_idx].append((available_start, available_start + required_duration))
                    
                    moved_events.add(event_idx)
                    
                    total_moved += 1
                    if move_info.get('is_whole_task'):
                        # 计算整体任务的NPU时间
                        npu_time = 0
                        for sub_id, start, end in test_event.sub_segment_schedule:
                            if 'npu' in sub_id.lower() or 'main' in sub_id:
                                npu_time += (end - start)
                        total_gap_utilized += npu_time
                    else:
                        total_gap_utilized += required_duration
                    
                    if move_info.get('is_whole_task'):
                        print(f"  ✓ 移动 {move_info['task_id']} (整体): "
                              f"{test_event.start_time:.1f} -> {available_start:.1f}ms")
                    else:
                        print(f"  ✓ 移动 {move_info['task_id']}.{segment['sub_id']}: "
                              f"{segment['start']:.1f} -> {available_start:.1f}ms")
                else:
                    if move_info.get('is_whole_task'):
                        print(f"  ✗ 跳过 {move_info['task_id']} (整体): 会造成冲突")
                    else:
                        print(f"  ✗ 跳过 {move_info['task_id']}.{segment['sub_id']}: 会造成冲突")
        
        print(f"\n📈 优化结果:")
        print(f"  - 移动了 {total_moved} 个段")
        print(f"  - 利用了 {total_gap_utilized:.1f}ms 的空隙时间")
        
        # 重新排序事件
        optimized.sort(key=lambda x: x.start_time)
        
        return optimized
    
    def _check_no_conflicts_after_move(self, schedule: List[TaskScheduleInfo], 
                                      moved_event_idx: int, moved_sub_id: Optional[str]) -> bool:
        """检查移动后是否会造成资源冲突"""
        moved_event = schedule[moved_event_idx]
        
        # 如果是整体移动，检查所有段
        if moved_sub_id is None:
            # 检查该事件的所有段
            for sub_id, start, end in moved_event.sub_segment_schedule:
                if not self._check_segment_no_conflict(schedule, moved_event_idx, sub_id, start, end):
                    return False
            return True
        else:
            # 只检查特定的段
            for sub_id, start, end in moved_event.sub_segment_schedule:
                if sub_id == moved_sub_id:
                    return self._check_segment_no_conflict(schedule, moved_event_idx, sub_id, start, end)
            return True
    
    def _check_segment_no_conflict(self, schedule: List[TaskScheduleInfo], 
                                  event_idx: int, sub_id: str, start: float, end: float) -> bool:
        """检查特定段是否有冲突"""
        event = schedule[event_idx]
        task = self.scheduler.tasks.get(event.task_id)
        if not task:
            return True
        
        # 找到段的资源类型
        resource_type = None
        for sub_seg in task.get_sub_segments_for_scheduling():
            if sub_seg.sub_id == sub_id:
                resource_type = sub_seg.resource_type
                break
        
        if not resource_type:
            return True
        
        # 检查与其他事件的冲突
        for idx, other_event in enumerate(schedule):
            if idx == event_idx:
                continue
            
            if hasattr(other_event, 'sub_segment_schedule'):
                for other_sub_id, other_start, other_end in other_event.sub_segment_schedule:
                    # 找到这个子段的资源类型
                    other_task = self.scheduler.tasks.get(other_event.task_id)
                    if other_task:
                        for sub_seg in other_task.get_sub_segments_for_scheduling():
                            if sub_seg.sub_id == other_sub_id and sub_seg.resource_type == resource_type:
                                # 检查时间重叠
                                if not (other_end <= start or other_start >= end):
                                    return False  # 有冲突
                                break
        
        return True  # 无冲突


def main():
    """主测试函数 - 使用真实任务测试空隙感知调度"""
    print("=" * 80)
    print("🚀 空隙感知调度测试 - 真实任务场景（含依赖关系）")
    print("=" * 80)
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=40.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    # 应用基础修复
    print("\n应用调度修复...")
    fix_manager = apply_basic_fixes(scheduler)
    
    # 创建真实任务
    print("\n创建真实任务...")
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    # 应用FIFO和严格资源冲突修复（参考main_genetic.py）
    apply_minimal_fifo_fix(scheduler)
    apply_strict_resource_conflict_fix(scheduler)
    
    # 打印任务依赖关系
    print("\n📊 任务依赖关系:")
    for task in tasks:
        if task.dependencies:
            print(f"  {task.task_id} ({task.name}) 依赖于: {list(task.dependencies)}")
    
    # 执行基础调度
    print("\n=== 第一阶段：基础调度 ===")
    time_window = 200.0
    scheduler.schedule_history.clear()
    
    try:
        baseline_results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    except Exception as e:
        print(f"\n❌ 调度失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 验证基础调度
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    if not is_valid:
        print(f"\n⚠️ 基础调度有冲突：{len(conflicts)}个")
        for conflict in conflicts[:3]:
            print(f"  - {conflict}")
    else:
        print("✅ 基础调度无冲突")
    
    # 分析基础调度
    print("\n📊 基础调度分析:")
    task_counts = defaultdict(int)
    total_duration = 0
    for event in scheduler.schedule_history:
        task_counts[event.task_id] += 1
        total_duration += (event.end_time - event.start_time)
    
    print(f"  - 总事件数: {len(scheduler.schedule_history)}")
    print(f"  - 总执行时间: {total_duration:.1f}ms")
    print(f"  - 任务执行次数:")
    for task_id in sorted(task_counts.keys()):
        task = scheduler.tasks[task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        actual = task_counts[task_id]
        fps_rate = actual / expected if expected > 0 else 1.0
        print(f"    {task_id} ({task.name}): {actual}/{expected} ({fps_rate:.1%})")
    
    # 保存基础调度
    baseline_schedule = copy.deepcopy(scheduler.schedule_history)
    
    # 创建依赖感知的空隙调度器
    gap_scheduler = DependencyAwareGapScheduler(scheduler)
    
    # === 第二阶段：空隙优化 ===
    print("\n=== 第二阶段：依赖感知的空隙优化 ===")
    
    # 执行空隙优化
    optimized_schedule = gap_scheduler.create_gap_aware_schedule(baseline_schedule)
    scheduler.schedule_history = optimized_schedule
    
    # 验证优化后的调度
    is_valid_after, conflicts_after = validate_schedule_correctly(scheduler)
    if not is_valid_after:
        print(f"\n⚠️ 优化后有冲突：{len(conflicts_after)}个")
        for conflict in conflicts_after[:3]:
            print(f"  - {conflict}")
    else:
        print("✅ 优化后无冲突")
    
    # 验证依赖关系
    print("\n🔍 验证依赖关系:")
    dependency_violations = 0
    for event in optimized_schedule:
        task = scheduler.tasks[event.task_id]
        for dep_id in task.dependencies:
            # 找到依赖任务的所有执行
            dep_events = [e for e in optimized_schedule if e.task_id == dep_id]
            if dep_events:
                # 检查是否有依赖任务在当前任务之前完成
                valid_dep = any(dep_e.end_time <= event.start_time for dep_e in dep_events)
                if not valid_dep:
                    print(f"  ❌ {event.task_id} 违反了对 {dep_id} 的依赖")
                    dependency_violations += 1
    
    if dependency_violations == 0:
        print("  ✅ 所有依赖关系都得到满足")
    
    # 分析优化效果
    print("\n📊 优化效果分析:")
    
    # 计算资源利用率
    resource_busy = defaultdict(lambda: defaultdict(float))  # resource_type -> resource_id -> busy_time
    for event in optimized_schedule:
        task = scheduler.tasks.get(event.task_id)
        if task and hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
            for sub_id, start, end in event.sub_segment_schedule:
                duration = end - start
                # 找到对应的资源
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        res_type = sub_seg.resource_type
                        if res_type in event.assigned_resources:
                            res_id = event.assigned_resources[res_type]
                            resource_busy[res_type.value][res_id] += duration
                        break
    
    print(f"  - 资源利用率:")
    for res_type in ['NPU', 'DSP']:
        if res_type in resource_busy:
            total_busy = sum(resource_busy[res_type].values())
            num_resources = len([r for r in scheduler.resources.get(ResourceType[res_type], [])])
            if num_resources > 0:
                utilization = total_busy / (time_window * num_resources) * 100
                print(f"    {res_type}: {utilization:.1f}% (总忙碌时间: {total_busy:.1f}ms)")
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 16))
    
    # 基线调度
    scheduler.schedule_history = baseline_schedule
    viz1 = ElegantSchedulerVisualizer(scheduler)
    plt.sca(ax1)
    viz1.plot_elegant_gantt(time_window=time_window, show_all_labels=True)
    ax1.set_title('Baseline Schedule', fontsize=16, pad=20)
    
    # 优化后调度
    scheduler.schedule_history = optimized_schedule
    viz2 = ElegantSchedulerVisualizer(scheduler)
    plt.sca(ax2)
    viz2.plot_elegant_gantt(time_window=time_window, show_all_labels=True)
    ax2.set_title('Gap-Aware Optimized Schedule (Dependency-Aware)', fontsize=16, pad=20)
    
    plt.tight_layout()
    plt.savefig('gap_aware_real_tasks_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 生成Chrome trace
    viz2.export_chrome_tracing('gap_aware_real_tasks_optimized.json')
    
    print("\n✅ 测试完成！")
    print("\n生成的文件：")
    print("  - gap_aware_real_tasks_comparison.png")
    print("  - gap_aware_real_tasks_optimized.json")
    
    # 最终统计
    print("\n" + "=" * 60)
    print("📊 最终统计")
    print("=" * 60)
    
    # 任务执行统计
    optimized_task_counts = defaultdict(int)
    for event in optimized_schedule:
        optimized_task_counts[event.task_id] += 1
    
    print("\n任务执行对比:")
    print(f"{'任务':<8} {'基线':<10} {'优化后':<10} {'FPS要求':<10} {'满足率':<10}")
    print("-" * 50)
    
    for task_id in sorted(task_counts.keys()):
        task = scheduler.tasks[task_id]
        baseline_count = task_counts[task_id]
        optimized_count = optimized_task_counts[task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        satisfaction = optimized_count / expected if expected > 0 else 1.0
        
        print(f"{task_id:<8} {baseline_count:<10} {optimized_count:<10} "
              f"{expected:<10} {satisfaction:.1%}")


if __name__ == "__main__":
    main()
