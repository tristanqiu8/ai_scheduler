#!/usr/bin/env python3
"""
正确的空隙填充调度器
基于现有调度，仅在空隙中添加缺失的任务执行
"""

import sys
import os
import copy
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict

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


class CorrectGapFiller:
    """正确的空隙填充器 - 只在现有调度的空隙中添加任务"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        self.resource_busy_times = defaultdict(list)  # resource_id -> [(start, end)]
        
    def analyze_and_fill_gaps(self, baseline_schedule: List[TaskScheduleInfo], 
                             time_window: float = 200.0) -> List[TaskScheduleInfo]:
        """分析空隙并填充缺失的任务执行"""
        print("\n🔍 分析现有调度并填充空隙...")
        
        # 1. 构建资源占用时间线
        self._build_resource_timeline(baseline_schedule)
        
        # 先打印资源占用情况
        print("\n📊 资源占用分析:")
        for res_id in ['NPU_0', 'DSP_0']:
            busy_times = self.resource_busy_times.get(res_id, [])
            total_busy = sum(end - start for start, end in busy_times)
            print(f"  {res_id}: {len(busy_times)}个忙碌时段，总计{total_busy:.1f}ms")
            
            # 找出较大的空隙
            gaps = []
            if busy_times:
                if busy_times[0][0] > 5:
                    gaps.append((0, busy_times[0][0]))
                for i in range(len(busy_times) - 1):
                    gap_start = busy_times[i][1]
                    gap_end = busy_times[i + 1][0]
                    if gap_end - gap_start > 5:
                        gaps.append((gap_start, gap_end))
                if busy_times[-1][1] < time_window - 5:
                    gaps.append((busy_times[-1][1], time_window))
            else:
                gaps.append((0, time_window))
                
            if gaps:
                print(f"    主要空隙: ", end="")
                for start, end in gaps[:3]:
                    print(f"{start:.1f}-{end:.1f}ms({end-start:.1f}ms) ", end="")
                print()
        
        # 2. 找出需要更多执行的任务
        tasks_needing_runs = self._find_tasks_needing_more_runs(baseline_schedule, time_window)
        
        if not tasks_needing_runs:
            print("  ✅ 所有任务已满足FPS要求")
            return baseline_schedule
        
        print(f"\n📋 需要额外执行的任务:")
        for task_id, info in tasks_needing_runs.items():
            print(f"  {task_id} ({info['task'].name}): 需要{info['needed']}次 "
                  f"(当前{info['current']}/{info['expected']})")
        
        # 3. 复制基线调度（保留所有现有事件）
        new_schedule = copy.deepcopy(baseline_schedule)
        
        # 4. 为每个缺失的任务执行寻找空隙
        total_added = 0
        
        for task_id, info in sorted(tasks_needing_runs.items(), 
                                   key=lambda x: (x[1]['task'].priority.value, -x[1]['needed'])):
            task = info['task']
            needed = info['needed']
            
            print(f"\n为 {task_id} 寻找 {needed} 个空隙:")
            
            # 获取任务现有的执行时间
            existing_times = []
            for event in new_schedule:
                if event.task_id == task_id:
                    existing_times.append(event.start_time)
            existing_times.sort()
            
            # 寻找可用的空隙
            added_count = 0
            attempts = 0
            max_attempts = 50  # 防止无限循环
            
            while added_count < needed and attempts < max_attempts:
                attempts += 1
                gap_found = self._find_gap_for_task(task, existing_times, time_window)
                
                if gap_found:
                    start_time, resources = gap_found
                    
                    # 创建新的任务执行事件
                    new_event = self._create_task_event(task, start_time, resources)
                    new_schedule.append(new_event)
                    existing_times.append(start_time)
                    existing_times.sort()
                    
                    # 更新资源占用
                    self._update_resource_timeline(new_event, task)
                    
                    added_count += 1
                    total_added += 1
                    print(f"  ✓ 找到空隙 {start_time:.1f}ms")
                else:
                    print(f"  ✗ 无法找到第{added_count+1}个合适的空隙")
                    break
            
            if added_count < needed:
                print(f"  ⚠️ 只找到 {added_count}/{needed} 个空隙")
        
        print(f"\n📊 总计在空隙中添加了 {total_added} 次任务执行")
        
        # 5. 按时间排序
        new_schedule.sort(key=lambda x: x.start_time)
        
        return new_schedule
    
    def _build_resource_timeline(self, schedule: List[TaskScheduleInfo]):
        """构建资源占用时间线"""
        self.resource_busy_times.clear()
        
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
                                self.resource_busy_times[res_id].append((start, end))
                            break
            else:
                # 非分段任务
                for seg in task.segments:
                    if seg.resource_type in event.assigned_resources:
                        res_id = event.assigned_resources[seg.resource_type]
                        resource = next((r for r in self.scheduler.resources[seg.resource_type] 
                                       if r.unit_id == res_id), None)
                        if resource:
                            duration = seg.get_duration(resource.bandwidth)
                            start_time = event.start_time + seg.start_time
                            end_time = start_time + duration
                            self.resource_busy_times[res_id].append((start_time, end_time))
        
        # 排序时间线
        for res_id in self.resource_busy_times:
            self.resource_busy_times[res_id].sort()
    
    def _find_tasks_needing_more_runs(self, schedule: List[TaskScheduleInfo], 
                                    time_window: float) -> Dict[str, Dict]:
        """找出需要更多执行次数的任务"""
        task_counts = defaultdict(int)
        for event in schedule:
            task_counts[event.task_id] += 1
        
        tasks_needing_runs = {}
        for task_id, task in self.scheduler.tasks.items():
            expected = int((time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            if actual < expected:
                tasks_needing_runs[task_id] = {
                    'task': task,
                    'needed': expected - actual,
                    'current': actual,
                    'expected': expected
                }
        
        return tasks_needing_runs
    
    def _find_gap_for_task(self, task: NNTask, existing_times: List[float], 
                          time_window: float) -> Optional[Tuple[float, Dict[ResourceType, str]]]:
        """为任务找到合适的空隙"""
        # 获取任务需要的资源和执行时间
        required_resources = {}
        resource_durations = {}
        
        for seg in task.segments:
            resources = self.scheduler.resources.get(seg.resource_type, [])
            if resources:
                # 暂时分配第一个资源
                required_resources[seg.resource_type] = resources[0].unit_id
                resource_durations[resources[0].unit_id] = seg.get_duration(resources[0].bandwidth)
        
        # 计算任务的总执行时间
        task_duration = max(seg.get_duration(40.0) + seg.start_time for seg in task.segments)
        
        # 收集所有资源的空闲时间段
        all_gaps = []
        
        # 对每个需要的资源，找出其空闲时段
        for res_id in resource_durations:
            busy_times = self.resource_busy_times.get(res_id, [])
            if not busy_times:
                # 资源完全空闲
                all_gaps.append((0, time_window))
                continue
                
            # 找出空闲时段
            gaps = []
            if busy_times[0][0] > 0:
                gaps.append((0, busy_times[0][0]))
            
            for i in range(len(busy_times) - 1):
                gap_start = busy_times[i][1]
                gap_end = busy_times[i + 1][0]
                if gap_end - gap_start > task_duration:
                    gaps.append((gap_start, gap_end))
            
            if busy_times[-1][1] < time_window:
                gaps.append((busy_times[-1][1], time_window))
            
            all_gaps.extend(gaps)
        
        # 去重并排序空隙
        unique_gaps = list(set(all_gaps))
        unique_gaps.sort()
        
        # 在每个空隙中尝试放置任务
        for gap_start, gap_end in unique_gaps:
            # 检查是否有足够的空间
            if gap_end - gap_start < task_duration:
                continue
                
            # 在空隙内搜索有效的开始时间
            test_time = gap_start
            
            while test_time + task_duration <= gap_end and test_time + task_duration <= time_window:
                # 检查最小间隔约束
                valid = True
                for exist_time in existing_times:
                    if abs(test_time - exist_time) < task.min_interval_ms:
                        valid = False
                        break
                
                if valid:
                    # 检查所有资源在这个时间是否都可用
                    all_available = True
                    for res_id, duration in resource_durations.items():
                        if self._is_resource_busy(res_id, test_time, test_time + duration):
                            all_available = False
                            break
                    
                    if all_available:
                        return (test_time, required_resources)
                
                # 尝试下一个时间点
                test_time += 1.0  # 以1ms为步长搜索
        
        return None
    
    def _is_resource_busy(self, resource_id: str, start_time: float, end_time: float) -> bool:
        """检查资源在指定时间段是否忙碌"""
        for busy_start, busy_end in self.resource_busy_times.get(resource_id, []):
            # 检查时间段是否有重叠
            if not (end_time <= busy_start or start_time >= busy_end):
                return True
        return False
    
    def _create_task_event(self, task: NNTask, start_time: float, 
                          resources: Dict[ResourceType, str]) -> TaskScheduleInfo:
        """创建任务执行事件"""
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
                resource = next((r for r in self.scheduler.resources[seg.resource_type] 
                               if r.unit_id == resources[seg.resource_type]), None)
                if resource:
                    duration = seg.get_duration(resource.bandwidth)
                    seg_start = start_time + seg.start_time
                    seg_end = seg_start + duration
                    end_time = max(end_time, seg_end)
                    sub_schedule.append((f"{seg.segment_id}_0", seg_start, seg_end))
        
        event = TaskScheduleInfo(
            task_id=task.task_id,
            start_time=start_time,
            end_time=end_time,
            assigned_resources=resources,
            actual_latency=end_time - start_time,
            runtime_type=task.runtime_type
        )
        
        if sub_schedule:
            event.sub_segment_schedule = sub_schedule
        
        return event
    
    def _update_resource_timeline(self, event: TaskScheduleInfo, task: NNTask):
        """更新资源占用时间线"""
        if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
            for sub_id, start, end in event.sub_segment_schedule:
                # 找到对应的资源
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        res_type = sub_seg.resource_type
                        if res_type in event.assigned_resources:
                            res_id = event.assigned_resources[res_type]
                            self.resource_busy_times[res_id].append((start, end))
                            self.resource_busy_times[res_id].sort()
                        break
        else:
            # 非分段任务
            for seg in task.segments:
                if seg.resource_type in event.assigned_resources:
                    res_id = event.assigned_resources[seg.resource_type]
                    resource = next((r for r in self.scheduler.resources[seg.resource_type] 
                                   if r.unit_id == res_id), None)
                    if resource:
                        duration = seg.get_duration(resource.bandwidth)
                        start_time = event.start_time + seg.start_time
                        end_time = start_time + duration
                        self.resource_busy_times[res_id].append((start_time, end_time))
                        self.resource_busy_times[res_id].sort()


def main():
    """主测试函数"""
    print("=" * 80)
    print("🎯 正确的空隙填充调度测试")
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
    
    # 应用FIFO和严格资源冲突修复
    apply_minimal_fifo_fix(scheduler)
    apply_strict_resource_conflict_fix(scheduler)
    
    # 执行基础调度
    print("\n=== 第一阶段：基础调度 ===")
    time_window = 200.0
    scheduler.schedule_history.clear()
    baseline_results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    # 验证基础调度
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    print(f"基础调度验证: {'✅ 无冲突' if is_valid else f'❌ {len(conflicts)}个冲突'}")
    
    # 分析基础调度FPS
    print("\n📊 基础调度FPS分析:")
    task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        task_counts[event.task_id] += 1
    
    for task_id in sorted(task_counts.keys()):
        task = scheduler.tasks[task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        actual = task_counts[task_id]
        fps_rate = actual / expected if expected > 0 else 1.0
        status = "✅" if fps_rate >= 0.95 else "❌"
        print(f"  {status} {task_id} ({task.name}): {actual}/{expected} ({fps_rate:.1%})")
    
    # 保存基础调度
    baseline_schedule = copy.deepcopy(scheduler.schedule_history)
    
    # === 第二阶段：空隙填充 ===
    print("\n=== 第二阶段：空隙填充 ===")
    
    gap_filler = CorrectGapFiller(scheduler)
    filled_schedule = gap_filler.analyze_and_fill_gaps(baseline_schedule, time_window)
    
    # 更新调度历史
    scheduler.schedule_history = filled_schedule
    
    # 验证填充后的调度
    is_valid_after, conflicts_after = validate_schedule_correctly(scheduler)
    print(f"\n填充后验证: {'✅ 无冲突' if is_valid_after else f'❌ {len(conflicts_after)}个冲突'}")
    
    # 最终FPS分析
    print("\n📊 填充后FPS分析:")
    final_task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        final_task_counts[event.task_id] += 1
    
    all_fps_ok = True
    for task_id in sorted(final_task_counts.keys()):
        task = scheduler.tasks[task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        actual = final_task_counts[task_id]
        fps_rate = actual / expected if expected > 0 else 1.0
        status = "✅" if fps_rate >= 0.95 else "❌"
        if fps_rate < 0.95:
            all_fps_ok = False
        
        baseline_count = task_counts.get(task_id, 0)
        improvement = actual - baseline_count
        improvement_str = f"[+{improvement}]" if improvement > 0 else ""
        print(f"  {status} {task_id} ({task.name}): {actual}/{expected} ({fps_rate:.1%}) {improvement_str}")
    
    # 生成可视化
    print("\n生成可视化...")
    viz = ElegantSchedulerVisualizer(scheduler)
    plt.figure(figsize=(20, 10))
    viz.plot_elegant_gantt(time_window=time_window, show_all_labels=True)
    plt.title('Gap-Filled Schedule (Correct Approach)', fontsize=16, pad=20)
    plt.savefig('correct_gap_filled_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chrome trace
    viz.export_chrome_tracing('correct_gap_filled_schedule.json')
    
    # 资源利用率分析
    print("\n📊 资源利用率分析:")
    resource_busy = defaultdict(float)
    for event in scheduler.schedule_history:
        task = scheduler.tasks.get(event.task_id)
        if task and hasattr(event, 'sub_segment_schedule'):
            for sub_id, start, end in event.sub_segment_schedule:
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        res_type = sub_seg.resource_type
                        if res_type in event.assigned_resources:
                            res_id = event.assigned_resources[res_type]
                            resource_busy[res_id] += (end - start)
                        break
    
    for res_id in ['NPU_0', 'DSP_0']:
        if res_id in resource_busy:
            utilization = resource_busy[res_id] / time_window * 100
            print(f"  {res_id}: {utilization:.1f}% (忙碌 {resource_busy[res_id]:.1f}ms)")
    
    print("\n✅ 测试完成！")
    print(f"\n📊 优化效果总结:")
    print(f"  - FPS达标: {'是 ✅' if all_fps_ok else '否 ❌'}")
    print(f"  - 资源冲突: {'无 ✅' if is_valid_after else '有 ❌'}")
    
    print("\n生成的文件：")
    print("  - correct_gap_filled_schedule.png")
    print("  - correct_gap_filled_schedule.json")


if __name__ == "__main__":
    main()
