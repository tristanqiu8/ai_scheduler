#!/usr/bin/env python3
"""
完整的调度优化流程
流程：贪心调度 → 插空隙 → compactor → 满足帧率 → 插空隙 → compactor
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
from core.debug_compactor import DebugCompactor
from scenario.real_task import create_real_tasks
from viz.elegant_visualization import ElegantSchedulerVisualizer
import matplotlib.pyplot as plt


class CompleteSchedulerOptimizer:
    """完整的调度优化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        self.resource_busy_times = defaultdict(list)  # resource_id -> [(start, end, task_id)]
        
    def optimize_complete(self, max_iterations: int = 3) -> List[TaskScheduleInfo]:
        """完整优化流程"""
        print("\n🚀 开始完整的调度优化流程")
        print("=" * 60)
        
        # 第一步：贪心调度
        print("\n[步骤1] 执行贪心调度...")
        self.scheduler.schedule_history.clear()
        current_schedule = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
        self._print_fps_status(current_schedule, "贪心调度")
        
        # 迭代优化
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"第 {iteration + 1} 轮优化")
            print(f"{'='*60}")
            
            # 步骤2：插空隙
            print(f"\n[步骤2-{iteration+1}] 第一次插空隙...")
            current_schedule = self._fill_gaps(current_schedule)
            self._print_fps_status(current_schedule, f"第{iteration+1}轮插空隙")
            
            # 步骤3：紧凑化
            print(f"\n[步骤3-{iteration+1}] 执行紧凑化...")
            current_schedule, idle_time = self._compact_schedule(current_schedule)
            print(f"  ✓ 紧凑化完成，末尾空闲时间: {idle_time:.1f}ms ({idle_time/self.time_window*100:.1f}%)")
            self._print_fps_status(current_schedule, f"第{iteration+1}轮紧凑化")
            
            # 步骤4：满足帧率（贪心补充）
            print(f"\n[步骤4-{iteration+1}] 贪心补充未达标任务...")
            current_schedule = self._greedy_fill_fps(current_schedule)
            self._print_fps_status(current_schedule, f"第{iteration+1}轮贪心补充")
            
            # 步骤5：再次插空隙
            print(f"\n[步骤5-{iteration+1}] 第二次插空隙...")
            current_schedule = self._fill_gaps(current_schedule)
            self._print_fps_status(current_schedule, f"第{iteration+1}轮第二次插空隙")
            
            # 步骤6：最终紧凑化
            print(f"\n[步骤6-{iteration+1}] 最终紧凑化...")
            current_schedule, idle_time = self._compact_schedule(current_schedule)
            print(f"  ✓ 最终空闲时间: {idle_time:.1f}ms ({idle_time/self.time_window*100:.1f}%)")
            
            # 检查是否所有任务都达标
            if self._check_all_fps_satisfied(current_schedule):
                print(f"\n✅ 第{iteration+1}轮优化后所有任务FPS达标！")
                break
            else:
                print(f"\n⚠️ 第{iteration+1}轮优化后仍有任务未达标，继续优化...")
        
        return current_schedule
    
    def _compact_schedule(self, schedule: List[TaskScheduleInfo]) -> Tuple[List[TaskScheduleInfo], float]:
        """使用DebugCompactor进行紧凑化"""
        # 临时更新调度器的历史
        self.scheduler.schedule_history = copy.deepcopy(schedule)
        
        # 创建紧凑化器
        compactor = DebugCompactor(self.scheduler, self.time_window)
        
        # 执行紧凑化
        compacted_events, idle_time = compactor.simple_compact()
        
        return compacted_events, idle_time
    
    def _fill_gaps(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """在空隙中填充任务"""
        # 重建资源时间线
        self._rebuild_resource_timeline(schedule)
        
        # 找出需要更多执行的任务
        tasks_needing_runs = self._find_tasks_needing_more_runs(schedule)
        
        if not tasks_needing_runs:
            return schedule
        
        new_schedule = copy.deepcopy(schedule)
        
        # 为每个任务寻找空隙
        for task_id, info in sorted(tasks_needing_runs.items(), 
                                   key=lambda x: (x[1]['task'].priority.value, -x[1]['needed'])):
            task = info['task']
            needed = info['needed']
            
            # 获取已有执行时间
            existing_times = [e.start_time for e in new_schedule if e.task_id == task_id]
            
            added = 0
            for _ in range(needed):
                gap_found = self._find_gap_for_task(task, existing_times)
                if gap_found:
                    start_time, resources = gap_found
                    new_event = self._create_task_event(task, start_time, resources)
                    new_schedule.append(new_event)
                    existing_times.append(start_time)
                    self._update_resource_timeline(new_event, task)
                    added += 1
                else:
                    break
            
            if added > 0:
                print(f"    {task_id}: 在空隙中添加了 {added} 次执行")
        
        new_schedule.sort(key=lambda x: x.start_time)
        return new_schedule
    
    def _greedy_fill_fps(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """贪心地补充未达FPS要求的任务"""
        # 重建资源时间线
        self._rebuild_resource_timeline(schedule)
        
        # 找出未达标的任务
        tasks_needing_runs = self._find_tasks_needing_more_runs(schedule)
        
        if not tasks_needing_runs:
            return schedule
        
        new_schedule = copy.deepcopy(schedule)
        
        # 按优先级排序任务
        for task_id, info in sorted(tasks_needing_runs.items(), 
                                   key=lambda x: x[1]['task'].priority.value):
            task = info['task']
            needed = info['needed']
            
            # 获取已有执行时间
            existing_times = [e.start_time for e in new_schedule if e.task_id == task_id]
            
            added = 0
            for _ in range(needed):
                # 贪心策略：找最早的可用时间
                earliest_time = self._find_earliest_available_time(task, existing_times)
                if earliest_time is not None and earliest_time < self.time_window:
                    resources = self._allocate_resources(task)
                    new_event = self._create_task_event(task, earliest_time, resources)
                    new_schedule.append(new_event)
                    existing_times.append(earliest_time)
                    self._update_resource_timeline(new_event, task)
                    added += 1
                else:
                    break
            
            if added > 0:
                print(f"    {task_id}: 贪心添加了 {added} 次执行")
        
        new_schedule.sort(key=lambda x: x.start_time)
        return new_schedule
    
    def _find_earliest_available_time(self, task: NNTask, existing_times: List[float]) -> Optional[float]:
        """找到任务的最早可用时间"""
        # 计算任务执行时间
        task_duration = max(seg.get_duration(40.0) + seg.start_time for seg in task.segments)
        
        # 从0开始搜索
        test_time = 0.0
        
        while test_time + task_duration <= self.time_window:
            # 检查最小间隔
            valid = True
            for exist_time in existing_times:
                if abs(test_time - exist_time) < task.min_interval_ms:
                    valid = False
                    break
            
            if valid:
                # 检查资源可用性
                resources_available = True
                for seg in task.segments:
                    res_type = seg.resource_type
                    resources = self.scheduler.resources.get(res_type, [])
                    if resources:
                        res_id = resources[0].unit_id
                        seg_start = test_time + seg.start_time
                        seg_duration = seg.get_duration(40.0)
                        if self._is_resource_busy(res_id, seg_start, seg_start + seg_duration):
                            resources_available = False
                            break
                
                if resources_available:
                    return test_time
            
            test_time += 1.0
        
        return None
    
    def _rebuild_resource_timeline(self, schedule: List[TaskScheduleInfo]):
        """重建资源占用时间线"""
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
                                self.resource_busy_times[res_id].append((start, end, event.task_id))
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
                            self.resource_busy_times[res_id].append((start_time, end_time, event.task_id))
        
        # 排序
        for res_id in self.resource_busy_times:
            self.resource_busy_times[res_id].sort()
    
    def _find_tasks_needing_more_runs(self, schedule: List[TaskScheduleInfo]) -> Dict[str, Dict]:
        """找出需要更多执行的任务"""
        task_counts = defaultdict(int)
        for event in schedule:
            task_counts[event.task_id] += 1
        
        tasks_needing_runs = {}
        for task_id, task in self.scheduler.tasks.items():
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            if actual < expected:
                tasks_needing_runs[task_id] = {
                    'task': task,
                    'needed': expected - actual,
                    'current': actual,
                    'expected': expected
                }
        
        return tasks_needing_runs
    
    def _find_gap_for_task(self, task: NNTask, existing_times: List[float]) -> Optional[Tuple[float, Dict[ResourceType, str]]]:
        """为任务找到合适的空隙"""
        # 获取任务需要的资源
        required_resources = {}
        resource_durations = {}
        
        for seg in task.segments:
            resources = self.scheduler.resources.get(seg.resource_type, [])
            if resources:
                required_resources[seg.resource_type] = resources[0].unit_id
                resource_durations[resources[0].unit_id] = (seg.start_time, seg.get_duration(40.0))
        
        # 计算任务总时长
        task_duration = max(seg.get_duration(40.0) + seg.start_time for seg in task.segments)
        
        # 查找所有资源的空闲时段交集
        all_gaps = self._find_resource_gaps()
        
        # 在空隙中寻找合适的位置
        for gap_start, gap_end in all_gaps:
            if gap_end - gap_start < task_duration:
                continue
            
            test_time = gap_start
            while test_time + task_duration <= gap_end:
                # 检查最小间隔
                valid = True
                for exist_time in existing_times:
                    if abs(test_time - exist_time) < task.min_interval_ms:
                        valid = False
                        break
                
                if valid:
                    # 检查所有资源段是否可用
                    all_available = True
                    for res_id, (offset, duration) in resource_durations.items():
                        seg_start = test_time + offset
                        if self._is_resource_busy(res_id, seg_start, seg_start + duration):
                            all_available = False
                            break
                    
                    if all_available:
                        return (test_time, required_resources)
                
                test_time += 1.0
        
        return None
    
    def _find_resource_gaps(self) -> List[Tuple[float, float]]:
        """找出所有资源的公共空闲时段"""
        # 简化：只考虑主要资源
        gaps = []
        
        # 获取NPU的空闲时段
        npu_gaps = self._get_resource_gaps('NPU_0')
        
        # 对每个NPU空闲时段，检查其他资源是否也有空闲
        for start, end in npu_gaps:
            gaps.append((start, end))
        
        return gaps
    
    def _get_resource_gaps(self, resource_id: str) -> List[Tuple[float, float]]:
        """获取单个资源的空闲时段"""
        busy_times = self.resource_busy_times.get(resource_id, [])
        if not busy_times:
            return [(0, self.time_window)]
        
        gaps = []
        if busy_times[0][0] > 0:
            gaps.append((0, busy_times[0][0]))
        
        for i in range(len(busy_times) - 1):
            gap_start = busy_times[i][1]
            gap_end = busy_times[i + 1][0]
            if gap_end - gap_start > 1:
                gaps.append((gap_start, gap_end))
        
        if busy_times[-1][1] < self.time_window:
            gaps.append((busy_times[-1][1], self.time_window))
        
        return gaps
    
    def _is_resource_busy(self, resource_id: str, start_time: float, end_time: float) -> bool:
        """检查资源在指定时间段是否忙碌"""
        for busy_start, busy_end, _ in self.resource_busy_times.get(resource_id, []):
            if not (end_time <= busy_start or start_time >= busy_end):
                return True
        return False
    
    def _create_task_event(self, task: NNTask, start_time: float, 
                          resources: Dict[ResourceType, str]) -> TaskScheduleInfo:
        """创建任务事件"""
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
            assigned_resources=resources,
            actual_latency=end_time - start_time,
            runtime_type=task.runtime_type
        )
        
        if sub_schedule:
            event.sub_segment_schedule = sub_schedule
        
        return event
    
    def _allocate_resources(self, task: NNTask) -> Dict[ResourceType, str]:
        """为任务分配资源"""
        resources = {}
        for seg in task.segments:
            res_list = self.scheduler.resources.get(seg.resource_type, [])
            if res_list:
                resources[seg.resource_type] = res_list[0].unit_id
        return resources
    
    def _update_resource_timeline(self, event: TaskScheduleInfo, task: NNTask):
        """更新资源时间线"""
        if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
            for sub_id, start, end in event.sub_segment_schedule:
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        res_type = sub_seg.resource_type
                        if res_type in event.assigned_resources:
                            res_id = event.assigned_resources[res_type]
                            self.resource_busy_times[res_id].append((start, end, event.task_id))
                            self.resource_busy_times[res_id].sort()
                        break
        else:
            for seg in task.segments:
                if seg.resource_type in event.assigned_resources:
                    res_id = event.assigned_resources[seg.resource_type]
                    duration = seg.get_duration(40.0)
                    start_time = event.start_time + seg.start_time
                    end_time = start_time + duration
                    self.resource_busy_times[res_id].append((start_time, end_time, event.task_id))
                    self.resource_busy_times[res_id].sort()
    
    def _check_all_fps_satisfied(self, schedule: List[TaskScheduleInfo]) -> bool:
        """检查是否所有任务都满足FPS要求"""
        task_counts = defaultdict(int)
        for event in schedule:
            task_counts[event.task_id] += 1
        
        for task_id, task in self.scheduler.tasks.items():
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            if actual < expected * 0.95:  # 95%容忍度
                return False
        return True
    
    def _print_fps_status(self, schedule: List[TaskScheduleInfo], stage_name: str):
        """打印FPS状态"""
        print(f"\n  {stage_name} FPS状态:")
        task_counts = defaultdict(int)
        for event in schedule:
            task_counts[event.task_id] += 1
        
        unsatisfied = []
        for task_id in sorted(self.scheduler.tasks.keys()):
            task = self.scheduler.tasks[task_id]
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            fps_rate = actual / expected if expected > 0 else 1.0
            
            if fps_rate < 0.95:
                unsatisfied.append(f"{task_id}:{actual}/{expected}({fps_rate:.0%})")
        
        if unsatisfied:
            print(f"    未达标: {', '.join(unsatisfied)}")
        else:
            print(f"    ✅ 所有任务达标")


def main():
    """主测试函数"""
    print("=" * 80)
    print("🚀 完整调度优化流程测试")
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
    
    # 创建优化器并执行完整优化
    optimizer = CompleteSchedulerOptimizer(scheduler, 200.0)
    final_schedule = optimizer.optimize_complete(max_iterations=3)
    
    # 更新调度器
    scheduler.schedule_history = final_schedule
    
    # 最终验证
    print("\n" + "=" * 60)
    print("📊 最终验证")
    print("=" * 60)
    
    # 验证冲突
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    print(f"\n资源冲突检查: {'✅ 无冲突' if is_valid else f'❌ {len(conflicts)}个冲突'}")
    
    # 验证FPS
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
    
    # 计算最终空闲时间
    if scheduler.schedule_history:
        last_end = max(e.end_time for e in scheduler.schedule_history)
        final_idle = 200.0 - last_end
        print(f"\n最终空闲时间: {final_idle:.1f}ms ({final_idle/200.0*100:.1f}%)")
    
    # 生成可视化
    print("\n生成可视化...")
    viz = ElegantSchedulerVisualizer(scheduler)
    plt.figure(figsize=(20, 10))
    viz.plot_elegant_gantt(time_window=200.0, show_all_labels=True)
    plt.title('Complete Optimized Schedule (Greedy→Gap→Compact→FPS→Gap→Compact)', fontsize=16, pad=20)
    plt.savefig('complete_optimized_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chrome trace
    viz.export_chrome_tracing('complete_optimized_schedule.json')
    
    print("\n✅ 优化完成！")
    print(f"\n优化结果: {'所有任务FPS达标' if all_satisfied else '仍有任务未达标'}")
    print("\n生成的文件：")
    print("  - complete_optimized_schedule.png")
    print("  - complete_optimized_schedule.json")


if __name__ == "__main__":
    main()
