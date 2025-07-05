#!/usr/bin/env python3
"""
修复版增强空隙感知调度测试
修复资源冲突问题，确保所有任务既满足FPS又无冲突
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


class SafeGapAwareScheduler:
    """安全的空隙感知调度器 - 避免资源冲突"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        self.dependency_graph = self._build_dependency_graph()
        # 资源占用时间线，用于冲突检测
        self.resource_timeline = defaultdict(list)  # resource_id -> [(start, end, task_id)]
        
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """构建任务依赖图"""
        dep_graph = defaultdict(set)
        for task_id, task in self.scheduler.tasks.items():
            for dep in task.dependencies:
                dep_graph[dep].add(task_id)
        return dict(dep_graph)
    
    def _rebuild_resource_timeline(self, schedule: List[TaskScheduleInfo]):
        """重建资源占用时间线"""
        self.resource_timeline.clear()
        
        for event in schedule:
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
            
            # 处理分段任务
            if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
                for sub_id, start, end in event.sub_segment_schedule:
                    # 找到对应的资源
                    for sub_seg in task.get_sub_segments_for_scheduling():
                        if sub_seg.sub_id == sub_id:
                            res_type = sub_seg.resource_type
                            if res_type in event.assigned_resources:
                                res_id = event.assigned_resources[res_type]
                                self.resource_timeline[res_id].append((start, end, event.task_id))
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
                            self.resource_timeline[res_id].append((start_time, end_time, event.task_id))
        
        # 排序时间线
        for res_id in self.resource_timeline:
            self.resource_timeline[res_id].sort()
    
    def _check_resource_conflict(self, resource_id: str, start_time: float, end_time: float, 
                                exclude_task: Optional[str] = None) -> bool:
        """检查指定时间段内是否有资源冲突"""
        for existing_start, existing_end, task_id in self.resource_timeline.get(resource_id, []):
            if exclude_task and task_id == exclude_task:
                continue
            # 检查时间重叠
            if not (end_time <= existing_start or start_time >= existing_end):
                return True  # 有冲突
        return False  # 无冲突
    
    def _find_safe_time_slot(self, resource_id: str, duration: float, 
                           earliest_start: float = 0.0) -> Optional[float]:
        """找到资源的安全时间槽"""
        current_time = earliest_start
        
        # 获取该资源的占用时间线
        busy_periods = sorted(self.resource_timeline.get(resource_id, []))
        
        for start, end, _ in busy_periods:
            if current_time + duration <= start:
                # 找到空隙
                return current_time
            current_time = max(current_time, end)
        
        # 检查最后一个忙碌时段后是否有空间
        if current_time + duration <= self.time_window:
            return current_time
        
        return None
    
    def create_safe_enhanced_schedule(self, baseline_schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """创建安全的增强调度（避免冲突）"""
        print("\n🛡️ 创建安全的增强调度...")
        
        # 初始化资源时间线
        self._rebuild_resource_timeline(baseline_schedule)
        
        # 1. 分析FPS缺口
        fps_deficit = self._analyze_fps_deficit(baseline_schedule)
        print(f"\n📊 FPS缺口分析:")
        for task_id, info in fps_deficit.items():
            print(f"  {task_id}: 需要额外 {info['deficit']} 次执行 (当前{info['fps_rate']:.1%})")
        
        # 2. 为未达标任务安排额外执行（使用安全的方法）
        schedule = copy.deepcopy(baseline_schedule)
        
        for task_id, deficit_info in sorted(fps_deficit.items(), 
                                           key=lambda x: (x[1]['task'].priority.value, -x[1]['deficit'])):
            task = deficit_info['task']
            needed = deficit_info['deficit']
            
            print(f"\n处理 {task_id} (需要{needed}次额外执行):")
            scheduled_count = 0
            
            # 为该任务找到安全的执行时间
            for i in range(needed):
                safe_time = self._find_safe_time_for_task(task)
                if safe_time is not None:
                    # 创建新事件
                    new_event = self._create_safe_task_event(task, safe_time)
                    if new_event:
                        schedule.append(new_event)
                        # 更新资源时间线
                        self._update_timeline_with_event(new_event, task)
                        scheduled_count += 1
                        print(f"  ✓ 安排在 {safe_time:.1f}ms 执行")
                else:
                    print(f"  ✗ 无法找到第{i+1}次执行的安全时间槽")
                    break
            
            if scheduled_count < needed:
                print(f"  ⚠️ 只能安排 {scheduled_count}/{needed} 次执行")
        
        # 3. 重新排序
        schedule.sort(key=lambda x: x.start_time)
        
        return schedule
    
    def _find_safe_time_for_task(self, task: NNTask) -> Optional[float]:
        """为任务找到安全的执行时间"""
        # 获取任务需要的所有资源
        required_resources = {}
        for seg in task.segments:
            resources = self.scheduler.resources.get(seg.resource_type, [])
            if resources:
                # 选择第一个可用资源
                required_resources[seg.resource_type] = resources[0].unit_id
        
        # 估算任务执行时间
        task_duration = self._estimate_task_duration(task)
        
        # 从当前时间开始搜索
        search_time = 0.0
        while search_time + task_duration <= self.time_window:
            # 检查所有需要的资源在这个时间段是否都可用
            all_available = True
            
            for res_type, res_id in required_resources.items():
                if self._check_resource_conflict(res_id, search_time, search_time + task_duration):
                    all_available = False
                    break
            
            if all_available:
                # 还需要检查依赖关系
                if self._check_dependencies_satisfied(task, search_time):
                    return search_time
            
            # 向前推进搜索时间
            search_time += 0.1
        
        return None
    
    def _check_dependencies_satisfied(self, task: NNTask, start_time: float) -> bool:
        """检查任务的依赖是否满足"""
        for dep_id in task.dependencies:
            # 检查依赖任务是否已经执行
            dep_executed = False
            for _, end, task_id in self.resource_timeline.get('NPU_0', []) + self.resource_timeline.get('DSP_0', []):
                if task_id == dep_id and end <= start_time:
                    dep_executed = True
                    break
            if not dep_executed:
                return False
        return True
    
    def _create_safe_task_event(self, task: NNTask, start_time: float) -> Optional[TaskScheduleInfo]:
        """创建安全的任务事件"""
        # 分配资源
        assigned_resources = {}
        for seg in task.segments:
            resources = self.scheduler.resources.get(seg.resource_type, [])
            if resources:
                assigned_resources[seg.resource_type] = resources[0].unit_id
        
        # 计算结束时间
        end_time = start_time
        sub_schedule = []
        
        if task.is_segmented:
            # 处理分段任务
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
            # 非分段任务
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
    
    def _update_timeline_with_event(self, event: TaskScheduleInfo, task: NNTask):
        """更新资源时间线"""
        if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
            for sub_id, start, end in event.sub_segment_schedule:
                # 找到对应的资源
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        res_type = sub_seg.resource_type
                        if res_type in event.assigned_resources:
                            res_id = event.assigned_resources[res_type]
                            self.resource_timeline[res_id].append((start, end, event.task_id))
                            self.resource_timeline[res_id].sort()
                        break
        else:
            # 处理非分段任务
            for seg in task.segments:
                if seg.resource_type in event.assigned_resources:
                    res_id = event.assigned_resources[seg.resource_type]
                    duration = seg.get_duration(40.0)
                    start_time = event.start_time + seg.start_time
                    end_time = start_time + duration
                    self.resource_timeline[res_id].append((start_time, end_time, event.task_id))
                    self.resource_timeline[res_id].sort()
    
    def _analyze_fps_deficit(self, baseline_schedule: List[TaskScheduleInfo]) -> Dict[str, Dict]:
        """分析FPS缺口"""
        task_counts = defaultdict(int)
        for event in baseline_schedule:
            task_counts[event.task_id] += 1
        
        fps_deficit = {}
        for task_id, task in self.scheduler.tasks.items():
            expected = int((self.time_window / 1000.0) * task.fps_requirement)
            actual = task_counts[task_id]
            if actual < expected:
                fps_deficit[task_id] = {
                    'task': task,
                    'expected': expected,
                    'actual': actual,
                    'deficit': expected - actual,
                    'fps_rate': actual / expected if expected > 0 else 1.0
                }
        
        return fps_deficit
    
    def _estimate_task_duration(self, task: NNTask) -> float:
        """估算任务执行时间"""
        total_duration = 0
        for seg in task.segments:
            duration = seg.get_duration(40.0)
            total_duration = max(total_duration, seg.start_time + duration)
        return total_duration


def safe_final_compaction(scheduler: MultiResourceScheduler, time_window: float):
    """安全的最终紧凑化（避免资源冲突）"""
    print("\n🔨 应用安全的最终紧凑化...")
    
    # 使用我们的安全调度器进行紧凑化
    safe_scheduler = SafeGapAwareScheduler(scheduler, time_window)
    safe_scheduler._rebuild_resource_timeline(scheduler.schedule_history)
    
    # 按开始时间排序事件
    events = sorted(scheduler.schedule_history, key=lambda x: x.start_time)
    compacted = []
    
    for event in events:
        task = scheduler.tasks.get(event.task_id)
        if not task:
            compacted.append(event)
            continue
        
        # 找到最早的安全时间
        earliest_safe_time = 0.0
        
        # 考虑依赖关系
        for dep_id in task.dependencies:
            for comp_event in compacted:
                if comp_event.task_id == dep_id:
                    earliest_safe_time = max(earliest_safe_time, comp_event.end_time)
        
        # 考虑同任务的前一次执行
        for comp_event in compacted:
            if comp_event.task_id == event.task_id:
                min_interval = task.min_interval_ms
                earliest_safe_time = max(earliest_safe_time, comp_event.start_time + min_interval)
        
        # 找到所有资源都可用的时间
        safe_time = safe_scheduler._find_safe_time_for_task(task)
        if safe_time is not None and safe_time >= earliest_safe_time:
            # 创建新事件
            duration = event.end_time - event.start_time
            new_event = copy.deepcopy(event)
            new_event.start_time = safe_time
            new_event.end_time = safe_time + duration
            
            # 调整子段时间
            if hasattr(new_event, 'sub_segment_schedule') and new_event.sub_segment_schedule:
                time_shift = safe_time - event.start_time
                new_sub_schedule = []
                for sub_id, start, end in new_event.sub_segment_schedule:
                    new_sub_schedule.append((sub_id, start + time_shift, end + time_shift))
                new_event.sub_segment_schedule = new_sub_schedule
            
            compacted.append(new_event)
            safe_scheduler._update_timeline_with_event(new_event, task)
        else:
            # 保持原位置
            compacted.append(event)
    
    # 计算空闲时间
    if compacted:
        last_end = max(e.end_time for e in compacted)
        idle_time = time_window - last_end
        print(f"  ✓ 安全紧凑化完成，末尾空闲时间: {idle_time:.1f}ms")
    else:
        idle_time = time_window
    
    scheduler.schedule_history = compacted


def main():
    """主测试函数"""
    print("=" * 80)
    print("🛡️ 安全增强版空隙感知调度测试")
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
    
    # 强制YOLO任务分段
    print("\n🔧 强制YOLO任务分段...")
    for task_id in ['T2', 'T3']:
        task = scheduler.tasks.get(task_id)
        if task:
            task.segmentation_strategy = SegmentationStrategy.FORCED_SEGMENTATION
            for segment in task.segments:
                if segment.segment_id == "main":
                    available_cuts = segment.get_available_cuts()
                    segment.apply_segmentation(available_cuts)
                    print(f"  ✓ {task_id} 分段为 {len(segment.sub_segments)} 个子段")
    
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
    
    # === 第二阶段：安全的空隙优化 ===
    print("\n=== 第二阶段：安全的空隙优化 ===")
    
    # 创建安全调度器
    safe_scheduler = SafeGapAwareScheduler(scheduler, time_window)
    
    # 执行安全优化
    optimized_schedule = safe_scheduler.create_safe_enhanced_schedule(baseline_schedule)
    scheduler.schedule_history = optimized_schedule
    
    # 验证优化后的调度
    is_valid_after, conflicts_after = validate_schedule_correctly(scheduler)
    print(f"\n优化后验证: {'✅ 无冲突' if is_valid_after else f'❌ {len(conflicts_after)}个冲突'}")
    if not is_valid_after:
        for conflict in conflicts_after[:5]:
            print(f"  - {conflict}")
    
    # === 第三阶段：安全的最终紧凑化 ===
    print("\n=== 第三阶段：安全的最终紧凑化 ===")
    safe_final_compaction(scheduler, time_window)
    
    # 最终验证
    is_valid_final, conflicts_final = validate_schedule_correctly(scheduler)
    print(f"\n最终验证: {'✅ 无冲突' if is_valid_final else f'❌ {len(conflicts_final)}个冲突'}")
    
    # 最终FPS验证
    print("\n📊 最终FPS验证:")
    final_task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        final_task_counts[event.task_id] += 1
    
    all_fps_met = True
    for task_id in sorted(final_task_counts.keys()):
        task = scheduler.tasks[task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        actual = final_task_counts[task_id]
        fps_rate = actual / expected if expected > 0 else 1.0
        
        status = "✅" if fps_rate >= 0.95 else "❌"
        print(f"  {status} {task_id} ({task.name}): {actual}/{expected} ({fps_rate:.1%})")
        
        if fps_rate < 0.95:
            all_fps_met = False
    
    print(f"\n{'✅ 所有任务FPS达标且无冲突！' if (all_fps_met and is_valid_final) else '⚠️ 存在问题需要解决'}")
    
    # 生成可视化
    print("\n生成可视化...")
    viz = ElegantSchedulerVisualizer(scheduler)
    plt.figure(figsize=(20, 10))
    viz.plot_elegant_gantt(time_window=time_window, show_all_labels=True)
    plt.title('Safe Enhanced Gap-Aware Schedule (No Conflicts)', fontsize=16, pad=20)
    plt.savefig('safe_enhanced_gap_aware_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chrome trace
    viz.export_chrome_tracing('safe_enhanced_gap_aware_schedule.json')
    
    # 资源利用率统计
    print("\n📊 资源利用率:")
    resource_busy = defaultdict(float)
    for event in scheduler.schedule_history:
        duration = event.end_time - event.start_time
        for res_type, res_id in event.assigned_resources.items():
            resource_busy[res_id] += duration
    
    for res_id in ['NPU_0', 'DSP_0']:
        if res_id in resource_busy:
            utilization = resource_busy[res_id] / time_window * 100
            print(f"  {res_id}: {utilization:.1f}%")
    
    print("\n✅ 测试完成！")
    print("\n生成的文件：")
    print("  - safe_enhanced_gap_aware_schedule.png")
    print("  - safe_enhanced_gap_aware_schedule.json")


if __name__ == "__main__":
    main()
