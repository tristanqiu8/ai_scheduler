#!/usr/bin/env python3
"""
修复冲突的增强优化器
解决分段任务导致的资源冲突问题
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


class ConflictFreeOptimizer:
    """修复冲突的优化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        self.resource_busy_times = defaultdict(list)  # resource_id -> [(start, end, task_id)]
        
    def optimize_complete(self, max_iterations: int = 3) -> List[TaskScheduleInfo]:
        """完整优化流程（修复冲突版）"""
        print("\n🚀 开始无冲突的完整调度优化流程")
        print("=" * 60)
        
        # 预处理：强制T2和T3使用最大分段
        self._force_segmentation_for_long_tasks()
        
        # 第一步：贪心调度
        print("\n[步骤1] 执行贪心调度...")
        self.scheduler.schedule_history.clear()
        current_schedule = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
        self._print_fps_status(current_schedule, "贪心调度")
        self._validate_and_print_conflicts(current_schedule, "贪心调度")
        
        # 迭代优化
        for iteration in range(max_iterations):
            print(f"\n{'='*60}")
            print(f"第 {iteration + 1} 轮优化")
            print(f"{'='*60}")
            
            # 步骤2：增强的插空隙（带冲突检查）
            print(f"\n[步骤2-{iteration+1}] 增强的插空隙...")
            current_schedule = self._enhanced_fill_gaps_safe(current_schedule)
            self._print_fps_status(current_schedule, f"第{iteration+1}轮插空隙")
            self._validate_and_print_conflicts(current_schedule, f"第{iteration+1}轮插空隙")
            
            # 步骤3：紧凑化
            print(f"\n[步骤3-{iteration+1}] 执行紧凑化...")
            current_schedule, idle_time = self._compact_schedule(current_schedule)
            print(f"  ✓ 紧凑化完成，末尾空闲时间: {idle_time:.1f}ms ({idle_time/self.time_window*100:.1f}%)")
            self._validate_and_print_conflicts(current_schedule, f"第{iteration+1}轮紧凑化")
            
            # 步骤4：满足帧率（贪心补充）
            print(f"\n[步骤4-{iteration+1}] 贪心补充未达标任务...")
            current_schedule = self._greedy_fill_fps_safe(current_schedule)
            self._print_fps_status(current_schedule, f"第{iteration+1}轮贪心补充")
            self._validate_and_print_conflicts(current_schedule, f"第{iteration+1}轮贪心补充")
            
            # 步骤5：最终紧凑化
            print(f"\n[步骤5-{iteration+1}] 最终紧凑化...")
            current_schedule, idle_time = self._compact_schedule(current_schedule)
            print(f"  ✓ 最终空闲时间: {idle_time:.1f}ms ({idle_time/self.time_window*100:.1f}%)")
            
            # 检查是否所有任务都达标
            if self._check_all_fps_satisfied(current_schedule):
                print(f"\n✅ 第{iteration+1}轮优化后所有任务FPS达标！")
                break
        
        return current_schedule
    
    def _validate_and_print_conflicts(self, schedule: List[TaskScheduleInfo], stage_name: str):
        """验证并打印冲突信息"""
        # 临时更新调度器历史以使用验证函数
        self.scheduler.schedule_history = schedule
        is_valid, conflicts = validate_schedule_correctly(self.scheduler)
        
        if not is_valid:
            print(f"\n  ⚠️ {stage_name}后发现{len(conflicts)}个冲突:")
            for i, conflict in enumerate(conflicts[:3]):  # 只显示前3个
                print(f"    - {conflict}")
            if len(conflicts) > 3:
                print(f"    ... 还有{len(conflicts)-3}个冲突")
        else:
            print(f"  ✅ {stage_name}后无冲突")
    
    def _force_segmentation_for_long_tasks(self):
        """强制T2和T3使用最大分段"""
        print("\n[预处理] 强制长任务分段...")
        
        for task_id in ['T2', 'T3']:
            task = self.scheduler.tasks.get(task_id)
            if task:
                # 确保使用CUSTOM_SEGMENTATION策略
                task.segmentation_strategy = SegmentationStrategy.CUSTOM_SEGMENTATION
                
                # 选择最大分段配置
                if hasattr(task, 'preset_cut_configurations'):
                    for seg_id, configs in task.preset_cut_configurations.items():
                        if configs:
                            # 选择cut点最多的配置（通常是最后一个）
                            max_cuts_idx = len(configs) - 1
                            task.select_cut_configuration(seg_id, max_cuts_idx)
                            print(f"  ✓ {task_id}: 选择配置{max_cuts_idx}（{len(configs[max_cuts_idx])}个切分点）")
                
                # 确保分段被应用
                for segment in task.segments:
                    if segment.segment_id == "main" and segment.cut_points:
                        # 获取所有可用的cut点
                        all_cuts = [cp.op_id for cp in segment.cut_points]
                        segment.apply_segmentation(all_cuts)
                        print(f"    已应用{len(segment.sub_segments)}个子段")
    
    def _enhanced_fill_gaps_safe(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """安全的增强空隙填充（避免冲突）"""
        # 重建资源时间线
        self._rebuild_resource_timeline(schedule)
        
        # 找出DSP忙碌时的NPU空隙
        npu_gaps_during_dsp = self._find_npu_gaps_during_dsp_busy(schedule)
        
        if not npu_gaps_during_dsp:
            return schedule
        
        print(f"\n  发现{len(npu_gaps_during_dsp)}个DSP忙碌时的NPU空隙")
        
        new_schedule = copy.deepcopy(schedule)
        
        # 优先级排序的纯NPU任务
        pure_npu_tasks = [
            ('T6', 0.778),  # HIGH优先级，最短
            ('T4', 0.364),  # NORMAL优先级，很短
            ('T5', 0.755),  # NORMAL优先级，短
            ('T7', 3.096),  # NORMAL优先级，较长
        ]
        
        # 为每个空隙填充任务
        for gap_start, gap_end in npu_gaps_during_dsp:
            gap_duration = gap_end - gap_start
            if gap_duration < 0.5:  # 忽略太小的空隙
                continue
                
            print(f"\n  处理NPU空隙: {gap_start:.1f}-{gap_end:.1f}ms (持续{gap_duration:.1f}ms)")
            
            # 获取每个任务当前的执行次数
            task_counts = defaultdict(int)
            for event in new_schedule:
                task_counts[event.task_id] += 1
            
            # 尝试填充任务
            gap_used = gap_start
            for task_id, duration in pure_npu_tasks:
                if gap_end - gap_used < duration + 0.1:  # 留一点余量
                    continue
                
                task = self.scheduler.tasks.get(task_id)
                if not task:
                    continue
                
                # 检查是否需要更多执行
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                current = task_counts[task_id]
                
                if current < expected:
                    # 检查最小间隔约束
                    existing_times = [e.start_time for e in new_schedule if e.task_id == task_id]
                    valid = True
                    for exist_time in existing_times:
                        if abs(gap_used - exist_time) < task.min_interval_ms:
                            valid = False
                            break
                    
                    if valid:
                        # 再次确认资源真的空闲
                        if self._verify_resource_available('NPU_0', gap_used, gap_used + duration):
                            # 创建新事件
                            resources = {ResourceType.NPU: 'NPU_0'}
                            new_event = self._create_safe_task_event(task, gap_used, resources)
                            new_schedule.append(new_event)
                            # 更新资源时间线
                            self._update_resource_timeline(new_event, task)
                            print(f"    ✓ 在{gap_used:.1f}ms处插入{task_id}")
                            
                            gap_used += duration + 0.1
                            task_counts[task_id] += 1
        
        new_schedule.sort(key=lambda x: x.start_time)
        return new_schedule
    
    def _greedy_fill_fps_safe(self, schedule: List[TaskScheduleInfo]) -> List[TaskScheduleInfo]:
        """安全的贪心FPS补充"""
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
                earliest_time = self._find_earliest_safe_time(task, existing_times)
                if earliest_time is not None and earliest_time < self.time_window:
                    resources = self._allocate_resources(task)
                    new_event = self._create_safe_task_event(task, earliest_time, resources)
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
    
    def _find_earliest_safe_time(self, task: NNTask, existing_times: List[float]) -> Optional[float]:
        """找到任务的最早安全时间（无冲突）"""
        # 计算任务总执行时间
        task_duration = self._get_task_total_duration(task)
        
        # 从0开始搜索
        test_time = 0.0
        step = 0.5  # 搜索步长
        
        while test_time + task_duration <= self.time_window:
            # 检查最小间隔
            valid = True
            for exist_time in existing_times:
                if abs(test_time - exist_time) < task.min_interval_ms:
                    valid = False
                    break
            
            if valid:
                # 检查所有需要的资源是否可用
                if self._check_all_resources_available(task, test_time):
                    return test_time
            
            test_time += step
        
        return None
    
    def _check_all_resources_available(self, task: NNTask, start_time: float) -> bool:
        """检查任务所需的所有资源是否可用"""
        current_time = start_time
        
        if task.is_segmented:
            # 分段任务：按顺序检查每个子段
            for seg in task.segments:
                if seg.is_segmented and seg.sub_segments:
                    for sub_seg in seg.sub_segments:
                        duration = sub_seg.get_duration(40.0)
                        res_type = sub_seg.resource_type
                        if res_type == ResourceType.NPU:
                            if not self._verify_resource_available('NPU_0', current_time, current_time + duration):
                                return False
                        elif res_type == ResourceType.DSP:
                            if not self._verify_resource_available('DSP_0', current_time, current_time + duration):
                                return False
                        current_time += duration
                else:
                    duration = seg.get_duration(40.0)
                    res_type = seg.resource_type
                    seg_time = start_time + seg.start_time
                    if res_type == ResourceType.NPU:
                        if not self._verify_resource_available('NPU_0', seg_time, seg_time + duration):
                            return False
                    elif res_type == ResourceType.DSP:
                        if not self._verify_resource_available('DSP_0', seg_time, seg_time + duration):
                            return False
        else:
            # 非分段任务
            for seg in task.segments:
                duration = seg.get_duration(40.0)
                seg_time = start_time + seg.start_time
                res_type = seg.resource_type
                if res_type == ResourceType.NPU:
                    if not self._verify_resource_available('NPU_0', seg_time, seg_time + duration):
                        return False
                elif res_type == ResourceType.DSP:
                    if not self._verify_resource_available('DSP_0', seg_time, seg_time + duration):
                        return False
        
        return True
    
    def _verify_resource_available(self, resource_id: str, start_time: float, end_time: float) -> bool:
        """验证资源在指定时间段是否真的可用"""
        for busy_start, busy_end, _ in self.resource_busy_times.get(resource_id, []):
            # 检查是否有任何重叠
            if not (end_time <= busy_start + 0.001 or start_time >= busy_end - 0.001):
                return False
        return True
    
    def _get_task_total_duration(self, task: NNTask) -> float:
        """获取任务的总持续时间"""
        if task.is_segmented:
            total = 0
            for seg in task.segments:
                if seg.is_segmented and seg.sub_segments:
                    for sub_seg in seg.sub_segments:
                        total += sub_seg.get_duration(40.0)
                else:
                    total += seg.get_duration(40.0)
            return total
        else:
            # 对于非分段任务，返回最晚结束时间
            max_end = 0
            for seg in task.segments:
                seg_end = seg.start_time + seg.get_duration(40.0)
                max_end = max(max_end, seg_end)
            return max_end
    
    def _create_safe_task_event(self, task: NNTask, start_time: float, 
                               resources: Dict[ResourceType, str]) -> TaskScheduleInfo:
        """创建安全的任务事件（确保分段任务正确处理）"""
        end_time = start_time
        sub_schedule = []
        
        if task.is_segmented:
            current_time = start_time
            # 分段任务必须按顺序执行每个子段
            for seg in task.segments:
                if seg.is_segmented and seg.sub_segments:
                    # 处理已分段的segment
                    for sub_seg in seg.sub_segments:
                        if sub_seg.resource_type in resources:
                            duration = sub_seg.get_duration(40.0)
                            sub_schedule.append((sub_seg.sub_id, current_time, current_time + duration))
                            current_time += duration
                            end_time = current_time
                else:
                    # 未分段的segment
                    if seg.resource_type in resources:
                        duration = seg.get_duration(40.0)
                        sub_schedule.append((f"{seg.segment_id}_0", current_time, current_time + duration))
                        current_time += duration
                        end_time = current_time
            
            # 对于混合任务（如T2：NPU+DSP），确保DSP部分在NPU之后
            if 'postprocess' in [s[0] for s in sub_schedule]:
                # 重新排序，确保postprocess在最后
                main_parts = [s for s in sub_schedule if 'postprocess' not in s[0]]
                post_parts = [s for s in sub_schedule if 'postprocess' in s[0]]
                
                # 调整postprocess的时间
                if main_parts and post_parts:
                    last_main_end = main_parts[-1][2]
                    for i, (sub_id, _, _) in enumerate(post_parts):
                        duration = post_parts[i][2] - post_parts[i][1]
                        new_start = last_main_end
                        post_parts[i] = (sub_id, new_start, new_start + duration)
                        last_main_end = new_start + duration
                        end_time = last_main_end
                
                sub_schedule = main_parts + post_parts
        else:
            # 非分段任务
            for seg in task.segments:
                if seg.resource_type in resources:
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
    
    def _find_npu_gaps_during_dsp_busy(self, schedule: List[TaskScheduleInfo]) -> List[Tuple[float, float]]:
        """找出DSP忙碌时的NPU空隙"""
        # 找出DSP忙碌时段
        dsp_busy_periods = []
        for event in schedule:
            task = self.scheduler.tasks.get(event.task_id)
            if not task:
                continue
            
            if hasattr(event, 'sub_segment_schedule'):
                for sub_id, start, end in event.sub_segment_schedule:
                    if 'dsp' in sub_id.lower():
                        dsp_busy_periods.append((start, end))
        
        # 对每个DSP忙碌时段，检查NPU是否空闲
        npu_gaps = []
        for dsp_start, dsp_end in dsp_busy_periods:
            # 在这个DSP时段内查找NPU空隙
            npu_free_start = dsp_start
            npu_free_end = dsp_end
            
            # 检查NPU占用情况
            for res_start, res_end, _ in self.resource_busy_times.get('NPU_0', []):
                if res_start <= dsp_start and res_end >= dsp_end:
                    # NPU完全占用这个时段
                    npu_free_start = npu_free_end = 0
                    break
                elif res_start <= dsp_start < res_end < dsp_end:
                    # 部分重叠，调整开始时间
                    npu_free_start = max(npu_free_start, res_end)
                elif dsp_start < res_start < dsp_end <= res_end:
                    # 部分重叠，调整结束时间
                    npu_free_end = min(npu_free_end, res_start)
                elif dsp_start < res_start < res_end < dsp_end:
                    # NPU占用在中间，取前半部分
                    npu_free_end = res_start
            
            if npu_free_end > npu_free_start + 0.1:  # 至少0.1ms的空隙才有意义
                npu_gaps.append((npu_free_start, npu_free_end))
        
        # 合并相邻空隙
        if npu_gaps:
            npu_gaps.sort()
            merged_gaps = []
            current_start, current_end = npu_gaps[0]
            
            for start, end in npu_gaps[1:]:
                if start <= current_end + 0.1:
                    current_end = max(current_end, end)
                else:
                    merged_gaps.append((current_start, current_end))
                    current_start, current_end = start, end
            
            merged_gaps.append((current_start, current_end))
            return merged_gaps
        
        return []
    
    # 其他必要的辅助方法
    def _compact_schedule(self, schedule: List[TaskScheduleInfo]) -> Tuple[List[TaskScheduleInfo], float]:
        """使用DebugCompactor进行紧凑化"""
        self.scheduler.schedule_history = copy.deepcopy(schedule)
        compactor = DebugCompactor(self.scheduler, self.time_window)
        compacted_events, idle_time = compactor.simple_compact()
        return compacted_events, idle_time
    
    def _rebuild_resource_timeline(self, schedule: List[TaskScheduleInfo]):
        """重建资源占用时间线"""
        self.resource_busy_times.clear()
        
        for event in schedule:
            self._update_resource_timeline(event, self.scheduler.tasks.get(event.task_id))
    
    def _update_resource_timeline(self, event: TaskScheduleInfo, task: Optional[NNTask]):
        """更新资源时间线"""
        if not task:
            return
            
        if hasattr(event, 'sub_segment_schedule') and event.sub_segment_schedule:
            for sub_id, start, end in event.sub_segment_schedule:
                # 通过sub_id确定资源类型
                for sub_seg in task.get_sub_segments_for_scheduling():
                    if sub_seg.sub_id == sub_id:
                        res_type = sub_seg.resource_type
                        if res_type in event.assigned_resources:
                            res_id = event.assigned_resources[res_type]
                            # 添加到时间线并保持排序
                            if res_id not in self.resource_busy_times:
                                self.resource_busy_times[res_id] = []
                            self.resource_busy_times[res_id].append((start, end, event.task_id))
                            self.resource_busy_times[res_id].sort()
                        break
    
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
    
    def _allocate_resources(self, task: NNTask) -> Dict[ResourceType, str]:
        """为任务分配资源"""
        resources = {}
        for seg in task.segments:
            res_list = self.scheduler.resources.get(seg.resource_type, [])
            if res_list:
                resources[seg.resource_type] = res_list[0].unit_id
        return resources
    
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
    print("🚀 无冲突的完整调度优化测试")
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
    
    # 创建无冲突优化器并执行优化
    optimizer = ConflictFreeOptimizer(scheduler, 200.0)
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
    if not is_valid:
        print("冲突详情:")
        for conflict in conflicts[:5]:
            print(f"  - {conflict}")
    
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
    plt.title('Conflict-Free Optimized Schedule', fontsize=16, pad=20)
    plt.savefig('conflict_free_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chrome trace
    viz.export_chrome_tracing('conflict_free_schedule.json')
    
    print("\n✅ 优化完成！")
    print(f"\n优化结果: {'所有任务FPS达标' if all_satisfied else '仍有任务未达标'}")
    print(f"最终状态: {'无冲突' if is_valid else f'有{len(conflicts)}个冲突'}")
    print("\n生成的文件：")
    print("  - conflict_free_schedule.png")
    print("  - conflict_free_schedule.json")


if __name__ == "__main__":
    main()