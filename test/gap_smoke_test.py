#!/usr/bin/env python3
"""
完整的空隙调度器 - 确保不丢失任何段
"""

import sys
import os
import copy
from typing import List, Dict, Tuple, Optional
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
from viz.elegant_visualization import ElegantSchedulerVisualizer
import matplotlib.pyplot as plt


def create_gap_filling_schedule(baseline_schedule: List[TaskScheduleInfo], 
                               dsp_busy_periods: List[Tuple[float, float]]) -> List[TaskScheduleInfo]:
    """
    创建空隙填充的优化调度
    重要：确保所有段都被保留
    """
    print("\n创建优化调度...")
    optimized_schedule = []
    
    # 对每个DSP忙碌时段，尝试填充任务B的段
    for dsp_start, dsp_end in dsp_busy_periods:
        dsp_duration = dsp_end - dsp_start
        print(f"\n处理DSP时段 {dsp_start:.1f}-{dsp_end:.1f}ms (持续{dsp_duration:.1f}ms)")
        
        # 找到可以移动的任务B事件
        for event in baseline_schedule:
            if event.task_id == 'B' and event.start_time > dsp_end:
                print(f"  找到任务B事件: {event.start_time:.1f}-{event.end_time:.1f}ms")
                
                if hasattr(event, 'sub_segment_schedule'):
                    # 计算哪些段可以放入DSP空隙
                    segments_in_gap = []
                    segments_after_gap = []
                    current_time = dsp_start
                    
                    for sub_id, start, end in event.sub_segment_schedule:
                        duration = end - start
                        
                        if current_time + duration <= dsp_end:
                            # 这个段可以完全放入空隙
                            segments_in_gap.append({
                                'sub_id': sub_id,
                                'new_start': current_time,
                                'new_end': current_time + duration,
                                'duration': duration
                            })
                            current_time += duration
                            print(f"    ✓ {sub_id} 可以放入空隙 ({current_time-duration:.1f}-{current_time:.1f}ms)")
                        else:
                            # 这个段不能放入空隙，需要另外处理
                            segments_after_gap.append({
                                'sub_id': sub_id,
                                'duration': duration,
                                'original_start': start
                            })
                            print(f"    ✗ {sub_id} 无法放入空隙 (需要{duration:.1f}ms，剩余{dsp_end-current_time:.1f}ms)")
                    
                    # 如果有段可以放入空隙
                    if segments_in_gap:
                        # 创建两个新事件：一个在空隙中，一个在原位置（包含剩余的段）
                        
                        # 1. 空隙中的事件
                        gap_event = copy.deepcopy(event)
                        gap_event.sub_segment_schedule = [
                            (seg['sub_id'], seg['new_start'], seg['new_end']) 
                            for seg in segments_in_gap
                        ]
                        gap_event.start_time = segments_in_gap[0]['new_start']
                        gap_event.end_time = segments_in_gap[-1]['new_end']
                        
                        print(f"\n  创建空隙事件: {gap_event.start_time:.1f}-{gap_event.end_time:.1f}ms")
                        for sub_id, start, end in gap_event.sub_segment_schedule:
                            print(f"    {sub_id}: {start:.1f}-{end:.1f}ms")
                        
                        # 2. 剩余段的事件（如果有）
                        if segments_after_gap:
                            remaining_event = copy.deepcopy(event)
                            # 保持原始时间，或者稍微调整以避免冲突
                            remaining_start = event.start_time
                            remaining_schedule = []
                            
                            for seg in segments_after_gap:
                                seg_start = remaining_start
                                seg_end = seg_start + seg['duration']
                                remaining_schedule.append((seg['sub_id'], seg_start, seg_end))
                                remaining_start = seg_end
                            
                            remaining_event.sub_segment_schedule = remaining_schedule
                            remaining_event.start_time = remaining_schedule[0][1]
                            remaining_event.end_time = remaining_schedule[-1][2]
                            
                            print(f"\n  创建剩余段事件: {remaining_event.start_time:.1f}-{remaining_event.end_time:.1f}ms")
                            for sub_id, start, end in remaining_event.sub_segment_schedule:
                                print(f"    {sub_id}: {start:.1f}-{end:.1f}ms")
                        
                        # 返回优化后的调度
                        optimized = []
                        for e in baseline_schedule:
                            if e == event:
                                # 替换原事件为新事件
                                optimized.append(gap_event)
                                if segments_after_gap:
                                    optimized.append(remaining_event)
                            else:
                                optimized.append(e)
                        
                        # 按时间排序
                        optimized.sort(key=lambda x: x.start_time)
                        return optimized
    
    # 如果没有找到优化机会，返回原调度
    return baseline_schedule


def main():
    """主测试函数"""
    print("=" * 80)
    print("🚀 完整的空隙调度测试")
    print("=" * 80)
    
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=40.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    # 应用所有修复
    print("\n应用调度修复...")
    fix_manager = apply_basic_fixes(scheduler)
    apply_minimal_fifo_fix(scheduler)
    apply_strict_resource_conflict_fix(scheduler)
    print("✓ 修复已应用")
    
    # 创建测试任务
    print("\n创建测试任务...")
    
    # 任务A
    taskA = NNTask("A", "Mixed_DSP_NPU",
                   priority=TaskPriority.HIGH,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.NO_SEGMENTATION)
    
    taskA.set_dsp_npu_sequence([
        (ResourceType.NPU, {40: 5.0}, 0, "npu_seg1"),
        (ResourceType.DSP, {40: 10.0}, 5.0, "dsp_seg1"),
        (ResourceType.NPU, {40: 5.0}, 15.0, "npu_seg2"),
    ])
    taskA.set_performance_requirements(fps=25, latency=40)
    
    # 任务B - 分3段
    taskB = NNTask("B", "Pure_NPU_Segmentable", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    
    taskB.set_npu_only({40: 15.0}, "main")
    taskB.add_cut_points_to_segment("main", [
        ("cut1", 0.33, 0),
        ("cut2", 0.66, 0),
    ])
    taskB.set_preset_cut_configurations("main", [
        [],
        ["cut1"],
        ["cut2"],
        ["cut1", "cut2"],
    ])
    taskB.select_cut_configuration("main", 3)
    
    # 应用分段
    segment = taskB.get_segment_by_id("main")
    if segment:
        segment.apply_segmentation(["cut1", "cut2"])
        print(f"  ✓ 任务B已分段为 {len(segment.sub_segments)} 个子段")
        for i, sub_seg in enumerate(segment.sub_segments):
            print(f"    - {sub_seg.sub_id}: {sub_seg.get_duration(40)}ms")
    
    taskB.set_performance_requirements(fps=25, latency=40)
    
    # 添加任务
    scheduler.add_task(taskA)
    scheduler.add_task(taskB)
    
    # 执行基础调度
    print("\n=== 第一阶段：基础调度 ===")
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(100.0)
    
    # 验证基础调度
    is_valid, conflicts = validate_schedule_correctly(scheduler)
    if not is_valid:
        print(f"\n❌ 基础调度有冲突：{conflicts}")
        return
    else:
        print("✅ 基础调度无冲突")
    
    # 统计任务B的所有段
    print("\n统计任务B的段:")
    b_segments_count = 0
    for event in scheduler.schedule_history:
        if event.task_id == 'B' and hasattr(event, 'sub_segment_schedule'):
            for sub_id, start, end in event.sub_segment_schedule:
                b_segments_count += 1
                print(f"  {sub_id}: {start:.1f}-{end:.1f}ms")
    print(f"  总计: {b_segments_count} 个段")
    
    # 保存基础调度
    baseline_schedule = copy.deepcopy(scheduler.schedule_history)
    
    # 找出DSP忙碌时段
    dsp_busy_periods = []
    for event in baseline_schedule:
        if event.task_id == 'A' and hasattr(event, 'sub_segment_schedule'):
            for sub_id, start, end in event.sub_segment_schedule:
                if 'dsp' in sub_id.lower():
                    dsp_busy_periods.append((start, end))
    
    print(f"\nDSP忙碌时段: {dsp_busy_periods}")
    
    # === 第二阶段：空隙优化 ===
    print("\n=== 第二阶段：空隙优化 ===")
    
    # 创建优化调度
    optimized_schedule = create_gap_filling_schedule(baseline_schedule, dsp_busy_periods)
    scheduler.schedule_history = optimized_schedule
    
    # 再次统计任务B的段
    print("\n优化后统计任务B的段:")
    b_segments_after = 0
    for event in scheduler.schedule_history:
        if event.task_id == 'B' and hasattr(event, 'sub_segment_schedule'):
            for sub_id, start, end in event.sub_segment_schedule:
                b_segments_after += 1
                print(f"  {sub_id}: {start:.1f}-{end:.1f}ms")
    print(f"  总计: {b_segments_after} 个段")
    
    if b_segments_after != b_segments_count:
        print(f"\n⚠️ 警告：段数不匹配！原始{b_segments_count}个，优化后{b_segments_after}个")
    else:
        print(f"\n✅ 所有段都被保留")
    
    # 验证优化后的调度
    is_valid_after, conflicts_after = validate_schedule_correctly(scheduler)
    if not is_valid_after:
        print(f"\n❌ 优化后有冲突：{conflicts_after}")
    else:
        print("✅ 优化后无冲突")
    
    # 生成可视化
    print("\n生成可视化...")
    viz = ElegantSchedulerVisualizer(scheduler)
    plt.figure(figsize=(20, 8))
    viz.plot_elegant_gantt(time_window=100.0, show_all_labels=True)
    
    # 标注DSP忙碌时段
    ax = plt.gca()
    for start, end in dsp_busy_periods:
        ax.axvspan(start, end, alpha=0.1, color='red')
        ax.text((start+end)/2, ax.get_ylim()[1]*0.95, 'DSP Busy', 
                ha='center', va='top', fontsize=10, color='red')
    
    plt.title('Complete Gap-Aware Schedule (No Lost Segments)', fontsize=16, pad=20)
    plt.savefig('complete_gap_schedule.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Chrome trace
    viz.export_chrome_tracing('complete_gap_schedule.json')
    
    print("\n✅ 测试完成！")
    print("\n生成的文件：")
    print("  - complete_gap_schedule.png")
    print("  - complete_gap_schedule.json")
    
    # 最终验证
    print("\n" + "=" * 60)
    print("📊 最终验证")
    print("=" * 60)
    
    # 检查空隙利用情况
    gap_utilized = 0
    for event in scheduler.schedule_history:
        if event.task_id == 'B' and hasattr(event, 'sub_segment_schedule'):
            for sub_id, start, end in event.sub_segment_schedule:
                for dsp_start, dsp_end in dsp_busy_periods:
                    if start >= dsp_start and end <= dsp_end:
                        print(f"✨ {sub_id}在DSP空隙中: {start:.1f}-{end:.1f}ms")
                        gap_utilized += 1
    
    print(f"\n总结: 利用了{gap_utilized}个空隙，保留了{b_segments_after}个段")


if __name__ == "__main__":
    main()
