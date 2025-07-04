#!/usr/bin/env python3
"""
修复分段不生效的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy
from core.task import NNTask
from core.scheduler import MultiResourceScheduler
from core.modular_scheduler_fixes import apply_basic_fixes
from core.minimal_fifo_fix_corrected import apply_minimal_fifo_fix
from core.strict_resource_conflict_fix import apply_strict_resource_conflict_fix


def debug_segmentation_flow(task, segment_id, cuts):
    """调试分段流程"""
    print(f"\n[DEBUG] 调试任务{task.task_id}的分段流程")
    print(f"  段ID: {segment_id}")
    print(f"  切点: {cuts}")
    
    # 获取段
    segment = task.get_segment_by_id(segment_id)
    if not segment:
        print(f"  ❌ 找不到段 {segment_id}")
        return
    
    print(f"  段的切点: {[cp.op_id for cp in segment.cut_points]}")
    print(f"  段的持续时间: {segment.get_duration(40.0)}ms")
    
    # 手动应用分段
    sub_segments = segment.apply_segmentation(cuts)
    print(f"  应用分段后的子段数: {len(sub_segments)}")
    for i, sub_seg in enumerate(sub_segments):
        print(f"    子段{i}: {sub_seg.sub_id}, 持续时间={sub_seg.get_duration(40.0)}ms")
    
    return sub_segments


def fix_task_segmentation(scheduler):
    """修复任务分段问题"""
    # 在调度器的make_segmentation_decision方法中添加实际的分段应用
    original_make_decision = scheduler.make_segmentation_decision
    
    def enhanced_make_segmentation_decision(task, current_time):
        # 调用原始方法获取决策
        decisions = original_make_decision(task, current_time)
        
        # 确保CUSTOM_SEGMENTATION策略的任务应用其预设配置
        if task.segmentation_strategy == SegmentationStrategy.CUSTOM_SEGMENTATION:
            # 对于每个段，确保应用了选定的切点
            for segment in task.segments:
                seg_id = segment.segment_id
                if seg_id in task.selected_cut_config_index:
                    config_idx = task.selected_cut_config_index[seg_id]
                    if seg_id in task.preset_cut_configurations:
                        cuts = task.preset_cut_configurations[seg_id][config_idx]
                        # 立即应用分段
                        segment.apply_segmentation(cuts)
                        print(f"[DEBUG] 为任务{task.task_id}的段{seg_id}应用了{len(cuts)}个切点")
        
        return decisions
    
    # 替换方法
    scheduler.make_segmentation_decision = enhanced_make_segmentation_decision
    return scheduler


def create_and_test_fixed_tasks():
    """创建并测试修复后的任务"""
    # 创建调度器
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=40.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    # 应用修复
    fix_manager = apply_basic_fixes(scheduler)
    apply_minimal_fifo_fix(scheduler)
    apply_strict_resource_conflict_fix(scheduler)
    
    # 应用分段修复
    scheduler = fix_task_segmentation(scheduler)
    
    # 创建任务
    tasks = []
    
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
    tasks.append(taskA)
    
    # 任务B - 确保正确初始化
    taskB = NNTask("B", "Pure_NPU_Segmentable", 
                   priority=TaskPriority.NORMAL,
                   runtime_type=RuntimeType.ACPU_RUNTIME,
                   segmentation_strategy=SegmentationStrategy.CUSTOM_SEGMENTATION)
    
    taskB.set_npu_only({40: 15.0}, "main")
    
    # 正确添加切点
    taskB.add_cut_points_to_segment("main", [
        ("cut1", 0.33, 0),
        ("cut2", 0.66, 0),
    ])
    
    # 设置预定义配置
    taskB.set_preset_cut_configurations("main", [
        [],
        ["cut1"],
        ["cut2"],
        ["cut1", "cut2"],
    ])
    
    # 选择配置3（完全分段）
    taskB.select_cut_configuration("main", 3)
    
    # 手动触发分段以验证
    segment = taskB.get_segment_by_id("main")
    if segment:
        print("\n[手动测试] 分段前:")
        print(f"  段持续时间: {segment.get_duration(40.0)}ms")
        print(f"  是否已分段: {segment.is_segmented}")
        
        # 手动应用分段
        cuts = taskB.preset_cut_configurations["main"][3]
        sub_segs = segment.apply_segmentation(cuts)
        
        print(f"\n[手动测试] 分段后:")
        print(f"  子段数量: {len(sub_segs)}")
        for sub_seg in sub_segs:
            print(f"  - {sub_seg.sub_id}: {sub_seg.get_duration(40.0)}ms")
    
    taskB.set_performance_requirements(fps=25, latency=40)
    tasks.append(taskB)
    
    # 添加任务到调度器
    for task in tasks:
        scheduler.add_task(task)
    
    return scheduler, tasks


def test_with_explicit_segmentation():
    """测试显式分段应用"""
    print("=" * 80)
    print("🔧 测试修复后的分段功能")
    print("=" * 80)
    
    scheduler, tasks = create_and_test_fixed_tasks()
    
    # 在调度前确保任务B已分段
    task_b = scheduler.tasks["B"]
    
    # 方法1：通过apply_segmentation_decision强制应用
    segmentation_decisions = {"main": ["cut1", "cut2"]}
    overhead = task_b.apply_segmentation_decision(segmentation_decisions)
    print(f"\n应用分段决策，开销: {overhead}ms")
    
    # 验证分段结果
    sub_segments = task_b.get_sub_segments_for_scheduling()
    print(f"任务B的子段数: {len(sub_segments)}")
    for sub_seg in sub_segments:
        print(f"  {sub_seg.sub_id}: {sub_seg.get_duration(40.0)}ms, 开始时间: {sub_seg.start_time}ms")
    
    # 运行调度
    print("\n运行调度...")
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(100.0)
    
    # 分析结果
    print("\n调度结果分析:")
    task_a_dsp_time = None
    task_b_schedules = []
    
    for event in scheduler.schedule_history:
        if event.task_id == "A":
            print(f"\n任务A事件:")
            for sub_seg_id, start, end in event.sub_segment_schedule:
                print(f"  {sub_seg_id}: {start:.1f}-{end:.1f}ms")
                if "dsp" in sub_seg_id:
                    task_a_dsp_time = (start, end)
                    
        elif event.task_id == "B":
            print(f"\n任务B事件:")
            for sub_seg_id, start, end in event.sub_segment_schedule:
                print(f"  {sub_seg_id}: {start:.1f}-{end:.1f}ms")
                task_b_schedules.append((sub_seg_id, start, end))
    
    # 检查是否有插入
    if task_a_dsp_time:
        print(f"\n任务A的DSP时间窗口: {task_a_dsp_time[0]:.1f}-{task_a_dsp_time[1]:.1f}ms")
        for seg_id, start, end in task_b_schedules:
            if start >= task_a_dsp_time[0] and end <= task_a_dsp_time[1]:
                print(f"✅ 成功！任务B的{seg_id}插入到了DSP空隙中")
                return True
    
    print("\n❌ 任务B的段仍未插入到DSP空隙中")
    print("\n可能需要进一步修改调度算法来实现空隙感知调度")
    return False


if __name__ == "__main__":
    # 运行测试
    success = test_with_explicit_segmentation()
    
    if not success:
        print("\n建议下一步：实现空隙感知的调度算法")