#!/usr/bin/env python3
"""
简单测试 - 验证段交织调度的有效性
专门设计来展示不同网络段交织的优势
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enums import ResourceType, TaskPriority
from core.task import create_mixed_task
from viz.schedule_visualizer import ScheduleVisualizer

# 尝试导入不同的调度器
try:
    from archive.executor_old import ScheduleExecutor
    from archive.segment_aware_executor import SegmentAwareExecutor
    from archive.true_segment_scheduler import TrueSegmentScheduler
except ImportError:
    try:
        from true_segment_scheduler import TrueSegmentScheduler
    except:
        print("请确保 true_segment_scheduler.py 在当前目录或 core 目录")
        sys.exit(1)


def create_test_scenario():
    """创建测试场景 - 设计来展示段交织的优势"""
    
    # 任务A: 短NPU + 长DSP
    task_a = create_mixed_task(
        "TaskA", "短NPU+长DSP",
        segments=[
            (ResourceType.NPU, {60: 3.0}, "a_npu"),    # 3ms
            (ResourceType.DSP, {40: 10.0}, "a_dsp"),   # 10ms
        ],
        priority=TaskPriority.NORMAL
    )
    task_a.set_performance_requirements(fps=10, latency=50)
    
    # 任务B: 长NPU + 短DSP
    task_b = create_mixed_task(
        "TaskB", "长NPU+短DSP", 
        segments=[
            (ResourceType.NPU, {60: 8.0}, "b_npu"),    # 8ms
            (ResourceType.DSP, {40: 2.0}, "b_dsp"),    # 2ms
        ],
        priority=TaskPriority.NORMAL
    )
    task_b.set_performance_requirements(fps=10, latency=50)
    
    # 任务C: 中等NPU + 中等DSP
    task_c = create_mixed_task(
        "TaskC", "中NPU+中DSP",
        segments=[
            (ResourceType.NPU, {60: 5.0}, "c_npu"),    # 5ms
            (ResourceType.DSP, {40: 5.0}, "c_dsp"),    # 5ms
        ],
        priority=TaskPriority.NORMAL
    )
    task_c.set_performance_requirements(fps=10, latency=50)
    
    return [task_a, task_b, task_c]


def run_test_with_scheduler(scheduler_name, scheduler_class, tasks, use_segment_mode=False):
    """使用指定调度器运行测试"""
    print(f"\n{'='*60}")
    print(f"测试调度器: {scheduler_name}")
    print(f"{'='*60}")
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 注册任务
    for task in tasks:
        launcher.register_task(task)
    
    # 创建发射计划
    time_window = 50.0
    plan = launcher.create_launch_plan(time_window, "eager")
    
    # 创建并运行调度器
    if scheduler_class == SegmentAwareExecutor:
        scheduler = scheduler_class(queue_manager, tracer, launcher.tasks)
        if use_segment_mode:
            scheduler.segment_launch_tasks = {"TaskA", "TaskB", "TaskC"}
    else:
        scheduler = scheduler_class(queue_manager, tracer, launcher.tasks)
    
    stats = scheduler.execute_plan(plan, time_window)
    
    # 显示结果
    print("\n执行时间线:")
    visualizer = ScheduleVisualizer(tracer)
    visualizer.print_gantt_chart(width=60)
    
    # 计算指标
    utilization = tracer.get_resource_utilization()
    makespan = tracer.get_statistics()['time_span']
    
    print(f"\n性能指标:")
    print(f"  完成实例: {stats.get('completed_instances', 'N/A')}")
    print(f"  总时间跨度: {makespan:.1f}ms")
    print(f"  NPU利用率: {utilization.get('NPU_0', 0):.1f}%")
    print(f"  DSP利用率: {utilization.get('DSP_0', 0):.1f}%")
    
    return {
        'completed': stats.get('completed_instances', 0),
        'makespan': makespan,
        'npu_util': utilization.get('NPU_0', 0),
        'dsp_util': utilization.get('DSP_0', 0),
        'tracer': tracer  # 返回tracer以供后续分析
    }


def analyze_segment_execution(tracer):
    """分析段执行的交织情况"""
    print("\n段执行分析:")
    
    # 收集NPU和DSP的执行时间线
    npu_timeline = []
    dsp_timeline = []
    
    for exec in tracer.executions:
        if "NPU" in exec.resource_id:
            npu_timeline.append((exec.start_time, exec.end_time, exec.task_id))
        elif "DSP" in exec.resource_id:
            dsp_timeline.append((exec.start_time, exec.end_time, exec.task_id))
    
    # 按开始时间排序
    npu_timeline.sort(key=lambda x: x[0])
    dsp_timeline.sort(key=lambda x: x[0])
    
    print("\nNPU执行序列:")
    for start, end, task in npu_timeline:
        print(f"  {start:>5.1f}-{end:>5.1f}ms: {task}")
    
    print("\nDSP执行序列:")
    for start, end, task in dsp_timeline:
        print(f"  {start:>5.1f}-{end:>5.1f}ms: {task}")
    
    # 计算并行执行时间
    parallel_time = 0
    for npu_start, npu_end, _ in npu_timeline:
        for dsp_start, dsp_end, _ in dsp_timeline:
            # 计算重叠时间
            overlap_start = max(npu_start, dsp_start)
            overlap_end = min(npu_end, dsp_end)
            if overlap_start < overlap_end:
                parallel_time += overlap_end - overlap_start
    
    total_time = tracer.get_statistics()['time_span']
    parallel_ratio = (parallel_time / total_time * 100) if total_time > 0 else 0
    
    print(f"\n并行执行分析:")
    print(f"  并行执行时间: {parallel_time:.1f}ms")
    print(f"  总执行时间: {total_time:.1f}ms")
    print(f"  并行度: {parallel_ratio:.1f}%")


def main():
    """主测试函数"""
    print("=== 段交织调度有效性测试 ===")
    print("\n测试场景:")
    print("  TaskA: NPU(3ms) → DSP(10ms)")
    print("  TaskB: NPU(8ms) → DSP(2ms)")
    print("  TaskC: NPU(5ms) → DSP(5ms)")
    print("\n理想执行顺序应该让NPU和DSP尽可能并行工作")
    
    # 创建测试任务
    tasks = create_test_scenario()
    
    results = {}
    
    # 1. 测试原始执行器
    results['original'] = run_test_with_scheduler(
        "原始执行器（整体发射）",
        ScheduleExecutor,
        tasks
    )
    
    # 2. 测试段感知执行器（整体模式）
    results['segment_aware_whole'] = run_test_with_scheduler(
        "段感知执行器（整体模式）",
        SegmentAwareExecutor,
        tasks,
        use_segment_mode=False
    )
    
    # 3. 测试段感知执行器（段级模式）
    results['segment_aware_segment'] = run_test_with_scheduler(
        "段感知执行器（段级模式）",
        SegmentAwareExecutor,
        tasks,
        use_segment_mode=True
    )
    
    # 4. 测试真正的段级调度器
    results['true_segment'] = run_test_with_scheduler(
        "真正的段级调度器",
        TrueSegmentScheduler,
        tasks
    )
    
    # 分析最后一个调度器的详细执行
    print("\n" + "="*60)
    print("真正的段级调度器的详细分析:")
    if 'tracer' in results['true_segment']:
        analyze_segment_execution(results['true_segment']['tracer'])
    else:
        print("警告：无法获取tracer进行详细分析")
    
    # 对比结果
    print("\n" + "="*60)
    print("性能对比总结:")
    print("="*60)
    print(f"{'调度器':<30} {'完成数':>8} {'时间跨度':>10} {'NPU利用':>10} {'DSP利用':>10}")
    print("-"*70)
    
    for name, result in results.items():
        print(f"{name:<30} {result['completed']:>8} {result['makespan']:>10.1f} "
              f"{result['npu_util']:>10.1f}% {result['dsp_util']:>10.1f}%")
    
    # 计算改进
    if results['original']['makespan'] > 0:
        improvement = ((results['original']['makespan'] - results['true_segment']['makespan']) 
                      / results['original']['makespan'] * 100)
        print(f"\n时间跨度改进: {improvement:.1f}%")
    
    # 理论最优分析
    print("\n理论最优分析:")
    print("  NPU总工作量: 3+8+5 = 16ms")
    print("  DSP总工作量: 10+2+5 = 17ms")
    print("  理论最短时间: max(16, 17) = 17ms")
    print(f"  实际最短时间: {results['true_segment']['makespan']:.1f}ms")
    
    # 关键洞察
    print("\n关键洞察:")
    if results['true_segment']['makespan'] < results['original']['makespan']:
        print("✅ 真正的段级调度器成功减少了总执行时间")
        print("✅ NPU和DSP资源得到了更好的并行利用")
    else:
        print("❌ 优化效果不明显，可能需要:")
        print("   - 更多的任务来增加交织机会")
        print("   - 更智能的段选择策略")
        print("   - 考虑任务特性的调度算法")


if __name__ == "__main__":
    main()
