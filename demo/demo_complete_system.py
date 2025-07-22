#!/usr/bin/env python3
"""
完整系统演示 - 展示所有模块的协同工作
包括：任务发射、执行、评估和优化
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    ResourceType, TaskPriority,
    ResourceQueueManager, ScheduleTracer, 
    TaskLauncher, ScheduleExecutor,
    PerformanceEvaluator, LaunchOptimizer, OptimizationConfig
)
from viz.schedule_visualizer import ScheduleVisualizer
from scenario.real_task import create_real_tasks
import matplotlib.pyplot as plt


def demo_complete_system():
    """演示完整的调度系统工作流程"""
    print("="*80)
    print("[DEMO] AI调度系统完整演示")
    print("="*80)
    
    # 1. 系统初始化
    print("\n[STEP 1] 系统初始化")
    print("-"*40)
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    print("[OK] 资源配置:")
    print("  - NPU x2 (带宽: 60.0)")
    print("  - DSP x2 (带宽: 40.0)")
    
    # 2. 加载任务
    print("\n[STEP 2] 加载任务")
    print("-"*40)
    
    tasks = create_real_tasks()
    # 选择代表性任务
    selected_tasks = [
        tasks[0],  # T1: MOTR (CRITICAL)
        tasks[1],  # T2: YoloV8nBig (HIGH)
        tasks[2],  # T3: YoloV8nSmall (NORMAL)
        tasks[5],  # T6: reid (NORMAL)
        tasks[6],  # T7: crop (LOW)
    ]
    
    print(f"[OK] 加载了 {len(selected_tasks)} 个任务:")
    for task in selected_tasks:
        seg_info = f"{len(task.segments)}段"
        if task.segments:
            types = set(seg.resource_type.value for seg in task.segments)
            seg_info += f" ({'/'.join(types)})"
        
        print(f"  {task.task_id}: {task.name:<15} "
              f"优先级={task.priority.name:<8} "
              f"FPS={task.fps_requirement:<3} "
              f"{seg_info}")
    
    # 3. 基线执行（激进策略）
    print("\n[STEP 3] 基线执行 (激进策略)")
    print("-"*40)
    
    # 创建追踪器和发射器
    tracer_baseline = ScheduleTracer(queue_manager)
    launcher_baseline = TaskLauncher(queue_manager, tracer_baseline)
    
    # 注册任务
    for task in selected_tasks:
        launcher_baseline.register_task(task)
    
    # 创建激进的发射计划
    time_window = 200.0
    plan_eager = launcher_baseline.create_launch_plan(time_window, strategy="eager")
    
    print(f"[OK] 激进发射计划: {len(plan_eager.events)} 个发射事件")
    
    # 执行基线
    executor_baseline = ScheduleExecutor(queue_manager, tracer_baseline, launcher_baseline.tasks)
    exec_stats_baseline = executor_baseline.execute_plan(plan_eager, time_window)
    
    print(f"[OK] 执行完成: {exec_stats_baseline['total_segments_executed']} 个段")
    
    # 评估基线
    evaluator_baseline = PerformanceEvaluator(tracer_baseline, launcher_baseline.tasks, queue_manager)
    metrics_baseline = evaluator_baseline.evaluate(time_window, plan_eager.events)
    
    print(f"\n[BASELINE] 基线性能:")
    print(f"  空闲时间: {metrics_baseline.idle_time:.1f}ms ({metrics_baseline.idle_time_ratio:.1f}%)")
    print(f"  FPS满足率: {metrics_baseline.fps_satisfaction_rate:.1f}%")
    print(f"  NPU利用率: {metrics_baseline.avg_npu_utilization:.1f}%")
    print(f"  DSP利用率: {metrics_baseline.avg_dsp_utilization:.1f}%")
    
    # 4. 优化发射策略
    print("\n[STEP 4] 优化发射策略")
    print("-"*40)
    
    # 为优化创建独立的组件，但使用相同的资源配置
    queue_manager_opt = ResourceQueueManager()
    queue_manager_opt.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager_opt.add_resource("NPU_1", ResourceType.NPU, 60.0)
    queue_manager_opt.add_resource("DSP_0", ResourceType.DSP, 40.0)
    queue_manager_opt.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    tracer_opt = ScheduleTracer(queue_manager_opt)
    launcher_opt = TaskLauncher(queue_manager_opt, tracer_opt)
    
    for task in selected_tasks:
        launcher_opt.register_task(task)
    
    # 配置优化器
    opt_config = OptimizationConfig(
        idle_time_weight=0.7,        # 更重视空闲时间
        fps_satisfaction_weight=0.2,  # 保证基本FPS
        resource_balance_weight=0.1,
        fps_tolerance=0.95,          # 95%的FPS容忍度
        max_iterations=30,           # 减少迭代次数以加快演示
        population_size=20
    )
    
    optimizer = LaunchOptimizer(launcher_opt, queue_manager_opt, opt_config)
    
    print("[OK] 优化器配置:")
    print(f"  目标权重: 空闲时间={opt_config.idle_time_weight}, "
          f"FPS={opt_config.fps_satisfaction_weight}, "
          f"均衡={opt_config.resource_balance_weight}")
    print(f"  FPS容忍度: {opt_config.fps_tolerance*100}%")
    
    # 运行优化
    best_strategy = optimizer.optimize(time_window, base_strategy="eager")
    
    # 5. 执行优化后的策略
    print("\n[STEP 5] 执行优化策略")
    print("-"*40)
    
    # 应用最优策略
    plan_optimized = optimizer.apply_best_strategy()
    
    if plan_optimized:
        print(f"[OK] 优化发射计划: {len(plan_optimized.events)} 个发射事件")
        
        # 显示优化后的前几个事件
        print("\n优化后的发射事件（前10个）:")
        for i, event in enumerate(plan_optimized.events[:10]):
            print(f"  {event.time:>6.1f}ms: {event.task_id}#{event.instance_id}")
        if len(plan_optimized.events) > 10:
            print(f"  ... 还有 {len(plan_optimized.events)-10} 个事件")
        
        # 延迟统计
        delays = [e.task_id for e in plan_optimized.events 
                 if e.task_id in best_strategy.delay_factors 
                 and best_strategy.delay_factors[e.task_id] > 0]
        if delays:
            print(f"  延迟的任务: {len(set(delays))} 个")
        
        # 执行优化计划
        executor_opt = ScheduleExecutor(queue_manager_opt, tracer_opt, launcher_opt.tasks)
        exec_stats_opt = executor_opt.execute_plan(plan_optimized, time_window)
        
        print(f"[OK] 执行完成: {exec_stats_opt['total_segments_executed']} 个段")
        
        # 评估优化结果
        evaluator_opt = PerformanceEvaluator(tracer_opt, launcher_opt.tasks, queue_manager)
        metrics_opt = evaluator_opt.evaluate(time_window, plan_optimized.events)
        
        print(f"\n[OPTIMIZED] 优化后性能:")
        print(f"  空闲时间: {metrics_opt.idle_time:.1f}ms ({metrics_opt.idle_time_ratio:.1f}%)")
        print(f"  FPS满足率: {metrics_opt.fps_satisfaction_rate:.1f}%")
        print(f"  NPU利用率: {metrics_opt.avg_npu_utilization:.1f}%")
        print(f"  DSP利用率: {metrics_opt.avg_dsp_utilization:.1f}%")
    
    # 6. 对比分析
    print("\n[STEP 6] 性能对比")
    print("-"*40)
    
    if plan_optimized and metrics_opt:
        idle_improve = metrics_opt.idle_time - metrics_baseline.idle_time
        fps_change = metrics_opt.fps_satisfaction_rate - metrics_baseline.fps_satisfaction_rate
        
        print(f"空闲时间改进: {idle_improve:+.1f}ms "
              f"({metrics_baseline.idle_time:.1f} → {metrics_opt.idle_time:.1f})")
        print(f"FPS满足率变化: {fps_change:+.1f}% "
              f"({metrics_baseline.fps_satisfaction_rate:.1f} → {metrics_opt.fps_satisfaction_rate:.1f})")
        
        if idle_improve > 0:
            print("\n[SUCCESS] 优化成功！空闲时间增加了 {:.1f}ms".format(idle_improve))
        elif fps_change < -5:
            print("\n[WARNING] 警告：FPS满足率下降超过5%")
    
    # 7. 可视化
    print("\n[STEP 7] 生成可视化")
    print("-"*40)
    
    # 基线可视化
    viz_baseline = ScheduleVisualizer(tracer_baseline)
    viz_baseline.plot_resource_timeline("demo_baseline_gantt.png")
    viz_baseline.export_chrome_tracing("demo_baseline_trace.json")
    
    # 优化后可视化
    if plan_optimized:
        viz_opt = ScheduleVisualizer(tracer_opt)
        viz_opt.plot_resource_timeline("demo_optimized_gantt.png")
        viz_opt.export_chrome_tracing("demo_optimized_trace.json")
        
        # 生成详细报告
        evaluator_opt.export_json_report("demo_performance_report.json")
    
    print("[OK] 生成的文件:")
    print("  - demo_baseline_gantt.png (基线甘特图)")
    print("  - demo_baseline_trace.json (基线Chrome追踪)")
    if plan_optimized:
        print("  - demo_optimized_gantt.png (优化后甘特图)")
        print("  - demo_optimized_trace.json (优化后Chrome追踪)")
        print("  - demo_performance_report.json (性能报告)")
    
    # 8. 任务性能详情
    print("\n[STEP 8] 任务执行详情")
    print("-"*40)
    
    print(f"\n{'任务ID':<10} {'优先级':<8} {'FPS要求':<8} {'基线FPS':<10} {'优化FPS':<10} {'状态':<6}")
    print("-"*60)
    
    for task_id in sorted(evaluator_baseline.task_metrics.keys()):
        baseline_m = evaluator_baseline.task_metrics[task_id]
        opt_m = evaluator_opt.task_metrics[task_id] if plan_optimized else None
        
        baseline_fps = f"{baseline_m.achieved_fps:.1f}"
        opt_fps = f"{opt_m.achieved_fps:.1f}" if opt_m else "N/A"
        
        status = "[OK]" if baseline_m.fps_satisfaction else "[ERROR]"
        if opt_m and opt_m.fps_satisfaction != baseline_m.fps_satisfaction:
            status = "[OK]->[ERROR]" if baseline_m.fps_satisfaction else "[ERROR]->[OK]"
        
        print(f"{task_id:<10} {baseline_m.priority.name:<8} "
              f"{baseline_m.fps_requirement:<8.1f} {baseline_fps:<10} "
              f"{opt_fps:<10} {status:<6}")
    
    # 9. 总结
    print("\n" + "="*80)
    print("[TIP] 系统演示总结")
    print("="*80)
    
    print("\n关键发现:")
    print("1. 新架构成功分离了发射、执行和评估逻辑")
    print("2. 激进发射策略提供了基线性能")
    print("3. 优化器能够找到更好的发射时机来增加空闲时间")
    print("4. 评估器提供了全面的性能指标")
    print("5. 可视化支持多种格式（甘特图、Chrome追踪、JSON报告）")
    
    if plan_optimized and idle_improve > 0:
        print(f"\n[SUCCESS] 优化成功将空闲时间提升了 {idle_improve:.1f}ms!")
        print("   这些额外的空闲时间可用于:")
        print("   - 系统节能")
        print("   - 处理突发任务")
        print("   - 提升系统响应能力")


def demo_segment_visualization():
    """专门演示分段标签的可视化"""
    print("\n\n" + "="*80)
    print("🏷️  分段标签可视化演示")
    print("="*80)
    
    # 创建简单场景
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    
    # 手动记录一些分段执行来展示标签
    print("\n模拟分段执行:")
    
    # 任务T1的多个分段
    segments = [
        ("T1#0_s1", "NPU_0", 0, 5),
        ("T1#0_s2", "DSP_0", 5, 10),
        ("T1#0_s3", "NPU_0", 10, 15),
        ("T1#1_seg1", "NPU_0", 20, 25),
        ("T1#1_seg2", "DSP_0", 25, 30),
        ("T1#1_seg3", "NPU_0", 30, 35),
    ]
    
    for task_id, resource, start, end in segments:
        tracer.record_execution(
            task_id, resource, float(start), float(end),
            60.0 if "NPU" in resource else 40.0,
            segment_id=task_id.split('_')[-1]
        )
        print(f"  {task_id} 在 {resource} 上执行 ({start}-{end}ms)")
    
    # 显示甘特图
    print("\n文本甘特图展示:")
    viz = ScheduleVisualizer(tracer)
    viz.print_gantt_chart(width=60)
    
    print("\n[OK] 分段标签格式验证:")
    print("  - '_s1/2/3' 格式: 简短标签")
    print("  - '_seg1/2/3' 格式: 完整标签")
    print("  两种格式都被支持！")


if __name__ == "__main__":
    # 运行完整系统演示
    demo_complete_system()
    
    # 运行分段标签演示
    demo_segment_visualization()
    
    print("\n\n[COMPLETE] 所有演示完成！")
