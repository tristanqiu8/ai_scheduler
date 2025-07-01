#!/usr/bin/env python3
"""
激进的遗传算法主程序 - 专注于最大化空闲时间
目标：最小化NPU工作时间，最大化末尾空闲
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入新的优化器
try:
    from core.aggressive_idle_optimizer import AggressiveIdleOptimizer
except ImportError:
    # 如果正常导入失败，尝试直接导入
    print("警告：无法从core包导入AggressiveIdleOptimizer，尝试直接导入...")
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'core'))
    from aggressive_idle_optimizer import AggressiveIdleOptimizer
from core.scheduler import MultiResourceScheduler
from scenario.real_task import create_real_tasks
from core.modular_scheduler_fixes import apply_basic_fixes
from core.minimal_fifo_fix_corrected import apply_minimal_fifo_fix
from core.strict_resource_conflict_fix import apply_strict_resource_conflict_fix
from core.fixed_validation_and_metrics import validate_schedule_correctly
from core.improved_genetic_optimizer import (
    calculate_detailed_utilization,
    print_detailed_utilization,
    analyze_fps_satisfaction
)
from viz.elegant_visualization import ElegantSchedulerVisualizer
from core.debug_compactor import DebugCompactor
import copy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def visualize_idle_comparison(baseline_idle, optimized_idle, time_window):
    """可视化空闲时间对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 饼图1：基线
    baseline_work = time_window - baseline_idle
    ax1.pie([baseline_work, baseline_idle], 
            labels=['工作时间', '空闲时间'],
            colors=['#FF6B6B', '#4ECDC4'],
            autopct='%1.1f%%',
            startangle=90)
    ax1.set_title(f'基线调度\n(空闲: {baseline_idle:.1f}ms)')
    
    # 饼图2：优化后
    optimized_work = time_window - optimized_idle
    ax2.pie([optimized_work, optimized_idle],
            labels=['工作时间', '空闲时间'],
            colors=['#FF6B6B', '#4ECDC4'],
            autopct='%1.1f%%',
            startangle=90)
    ax2.set_title(f'优化后调度\n(空闲: {optimized_idle:.1f}ms)')
    
    # 总标题
    improvement = optimized_idle - baseline_idle
    plt.suptitle(f'空闲时间优化效果\n改进: +{improvement:.1f}ms ({improvement/time_window*100:.1f}%)', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('idle_time_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 空闲时间对比图已保存到 idle_time_comparison.png")


def run_compaction_and_measure_idle(scheduler, time_window):
    """运行紧凑化并测量实际空闲时间"""
    compactor = DebugCompactor(scheduler, time_window)
    original_events = copy.deepcopy(scheduler.schedule_history)
    
    # 执行紧凑化
    compacted_events, idle_time = compactor.simple_compact()
    
    # 更新调度历史
    scheduler.schedule_history = compacted_events
    
    return idle_time, compacted_events, original_events


def main():
    """主函数"""
    print("=" * 80)
    print("🚀 激进遗传算法优化 - 最大化空闲时间")
    print("=" * 80)
    
    time_window = 200.0
    
    # 创建系统
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=40)
    scheduler.add_dsp("DSP_0", bandwidth=40)
    
    # 应用修复
    fix_manager = apply_basic_fixes(scheduler)
    
    # 创建任务
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    # 应用额外修复
    apply_minimal_fifo_fix(scheduler)
    apply_strict_resource_conflict_fix(scheduler)
    
    # ========== 基线评估 ==========
    print("\n📊 评估基线性能...")
    scheduler.schedule_history.clear()
    baseline_results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    # 验证基线
    is_valid, baseline_conflicts = validate_schedule_correctly(scheduler)
    baseline_stats = analyze_fps_satisfaction(scheduler, time_window)
    baseline_util = calculate_detailed_utilization(scheduler, time_window)
    
    # 运行紧凑化测量基线空闲时间
    baseline_idle, baseline_compacted, _ = run_compaction_and_measure_idle(scheduler, time_window)
    
    print(f"\n📈 基线结果:")
    print(f"  - 资源冲突: {len(baseline_conflicts)}")
    print(f"  - 平均FPS满足率: {baseline_stats['total_fps_rate'] / len(tasks):.1%}")
    print(f"  - NPU利用率: {baseline_util['NPU']['overall_utilization']:.1%}")
    print(f"  - DSP利用率: {baseline_util['DSP']['overall_utilization']:.1%}")
    print(f"  - 紧凑化后空闲时间: {baseline_idle:.1f}ms ({baseline_idle/time_window*100:.1f}%)")
    
    # 保存基线可视化
    try:
        # 恢复原始调度用于可视化
        scheduler.schedule_history = baseline_results
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt()
        plt.savefig('aggressive_baseline.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 紧凑化后的可视化
        scheduler.schedule_history = baseline_compacted
        viz2 = ElegantSchedulerVisualizer(scheduler)
        viz2.plot_elegant_gantt()
        plt.savefig('aggressive_baseline_compacted.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"⚠️ 基线可视化失败: {e}")
    
    # ========== 激进优化 ==========
    print("\n" + "=" * 80)
    print("🧬 开始激进遗传算法优化")
    print("=" * 80)
    
    # 恢复原始调度状态
    scheduler.schedule_history = baseline_results
    
    # 创建激进优化器
    optimizer = AggressiveIdleOptimizer(scheduler, tasks, time_window)
    optimizer.set_baseline_performance(baseline_stats, len(baseline_conflicts))
    
    # 运行激进优化
    best_individual = optimizer.optimize_for_idle_time()
    optimizer.print_idle_optimization_report()
    
    # ========== 评估优化结果 ==========
    print("\n📊 评估优化结果...")
    scheduler.schedule_history.clear()
    optimized_results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    # 验证优化结果
    is_valid, optimized_conflicts = validate_schedule_correctly(scheduler)
    optimized_stats = analyze_fps_satisfaction(scheduler, time_window)
    optimized_util = calculate_detailed_utilization(scheduler, time_window)
    
    # 运行紧凑化测量优化后空闲时间
    optimized_idle, optimized_compacted, _ = run_compaction_and_measure_idle(scheduler, time_window)
    
    print(f"\n📈 优化后结果:")
    print(f"  - 资源冲突: {len(optimized_conflicts)}")
    print(f"  - 平均FPS满足率: {optimized_stats['total_fps_rate'] / len(tasks):.1%}")
    print(f"  - NPU利用率: {optimized_util['NPU']['overall_utilization']:.1%}")
    print(f"  - DSP利用率: {optimized_util['DSP']['overall_utilization']:.1%}")
    print(f"  - 紧凑化后空闲时间: {optimized_idle:.1f}ms ({optimized_idle/time_window*100:.1f}%)")
    
    # 保存优化后可视化
    try:
        # 优化后原始调度
        scheduler.schedule_history = optimized_results
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt()
        plt.savefig('aggressive_optimized.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # 优化后紧凑化调度
        scheduler.schedule_history = optimized_compacted
        viz2 = ElegantSchedulerVisualizer(scheduler)
        viz2.plot_elegant_gantt()
        
        # 在图上标注空闲时间
        ax = plt.gca()
        if optimized_idle > 0:
            idle_start = time_window - optimized_idle
            ax.axvspan(idle_start, time_window, alpha=0.3, color='lightgreen')
            ax.text(idle_start + optimized_idle/2, ax.get_ylim()[1]*0.95,
                   f'{optimized_idle:.1f}ms\nIDLE\n({optimized_idle/time_window*100:.1f}%)', 
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
        
        plt.savefig('aggressive_optimized_compacted.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        # Chrome trace
        viz2.export_chrome_tracing('aggressive_optimized_trace.json')
    except Exception as e:
        print(f"⚠️ 优化可视化失败: {e}")
    
    # ========== 最终对比 ==========
    print("\n" + "=" * 80)
    print("📊 优化效果总结")
    print("=" * 80)
    
    print("\n指标对比:")
    print(f"{'指标':<20} {'基线':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 65)
    
    # 空闲时间对比（最重要）
    idle_improvement = optimized_idle - baseline_idle
    print(f"{'紧凑化后空闲时间':<20} {baseline_idle:.1f}ms ({baseline_idle/time_window*100:.1f}%)  "
          f"{optimized_idle:.1f}ms ({optimized_idle/time_window*100:.1f}%)  "
          f"+{idle_improvement:.1f}ms")
    
    # FPS对比
    baseline_avg_fps = baseline_stats['total_fps_rate'] / len(tasks)
    optimized_avg_fps = optimized_stats['total_fps_rate'] / len(tasks)
    fps_change = (optimized_avg_fps - baseline_avg_fps) * 100
    print(f"{'平均FPS满足率':<20} {baseline_avg_fps:.1%}{'':12} "
          f"{optimized_avg_fps:.1%}{'':12} "
          f"{fps_change:+.1f}%")
    
    # 资源利用率
    print(f"{'NPU利用率':<20} {baseline_util['NPU']['overall_utilization']:.1%}{'':12} "
          f"{optimized_util['NPU']['overall_utilization']:.1%}{'':12} "
          f"{(optimized_util['NPU']['overall_utilization'] - baseline_util['NPU']['overall_utilization']):.1%}")
    
    print(f"{'DSP利用率':<20} {baseline_util['DSP']['overall_utilization']:.1%}{'':12} "
          f"{optimized_util['DSP']['overall_utilization']:.1%}{'':12} "
          f"{(optimized_util['DSP']['overall_utilization'] - baseline_util['DSP']['overall_utilization']):.1%}")
    
    # 冲突数
    print(f"{'资源冲突数':<20} {len(baseline_conflicts):<15} {len(optimized_conflicts):<15} "
          f"{len(baseline_conflicts) - len(optimized_conflicts)}")
    
    # 生成空闲时间对比图
    visualize_idle_comparison(baseline_idle, optimized_idle, time_window)
    
    # 详细任务执行对比
    print("\n📋 任务执行详情:")
    print(f"{'任务':<8} {'FPS要求':<10} {'基线执行':<10} {'优化执行':<10} {'差异':<10}")
    print("-" * 50)
    
    for task_id in sorted(baseline_stats['task_fps'].keys()):
        baseline_info = baseline_stats['task_fps'][task_id]
        optimized_info = optimized_stats['task_fps'][task_id]
        task = next(t for t in tasks if t.task_id == task_id)
        
        diff = optimized_info['count'] - baseline_info['count']
        print(f"{task_id:<8} {task.fps_requirement:<10} "
              f"{baseline_info['count']:<10} {optimized_info['count']:<10} "
              f"{diff:+d}")
    
    print("\n📁 生成的文件:")
    print("  - aggressive_baseline.png: 基线调度图")
    print("  - aggressive_baseline_compacted.png: 基线紧凑化调度图")
    print("  - aggressive_optimized.png: 优化后调度图")
    print("  - aggressive_optimized_compacted.png: 优化后紧凑化调度图")
    print("  - aggressive_optimized_trace.json: Chrome追踪文件")
    print("  - idle_time_comparison.png: 空闲时间对比图")
    
    print(f"\n✨ 优化完成！空闲时间从 {baseline_idle:.1f}ms 提升到 {optimized_idle:.1f}ms "
          f"(+{idle_improvement:.1f}ms, 提升{idle_improvement/baseline_idle*100:.1f}%)")


if __name__ == "__main__":
    main()
