#!/usr/bin/env python3
"""
测试空隙感知优化
"""

import sys
import os
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    MultiResourceScheduler, GapAwareOptimizer,
    apply_basic_fixes, apply_minimal_fifo_fix, 
    apply_strict_resource_conflict_fix
)
from fix_segmentation_issue import create_and_test_fixed_tasks
from viz.elegant_visualization import ElegantSchedulerVisualizer
import matplotlib.pyplot as plt


def test_gap_aware_optimization():
    """测试空隙感知优化"""
    print("=" * 80)
    print("🚀 测试空隙感知优化")
    print("=" * 80)
    
    # 创建调度器和任务
    scheduler, tasks = create_and_test_fixed_tasks()
    
    # 第一阶段：原始调度
    print("\n📅 第一阶段：执行原始调度")
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(100.0)
    
    # 显示原始结果
    print(f"\n原始调度事件数: {len(results)}")
    
    # 保存原始调度用于对比
    original_schedule = copy.deepcopy(scheduler.schedule_history)
    
    # 第二阶段：空隙优化
    print("\n📅 第二阶段：空隙感知优化")
    optimizer = GapAwareOptimizer(scheduler)
    new_insertions = optimizer.optimize_schedule(100.0)
    
    # 生成对比可视化
    print("\n📊 生成可视化对比...")
    
    # 原始调度图
    scheduler.schedule_history = original_schedule
    viz1 = ElegantSchedulerVisualizer(scheduler)
    plt.figure(figsize=(20, 8))
    plt.subplot(2, 1, 1)
    viz1.plot_elegant_gantt(time_window=50.0)
    plt.title('原始调度（无空隙优化）')
    
    # 优化后调度图
    scheduler.schedule_history.extend(new_insertions)
    scheduler.schedule_history.sort(key=lambda x: x.start_time)
    viz2 = ElegantSchedulerVisualizer(scheduler)
    plt.subplot(2, 1, 2)
    viz2.plot_elegant_gantt(time_window=50.0)
    plt.title('优化后调度（空隙插入）')
    
    plt.tight_layout()
    plt.savefig('gap_optimization_comparison.png', dpi=150)
    plt.close()
    
    print("\n✅ 测试完成！查看 gap_optimization_comparison.png")
    
    # 导出Chrome trace
    viz2.export_chrome_tracing('gap_optimized_trace.json')
    print("✅ Chrome trace已导出到 gap_optimized_trace.json")


if __name__ == "__main__":
    test_gap_aware_optimization()