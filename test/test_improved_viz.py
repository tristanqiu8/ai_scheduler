#!/usr/bin/env python3
"""
演示可视化改进效果
特别是确保所有任务标签都能正确显示
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enums import ResourceType, TaskPriority
from viz.schedule_visualizer import ScheduleVisualizer


def demo_viz_improvements():
    """演示改进后的可视化效果"""
    print("=== 可视化改进演示 ===\n")
    
    # 创建资源和追踪器
    manager = ResourceQueueManager()
    for res_id in ["NPU_0", "NPU_1", "DSP_0", "DSP_1"]:
        res_type = ResourceType.NPU if "NPU" in res_id else ResourceType.DSP
        bandwidth = 60.0 if res_type == ResourceType.NPU else 40.0
        manager.add_resource(res_id, res_type, bandwidth)
    
    tracer = ScheduleTracer(manager)
    visualizer = ScheduleVisualizer(tracer)
    
    # 模拟一些执行记录
    executions = [
        # NPU_0: CRITICAL和NORMAL任务
        ("CRIT_DET", "NPU_0", 0.0, 5.0, TaskPriority.CRITICAL, 60.0),
        ("NORM_PROC", "NPU_0", 5.0, 13.0, TaskPriority.NORMAL, 60.0),
        
        # NPU_1: HIGH和NORMAL任务
        ("HIGH_VIDEO", "NPU_1", 2.0, 8.0, TaskPriority.HIGH, 60.0),
        ("NORM_CLEAN", "NPU_1", 10.0, 15.0, TaskPriority.NORMAL, 60.0),
        
        # DSP_0: 长HIGH任务
        ("HIGH_AUDIO", "DSP_0", 0.0, 10.0, TaskPriority.HIGH, 40.0),
        
        # DSP_1: NORMAL任务
        ("NORM_FILTER", "DSP_1", 3.0, 10.0, TaskPriority.NORMAL, 40.0),
    ]
    
    # 记录执行
    for task_id, res_id, start, end, priority, bw in executions:
        # 记录入队（用于统计）
        tracer.record_enqueue(task_id, res_id, priority, start, [])
        # 记录执行
        tracer.record_execution(task_id, res_id, start, end, bw)
    
    # 1. 生成matplotlib图表
    print("生成可视化输出...")
    visualizer.plot_resource_timeline("improved_timeline.png")
    
    # 2. 生成Chrome Tracing
    visualizer.export_chrome_tracing("improved_trace.json")
    
    # 3. 打印文本甘特图
    print("\n文本甘特图:")
    print("-" * 70)
    visualizer.print_gantt_chart(width=70)
    
    # 4. 显示统计
    stats = tracer.get_statistics()
    print(f"\n资源利用率:")
    for res_id in sorted(["NPU_0", "NPU_1", "DSP_0", "DSP_1"]):
        util = stats['resource_utilization'].get(res_id, 0)
        print(f"  {res_id}: {util:.1f}%")
    
    print("\n✅ 可视化文件已生成:")
    print("  - improved_timeline.png (Matplotlib图表)")
    print("  - improved_trace.json (Chrome Tracing)")


if __name__ == "__main__":
    demo_viz_improvements()
