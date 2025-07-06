#!/usr/bin/env python3
"""
演示 ScheduleTracer 的完整功能
包括甘特图和Chrome Tracing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enums import ResourceType, TaskPriority
from viz.schedule_visualizer import ScheduleVisualizer


def demo_schedule_tracer():
    """演示调度追踪器的所有功能"""
    print("=== ScheduleTracer 功能演示 ===\n")
    
    # 1. 创建资源系统
    manager = ResourceQueueManager()
    
    # 添加多个资源
    resources = [
        ("NPU_0", ResourceType.NPU, 60.0),
        ("NPU_1", ResourceType.NPU, 60.0),
        ("DSP_0", ResourceType.DSP, 40.0),
        ("DSP_1", ResourceType.DSP, 40.0)
    ]
    
    for res_id, res_type, bandwidth in resources:
        manager.add_resource(res_id, res_type, bandwidth)
    
    print("系统资源配置:")
    for rid, queue in sorted(manager.resource_queues.items()):
        print(f"  {rid}: {queue.resource_type.value}, 带宽={queue.bandwidth}")
    
    # 2. 创建追踪器和可视化器
    tracer = ScheduleTracer(manager)
    visualizer = ScheduleVisualizer(tracer)
    
    # 3. 模拟多样化的任务执行
    print("\n模拟任务执行:")
    
    executions = [
        # CRITICAL任务 - 最高优先级
        ("CRITICAL_DETECT", "NPU_0", 0.0, 4.0, TaskPriority.CRITICAL, 60.0),
        
        # HIGH优先级任务 - 视频处理
        ("VIDEO_PROC_1", "DSP_0", 0.0, 8.0, TaskPriority.HIGH, 40.0),
        ("VIDEO_PROC_2", "DSP_1", 5.0, 11.0, TaskPriority.HIGH, 40.0),
        
        # NORMAL优先级任务
        ("NORMAL_TASK_1", "NPU_1", 0.0, 5.0, TaskPriority.NORMAL, 60.0),
        ("NORMAL_TASK_2", "NPU_0", 4.0, 10.0, TaskPriority.NORMAL, 60.0),
        ("NORMAL_TASK_3", "NPU_1", 5.0, 12.0, TaskPriority.NORMAL, 60.0),
        
        # LOW优先级任务 - 后台任务
        ("BACKGROUND_CLEAN", "NPU_0", 10.0, 20.0, TaskPriority.LOW, 60.0),
        
        # 更多混合任务
        ("HIGH_COMPUTE", "NPU_1", 12.0, 18.0, TaskPriority.HIGH, 60.0),
        ("NORMAL_AUDIO", "DSP_0", 8.0, 15.0, TaskPriority.NORMAL, 40.0),
        ("CRITICAL_SAFETY", "DSP_1", 11.0, 14.0, TaskPriority.CRITICAL, 40.0),
    ]
    
    # 记录执行
    for task_id, res_id, start, end, priority, bw in executions:
        # 记录入队（用于统计）
        tracer.record_enqueue(task_id, res_id, priority, start, [])
        # 记录执行
        tracer.record_execution(task_id, res_id, start, end, bw)
        print(f"  {start:>5.1f} - {end:>5.1f}ms: {res_id} 执行 {task_id} ({priority.name})")
    
    print("\n" + "="*60)
    
    # 4. 显示文本甘特图
    print("\n文本格式甘特图:")
    print("-" * 80)
    visualizer.print_gantt_chart(width=80)
    
    # 5. 显示详细统计
    print("\n" + "="*60)
    print("调度统计分析")
    print("="*60)
    
    stats = tracer.get_statistics()
    
    print(f"\n基本统计:")
    print(f"  总任务数: {stats['total_tasks']}")
    print(f"  总执行次数: {stats['total_executions']}")
    print(f"  时间跨度: {stats['time_span']:.1f}ms")
    
    print(f"\n性能指标:")
    print(f"  平均等待时间: {stats['average_wait_time']:.2f}ms")
    print(f"  平均执行时间: {stats['average_execution_time']:.2f}ms")
    
    print(f"\n任务优先级分布:")
    for priority, count in sorted(stats['tasks_by_priority'].items()):
        print(f"  {priority}: {count} 个任务")
    
    print(f"\n资源利用率:")
    all_resources = ["NPU_0", "NPU_1", "DSP_0", "DSP_1"]
    for res_id in sorted(all_resources):
        util = stats['resource_utilization'].get(res_id, 0.0)
        if util > 0:
            print(f"  {res_id}: {util:.1f}%")
        else:
            print(f"  {res_id}: IDLE")
    
    # 6. 生成可视化文件
    print("\n生成可视化文件...")
    
    # Matplotlib图表
    visualizer.plot_resource_timeline("demo_timeline.png")
    
    # Chrome Tracing
    visualizer.export_chrome_tracing("demo_trace.json")
    
    # 详细报告
    visualizer.export_summary_report("demo_report.txt")
    
    print("\n✅ 可视化文件已生成:")
    print("  - demo_timeline.png (Matplotlib图表)")
    print("  - demo_trace.json (Chrome Tracing文件)")
    print("  - demo_report.txt (文本报告)")
    print("\n在Chrome浏览器中打开 chrome://tracing 并加载 demo_trace.json 查看详细时间线")
    
    # 7. 显示任务执行时间线详情
    print("\n" + "="*60)
    print("任务执行时间线详情")
    print("="*60)
    
    timeline = tracer.get_timeline()
    for resource_id in sorted(timeline.keys()):
        executions = timeline[resource_id]
        print(f"\n{resource_id}:")
        for exec in executions:
            print(f"  {exec.start_time:>6.1f} - {exec.end_time:>6.1f}ms: "
                  f"{exec.task_id:<20} (优先级: {exec.priority.name:<8}, "
                  f"带宽: {exec.bandwidth}, 时长: {exec.duration:.1f}ms)")


if __name__ == "__main__":
    demo_schedule_tracer()
