#!/usr/bin/env python3
"""
测试 ScheduleTracer 功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enums import ResourceType, TaskPriority
from viz.schedule_visualizer import ScheduleVisualizer


def test_schedule_tracer():
    """测试调度追踪器"""
    print("=== 测试 ScheduleTracer ===\n")
    
    # 创建资源队列管理器
    manager = ResourceQueueManager()
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    # 创建追踪器和可视化器
    tracer = ScheduleTracer(manager)
    visualizer = ScheduleVisualizer(tracer)
    
    # 测试执行记录
    print("测试任务执行记录:")
    
    test_executions = [
        # 基本功能测试
        ("TASK_1", "NPU_0", 0.0, 5.0, TaskPriority.CRITICAL, 60.0),
        ("TASK_2", "NPU_1", 2.0, 10.0, TaskPriority.HIGH, 60.0),
        ("TASK_3", "DSP_0", 0.0, 10.0, TaskPriority.NORMAL, 40.0),
        ("TASK_4", "NPU_0", 5.0, 11.0, TaskPriority.NORMAL, 60.0),
        
        # 测试并发执行
        ("TASK_5", "NPU_1", 10.0, 15.0, TaskPriority.HIGH, 60.0),
        ("TASK_6", "DSP_0", 10.0, 18.0, TaskPriority.NORMAL, 40.0),
        
        # 测试低优先级任务
        ("TASK_7", "NPU_0", 11.0, 20.0, TaskPriority.LOW, 60.0),
    ]
    
    # 记录执行
    for task_id, res_id, start, end, priority, bw in test_executions:
        # 记录入队
        tracer.record_enqueue(task_id, res_id, priority, start, [])
        # 记录执行
        tracer.record_execution(task_id, res_id, start, end, bw)
        print(f"  {start:.1f}-{end:.1f}ms: {res_id} 执行 {task_id}")
    
    # 测试文本甘特图
    print("\n测试文本甘特图:")
    print("-" * 60)
    visualizer.print_gantt_chart(width=60)
    
    # 测试统计功能
    print("\n测试统计功能:")
    stats = tracer.get_statistics()
    
    # 验证统计数据
    assert stats['total_tasks'] == 7, f"Expected 7 tasks, got {stats['total_tasks']}"
    assert stats['total_executions'] == 7, f"Expected 7 executions, got {stats['total_executions']}"
    print(f"  ✓ 总任务数: {stats['total_tasks']}")
    print(f"  ✓ 总执行次数: {stats['total_executions']}")
    print(f"  ✓ 时间跨度: {stats['time_span']:.1f}ms")
    
    # 测试资源利用率
    print("\n测试资源利用率计算:")
    for res_id in ["NPU_0", "NPU_1", "DSP_0"]:
        util = stats['resource_utilization'].get(res_id, 0)
        print(f"  {res_id}: {util:.1f}%")
        assert util > 0, f"Resource {res_id} should have utilization > 0"
    
    # 测试优先级分布
    print("\n测试优先级分布:")
    priority_dist = stats['tasks_by_priority']
    assert priority_dist.get('CRITICAL', 0) == 1, "Should have 1 CRITICAL task"
    assert priority_dist.get('HIGH', 0) == 2, "Should have 2 HIGH tasks"
    assert priority_dist.get('NORMAL', 0) == 3, "Should have 3 NORMAL tasks"
    assert priority_dist.get('LOW', 0) == 1, "Should have 1 LOW task"
    print("  ✓ 优先级分布正确")
    
    # 测试时间线获取
    print("\n测试时间线获取:")
    timeline = tracer.get_timeline()
    assert "NPU_0" in timeline, "NPU_0 should be in timeline"
    assert "NPU_1" in timeline, "NPU_1 should be in timeline"
    assert "DSP_0" in timeline, "DSP_0 should be in timeline"
    assert len(timeline["NPU_0"]) == 3, "NPU_0 should have 3 executions"
    assert len(timeline["NPU_1"]) == 2, "NPU_1 should have 2 executions"
    assert len(timeline["DSP_0"]) == 2, "DSP_0 should have 2 executions"
    print("  ✓ 时间线数据正确")
    
    # 测试任务时间线
    print("\n测试任务时间线:")
    task1_timeline = tracer.get_task_timeline("TASK_1")
    assert len(task1_timeline) == 1, "TASK_1 should have 1 execution"
    assert task1_timeline[0].start_time == 0.0, "TASK_1 should start at 0.0"
    assert task1_timeline[0].end_time == 5.0, "TASK_1 should end at 5.0"
    print("  ✓ 任务时间线查询正确")
    
    # 测试可视化输出
    print("\n测试可视化输出:")
    
    # Matplotlib图表
    visualizer.plot_resource_timeline("test_timeline.png")
    assert os.path.exists("test_timeline.png"), "Timeline plot should be created"
    print("  ✓ Matplotlib图表生成成功")
    
    # Chrome Tracing
    visualizer.export_chrome_tracing("test_trace.json")
    assert os.path.exists("test_trace.json"), "Chrome trace should be created"
    print("  ✓ Chrome Tracing文件生成成功")
    
    # 测试边界情况
    print("\n测试边界情况:")
    
    # 空追踪器
    empty_manager = ResourceQueueManager()
    empty_tracer = ScheduleTracer(empty_manager)
    empty_viz = ScheduleVisualizer(empty_tracer)
    
    # 应该能处理空数据
    empty_viz.print_gantt_chart(width=40)
    empty_stats = empty_tracer.get_statistics()
    assert empty_stats['total_tasks'] == 0, "Empty tracer should have 0 tasks"
    print("  ✓ 空数据处理正常")
    
    # 测试单个任务
    single_manager = ResourceQueueManager()
    single_manager.add_resource("TEST_RES", ResourceType.NPU, 100.0)
    single_tracer = ScheduleTracer(single_manager)
    single_tracer.record_execution("SINGLE_TASK", "TEST_RES", 0.0, 10.0, 100.0)
    
    single_stats = single_tracer.get_statistics()
    assert single_stats['total_tasks'] == 1, "Should have 1 task"
    assert single_stats['resource_utilization']['TEST_RES'] == 100.0, "Should have 100% utilization"
    print("  ✓ 单任务处理正常")
    
    print("\n✅ 所有测试通过！")
    
    # 清理测试文件
    import os
    if os.path.exists("test_timeline.png"):
        os.remove("test_timeline.png")
    if os.path.exists("test_trace.json"):
        os.remove("test_trace.json")
    print("\n已清理测试文件")


if __name__ == "__main__":
    test_schedule_tracer()
