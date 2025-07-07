#!/usr/bin/env python3
"""
修复延迟计算问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import PerformanceEvaluator
from collections import defaultdict


# 保存原始方法
original_evaluate_task_performance = PerformanceEvaluator._evaluate_task_performance


def fixed_evaluate_task_performance(self, launch_events=None):
    """修复后的任务性能评估方法"""
    # 初始化任务指标
    for task_id, task in self.tasks.items():
        self.task_metrics[task_id] = self.TaskPerformanceMetrics(
            task_id=task_id,
            task_name=task.name,
            priority=task.priority,
            fps_requirement=task.fps_requirement,
            latency_requirement=task.latency_requirement
        )
    
    # 构建发射时间映射
    launch_times = defaultdict(list)  # task_id -> [(instance_id, launch_time)]
    if launch_events:
        for event in launch_events:
            # 尝试获取时间属性（兼容不同的属性名）
            launch_time = None
            if hasattr(event, 'time'):
                launch_time = event.time
            elif hasattr(event, 'launch_time'):
                launch_time = event.launch_time
            
            if launch_time is not None and hasattr(event, 'task_id'):
                instance_id = getattr(event, 'instance_id', 0)
                launch_times[event.task_id].append((instance_id, launch_time))
    
    # 分析执行历史
    task_instances = defaultdict(lambda: defaultdict(list))  # task_id -> instance -> executions
    
    for execution in self.tracer.executions:
        # 解析任务ID和实例号
        if '#' in execution.task_id:
            base_task_id, instance_info = execution.task_id.split('#', 1)
            if '_' in instance_info:
                instance_num = int(instance_info.split('_')[0])
            else:
                instance_num = int(instance_info)
        else:
            base_task_id = execution.task_id
            instance_num = 0
        
        if base_task_id in self.task_metrics:
            task_instances[base_task_id][instance_num].append(execution)
    
    # 计算每个任务的指标
    for task_id, instances in task_instances.items():
        metrics = self.task_metrics[task_id]
        metrics.instance_count = len(instances)
        
        # 处理每个实例
        for instance_num, executions in instances.items():
            if not executions:
                continue
            
            # 找到该实例的第一次和最后一次执行
            first_exec = min(executions, key=lambda e: e.start_time)
            last_exec = max(executions, key=lambda e: e.end_time)
            
            # 查找对应的发射时间
            launch_time = None
            if task_id in launch_times:
                # 查找匹配的实例发射时间
                for inst_id, l_time in launch_times[task_id]:
                    if inst_id == instance_num:
                        launch_time = l_time
                        break
            
            # 计算延迟
            if launch_time is not None:
                # 等待时间：发射到首次执行
                wait_time = first_exec.start_time - launch_time
                metrics.wait_times.append(wait_time)
                
                # 总延迟：发射到完成
                total_latency = last_exec.end_time - launch_time
                metrics.latencies.append(total_latency)
                
                # 检查延迟违规
                if total_latency > metrics.latency_requirement:
                    metrics.latency_violations += 1
            
            # 累计执行时间
            for exec in executions:
                metrics.execution_times.append(exec.duration)
                metrics.total_execution_time += exec.duration
                metrics.execution_count += 1
        
        # 计算平均值
        if metrics.wait_times:
            metrics.avg_wait_time = sum(metrics.wait_times) / len(metrics.wait_times)
        
        if metrics.latencies:
            metrics.avg_latency = sum(metrics.latencies) / len(metrics.latencies)
            metrics.max_latency = max(metrics.latencies)
            metrics.latency_satisfaction_rate = 1.0 - (metrics.latency_violations / len(metrics.latencies))
        
        if metrics.execution_times:
            metrics.avg_execution_time = sum(metrics.execution_times) / len(metrics.execution_times)
        
        # 计算FPS
        if self.time_window > 0:
            metrics.achieved_fps = (metrics.instance_count * 1000.0) / self.time_window
            metrics.fps_achievement_rate = min(100.0, (metrics.achieved_fps / metrics.fps_requirement) * 100.0)
            metrics.fps_satisfaction = metrics.achieved_fps >= metrics.fps_requirement


# 应用修复
PerformanceEvaluator._evaluate_task_performance = fixed_evaluate_task_performance


if __name__ == "__main__":
    # 测试修复
    from core import (
        ResourceType, TaskPriority, NNTask,
        ResourceQueueManager, ScheduleTracer,
        TaskLauncher, ScheduleExecutor,
        TaskPerformanceMetrics
    )
    
    print("测试延迟计算修复")
    print("="*80)
    
    # 创建测试环境
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 创建测试任务
    task = NNTask("TEST", "测试任务", priority=TaskPriority.HIGH)
    task.add_segment(ResourceType.NPU, {60: 10.0}, "compute")
    task.set_performance_requirements(fps=10, latency=20)  # 20ms延迟要求
    launcher.register_task(task)
    
    # 创建发射计划
    time_window = 100.0
    plan = launcher.create_launch_plan(time_window, "eager")
    
    print(f"\n发射事件:")
    for event in plan.events[:3]:
        print(f"  {event.time:.1f}ms: {event.task_id}#{event.instance_id}")
    
    # 模拟延迟执行（在发射后5ms才开始执行）
    print("\n模拟延迟执行...")
    
    # 需要先导入TaskPerformanceMetrics
    PerformanceEvaluator.TaskPerformanceMetrics = TaskPerformanceMetrics
    
    # 模拟延迟执行 - 使用tracer的record_execution方法
    
    # 第一个实例：发射在0ms，执行在5ms
    tracer.record_execution(
        task_id="TEST#0_seg0",
        resource_id="NPU_0",
        start_time=5.0,
        end_time=15.0,
        bandwidth=60.0,
        segment_id="compute"
    )
    
    # 第二个实例：发射在100ms，执行在108ms（如果有的话）
    if len(plan.events) > 1:
        tracer.record_execution(
            task_id="TEST#1_seg0",
            resource_id="NPU_0",
            start_time=108.0,
            end_time=118.0,
            bandwidth=60.0,
            segment_id="compute"
        )
    
    # 评估性能
    evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
    metrics = evaluator.evaluate(time_window, plan.events)
    
    # 检查延迟计算
    print(f"\n任务性能指标:")
    test_metrics = evaluator.task_metrics.get("TEST")
    if test_metrics:
        print(f"  实例数: {test_metrics.instance_count}")
        print(f"  平均等待时间: {test_metrics.avg_wait_time:.1f}ms")
        print(f"  平均延迟: {test_metrics.avg_latency:.1f}ms")
        print(f"  最大延迟: {test_metrics.max_latency:.1f}ms")
        print(f"  延迟满足率: {test_metrics.latency_satisfaction_rate:.1%}")
        
        if test_metrics.avg_latency > 0:
            print("\n✅ 延迟计算修复成功！")
        else:
            print("\n❌ 延迟计算仍有问题")
    
    # 显示整体指标
    print(f"\n整体性能指标:")
    print(f"  平均等待时间: {metrics.avg_wait_time:.1f}ms")
    print(f"  平均延迟: {metrics.avg_latency:.1f}ms")
    print(f"  最大延迟: {metrics.max_latency:.1f}ms")
