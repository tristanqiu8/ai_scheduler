#!/usr/bin/env python3
"""验证 ScheduleExecutor 在段级模式下严格遵守任务依赖。"""

import os
import sys

import pytest


# 允许直接运行该测试文件
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from NNScheduler.core.enums import ResourceType, TaskPriority
from NNScheduler.core.task import NNTask
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.executor import ScheduleExecutor


def _make_tasks():
    """构造一个简单的依赖场景。"""
    # 1. 预处理任务：运行在 DSP，10ms，低 FPS
    t_pre = NNTask("T_PRE", "Preprocess", priority=TaskPriority.CRITICAL)
    t_pre.set_dsp_only({40.0: 10.0})
    t_pre.set_performance_requirements(fps=10.0, latency=100.0)

    # 2. 高优先级推理任务：依赖 T_PRE，运行在 NPU，5ms
    t_high = NNTask("T_HIGH", "High", priority=TaskPriority.CRITICAL)
    t_high.set_npu_only({40.0: 5.0})
    t_high.set_performance_requirements(fps=10.0, latency=20.0)
    t_high.add_dependency("T_PRE")

    # 3. 低优先级长任务：运行在 NPU，15ms
    t_low = NNTask("T_LOW", "Low", priority=TaskPriority.LOW)
    t_low.set_npu_only({40.0: 15.0})
    t_low.set_performance_requirements(fps=50.0, latency=100.0)

    return [t_pre, t_high, t_low]


def test_executor_respects_dependencies_segment_mode():
    """段级执行模式下，依赖任务必须先完成。"""

    # 资源与追踪器
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)

    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)

    for task in _make_tasks():
        launcher.register_task(task)

    # 急切计划会在 t=0 发射所有任务
    plan = launcher.create_launch_plan(200.0, "eager")

    executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
    executor.execute_plan(plan, 200.0, segment_mode=True)

    # 收集首个实例的执行段
    pre_exec = next(e for e in tracer.executions if e.task_id.startswith("T_PRE#0"))
    high_exec = next(e for e in tracer.executions if e.task_id.startswith("T_HIGH#0"))

    # T_HIGH 必须等到 T_PRE 完成后才能开始
    assert high_exec.start_time >= pre_exec.end_time - 1e-6


if __name__ == "__main__":
    pytest.main([__file__])
