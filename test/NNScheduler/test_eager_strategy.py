#!/usr/bin/env python3
"""验证 eager 策略下的依赖缓冲与发射逻辑"""

import pytest

from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.enums import ResourceType, TaskPriority
from NNScheduler.core.task import NNTask


def _setup_queue() -> ResourceQueueManager:
    queue = ResourceQueueManager()
    queue.add_resource("NPU_0", ResourceType.NPU, 60.0)
    return queue


def _make_task(task_id: str, fps: float, duration_ms: float, priority: TaskPriority) -> NNTask:
    task = NNTask(task_id=task_id, priority=priority)
    task.set_performance_requirements(fps=fps, latency=1000.0 / fps)
    task.add_segment(ResourceType.NPU, {60.0: duration_ms}, segment_id=f"{task_id}_seg")
    return task


def test_eager_dependency_guard_scales_with_duration():
    queue = _setup_queue()
    launcher = EnhancedTaskLauncher(queue)

    fast = _make_task("fast", fps=500.0, duration_ms=0.5, priority=TaskPriority.CRITICAL)
    slow = _make_task("slow", fps=500.0, duration_ms=4.0, priority=TaskPriority.HIGH)
    consumer = _make_task("consumer", fps=500.0, duration_ms=1.0, priority=TaskPriority.NORMAL)

    consumer.add_dependency("fast")
    consumer.add_dependency("slow")

    for task in (fast, slow, consumer):
        launcher.register_task(task)

    plan = launcher.create_launch_plan(30.0, "eager")

    fast_launch = plan.task_schedules["fast"][0]
    slow_launch = plan.task_schedules["slow"][0]
    consumer_launch = plan.task_schedules["consumer"][0]

    assert fast_launch == pytest.approx(0.0, abs=1e-6)
    assert slow_launch == pytest.approx(0.0, abs=1e-6)

    fast_completion = fast_launch + 0.5 * 1.1
    slow_completion = slow_launch + 4.0 * 1.1

    expected_guard_fast = max(0.2, min(3.0, fast_completion * 0.1))
    expected_guard_slow = max(0.2, min(3.0, slow_completion * 0.1))

    min_expected = max(fast_completion + expected_guard_fast, slow_completion + expected_guard_slow)

    assert consumer_launch == pytest.approx(min_expected, abs=1e-3)


def test_eager_dependency_guard_small_tasks_minimum():
    queue = _setup_queue()
    launcher = EnhancedTaskLauncher(queue)

    base = _make_task("base", fps=1000.0, duration_ms=0.1, priority=TaskPriority.CRITICAL)
    follower = _make_task("follower", fps=1000.0, duration_ms=0.1, priority=TaskPriority.HIGH)
    follower.add_dependency("base")

    launcher.register_task(base)
    launcher.register_task(follower)

    plan = launcher.create_launch_plan(5.0, "eager")
    base_launch = plan.task_schedules["base"][0]
    follower_launch = plan.task_schedules["follower"][0]

    completion = base_launch + 0.1 * 1.1
    guard = max(0.2, min(3.0, completion * 0.1))
    assert follower_launch == pytest.approx(completion + guard, abs=1e-3)
