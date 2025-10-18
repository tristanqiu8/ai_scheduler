#!/usr/bin/env python3
"""Tests for the fixed launch strategy."""

import pytest

from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.enums import ResourceType, TaskPriority
from NNScheduler.core.task import NNTask


def create_queue_with_resources() -> ResourceQueueManager:
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    queue_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    return queue_manager


def make_task(
    task_id: str,
    fps: float,
    offset: float,
    *,
    priority: TaskPriority = TaskPriority.NORMAL,
    respect_dependencies: bool = False,
    latency: float = 100.0,
    segment_duration_ms: float = 5.0,
) -> NNTask:
    task = NNTask(task_id=task_id, priority=priority)
    task.set_performance_requirements(fps=fps, latency=latency)
    task.set_launch_phase(offset_ms=offset, respect_dependencies=respect_dependencies)
    task.add_segment(ResourceType.NPU, {60.0: segment_duration_ms}, segment_id=f"{task_id}_npu")
    return task


def test_fixed_mode_simple_offsets():
    queue_manager = create_queue_with_resources()
    launcher = EnhancedTaskLauncher(queue_manager)

    task_fast = make_task("fast", fps=100.0, offset=5.0, priority=TaskPriority.CRITICAL)
    task_slow = make_task("slow", fps=50.0, offset=15.0, priority=TaskPriority.HIGH)

    for task in (task_fast, task_slow):
        launcher.register_task(task)

    time_window = 60.0
    plan = launcher.create_launch_plan(time_window, "fixed")

    fast_times = plan.task_schedules["fast"]
    slow_times = plan.task_schedules["slow"]

    expected_fast = [5.0 + i * 10.0 for i in range(6)]
    expected_slow = [15.0 + i * 20.0 for i in range(3)]

    assert len(fast_times) == len(expected_fast)
    assert len(slow_times) == len(expected_slow)
    assert fast_times == pytest.approx(expected_fast, abs=1e-6)
    assert slow_times == pytest.approx(expected_slow, abs=1e-6)


def test_fixed_mode_respects_dependencies_when_requested():
    queue_manager = create_queue_with_resources()
    launcher = EnhancedTaskLauncher(queue_manager)

    upstream = make_task(
        "upstream",
        fps=40.0,
        offset=0.0,
        priority=TaskPriority.CRITICAL,
        segment_duration_ms=8.0,
    )
    downstream = make_task(
        "downstream",
        fps=40.0,
        offset=0.0,
        respect_dependencies=True,
        priority=TaskPriority.HIGH,
        segment_duration_ms=4.0,
    )
    downstream.add_dependency("upstream")

    independent = make_task("independent", fps=25.0, offset=12.0, priority=TaskPriority.NORMAL)

    for task in (upstream, downstream, independent):
        launcher.register_task(task)

    time_window = 80.0
    plan = launcher.create_launch_plan(time_window, "fixed")

    upstream_times = plan.task_schedules["upstream"]
    downstream_times = plan.task_schedules["downstream"]
    independent_times = plan.task_schedules["independent"]

    expected_upstream = [0.0 + i * 25.0 for i in range(4)]
    expected_independent = [12.0 + i * 40.0 for i in range(2)]
    expected_downstream = [expected_upstream[i] + 8.8 + 1.0 for i in range(3)]

    assert upstream_times == pytest.approx(expected_upstream, abs=1e-6)
    assert independent_times == pytest.approx(expected_independent, abs=1e-6)
    assert downstream_times == pytest.approx(expected_downstream, abs=1e-3)
