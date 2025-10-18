#!/usr/bin/env python3
"""Tests covering launch_profile behavior across strategies."""

import pytest

from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.executor import ScheduleExecutor
from NNScheduler.core.task import NNTask
from NNScheduler.core.enums import ResourceType, TaskPriority


def _make_basic_task(task_id: str, fps: float, priority: TaskPriority = TaskPriority.NORMAL) -> NNTask:
    task = NNTask(task_id=task_id, priority=priority)
    task.set_performance_requirements(fps=fps, latency=1000.0 / fps)
    task.add_segment(ResourceType.NPU, {60.0: 5.0}, segment_id=f"{task_id}_npu")
    return task


def _setup_queue_with_npu() -> ResourceQueueManager:
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    return queue_manager


def test_fixed_strategy_applies_random_slack():
    queue_manager = _setup_queue_with_npu()
    tracer = ScheduleTracer(queue_manager)
    launcher = EnhancedTaskLauncher(queue_manager, tracer)

    task = _make_basic_task("fixed_task", fps=50.0, priority=TaskPriority.HIGH)
    task.set_launch_phase(offset_ms=10.0)
    launcher.register_task(task)

    plan = launcher.create_launch_plan(80.0, "fixed")
    assert plan.task_schedules["fixed_task"][0] == pytest.approx(10.0, abs=1e-6)

    executor = ScheduleExecutor(
        queue_manager,
        tracer,
        launcher.tasks,
        random_slack_enabled=True,
        random_slack_std=0.5,
        random_slack_seed=123,
        launch_strategy="fixed",
    )

    executor.execute_plan(plan, 80.0, segment_mode=True)

    task_info = tracer.task_info.get("fixed_task#0")
    assert task_info is not None
    jitter = task_info.get("jitter_ms")
    assert jitter is not None
    assert jitter == pytest.approx(0.202114, abs=1e-3)


def test_eager_and_balanced_respect_launch_profile_offsets():
    queue_manager = _setup_queue_with_npu()

    custom_task = _make_basic_task("custom", fps=50.0, priority=TaskPriority.HIGH)
    custom_task.set_launch_phase(offset_ms=12.0, respect_dependencies=False)

    auto_task = _make_basic_task("auto", fps=50.0, priority=TaskPriority.HIGH)

    # Eager plan
    eager_launcher = EnhancedTaskLauncher(queue_manager)
    eager_launcher.register_task(custom_task)
    eager_launcher.register_task(auto_task)
    eager_plan = eager_launcher.create_launch_plan(70.0, "eager")

    assert eager_plan.task_schedules["custom"][:3] == pytest.approx([12.0, 32.0, 52.0], abs=1e-6)
    assert eager_plan.task_schedules["auto"][0] == pytest.approx(0.0, abs=1e-6)

    # Balanced plan
    bal_launcher = EnhancedTaskLauncher(queue_manager)
    bal_launcher.register_task(custom_task)
    bal_launcher.register_task(auto_task)
    bal_plan = bal_launcher.create_launch_plan(70.0, "balanced")

    assert bal_plan.task_schedules["custom"][0] == pytest.approx(12.0, abs=1e-6)
    assert bal_plan.task_schedules["auto"][0] == pytest.approx(10.0, abs=1e-6)


def test_sync_and_fixed_offsets_differ():
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("ISP_0", ResourceType.ISP, 50.0)
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)

    def build_tasks():
        task_a = NNTask("task_a", priority=TaskPriority.CRITICAL)
        task_a.set_performance_requirements(fps=120.0, latency=8.0)
        task_a.add_segment(ResourceType.ISP, {50.0: 1.0}, segment_id="isp")
        task_a.add_segment(ResourceType.NPU, {60.0: 1.0}, segment_id="npu")

        task_b = NNTask("task_b", priority=TaskPriority.HIGH)
        task_b.set_performance_requirements(fps=120.0, latency=8.0)
        task_b.add_segment(ResourceType.ISP, {50.0: 2.0}, segment_id="isp")
        task_b.add_segment(ResourceType.NPU, {60.0: 1.0}, segment_id="npu")
        task_b.set_launch_phase(offset_ms=5.0)

        return task_a, task_b

    # Sync strategy ignores explicit offsets, derives from ISP duration
    sync_launcher = EnhancedTaskLauncher(queue_manager)
    for task in build_tasks():
        sync_launcher.register_task(task)
    sync_plan = sync_launcher.create_launch_plan(20.0, "sync")

    # Fixed strategy follows configured offsets
    fixed_launcher = EnhancedTaskLauncher(queue_manager)
    for task in build_tasks():
        fixed_launcher.register_task(task)
    fixed_plan = fixed_launcher.create_launch_plan(20.0, "fixed")

    assert sync_plan.task_schedules["task_b"][0] == pytest.approx(1.0, abs=1e-6)
    assert fixed_plan.task_schedules["task_b"][0] == pytest.approx(5.0, abs=1e-6)
