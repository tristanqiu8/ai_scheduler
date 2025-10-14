"""
Test directory conftest.py
Test fixtures and configuration
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from NNScheduler.core.artifacts import get_artifacts_root
from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.enums import ResourceType, TaskPriority
from NNScheduler.core.task import NNTask


# 标准化所有测试输出目录
ARTIFACTS_ROOT = get_artifacts_root().resolve()
os.environ["AI_SCHEDULER_ARTIFACTS_DIR"] = str(ARTIFACTS_ROOT)


@pytest.fixture
def queue_manager():
    """创建资源队列管理器fixture"""
    manager = ResourceQueueManager()
    manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    return manager


@pytest.fixture
def simple_task():
    """创建简单任务fixture"""
    task = NNTask("TEST", "Test Task", priority=TaskPriority.NORMAL)
    task.add_segment(ResourceType.NPU, {60: 5.0, 40: 7.5}, "main")
    task.set_performance_requirements(fps=10, latency=100)
    return task


@pytest.fixture
def mixed_task():
    """创建混合资源任务fixture"""
    task = NNTask("MIXED", "Mixed Task", priority=TaskPriority.HIGH)
    task.add_segment(ResourceType.NPU, {60: 5.0}, "npu_seg")
    task.add_segment(ResourceType.DSP, {40: 3.0}, "dsp_seg")
    task.set_performance_requirements(fps=20, latency=50)
    return task


@pytest.fixture
def tracer(queue_manager):
    """创建调度追踪器fixture"""
    return ScheduleTracer(queue_manager)


@pytest.fixture
def launcher(queue_manager, tracer):
    """创建任务发射器fixture"""
    return TaskLauncher(queue_manager, tracer)
