"""
pytest配置和共享fixtures
"""

import pytest
import sys
import os

# 确保项目根目录在Python路径中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.launcher import TaskLauncher
from core.enums import ResourceType, TaskPriority
from core.task import NNTask


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


# 自定义标记
def pytest_configure(config):
    """注册自定义标记"""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "visualization: marks tests that generate visualization")


# 测试收集钩子
def pytest_collection_modifyitems(config, items):
    """修改测试收集行为"""
    for item in items:
        # 自动为某些测试添加标记
        if "test_visualization" in item.nodeid:
            item.add_marker(pytest.mark.visualization)
        if "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)