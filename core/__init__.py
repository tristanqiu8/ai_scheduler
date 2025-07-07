# core package initialization
"""
Core components for AI Scheduler - New Architecture
"""

# 基础枚举和模型
from .enums import (
    ResourceType, 
    TaskPriority, 
    RuntimeType, 
    SegmentationStrategy, 
    CutPointStatus
)

from .models import (
    ResourceUnit, 
    TaskScheduleInfo, 
    SubSegment, 
    ResourceSegment, 
    CutPoint
)

# 核心任务定义
from .task import NNTask

# 带宽管理（在资源队列之前）
from .bandwidth_manager import BandwidthManager, BandwidthAllocation

# 资源队列
from .resource_queue import ResourceQueue, ResourceQueueManager, QueuedTask
from .schedule_tracer import ScheduleTracer

# 新架构组件
from .launcher import TaskLauncher, TaskLaunchConfig, LaunchPlan, LaunchEvent
from .executor import ScheduleExecutor, TaskInstance, SegmentCompletion
from .evaluator import PerformanceEvaluator, TaskPerformanceMetrics, ResourceUtilizationMetrics, OverallPerformanceMetrics
from .launch_optimizer import LaunchOptimizer, OptimizationConfig, LaunchStrategy

__all__ = [
    # Enums
    'ResourceType', 
    'TaskPriority', 
    'RuntimeType', 
    'SegmentationStrategy', 
    'CutPointStatus',
    
    # Models
    'ResourceUnit', 
    'TaskScheduleInfo', 
    'SubSegment', 
    'ResourceSegment', 
    'CutPoint',
    
    # Core classes
    'NNTask',
    
    # Bandwidth management
    'BandwidthManager',
    'BandwidthAllocation',
    
    # Resource queues
    'ResourceQueue',
    'ResourceQueueManager', 
    'QueuedTask',
    'ScheduleTracer',
    
    # Launcher components
    'TaskLauncher', 
    'TaskLaunchConfig', 
    'LaunchPlan',
    'LaunchEvent',
    
    # Future components (uncomment when ready)
    'ScheduleExecutor',
    'TaskInstance',
    'SegmentCompletion',
    'PerformanceEvaluator',
    'TaskPerformanceMetrics',
    'ResourceUtilizationMetrics',
    'OverallPerformanceMetrics',
    'LaunchOptimizer',
    'OptimizationConfig',
    'LaunchStrategy'
]
