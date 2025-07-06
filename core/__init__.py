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

# 新架构组件（当文件创建后取消注释）
# from .launcher import TaskLauncher, TaskLaunchConfig, LaunchPlan
# from .executor import ScheduleExecutor
# from .evaluator import PerformanceEvaluator, PerformanceMetrics
# from .launch_optimizer import LaunchOptimizer

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
    
    # New architecture (uncomment when ready)
    # 'TaskLauncher', 
    # 'TaskLaunchConfig', 
    # 'LaunchPlan',
    # 'ScheduleExecutor',
    # 'PerformanceEvaluator', 
    # 'PerformanceMetrics',
    # 'LaunchOptimizer'
]
