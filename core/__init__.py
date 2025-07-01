# core package initialization
"""
Core components for AI Scheduler
"""

# 首先导入枚举和基础模型
from .enums import ResourceType, TaskPriority, RuntimeType, SegmentationStrategy, CutPointStatus
from .models import (
    ResourceUnit, TaskScheduleInfo, ResourceBinding, 
    SegmentationDecision, SubSegment, ResourceSegment, CutPoint
)

# 然后导入其他核心组件
from .task import NNTask
from .scheduler import MultiResourceScheduler
from .priority_queue import ResourcePriorityQueues

# 导入修复模块
from .modular_scheduler_fixes import apply_basic_fixes, apply_performance_fixes
from .minimal_fifo_fix_corrected import apply_minimal_fifo_fix
from .fixed_validation_and_metrics import validate_schedule_correctly, calculate_resource_utilization
from .genetic_task_optimizer import GeneticTaskOptimizer, GeneticIndividual

# 导入改进的优化器
from .improved_genetic_optimizer import ImprovedGeneticOptimizer

# 导入新的激进优化器（如果文件存在）
try:
    from .aggressive_idle_optimizer import AggressiveIdleOptimizer
except ImportError:
    AggressiveIdleOptimizer = None  # 文件还未创建时的fallback

__all__ = [
    # Enums
    'ResourceType', 'TaskPriority', 'RuntimeType', 'SegmentationStrategy', 'CutPointStatus',
    # Models
    'ResourceUnit', 'TaskScheduleInfo', 'ResourceBinding', 
    'SegmentationDecision', 'SubSegment', 'ResourceSegment', 'CutPoint',
    # Core classes
    'NNTask', 'MultiResourceScheduler', 'ResourcePriorityQueues',
    # Fixes
    'apply_basic_fixes', 'apply_performance_fixes', 'apply_minimal_fifo_fix',
    # Utilities
    'validate_schedule_correctly', 'calculate_resource_utilization',
    # Optimizers
    'GeneticTaskOptimizer', 'GeneticIndividual',
    'ImprovedGeneticOptimizer',
    'AggressiveIdleOptimizer'  # 新增
]