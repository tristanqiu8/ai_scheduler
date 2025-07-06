#!/usr/bin/env python3
"""
模块化的调度器修复集合
将各种修复独立成可选的模块，便于组合使用
"""

from typing import Dict, List, Optional, Callable
from collections import defaultdict


class SchedulerFixModule:
    """调度器修复模块基类"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.enabled = True
        
    def apply(self, scheduler) -> bool:
        """应用修复，返回是否成功"""
        raise NotImplementedError
        
    def remove(self, scheduler) -> bool:
        """移除修复，返回是否成功"""
        raise NotImplementedError


class FIFOOrderFix(SchedulerFixModule):
    """FIFO执行顺序修复"""
    
    def __init__(self):
        super().__init__(
            "FIFO Order Fix",
            "确保同优先级任务按FIFO顺序执行"
        )
        self.original_methods = {}
        
    def apply(self, scheduler) -> bool:
        """应用FIFO修复"""
        print(f"🔧 应用 {self.name}...")
        
        # 为任务设置FIFO顺序
        fifo_order = 0
        for task_id in sorted(scheduler.tasks.keys()):
            task = scheduler.tasks[task_id]
            task._fifo_order = fifo_order
            fifo_order += 1
            
        print(f"  ✓ 为 {len(scheduler.tasks)} 个任务设置了执行顺序")
        return True
        
    def remove(self, scheduler) -> bool:
        """移除FIFO修复"""
        for task in scheduler.tasks.values():
            if hasattr(task, '_fifo_order'):
                delattr(task, '_fifo_order')
        return True


class ResourceConflictPrevention(SchedulerFixModule):
    """资源冲突预防"""
    
    def __init__(self):
        super().__init__(
            "Resource Conflict Prevention",
            "防止多个任务同时使用同一资源"
        )
        
    def apply(self, scheduler) -> bool:
        """应用资源冲突预防"""
        print(f"🔧 应用 {self.name}...")
        
        # 添加资源占用跟踪
        if not hasattr(scheduler, '_resource_occupancy'):
            from collections import defaultdict
            scheduler._resource_occupancy = defaultdict(list)
            
        # 保存原始的资源查找方法
        if hasattr(scheduler, 'find_available_resources_for_task'):
            self.original_find_resources = scheduler.find_available_resources_for_task
            
            def safe_find_resources(task, current_time):
                # 调用原始方法
                result = self.original_find_resources(task, current_time)
                
                if result:
                    # 检查资源是否真的可用
                    for res_type, res_id in result.items():
                        # 简单检查：该资源是否在使用中
                        if res_id in scheduler._resource_occupancy:
                            for start, end in scheduler._resource_occupancy[res_id]:
                                if start <= current_time < end:
                                    return None  # 资源忙，不能分配
                
                return result
            
            scheduler.find_available_resources_for_task = safe_find_resources
            
        print("  ✓ 资源冲突预防已启用")
        return True
        
    def remove(self, scheduler) -> bool:
        """移除资源冲突预防"""
        if hasattr(self, 'original_find_resources'):
            scheduler.find_available_resources_for_task = self.original_find_resources
        if hasattr(scheduler, '_resource_occupancy'):
            delattr(scheduler, '_resource_occupancy')
        return True


class DependencyRelaxation(SchedulerFixModule):
    """依赖关系放宽（仅用于特定高FPS任务）"""
    
    def __init__(self, high_fps_threshold: float = 50.0):
        super().__init__(
            "Dependency Relaxation",
            f"放宽高FPS任务（>={high_fps_threshold}）的依赖检查"
        )
        self.high_fps_threshold = high_fps_threshold
        
    def apply(self, scheduler) -> bool:
        """应用依赖放宽"""
        print(f"🔧 应用 {self.name}...")
        
        # 标记高FPS任务
        high_fps_count = 0
        for task in scheduler.tasks.values():
            if task.fps_requirement >= self.high_fps_threshold:
                task._relaxed_dependency = True
                high_fps_count += 1
                print(f"  ✓ {task.task_id} ({task.name}) 标记为高FPS任务")
                
        print(f"  ✓ 共 {high_fps_count} 个高FPS任务将使用放宽的依赖检查")
        return True
        
    def remove(self, scheduler) -> bool:
        """移除依赖放宽"""
        for task in scheduler.tasks.values():
            if hasattr(task, '_relaxed_dependency'):
                delattr(task, '_relaxed_dependency')
        return True


class SegmentationEnhancement(SchedulerFixModule):
    """分段功能增强"""
    
    def __init__(self):
        super().__init__(
            "Segmentation Enhancement",
            "增强任务分段功能，提高调度灵活性"
        )
        
    def apply(self, scheduler) -> bool:
        """应用分段增强"""
        print(f"🔧 应用 {self.name}...")
        
        # 确保分段功能开启
        scheduler.enable_segmentation = True
        
        # 设置合理的分段开销比例
        if hasattr(scheduler, 'max_segmentation_overhead_ratio'):
            scheduler.max_segmentation_overhead_ratio = 0.2  # 20%
            
        print("  ✓ 分段功能已增强")
        return True
        
    def remove(self, scheduler) -> bool:
        """移除分段增强"""
        scheduler.enable_segmentation = False
        return True


class PriorityBoost(SchedulerFixModule):
    """动态优先级提升"""
    
    def __init__(self):
        super().__init__(
            "Priority Boost",
            "为未满足FPS的任务动态提升优先级"
        )
        self.original_priorities = {}
        
    def apply(self, scheduler) -> bool:
        """应用优先级提升"""
        print(f"🔧 应用 {self.name}...")
        
        # 保存原始优先级
        for task_id, task in scheduler.tasks.items():
            self.original_priorities[task_id] = task.priority
            
        # 这里只是准备，实际提升将在调度过程中动态进行
        scheduler._priority_boost_enabled = True
        
        print("  ✓ 动态优先级提升已准备")
        return True
        
    def remove(self, scheduler) -> bool:
        """移除优先级提升"""
        # 恢复原始优先级
        for task_id, priority in self.original_priorities.items():
            if task_id in scheduler.tasks:
                scheduler.tasks[task_id].priority = priority
                
        scheduler._priority_boost_enabled = False
        return True


class ModularSchedulerFixes:
    """模块化的调度器修复管理器"""
    
    def __init__(self):
        self.modules = {}
        self.applied_modules = []
        
        # 注册所有可用模块
        self._register_default_modules()
        
    def _register_default_modules(self):
        """注册默认修复模块"""
        self.register_module(FIFOOrderFix())
        self.register_module(ResourceConflictPrevention())
        self.register_module(DependencyRelaxation())
        self.register_module(SegmentationEnhancement())
        self.register_module(PriorityBoost())
        
    def register_module(self, module: SchedulerFixModule):
        """注册新模块"""
        self.modules[module.name] = module
        
    def apply_fixes(self, scheduler, module_names: Optional[List[str]] = None):
        """应用指定的修复模块"""
        
        print("\n🛠️  应用模块化修复")
        print("=" * 60)
        
        if module_names is None:
            # 默认应用所有模块
            module_names = list(self.modules.keys())
            
        for name in module_names:
            if name not in self.modules:
                print(f"⚠️  未知模块: {name}")
                continue
                
            module = self.modules[name]
            if module.apply(scheduler):
                self.applied_modules.append(name)
                print(f"✅ {name} 应用成功\n")
            else:
                print(f"❌ {name} 应用失败\n")
                
    def remove_fixes(self, scheduler):
        """移除所有已应用的修复"""
        
        print("\n🔄 移除已应用的修复")
        
        for name in reversed(self.applied_modules):
            module = self.modules[name]
            if module.remove(scheduler):
                print(f"  ✓ {name} 已移除")
            else:
                print(f"  ✗ {name} 移除失败")
                
        self.applied_modules.clear()
        
    def list_modules(self):
        """列出所有可用模块"""
        
        print("\n📋 可用的修复模块:")
        print("=" * 60)
        
        for name, module in self.modules.items():
            status = "✅ 已应用" if name in self.applied_modules else "⭕ 未应用"
            print(f"{status} {name}")
            print(f"   {module.description}")
            print()


def create_scheduler_with_fixes(scheduler, selected_fixes: Optional[List[str]] = None):
    """便捷函数：创建带有指定修复的调度器"""
    
    fix_manager = ModularSchedulerFixes()
    
    # 如果没有指定，使用默认的基础修复
    if selected_fixes is None:
        selected_fixes = [
            "FIFO Order Fix",
            "Resource Conflict Prevention",
            "Segmentation Enhancement"
        ]
    
    fix_manager.apply_fixes(scheduler, selected_fixes)
    
    return fix_manager


# 导出便捷函数
def apply_basic_fixes(scheduler):
    """应用基础修复（FIFO、资源冲突预防、分段增强）"""
    return create_scheduler_with_fixes(scheduler, [
        "FIFO Order Fix",
        "Resource Conflict Prevention", 
        "Segmentation Enhancement"
    ])


def apply_performance_fixes(scheduler):
    """应用性能优化修复（包括依赖放宽和优先级提升）"""
    return create_scheduler_with_fixes(scheduler, [
        "FIFO Order Fix",
        "Resource Conflict Prevention",
        "Segmentation Enhancement",
        "Dependency Relaxation",
        "Priority Boost"
    ])


if __name__ == "__main__":
    print("模块化调度器修复集合")
    print("\n特性：")
    print("1. 独立的修复模块，可自由组合")
    print("2. 支持动态添加/移除修复")
    print("3. 为机器学习优化提供灵活的基础")
    
    # 创建管理器并列出模块
    manager = ModularSchedulerFixes()
    manager.list_modules()
