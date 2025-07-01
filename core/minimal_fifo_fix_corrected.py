#!/usr/bin/env python3
"""
最小化FIFO修复（修正版）
通过最少的代码改动解决同优先级任务的执行顺序问题
修复返回值类型问题
"""

from typing import Dict, List
from collections import defaultdict


def apply_minimal_fifo_fix(scheduler):
    """应用最小化的FIFO修复"""
    
    print("🔧 应用最小化FIFO修复...")
    
    # 1. 为任务添加执行顺序
    setup_task_order(scheduler)
    
    # 2. 修改调度逻辑
    patch_scheduling_logic(scheduler)
    
    print("✅ 最小化FIFO修复已应用")


def setup_task_order(scheduler):
    """设置任务执行顺序"""
    
    if not scheduler.tasks:
        print("  ⚠️ 没有任务，跳过顺序设置")
        return
    
    # 基于任务ID创建稳定的执行顺序
    for task_id, task in scheduler.tasks.items():
        # 提取任务编号
        try:
            if task_id.startswith('T'):
                order = int(task_id[1:])
            else:
                order = 1000 + hash(task_id) % 1000
        except:
            order = 1000 + hash(task_id) % 1000
        
        task._fifo_order = order
    
    print(f"  ✓ 为 {len(scheduler.tasks)} 个任务设置了执行顺序")


def patch_scheduling_logic(scheduler):
    """修改调度逻辑以支持FIFO"""
    
    # 保存原始的调度方法
    original_schedule = scheduler.priority_aware_schedule_with_segmentation
    
    def fifo_enhanced_schedule(time_window):
        """FIFO增强的调度方法"""
        
        # 修改任务排序逻辑
        original_find_resources = scheduler.find_available_resources_for_task_with_segmentation
        
        # 在调度过程中记录同优先级任务的执行顺序
        priority_last_scheduled = defaultdict(lambda: -1)  # {priority: last_scheduled_order}
        
        def should_defer_task(task, current_time):
            """判断任务是否应该推迟（让位给更早的同优先级任务）"""
            
            # 获取任务的FIFO顺序
            task_order = getattr(task, '_fifo_order', 999)
            
            # 检查是否有更早的同优先级任务等待
            for other_task in scheduler.tasks.values():
                if other_task.task_id == task.task_id:
                    continue
                    
                # 只考虑同优先级的任务
                if other_task.priority != task.priority:
                    continue
                
                other_order = getattr(other_task, '_fifo_order', 999)
                
                # 如果有更早的任务还未执行或未充分执行
                if other_order < task_order:
                    # 检查其他任务是否准备就绪
                    if (other_task.last_execution_time + other_task.min_interval_ms <= current_time):
                        # 检查执行次数差异
                        task_count = len([e for e in scheduler.schedule_history if e.task_id == task.task_id])
                        other_count = len([e for e in scheduler.schedule_history if e.task_id == other_task.task_id])
                        
                        # 如果其他任务执行次数更少，让它先执行
                        if other_count < task_count:
                            return True
            
            return False
        
        # 临时保存原始方法
        original_find = scheduler.find_available_resources_for_task_with_segmentation
        
        def fifo_aware_find_resources(task, current_time):
            """FIFO感知的资源查找"""
            
            # 如果应该推迟这个任务，返回None（不是空字典元组！）
            if should_defer_task(task, current_time):
                return None
            
            # 否则使用原始方法
            return original_find(task, current_time)
        
        # 临时替换方法
        scheduler.find_available_resources_for_task_with_segmentation = fifo_aware_find_resources
        
        # 执行原始调度
        try:
            results = original_schedule(time_window)
        finally:
            # 恢复原始方法
            scheduler.find_available_resources_for_task_with_segmentation = original_find
        
        return results
    
    # 替换调度方法
    scheduler.priority_aware_schedule_with_segmentation = fifo_enhanced_schedule


def analyze_task_execution_order(scheduler, results):
    """分析任务执行顺序（用于调试）"""
    
    print("\n📊 任务执行顺序分析:")
    
    # 统计每个任务的首次执行时间
    first_execution = {}
    execution_counts = defaultdict(int)
    
    for event in results:
        task_id = event.task_id
        execution_counts[task_id] += 1
        
        if task_id not in first_execution:
            first_execution[task_id] = event.start_time
    
    # 按优先级分组
    priority_groups = defaultdict(list)
    for task_id, task in scheduler.tasks.items():
        priority_groups[task.priority].append({
            'id': task_id,
            'order': getattr(task, '_fifo_order', 999),
            'first_exec': first_execution.get(task_id, float('inf')),
            'count': execution_counts[task_id]
        })
    
    # 显示结果
    for priority in sorted(priority_groups.keys(), key=lambda p: p.value):
        tasks = sorted(priority_groups[priority], key=lambda t: t['order'])
        
        print(f"\n{priority.name}优先级任务:")
        print(f"{'任务':<6} {'顺序':<6} {'首次执行':<12} {'执行次数':<10}")
        print("-" * 40)
        
        for task_info in tasks:
            if task_info['first_exec'] < float('inf'):
                print(f"{task_info['id']:<6} {task_info['order']:<6} "
                      f"{task_info['first_exec']:<12.1f} {task_info['count']:<10}")
            else:
                print(f"{task_info['id']:<6} {task_info['order']:<6} "
                      f"{'未执行':<12} {task_info['count']:<10}")


if __name__ == "__main__":
    print("最小化FIFO修复")
    print("通过最少的改动确保同优先级任务按顺序执行")
