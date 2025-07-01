#!/usr/bin/env python3
"""
修复的验证器和度量计算
解决错误的冲突检测和利用率计算问题
"""

from collections import defaultdict
from typing import List, Dict, Tuple
from decimal import Decimal, ROUND_HALF_UP


def validate_schedule_correctly(scheduler) -> Tuple[bool, List[str]]:
    """
    正确的调度验证，修复浮点精度问题
    """
    errors = []
    
    # 为每个资源构建时间线
    resource_timeline = defaultdict(list)
    
    for schedule in scheduler.schedule_history:
        task = scheduler.tasks[schedule.task_id]
        
        if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
            # 处理子段调度
            for sub_seg_id, start_time, end_time in schedule.sub_segment_schedule:
                # 找到对应的资源
                sub_seg = None
                for ss in task.get_sub_segments_for_scheduling():
                    if ss.sub_id == sub_seg_id:
                        sub_seg = ss
                        break
                
                if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                    resource_id = schedule.assigned_resources[sub_seg.resource_type]
                    # 使用高精度Decimal避免浮点误差
                    resource_timeline[resource_id].append((
                        Decimal(str(start_time)).quantize(Decimal('0.001')),
                        Decimal(str(end_time)).quantize(Decimal('0.001')),
                        task.task_id,
                        sub_seg_id
                    ))
    
    # 检查每个资源上的冲突
    for resource_id, timeline in resource_timeline.items():
        # 按开始时间排序
        timeline.sort(key=lambda x: x[0])
        
        # 检查重叠（使用Decimal精确比较）
        for i in range(len(timeline) - 1):
            curr_start, curr_end, curr_task, curr_seg = timeline[i]
            next_start, next_end, next_task, next_seg = timeline[i + 1]
            
            # 只有当当前结束时间严格大于下一个开始时间时才是冲突
            # 考虑到精度问题，允许0.001ms的容差
            if curr_end > next_start + Decimal('0.001'):
                overlap = float(curr_end - next_start)
                errors.append(
                    f"资源冲突: {resource_id} 上 {curr_seg} ({float(curr_start):.3f}-{float(curr_end):.3f}ms) "
                    f"与 {next_seg} ({float(next_start):.3f}-{float(next_end):.3f}ms) 重叠 {overlap:.3f}ms"
                )
    
    return len(errors) == 0, errors


def calculate_resource_utilization(scheduler, time_window: float) -> Dict[str, float]:
    """
    正确计算资源利用率
    """
    resource_busy_time = defaultdict(float)
    
    # 计算每个资源的忙碌时间
    for schedule in scheduler.schedule_history:
        if hasattr(schedule, 'sub_segment_schedule') and schedule.sub_segment_schedule:
            # 处理子段调度
            for sub_seg_id, start_time, end_time in schedule.sub_segment_schedule:
                # 找到对应的资源
                for task in scheduler.tasks.values():
                    if task.task_id == schedule.task_id:
                        for ss in task.get_sub_segments_for_scheduling():
                            if ss.sub_id == sub_seg_id:
                                if ss.resource_type in schedule.assigned_resources:
                                    resource_id = schedule.assigned_resources[ss.resource_type]
                                    duration = end_time - start_time
                                    resource_busy_time[resource_id] += duration
                                break
                        break
    
    # 计算利用率
    utilization = {}
    for resource_id, busy_time in resource_busy_time.items():
        utilization[resource_id] = (busy_time / time_window) * 100
    
    return utilization


def print_schedule_analysis(scheduler, time_window: float = 200.0):
    """
    打印调度分析报告
    """
    print("\n" + "=" * 60)
    print("调度分析报告")
    print("=" * 60)
    
    # 1. 验证结果
    is_valid, errors = validate_schedule_correctly(scheduler)
    
    if is_valid:
        print("\n✅ 资源冲突检查: 通过")
    else:
        print(f"\n❌ 资源冲突检查: 失败 ({len(errors)} 个冲突)")
        for error in errors[:3]:
            print(f"  - {error}")
    
    # 2. 资源利用率
    utilization = calculate_resource_utilization(scheduler, time_window)
    
    print("\n📊 资源利用率:")
    total_util = 0.0
    count = 0
    
    for resource_id in sorted(utilization.keys()):
        util = utilization[resource_id]
        print(f"  {resource_id}: {util:5.1f}%")
        total_util += util
        count += 1
    
    if count > 0:
        avg_util = total_util / count
        print(f"\n  平均利用率: {avg_util:5.1f}%")
    
    # 3. 任务执行统计
    task_stats = defaultdict(lambda: {'count': 0, 'total_time': 0.0})
    
    for schedule in scheduler.schedule_history:
        task_id = schedule.task_id
        task_stats[task_id]['count'] += 1
        task_stats[task_id]['total_time'] += (schedule.end_time - schedule.start_time)
    
    print("\n📋 任务执行统计:")
    for task_id in sorted(task_stats.keys()):
        stats = task_stats[task_id]
        task = scheduler.tasks[task_id]
        print(f"  {task_id} ({task.priority.name}): "
              f"{stats['count']} 次执行, "
              f"总时间 {stats['total_time']:.1f}ms")
    
    # 4. 时间线分析
    print("\n⏱️ 时间线分析:")
    
    # 找出最早和最晚的时间
    min_time = float('inf')
    max_time = 0.0
    
    for schedule in scheduler.schedule_history:
        min_time = min(min_time, schedule.start_time)
        max_time = max(max_time, schedule.end_time)
    
    actual_time_span = max_time - min_time
    print(f"  实际时间跨度: {actual_time_span:.1f}ms")
    print(f"  调度时间窗口: {time_window:.1f}ms")
    print(f"  时间利用率: {(actual_time_span / time_window * 100):.1f}%")


def fix_schedule_validator(scheduler):
    """
    替换错误的验证器
    """
    # 添加正确的验证方法
    scheduler.validate_schedule = lambda: validate_schedule_correctly(scheduler)[0]
    
    # 添加分析方法
    scheduler.analyze_schedule = lambda: print_schedule_analysis(scheduler)
    
    print("✅ 调度验证器已修复")


if __name__ == "__main__":
    print("修复的验证器和度量计算")
    print("主要修复:")
    print("1. 使用Decimal避免浮点精度问题")
    print("2. 正确计算资源利用率")
    print("3. 提供详细的调度分析")
