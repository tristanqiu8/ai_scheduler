#!/usr/bin/env python3
"""
调试版紧凑化算法 - 找出问题所在
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
import copy
from collections import defaultdict

from scheduler import MultiResourceScheduler
from .enums import ResourceType


class DebugCompactor:
    """调试版紧凑化器"""
    
    def __init__(self, scheduler: MultiResourceScheduler, time_window: float = 200.0):
        self.scheduler = scheduler
        self.time_window = time_window
        
    def analyze_schedule(self):
        """分析当前调度"""
        print("\n📊 调度分析:")
        print("=" * 60)
        
        events = self.scheduler.schedule_history
        first_window = [e for e in events if e.start_time < self.time_window]
        
        # 1. 统计任务
        task_counts = defaultdict(int)
        for event in first_window:
            task_counts[event.task_id] += 1
        
        print(f"\n任务执行情况:")
        for task_id in sorted(task_counts.keys()):
            print(f"  {task_id}: {task_counts[task_id]} 次")
        
        # 2. 查找空隙
        gaps = self._find_gaps(first_window)
        print(f"\n发现 {len(gaps)} 个空隙:")
        for gap in gaps:
            print(f"  {gap['start']:.1f}-{gap['end']:.1f}ms (持续 {gap['duration']:.1f}ms)")
        
        # 3. 分析依赖
        print(f"\n任务依赖关系:")
        for task_id, task in self.scheduler.tasks.items():
            if task.dependencies:
                print(f"  {task_id} 依赖: {list(task.dependencies)}")
        
        return gaps
    
    def _find_gaps(self, events):
        """查找所有资源都空闲的时间段"""
        if not events:
            return [{'start': 0, 'end': self.time_window, 'duration': self.time_window}]
        
        # 获取所有资源ID
        all_resources = set()
        for event in events:
            for res_type, res_id in event.assigned_resources.items():
                all_resources.add(res_id)
        
        print(f"\n资源列表: {sorted(all_resources)}")
        
        # 创建时间线
        timeline = []
        for event in events:
            timeline.append(('start', event.start_time, event))
            timeline.append(('end', event.end_time, event))
        
        timeline.sort(key=lambda x: x[1])
        
        # 扫描时间线找空隙
        active_resources = set()
        gaps = []
        last_time = 0
        
        for event_type, time, event in timeline:
            # 检查是否有空隙
            if last_time < time and len(active_resources) == 0:
                gaps.append({
                    'start': last_time,
                    'end': time,
                    'duration': time - last_time
                })
            
            if event_type == 'start':
                for res_type, res_id in event.assigned_resources.items():
                    active_resources.add(res_id)
            else:  # end
                for res_type, res_id in event.assigned_resources.items():
                    active_resources.discard(res_id)
            
            last_time = time
        
        # 检查末尾
        if last_time < self.time_window and len(active_resources) == 0:
            gaps.append({
                'start': last_time,
                'end': self.time_window,
                'duration': self.time_window - last_time
            })
        
        return gaps
    
    def simple_compact(self):
        """简单的贪心紧凑化算法"""
        print("\n🔧 执行简单贪心紧凑化...")
        
        events = copy.deepcopy(self.scheduler.schedule_history)
        first_window = [e for e in events if e.start_time < self.time_window]
        
        if not first_window:
            return events
        
        # 按开始时间排序
        first_window.sort(key=lambda e: (e.start_time, e.task_id))
        
        # 初始化资源时间线
        resource_timeline = defaultdict(float)  # 资源ID -> 最早可用时间
        
        # 贪心调度
        compacted = []
        
        print(f"\n开始紧凑化 {len(first_window)} 个事件...")
        
        for i, event in enumerate(first_window):
            # 找到所有需要资源的最早可用时间
            earliest_start = 0
            needed_resources = []
            
            for res_type, res_id in event.assigned_resources.items():
                earliest_start = max(earliest_start, resource_timeline[res_id])
                needed_resources.append(res_id)
            
            # 检查依赖（简化：假设同任务的事件必须保持顺序）
            for prev_event in compacted:
                if prev_event.task_id == event.task_id:
                    earliest_start = max(earliest_start, prev_event.end_time)
            
            # 计算新的时间
            duration = event.end_time - event.start_time
            new_start = earliest_start
            new_end = new_start + duration
            
            # 确保不超出窗口
            if new_start >= self.time_window:
                print(f"  事件 {i}: {event.task_id} 无法放入窗口，保持原位")
                compacted.append(event)
            else:
                # 创建新事件
                new_event = copy.deepcopy(event)
                new_event.start_time = new_start
                new_event.end_time = new_end
                compacted.append(new_event)
                
                # 更新资源时间线
                for res_id in needed_resources:
                    resource_timeline[res_id] = new_end
                
                if abs(new_start - event.start_time) > 0.1:
                    print(f"  事件 {i}: {event.task_id} 从 {event.start_time:.1f}ms 移动到 {new_start:.1f}ms")
        
        # 合并其他窗口的事件
        other_events = [e for e in events if e.start_time >= self.time_window]
        compacted.extend(other_events)
        
        # 计算末尾空闲
        if compacted:
            first_window_compacted = [e for e in compacted if e.start_time < self.time_window]
            if first_window_compacted:
                last_end = max(e.end_time for e in first_window_compacted)
                idle_time = self.time_window - last_end
            else:
                idle_time = self.time_window
        else:
            idle_time = self.time_window
        
        print(f"\n✅ 紧凑化完成，末尾空闲: {idle_time:.1f}ms")
        
        return compacted, idle_time


def test_debug_compactor(scheduler):
    """测试调试版紧凑化器"""
    from viz.elegant_visualization import ElegantSchedulerVisualizer
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    # 创建调试器
    debugger = DebugCompactor(scheduler)
    
    # 分析原始调度
    print("\n" + "="*80)
    print("原始调度分析")
    print("="*80)
    original_gaps = debugger.analyze_schedule()
    
    # 保存原始调度
    original_events = copy.deepcopy(scheduler.schedule_history)
    
    # 执行紧凑化
    compacted_events, idle_time = debugger.simple_compact()
    
    # 更新调度
    scheduler.schedule_history = compacted_events
    
    # 分析紧凑化后的调度
    print("\n" + "="*80)
    print("紧凑化后分析")
    print("="*80)
    compacted_gaps = debugger.analyze_schedule()
    
    # 生成可视化
    print("\n生成可视化...")
    
    # 原始调度图
    first_window_orig = [e for e in original_events if e.start_time < 200.0]
    scheduler.schedule_history = first_window_orig
    viz1 = ElegantSchedulerVisualizer(scheduler)
    viz1.plot_elegant_gantt()
    plt.title('Debug: Original Schedule')
    plt.xlim(0, 200)
    plt.savefig('debug_original.png', dpi=120)
    plt.close()
    
    # 紧凑化调度图
    first_window_comp = [e for e in compacted_events if e.start_time < 200.0]
    scheduler.schedule_history = first_window_comp
    viz2 = ElegantSchedulerVisualizer(scheduler)
    viz2.plot_elegant_gantt()
    
    # 标记空闲
    if idle_time > 0:
        idle_start = 200 - idle_time
        plt.axvspan(idle_start, 200, alpha=0.3, color='lightgreen')
        plt.text(idle_start + idle_time/2, plt.ylim()[1]*0.95,
                f'{idle_time:.1f}ms\nIDLE', 
                ha='center', va='top', fontsize=14, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8))
    
    plt.title(f'Debug: Compacted Schedule ({idle_time:.1f}ms idle)')
    plt.xlim(0, 200)
    plt.savefig('debug_compacted.png', dpi=120)
    plt.close()
    
    # 恢复完整调度并生成trace
    scheduler.schedule_history = compacted_events
    viz3 = ElegantSchedulerVisualizer(scheduler)
    viz3.export_chrome_tracing('debug_compacted_trace.json')
    
    print("\n✅ 调试文件已生成:")
    print("  - debug_original.png")
    print("  - debug_compacted.png")
    print("  - debug_compacted_trace.json")
    
    return compacted_events, idle_time


if __name__ == "__main__":
    print("调试版紧凑化器")
    # 需要实际的scheduler实例来测试
