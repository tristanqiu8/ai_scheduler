#!/usr/bin/env python3
"""
修复后的优雅可视化模块 - 改进任务标签显示
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from core.enums import ResourceType, TaskPriority, RuntimeType
from core.scheduler import MultiResourceScheduler


class ElegantSchedulerVisualizer:
    """优雅的可视化器，修复了标签显示问题"""
    
    def __init__(self, scheduler: MultiResourceScheduler):
        self.scheduler = scheduler
        
        # Dragon4 priority color palette (红、橙、绿、蓝)
        self.priority_colors = {
            TaskPriority.CRITICAL: '#DC2626',    # 红色 - CRITICAL
            TaskPriority.HIGH: '#EA580C',        # 橙色 - HIGH  
            TaskPriority.NORMAL: '#16A34A',      # 绿色 - NORMAL
            TaskPriority.LOW: '#2563EB'          # 蓝色 - LOW
        }
        
        # Dragon4 alternative palette (更鲜艳的红、橙、绿、蓝)
        self.alt_priority_colors = {
            TaskPriority.CRITICAL: '#EF4444',    # 鲜红色 - CRITICAL
            TaskPriority.HIGH: '#F97316',        # 鲜橙色 - HIGH
            TaskPriority.NORMAL: '#22C55E',      # 鲜绿色 - NORMAL
            TaskPriority.LOW: '#3B82F6'          # 鲜蓝色 - LOW
        }
        
        # 用于跟踪每个资源行上的标签位置，避免重叠
        self.label_positions = {}
    
    def _get_dragon4_task_name(self, task, segment_index=None, total_segments=None):
        """获取Dragon4格式的任务名称"""
        # 基础名称
        base_name = task.task_id
        
        # DSP Runtime前缀
        if hasattr(task, 'runtime_type') and task.runtime_type == RuntimeType.DSP_RUNTIME:
            base_name = f"X: {base_name}"
        
        # 分段后缀
        if segment_index is not None and total_segments is not None and total_segments > 1:
            base_name = f"{base_name}_{segment_index + 1}"
        
        return base_name
    
    def _check_label_overlap(self, resource_id: str, start_x: float, end_x: float, 
                           label_text: str) -> bool:
        """
        检查标签是否会与现有标签重叠
        """
        if resource_id not in self.label_positions:
            self.label_positions[resource_id] = []
        
        # 检查是否与现有标签重叠
        for existing_start, existing_end, _ in self.label_positions[resource_id]:
            # 如果有任何重叠，返回True
            if not (end_x < existing_start or start_x > existing_end):
                return True
        
        return False
    
    def _add_label_position(self, resource_id: str, start_x: float, end_x: float, label: str):
        """
        记录标签位置
        """
        if resource_id not in self.label_positions:
            self.label_positions[resource_id] = []
        self.label_positions[resource_id].append((start_x, end_x, label))
    
    def plot_elegant_gantt(self, 
                          time_window: float = None, 
                          bar_height: float = 0.35,
                          spacing: float = 0.8,
                          use_alt_colors: bool = False,
                          show_all_labels: bool = True):  # 新增参数
        """创建优雅的甘特图，标签显示在色块上方"""
        if not self.scheduler.schedule_history:
            print("No schedule history available")
            return
        
        # 重置标签位置跟踪
        self.label_positions = {}
        
        # Select color palette
        colors = self.alt_priority_colors if use_alt_colors else self.priority_colors
        
        # Determine time window
        if time_window is None:
            time_window = max(s.end_time for s in self.scheduler.schedule_history) * 1.1
        
        # Prepare resources
        npu_resources = []
        dsp_resources = []
        
        for resource in self.scheduler.resources[ResourceType.NPU]:
            npu_resources.append(resource.unit_id)
        for resource in self.scheduler.resources[ResourceType.DSP]:
            dsp_resources.append(resource.unit_id)
        
        all_resources = npu_resources + dsp_resources
        
        # Create figure with extra height for labels
        fig_height = len(all_resources) * spacing + 3  # 增加高度以容纳标签
        fig, ax = plt.subplots(figsize=(20, fig_height))
        
        # Subtle background
        fig.patch.set_facecolor('#FAFAFA')
        ax.set_facecolor('#FFFFFF')
        
        # Resource positions
        y_positions = {}
        current_y = 0
        
        # NPU group
        for i, res in enumerate(npu_resources):
            y_positions[res] = current_y
            current_y += spacing
        
        # Add subtle separator
        if npu_resources and dsp_resources:
            separator_y = current_y - spacing/2
            ax.axhline(y=separator_y, color='#E5E7EB', linewidth=1, alpha=0.5)
        
        # DSP group
        for i, res in enumerate(dsp_resources):
            y_positions[res] = current_y
            current_y += spacing
        
        # Plot tasks with elegant styling
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            if task.is_segmented and schedule.sub_segment_schedule:
                self._plot_elegant_segmented_task(ax, task, schedule, y_positions, 
                                                bar_height, colors, show_all_labels)
            else:
                self._plot_elegant_regular_task(ax, task, schedule, y_positions, 
                                              bar_height, colors, show_all_labels)
        
        # Draw minimal bindings
        self._draw_elegant_bindings(ax, y_positions, time_window, spacing)
        
        # Configure axes with minimal style - 调整Y轴范围以容纳标签
        ax.set_ylim(-spacing/2, len(all_resources) * spacing + spacing/2)  # 增加上方空间
        ax.set_xlim(-5, time_window * 1.05)
        
        # Y-axis: Minimal resource labels
        ax.set_yticks([y_positions[res] for res in all_resources])
        ax.set_yticklabels(all_resources, fontsize=10, color='#4B5563')
        
        # Subtle resource type indicators
        for i, res in enumerate(all_resources):
            if 'NPU' in res:
                ax.plot(-3, y_positions[res], marker='o', markersize=6, 
                       color='#3B82F6', alpha=0.6)
            else:
                ax.plot(-3, y_positions[res], marker='s', markersize=6, 
                       color='#8B5CF6', alpha=0.6)
        
        # X-axis with minimal styling
        ax.set_xlabel('Time (ms)', fontsize=11, color='#374151')
        ax.set_title('Task Scheduling Timeline', fontsize=14, fontweight='600', 
                    color='#111827', pad=15)
        
        # Minimal grid
        ax.grid(True, axis='x', alpha=0.15, linestyle='-', linewidth=0.5, color='#E5E7EB')
        ax.set_axisbelow(True)
        
        # Clean spines
        for spine in ax.spines.values():
            spine.set_color('#E5E7EB')
            spine.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Time markers - subtle
        for t in range(0, int(time_window), 50):
            if t > 0:  # Skip 0
                ax.text(t, -spacing/3, str(t), ha='center', va='top', 
                       fontsize=8, color='#9CA3AF')
        
        # Minimal legend
        self._create_elegant_legend(ax, colors)
        
        # Compact metrics
        self._add_elegant_metrics(ax, time_window)
        
        plt.tight_layout()
        plt.show()
    
    def _plot_elegant_segmented_task(self, ax, task, schedule, y_positions, 
                                   bar_height, colors, show_all_labels):
        """Plot segmented task with elegant minimal style and better labels"""
        base_color = colors[task.priority]
        
        # Calculate color variations for segments
        num_segments = len(schedule.sub_segment_schedule)
        
        # 收集所有段的信息以便统一处理标签
        segments_info = []
        
        for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
            # Find resource
            sub_seg = None
            for ss in task.get_sub_segments_for_scheduling():
                if ss.sub_id == sub_seg_id:
                    sub_seg = ss
                    break
            
            if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[sub_seg.resource_type]
                if resource_id in y_positions:
                    segments_info.append({
                        'index': i,
                        'sub_id': sub_seg_id,  # 保存段ID
                        'start': start_time,
                        'end': end_time,
                        'resource_id': resource_id,
                        'y_pos': y_positions[resource_id]
                    })
        
        # 绘制所有段
        for seg_info in segments_info:
            i = seg_info['index']
            start_time = seg_info['start']
            end_time = seg_info['end']
            y_pos = seg_info['y_pos']
            resource_id = seg_info['resource_id']
            sub_id = seg_info['sub_id']
            duration = end_time - start_time
            
            # 为不同的段使用略微不同的颜色
            # 使用HSL调整亮度
            import matplotlib.colors as mcolors
            import colorsys
            
            # 转换为HSL
            rgb = mcolors.hex2color(base_color)
            h, l, s = colorsys.rgb_to_hls(*rgb)
            
            # 根据段索引调整亮度
            l_adjusted = l * (1 - 0.1 * (i % 3))  # 轻微变化
            rgb_adjusted = colorsys.hls_to_rgb(h, l_adjusted, s)
            segment_color = mcolors.rgb2hex(rgb_adjusted)
            
            # Main rectangle with rounded corners
            rect = patches.FancyBboxPatch(
                (start_time, y_pos - bar_height/2), 
                duration, 
                bar_height,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor=segment_color,
                edgecolor='white',  # 白色边框
                alpha=0.9,
                linewidth=1.5
            )
            ax.add_patch(rect)
            
            # 添加段内标签（如果空间足够）
            if duration > 2.0:  # 只在段足够宽时显示
                # 提取段编号
                seg_num = sub_id.split('_')[-1] if '_' in sub_id else str(i)
                
                ax.text(start_time + duration/2, y_pos, 
                    f"S{seg_num}",
                    ha='center', va='center', 
                    fontsize=7,
                    color='white',
                    fontweight='bold',
                    alpha=0.8)
            
            # 添加垂直分割线（除了最后一段）
            if i < num_segments - 1:
                ax.plot([end_time, end_time], 
                    [y_pos - bar_height/2 + 1, y_pos + bar_height/2 - 1],
                    color='white', 
                    linewidth=2, 
                    linestyle='--',
                    alpha=0.7,
                    solid_capstyle='round')
        
        # 统一处理任务标签 - 显示在色块上方
        if show_all_labels and segments_info:
            first_seg = segments_info[0]
            last_seg = segments_info[-1]
            
            label = self._get_dragon4_task_name(task)
            
            # 添加段数信息
            if num_segments > 1:
                label += f" [{num_segments}段]"
            
            # 计算标签位置（整个任务的中心，但在色块上方）
            label_x = (first_seg['start'] + last_seg['end']) / 2
            label_y = first_seg['y_pos'] + bar_height/2 + 0.15  # 色块上方
            
            # 检查标签是否会重叠
            label_width = len(label) * 5  # 估算标签宽度
            label_start = label_x - label_width / 2
            label_end = label_x + label_width / 2
            
            # 如果会重叠，调整位置
            if self._check_label_overlap(first_seg['resource_id'], label_start, label_end, label):
                # 尝试稍微提高标签位置
                label_y += 0.2
            
            # 记录标签位置
            self._add_label_position(first_seg['resource_id'], label_start, label_end, label)
            
            # 绘制标签
            ax.text(label_x, label_y, label,
                ha='center', va='bottom', fontsize=8,
                color='#374151', fontweight='600',
                bbox=dict(boxstyle='round,pad=0.15', 
                        facecolor='white', 
                        edgecolor=base_color,  # 使用任务颜色作为边框
                        linewidth=1,
                        alpha=0.9))
    
    def _plot_elegant_regular_task(self, ax, task, schedule, y_positions, 
                                 bar_height, colors, show_all_labels):
        """Plot regular task with elegant style and better labels"""
        base_color = colors[task.priority]
        
        # 收集任务段信息
        task_segments = []
        
        for seg in task.segments:
            if seg.resource_type in schedule.assigned_resources:
                resource_id = schedule.assigned_resources[seg.resource_type]
                if resource_id in y_positions:
                    y_pos = y_positions[resource_id]
                    
                    # Calculate duration
                    resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                        if r.unit_id == resource_id), None)
                    if resource_unit:
                        duration = seg.get_duration(resource_unit.bandwidth)
                        start_time = schedule.start_time + seg.start_time
                        
                        task_segments.append({
                            'start': start_time,
                            'end': start_time + duration,
                            'y_pos': y_pos,
                            'resource_id': resource_id
                        })
                        
                        # Rounded rectangle
                        rect = patches.FancyBboxPatch(
                            (start_time, y_pos - bar_height/2), 
                            duration, 
                            bar_height,
                            boxstyle="round,pad=0.01,rounding_size=0.02",
                            facecolor=base_color,
                            edgecolor='none',
                            alpha=0.9,
                            linewidth=0
                        )
                        ax.add_patch(rect)
        
        # 显示标签 - 在色块上方
        if show_all_labels and task_segments:
            first_seg = task_segments[0]
            last_seg = task_segments[-1] if len(task_segments) > 1 else first_seg
            
            label = self._get_dragon4_task_name(task)
            
            # 计算标签位置（整个任务的中心，但在色块上方）
            label_x = (first_seg['start'] + last_seg['end']) / 2
            label_y = first_seg['y_pos'] + bar_height/2 + 0.15  # 色块上方
            
            # 检查标签是否会重叠
            label_width = len(label) * 5  # 估算标签宽度
            label_start = label_x - label_width / 2
            label_end = label_x + label_width / 2
            
            # 如果会重叠，调整位置
            if self._check_label_overlap(first_seg['resource_id'], label_start, label_end, label):
                # 尝试稍微提高标签位置
                label_y += 0.2
            
            # 记录标签位置
            self._add_label_position(first_seg['resource_id'], label_start, label_end, label)
            
            # 绘制标签
            ax.text(label_x, label_y, label,
                   ha='center', va='bottom', fontsize=8,
                   color='#374151', fontweight='600',
                   bbox=dict(boxstyle='round,pad=0.15', 
                           facecolor='white', 
                           edgecolor='none',
                           alpha=0.85))
    
    def _draw_elegant_bindings(self, ax, y_positions, time_window, spacing):
        """Draw elegant binding indicators (if any)"""
        # Keep minimal - bindings can clutter the chart
        pass
    
    def _create_elegant_legend(self, ax, colors):
        """Create minimal elegant legend with segmentation info"""
        # Priority legend
        legend_elements = []
        for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                        TaskPriority.NORMAL, TaskPriority.LOW]:
            color = colors[priority]
            elem = patches.Rectangle((0, 0), 1, 1, 
                facecolor=color,
                edgecolor='none',
                alpha=0.85,
                label=priority.name
            )
            legend_elements.append(elem)
        
        # 添加分段说明
        if any(task.is_segmented for task in self.scheduler.tasks.values()):
            # 添加分段示例
            segmented_elem = patches.Rectangle((0, 0), 1, 1,
                facecolor='gray',
                edgecolor='white',
                linewidth=1.5,
                alpha=0.7,
                label='分段任务'
            )
            legend_elements.append(segmented_elem)
        
        # Create compact legend
        legend = ax.legend(handles=legend_elements, 
                        loc='upper right',
                        frameon=True,
                        fancybox=False,
                        shadow=False,
                        ncol=5,  # 增加列数
                        fontsize=9,
                        handlelength=1.5,
                        handletextpad=0.5,
                        columnspacing=1.0)
        
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_edgecolor('#E5E7EB')
        legend.get_frame().set_linewidth(0.5)
        legend.get_frame().set_alpha(0.95)
    
    def _add_elegant_metrics(self, ax, time_window):
        """Add compact metrics display"""
        utilization = self.scheduler.get_resource_utilization(time_window)
        avg_util = sum(utilization.values()) / len(utilization) if utilization else 0
        
        # Compact metrics line
        metrics = (f"Utilization: {avg_util:.0f}% | "
                  f"Segmented: {self.scheduler.segmentation_stats['segmented_tasks']} | "
                  f"Overhead: {self.scheduler.segmentation_stats['total_overhead']:.0f}ms | "
                  f"Complete: {self.scheduler.schedule_history[-1].end_time:.0f}ms")
        
        # Single line metrics
        ax.text(0.02, 0.02, metrics, 
               transform=ax.transAxes, 
               fontsize=9, 
               color='#6B7280',
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white', 
                        edgecolor='#E5E7EB',
                        linewidth=0.5,
                        alpha=0.95))
    
    def export_chrome_tracing(self, filename: str = "elegant_trace.json"):
        """Export to Chrome Tracing format"""
        if not self.scheduler.schedule_history:
            print("No schedule history to export")
            return
        
        events = []
        pid_counter = 1
        resource_mapping = {}
        
        # Create process structure
        for res_type in [ResourceType.NPU, ResourceType.DSP]:
            pid = pid_counter
            pid_counter += 1
            
            events.append({
                "name": "process_name",
                "ph": "M",
                "pid": pid,
                "tid": 0,
                "args": {"name": res_type.name}
            })
            
            for resource in self.scheduler.resources[res_type]:
                tid = int(resource.unit_id.split('_')[1]) + 1
                resource_mapping[resource.unit_id] = (pid, tid)
                
                events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": resource.unit_id}
                })
        
        # Add tasks
        for schedule in self.scheduler.schedule_history:
            task = self.scheduler.tasks[schedule.task_id]
            
            if task.is_segmented and schedule.sub_segment_schedule:
                # Segmented task
                for i, (sub_seg_id, start_time, end_time) in enumerate(schedule.sub_segment_schedule):
                    sub_seg = None
                    for ss in task.get_sub_segments_for_scheduling():
                        if ss.sub_id == sub_seg_id:
                            sub_seg = ss
                            break
                    
                    if sub_seg and sub_seg.resource_type in schedule.assigned_resources:
                        resource_id = schedule.assigned_resources[sub_seg.resource_type]
                        if resource_id in resource_mapping:
                            pid, tid = resource_mapping[resource_id]
                            
                            # Task name with segment info
                            name = self._get_dragon4_task_name(task, i, len(schedule.sub_segment_schedule))
                            
                            events.append({
                                "name": name,
                                "cat": task.priority.name,
                                "ph": "X",
                                "ts": int(start_time * 1000),
                                "dur": int((end_time - start_time) * 1000),
                                "pid": pid,
                                "tid": tid,
                                "args": {
                                    "priority": task.priority.name,
                                    "segment": f"{i+1}/{len(schedule.sub_segment_schedule)}",
                                    "overhead_ms": f"{schedule.segmentation_overhead:.2f}"
                                }
                            })
            else:
                # Regular task
                for seg in task.segments:
                    if seg.resource_type in schedule.assigned_resources:
                        resource_id = schedule.assigned_resources[seg.resource_type]
                        if resource_id in resource_mapping:
                            pid, tid = resource_mapping[resource_id]
                            
                            resource_unit = next((r for r in self.scheduler.resources[seg.resource_type] 
                                                if r.unit_id == resource_id), None)
                            if resource_unit:
                                duration = seg.get_duration(resource_unit.bandwidth)
                                start_time = schedule.start_time + seg.start_time
                                
                                name = self._get_dragon4_task_name(task)
                                
                                events.append({
                                    "name": name,
                                    "cat": task.priority.name,
                                    "ph": "X",
                                    "ts": int(start_time * 1000),
                                    "dur": int(duration * 1000),
                                    "pid": pid,
                                    "tid": tid,
                                    "args": {
                                        "priority": task.priority.name,
                                        "fps": task.fps_requirement,
                                        "latency_req": task.latency_requirement
                                    }
                                })
        
        # Write to file
        with open(filename, 'w') as f:
            json.dump({"traceEvents": events}, f, indent=2)
        
        print(f"Chrome tracing data exported to {filename}")
        print(f"Open chrome://tracing and load this file to visualize")
