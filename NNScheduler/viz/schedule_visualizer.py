#!/usr/bin/env python3
"""
调度可视化器 - 支持多种可视化格式
包括终端甘特图、Chrome Tracing、Matplotlib图表等
"""

import json
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import matplotlib
# 使用非交互后端以支持无GUI环境（如CI/服务器）
try:
    matplotlib.use('Agg')
except Exception:
    pass
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

from NNScheduler.core.schedule_tracer import ScheduleTracer, TaskExecution
from NNScheduler.core.enums import TaskPriority, ResourceType
from NNScheduler.core.artifacts import ensure_artifact_path


class ScheduleVisualizer:
    """调度可视化器"""
    
    def __init__(self, tracer: ScheduleTracer):
        self.tracer = tracer
        
    def print_gantt_chart(self, width: int = 80):
        """打印文本格式的甘特图"""
        if not self.tracer.executions:
            print("No executions to display")
            return
            
        timeline = self.tracer.get_timeline()
        start_time = self.tracer.start_time or 0
        end_time = self.tracer.end_time or 0
        total_duration = end_time - start_time
        
        if total_duration <= 0:
            print("Invalid duration")
            return
        
        # 获取所有资源（包括未使用的）
        all_resources = self._get_all_resources()
        
        print("\n" + "="*width)
        print(f"Schedule Timeline (Total: {total_duration:.1f}ms)")
        print("="*width)
        
        # 打印时间标尺
        print(f"{'Resource':<12} ", end="")
        for i in range(0, width-13, 10):
            time_point = (i / (width-13)) * total_duration
            print(f"{time_point:>9.1f}", end=" ")
        print()
        print("-"*width)
        
        # 打印每个资源的执行情况
        for resource_id in sorted(all_resources):
            print(f"{resource_id:<12} ", end="")
            
            # 创建时间线字符串
            timeline_str = [" "] * (width - 13)
            
            # 填充执行的任务
            if resource_id in timeline:
                for execution in timeline[resource_id]:
                    start_pos = int((execution.start_time - start_time) / total_duration * (width - 13))
                    end_pos = int((execution.end_time - start_time) / total_duration * (width - 13))
                    
                    # 根据优先级选择字符
                    char = self._get_priority_char(execution.priority)
                    
                    # 填充执行区间
                    for i in range(start_pos, min(end_pos + 1, width - 13)):
                        if 0 <= i < len(timeline_str):
                            timeline_str[i] = char
            
            print("".join(timeline_str))
        
        print("-"*width)
        print("Priority: C=CRITICAL, H=HIGH, N=NORMAL, L=LOW")
        
        # 打印利用率
        print("\nResource Utilization:")
        utilization = self.tracer.get_resource_utilization()
        for resource_id in sorted(all_resources):
            util = utilization.get(resource_id, 0.0)
            status = "IDLE" if util == 0 else f"{util:.1f}%"
            print(f"  {resource_id}: {status}")
    
    def export_chrome_tracing(self, filename: str):
        """导出Chrome Tracing格式的JSON文件（改进版）"""
        chrome_events = []
        
        # 获取所有资源
        all_resources = self._get_all_resources()
        
        # 为所有资源分配线程ID
        resource_tid_map = self._create_resource_tid_map(all_resources)
        
        # 添加进程和线程元数据
        # 进程名称
        chrome_events.append({
            "name": "process_name",
            "ph": "M",
            "pid": 0,
            "args": {
                "name": "AI Scheduler"
            }
        })
        
        # 为每个资源添加线程名称
        for resource_id, tid in resource_tid_map.items():
            chrome_events.append({
                "name": "thread_name",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {
                    "name": resource_id
                }
            })
            
            # 添加线程排序索引，确保显示顺序
            chrome_events.append({
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 0,
                "tid": tid,
                "args": {
                    "sort_index": tid
                }
            })
        
        # 转换执行记录为Chrome格式
        for execution in self.tracer.executions:
            tid = resource_tid_map.get(execution.resource_id, 0)
            
            # 主事件（完整持续时间）
            event = {
                "name": f"{execution.task_id}",
                "cat": execution.resource_type.value,
                "ph": "X",  # Complete event
                "ts": execution.start_time * 1000,  # 转换为微秒
                "dur": execution.duration * 1000,
                "pid": 0,
                "tid": tid,
                "args": {
                    "task_id": execution.task_id,
                    "priority": execution.priority.name,
                    "bandwidth": execution.bandwidth,
                    "segment": execution.segment_id or "main",
                    "duration_ms": execution.duration
                }
            }
            
            # 根据优先级设置颜色
            color = self._get_priority_color(execution.priority)
            if color:
                event["cname"] = color
            
            chrome_events.append(event)
            
            # 为CRITICAL任务添加标记
            if execution.priority == TaskPriority.CRITICAL:
                marker = {
                    "name": "!CRITICAL",
                    "cat": "priority",
                    "ph": "i",  # Instant event
                    "ts": execution.start_time * 1000,
                    "pid": 0,
                    "tid": tid,
                    "s": "t",  # Thread scope
                    "cname": "bad"
                }
                chrome_events.append(marker)
        
        output_path = ensure_artifact_path(filename)

        # 写入文件
        with open(output_path, 'w') as f:
            json.dump({
                "traceEvents": chrome_events,
                "displayTimeUnit": "ms",
                "systemTraceEvents": "SystemTraceData",
                "otherData": {
                    "version": "AI Scheduler v1.0"
                }
            }, f, indent=2)
        print(f"Chrome tracing exported to: {output_path}")
    
    def plot_resource_timeline(self, filename: Optional[str] = None, 
                             figsize: tuple = (14, 8), dpi: int = 120):
        """使用matplotlib绘制资源时间线图"""
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # 获取所有资源和时间线
        all_resources = sorted(self._get_all_resources())
        timeline = self.tracer.get_timeline()
        
        # 反转Y轴顺序，使NPU_0在上面
        all_resources_reversed = list(reversed(all_resources))
        
        # 设置Y轴
        y_positions = {res: i for i, res in enumerate(all_resources_reversed)}
        ax.set_yticks(range(len(all_resources_reversed)))
        ax.set_yticklabels(all_resources_reversed)
        
        # 调整Y轴标签样式
        ax.tick_params(axis='y', labelsize=10)
        
        # 颜色映射
        priority_colors = {
            TaskPriority.CRITICAL: '#FF4444',
            TaskPriority.HIGH: '#FF8844', 
            TaskPriority.NORMAL: '#4488FF',
            TaskPriority.LOW: '#888888'
        }
        
        # 绘制任务块
        for resource_id, executions in timeline.items():
            y_pos = y_positions[resource_id]
            
            for exec in executions:
                color = priority_colors.get(exec.priority, '#4488FF')
                
                # 创建矩形
                rect = Rectangle(
                    (exec.start_time, y_pos - 0.2),
                    exec.duration,
                    0.4,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(rect)
                
                # 添加任务标签（根据任务宽度决定是否显示）
                label_threshold = 3.0  # 最小宽度阈值
                if exec.duration > label_threshold:
                    # 标签在矩形内部
                    # 根据任务宽度调整标签
                    if exec.duration > 8:
                        # 宽任务：显示完整标签
                        label_text = exec.task_id
                    elif exec.duration > 5:
                        # 中等任务：显示简化标签
                        label_text = exec.task_id.replace('TASK_', '')
                    else:
                        # 较窄任务：只显示最关键部分
                        parts = exec.task_id.split('_')
                        if len(parts) >= 3:
                            label_text = f"{parts[1]}_{parts[2]}"
                        else:
                            label_text = parts[-1]
                    
                    ax.text(
                        exec.start_time + exec.duration / 2,
                        y_pos,
                        label_text,
                        ha='center',
                        va='center',
                        fontsize=8 if exec.duration > 8 else 7,
                        color='white' if exec.priority == TaskPriority.CRITICAL else 'black',
                        weight='bold' if exec.priority == TaskPriority.CRITICAL else 'normal'
                    )
                else:
                    # 标签在矩形上方（小任务）
                    ax.text(
                        exec.start_time + exec.duration / 2,
                        y_pos + 0.5,
                        exec.task_id.split('_')[-1],  # 只显示最后部分
                        ha='center',
                        va='bottom',
                        fontsize=6,
                        color=color
                    )
        
        # 设置图表属性
        ax.set_xlabel('Time (ms)', fontsize=12)
        ax.set_ylabel('Resources', fontsize=12)
        ax.set_title('Task Execution Timeline', fontsize=14, weight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        # 设置X轴范围
        if self.tracer.start_time is not None and self.tracer.end_time is not None:
            ax.set_xlim(self.tracer.start_time - 1, self.tracer.end_time + 1)
        
        ax.set_ylim(-0.5, len(all_resources_reversed) - 0.5)
        # 添加图例
        legend_elements = [
            patches.Patch(color=color, label=priority.name)
            for priority, color in priority_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # 调整布局
        plt.tight_layout()
        
        # 保存或显示
        if filename:
            output_path = ensure_artifact_path(filename)
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
            print(f"Timeline plot saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()
    
    def _get_all_resources(self) -> List[str]:
        """获取所有资源ID（包括未使用的）"""
        # 从tracer的queue_manager获取所有资源
        all_resources = set()
        
        if self.tracer.queue_manager:
            all_resources.update(self.tracer.queue_manager.resource_queues.keys())
        
        # 添加有执行记录的资源
        all_resources.update(self.tracer.resource_busy_time.keys())
        
        return sorted(list(all_resources))
    
    def _create_resource_tid_map(self, all_resources: List[str]) -> Dict[str, int]:
        """创建资源到线程ID的映射"""
        tid_map = {}
        tid = 1
        
        # 按类型分组
        npu_resources = sorted([r for r in all_resources if "NPU" in r])
        dsp_resources = sorted([r for r in all_resources if "DSP" in r])
        other_resources = sorted([r for r in all_resources 
                                if r not in npu_resources and r not in dsp_resources])
        
        # 分配线程ID
        for res_list in [npu_resources, dsp_resources, other_resources]:
            for res in res_list:
                tid_map[res] = tid
                tid += 1
        
        return tid_map
    
    def _get_priority_char(self, priority: TaskPriority) -> str:
        """根据优先级返回显示字符"""
        mapping = {
            TaskPriority.CRITICAL: "C",
            TaskPriority.HIGH: "H",
            TaskPriority.NORMAL: "N",
            TaskPriority.LOW: "L"
        }
        return mapping.get(priority, "?")
    
    def _get_priority_color(self, priority: TaskPriority) -> Optional[str]:
        """根据优先级返回Chrome Tracing颜色"""
        # Chrome Tracing预定义颜色
        color_mapping = {
            TaskPriority.CRITICAL: "terrible",     # 红色
            TaskPriority.HIGH: "bad",             # 橙色
            TaskPriority.NORMAL: "good",          # 蓝色
            TaskPriority.LOW: "generic_work"      # 灰色
        }
        return color_mapping.get(priority)
    
    def export_summary_report(self, filename: str):
        """导出详细的调度报告"""
        stats = self.tracer.get_statistics()
        timeline = self.tracer.get_timeline()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("AI Scheduler Execution Report\n")
            f.write("=" * 60 + "\n\n")
            
            # 基本信息
            f.write("Summary Statistics:\n")
            f.write(f"  Total Tasks: {stats['total_tasks']}\n")
            f.write(f"  Total Executions: {stats['total_executions']}\n")
            f.write(f"  Time Span: {stats['time_span']:.1f}ms\n")
            f.write(f"  Average Wait Time: {stats['average_wait_time']:.2f}ms\n")
            f.write(f"  Average Execution Time: {stats['average_execution_time']:.2f}ms\n\n")
            
            # 资源利用率
            f.write("Resource Utilization:\n")
            all_resources = self._get_all_resources()
            for res_id in sorted(all_resources):
                util = stats['resource_utilization'].get(res_id, 0.0)
                if util > 0:
                    f.write(f"  {res_id}: {util:.1f}%\n")
                else:
                    f.write(f"  {res_id}: IDLE\n")
            f.write("\n")
            
            # 任务执行详情
            f.write("Execution Timeline:\n")
            for resource_id in sorted(all_resources):
                f.write(f"\n{resource_id}:\n")
                if resource_id in timeline:
                    for exec in timeline[resource_id]:
                        f.write(f"  {exec.start_time:>6.1f} - {exec.end_time:>6.1f}ms: "
                              f"{exec.task_id} ({exec.priority.name})\n")
                else:
                    f.write("  No tasks executed\n")
        
        print(f"Summary report saved to: {filename}")
