#!/usr/bin/env python3
"""
调度可视化器 - 支持多种可视化格式
包括终端甘特图、Chrome Tracing、Matplotlib图表等
"""

import json
import math
import uuid
import importlib
from pathlib import Path
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
from google.protobuf import descriptor_pb2, message_factory
from google.protobuf.descriptor_pool import DescriptorPool

from NNScheduler.core.schedule_tracer import ScheduleTracer, TaskExecution
from NNScheduler.core.enums import TaskPriority, ResourceType
from NNScheduler.core.artifacts import ensure_artifact_path


PRIORITY_HEX_COLORS = {
    TaskPriority.CRITICAL: "#FF8A80",  # 轻柔红
    TaskPriority.HIGH: "#FFE082",      # 粉柔黄
    TaskPriority.NORMAL: "#90CAF9",    # 天空蓝
    TaskPriority.LOW: "#A5D6A7",       # 薄荷绿
}

PRIORITY_CHROME_COLORS = {
    TaskPriority.CRITICAL: "bad",
    TaskPriority.HIGH: "rail_load",
    TaskPriority.NORMAL: "rail_response",
    TaskPriority.LOW: "good",
}

RESOURCE_MARKERS = {
    ResourceType.NPU: "N",
    ResourceType.DSP: "D",
    ResourceType.ISP: "I",
    ResourceType.CPU: "C",
    ResourceType.GPU: "G",
    ResourceType.VPU: "V",
    ResourceType.FPGA: "F",
}

# Timeline padding reserved for labels in textual view
TEXT_TIMELINE_PADDING = 13


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
        timeline_width = max(width - TEXT_TIMELINE_PADDING, 10)
        print(f"{'Resource':<12} ", end="")
        for i in range(0, timeline_width, 10):
            time_point = (i / timeline_width) * total_duration
            print(f"{time_point:>9.1f}", end=" ")
        print()
        print("-"*width)
        
        # 打印每个资源的执行情况
        for resource_id in sorted(all_resources):
            print(f"{resource_id:<12} ", end="")
            
            # 创建时间线字符串
            timeline_str = [" "] * timeline_width
            
            # 填充执行的任务
            if resource_id in timeline:
                for execution in timeline[resource_id]:
                    start_pos = int((execution.start_time - start_time) / total_duration * timeline_width)
                    end_pos = int((execution.end_time - start_time) / total_duration * timeline_width)
                    
                    # 根据优先级选择字符
                    char = self._get_priority_char(execution.priority)
                    
                    # 填充执行区间
                    for i in range(start_pos, min(end_pos + 1, timeline_width)):
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
        
        # 打印任务流水线视图
        print("\nTask Latency Timeline:")
        print("-"*width)
        self._print_task_latency_timeline(timeline_width, start_time, end_time)
    
    def _print_task_latency_timeline(self, timeline_width: int, global_start: float, global_end: float):
        """打印任务级流水线视图"""
        summaries = self.tracer.get_task_latency_summary()
        if not summaries:
            print("  No task executions recorded")
            return
        
        total_duration = global_end - global_start
        if total_duration <= 0:
            print("  Invalid global duration for task timeline")
            return
        
        header = f"{'Task':<24} {'Start(ms)':>10} {'End(ms)':>10} {'Latency(ms)':>12} {'Priority':>10} {'Segs':>6}"
        print(header)
        timeline_indent = 25  # 24 chars for task label + single space separator
        print(" " * timeline_indent + "-" * timeline_width)
        
        for _, summary in sorted(summaries.items(), key=lambda item: (item[1]["first_start"], item[1]["task_id"])):
            start = summary["first_start"]
            end = summary["last_end"]
            latency = summary["latency"]
            priority_obj = summary.get("priority")
            priority = priority_obj.name if priority_obj else "UNKNOWN"
            segment_count = summary.get("segment_count", 0)
            display_name = summary.get("display_name") or summary.get("task_id", "UNKNOWN")
            
            print(f"{display_name:<24} {start:>10.2f} {end:>10.2f} {latency:>12.2f} {priority:>10} {segment_count:>6}")
            
            timeline_line = [" "] * timeline_width
            latency_start_idx = int((start - global_start) / total_duration * timeline_width)
            latency_end_idx = int(math.ceil((end - global_start) / total_duration * timeline_width))
            latency_start_idx = max(0, min(latency_start_idx, timeline_width - 1))
            latency_end_idx = max(latency_start_idx + 1, min(latency_end_idx, timeline_width))
            
            for idx in range(latency_start_idx, latency_end_idx):
                if 0 <= idx < timeline_width and timeline_line[idx] == " ":
                    timeline_line[idx] = "."
            
            first_segment_idx = latency_start_idx
            segments = summary.get("segments", [])
            for segment in segments:
                seg_start_idx = int((segment["start"] - global_start) / total_duration * timeline_width)
                seg_end_idx = int(math.ceil((segment["end"] - global_start) / total_duration * timeline_width))
                seg_start_idx = max(0, min(seg_start_idx, timeline_width - 1))
                seg_end_idx = max(seg_start_idx + 1, min(seg_end_idx, timeline_width))
                marker = self._get_resource_marker(segment.get("resource_type"))
                
                for idx in range(seg_start_idx, seg_end_idx):
                    if 0 <= idx < timeline_width:
                        timeline_line[idx] = marker
                
                if segment.get("is_first_segment"):
                    first_segment_idx = seg_start_idx
            
            pointer_line = [" "] * timeline_width
            if 0 <= first_segment_idx < timeline_width:
                pointer_line[first_segment_idx] = "^"
            
            print(" " * timeline_indent + "".join(timeline_line))
            print(" " * timeline_indent + "".join(pointer_line))
            gaps = summary.get("gaps", [])
            if gaps:
                gap_desc = ", ".join(f"{gap['duration']:.2f}ms idle" for gap in gaps)
                print(" " * timeline_indent + f"gaps: {gap_desc}")
            print()

    def export_chrome_tracing(self, filename: str):
        """导出Chrome Tracing格式的JSON文件（改进版）"""
        chrome_events: List[Dict[str, Any]] = []
        
        # 获取所有资源
        all_resources = self._get_all_resources()
        
        # 为所有资源分配线程ID
        resource_tid_map = self._create_resource_tid_map(all_resources)
        task_summaries = self.tracer.get_task_latency_summary()
        
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
            priority_hex = PRIORITY_HEX_COLORS.get(execution.priority)
            priority_argb = _hex_to_argb(priority_hex)
            
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
                    "duration_ms": execution.duration,
                    "hex_color": priority_hex
                }
            }
            
            # 根据优先级设置颜色
            color = self._get_priority_color(execution.priority)
            if color:
                event["cname"] = color
            if priority_argb is not None:
                event["color"] = priority_argb
            
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
                if priority_argb is not None:
                    marker["color"] = priority_argb
                chrome_events.append(marker)
        
        # 为每个任务创建独立线程，展示端到端时延
        if task_summaries:
            start_tid = (max(resource_tid_map.values()) if resource_tid_map else 0) + 1
            sort_base = len(resource_tid_map) + 1
            for offset, summary in enumerate(sorted(task_summaries.values(), key=lambda item: (item["first_start"], item["task_id"]))):
                display_name = summary.get("display_name") or summary.get("task_id")
                priority: TaskPriority = summary.get("priority", TaskPriority.NORMAL)
                tid = start_tid + offset
                sort_index = sort_base + offset
                priority_hex = PRIORITY_HEX_COLORS.get(priority)
                priority_argb = _hex_to_argb(priority_hex)
                
                chrome_events.append({
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 0,
                    "tid": tid,
                    "args": {
                        "name": display_name
                    }
                })
                chrome_events.append({
                    "name": "thread_sort_index",
                    "ph": "M",
                    "pid": 0,
                    "tid": tid,
                    "args": {
                        "sort_index": sort_index
                    }
                })
                
                latency_duration_us = max(summary["latency"], 0.0) * 1000
                task_event = {
                    "name": display_name,
                    "cat": "TaskLatency",
                    "ph": "X",
                    "ts": summary["first_start"] * 1000,
                    "dur": max(latency_duration_us, 1.0),
                    "pid": 0,
                    "tid": tid,
                    "args": {
                        "task_id": summary.get("task_id"),
                        "task_name": summary.get("task_name"),
                        "instance_id": summary.get("instance_id"),
                        "priority": priority.name,
                        "latency_ms": summary.get("latency"),
                        "segment_count": summary.get("segment_count"),
                        "first_resource": summary.get("first_resource_id"),
                        "first_segment_id": summary.get("first_segment_id"),
                        "gaps": summary.get("gaps"),
                        "hex_color": priority_hex,
                    }
                }
                color = self._get_priority_color(priority)
                if color:
                    task_event["cname"] = color
                if priority_argb is not None:
                    task_event["color"] = priority_argb
                chrome_events.append(task_event)
                
                highlight_event = {
                    "name": "dispatch",
                    "cat": "TaskDispatch",
                    "ph": "i",
                    "ts": summary["first_start"] * 1000,
                    "pid": 0,
                    "tid": tid,
                    "s": "p",
                    "args": {
                        "task_id": summary.get("task_id"),
                        "first_resource": summary.get("first_resource_id"),
                        "first_segment_id": summary.get("first_segment_id"),
                        "priority": priority.name,
                        "hex_color": priority_hex
                    }
                }
                if color:
                    highlight_event["cname"] = color
                if priority_argb is not None:
                    highlight_event["color"] = priority_argb
                chrome_events.append(highlight_event)
        
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
        
        perfetto_path = self._export_perfetto_trace(output_path, task_summaries)
        if perfetto_path:
            print(f"Perfetto trace exported to: {perfetto_path}")
    
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
        priority_colors = PRIORITY_HEX_COLORS
        
        # 绘制任务块
        for resource_id, executions in timeline.items():
            y_pos = y_positions[resource_id]
            
            for exec in executions:
                color = priority_colors.get(exec.priority, '#1E88E5')
                
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
        """根据优先级返回Chrome/Perfetto调色板名"""
        return PRIORITY_CHROME_COLORS.get(priority)
    
    def _get_resource_marker(self, resource_type: Optional[ResourceType]) -> str:
        """根据资源类型返回文本流水线标记"""
        if resource_type is None:
            return "="
        if isinstance(resource_type, str):
            for rt, marker in RESOURCE_MARKERS.items():
                if resource_type == rt.value or resource_type == rt.name:
                    return marker
            return resource_type[:1].upper() if resource_type else "="
        return RESOURCE_MARKERS.get(resource_type, "=")
    
    def _export_perfetto_trace(self, json_output_path: Path, task_summaries: Dict[str, Dict[str, Any]]) -> Optional[Path]:
        """使用Perfetto库导出更丰富的trace，若库不可用则忽略"""
        builder = _PerfettoTraceBuilder()
        if not builder.available:
            print("[WARN] Perfetto trace protobufs未找到，跳过 .pftrace 导出。")
            return None
        
        timeline = self.tracer.get_timeline()
        
        for resource_id, executions in timeline.items():
            track_uuid = builder.add_track(resource_id, hex_color=None)
            for execution in executions:
                builder.add_slice(
                    track_uuid=track_uuid,
                    name=execution.task_id,
                    category=execution.priority.name,
                    start_ms=execution.start_time,
                    duration_ms=execution.duration,
                    hex_color=PRIORITY_HEX_COLORS.get(execution.priority)
                )
        
        if task_summaries:
            for summary in sorted(task_summaries.values(), key=lambda item: (item["first_start"], item["task_id"])):
                display_name = summary.get("display_name") or summary.get("task_id")
                priority: TaskPriority = summary.get("priority", TaskPriority.NORMAL)
                color_hex = PRIORITY_HEX_COLORS.get(priority)
                track_uuid = builder.add_track(display_name, hex_color=color_hex)
                
                latency_ms = max(summary["latency"], 0.0)
                builder.add_slice(
                    track_uuid=track_uuid,
                    name=display_name,
                    category="TaskLatency",
                    start_ms=summary["first_start"],
                    duration_ms=max(latency_ms, 0.001),
                    hex_color=color_hex,
                    annotations={
                        "task_id": summary.get("task_id"),
                        "task_name": summary.get("task_name"),
                        "instance_id": summary.get("instance_id"),
                        "segment_count": summary.get("segment_count"),
                        "priority": priority.name,
                        "latency_ms": latency_ms,
                        "hex_color": color_hex
                    }
                )
                
                builder.add_instant(
                    track_uuid=track_uuid,
                    name="dispatch",
                    category="TaskDispatch",
                    timestamp_ms=summary["first_start"],
                    hex_color=color_hex,
                    annotations={
                        "task_id": summary.get("task_id"),
                        "first_resource": summary.get("first_resource_id"),
                        "first_segment_id": summary.get("first_segment_id"),
                        "priority": priority.name,
                        "hex_color": color_hex
                    }
                )
        
        perfetto_path = json_output_path.with_suffix(".pftrace")
        builder.serialize(perfetto_path)
        return perfetto_path
    
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


def _ms_to_ns(value_ms: float) -> int:
    """Convert milliseconds to nanoseconds with rounding."""
    return int(round(value_ms * 1_000_000))


def _hex_to_argb(hex_color: Optional[str]) -> Optional[int]:
    if not hex_color:
        return None
    value = hex_color.lstrip("#")
    if len(value) == 6:
        return (0xFF << 24) | int(value, 16)
    if len(value) == 8:
        return int(value, 16)
    return None


class _PerfettoTraceBuilder:
    """Lightweight helper wrapping Perfetto trace protobuf APIs (optional)."""

    def __init__(self):
        self.available = False
        self.trace = None
        self.track_event_type_values: Dict[str, int] = {}

        try:
            platform_mod = importlib.import_module("perfetto.trace_processor.platform")
        except ImportError:
            return

        delegate_cls = getattr(platform_mod, "PlatformDelegate", None)
        if delegate_cls is None:
            return

        try:
            descriptor_bytes = delegate_cls().get_resource("trace_processor.descriptor")
        except Exception:
            return

        file_set = descriptor_pb2.FileDescriptorSet()
        try:
            file_set.MergeFromString(descriptor_bytes)
        except Exception:
            return

        pool = DescriptorPool()
        for file_desc in file_set.file:
            try:
                pool.Add(file_desc)
            except Exception:
                continue

        factory = message_factory.MessageFactory(pool)

        def _resolve(name: str):
            try:
                desc = pool.FindMessageTypeByName(name)
            except KeyError:
                return None
            return factory.GetPrototype(desc)

        self.Trace = _resolve("perfetto.protos.Trace")
        self.TracePacket = _resolve("perfetto.protos.TracePacket")
        self.TrackDescriptor = _resolve("perfetto.protos.TrackDescriptor")
        self.TrackEvent = _resolve("perfetto.protos.TrackEvent")

        if not all([self.Trace, self.TracePacket, self.TrackDescriptor, self.TrackEvent]):
            return

        track_event_desc = self.TrackEvent.DESCRIPTOR
        track_event_type = track_event_desc.enum_types_by_name.get("Type") if track_event_desc else None
        if track_event_type:
            self.track_event_type_values = {
                value.name: value.number for value in track_event_type.values
            }

        self.trace = self.Trace()
        self.available = True

    def add_track(self, name: str, hex_color: Optional[str] = None) -> int:
        if not self.available or self.trace is None:
            return 0
        track_uuid = self._generate_uuid()
        packet = self.trace.packet.add()
        descriptor = packet.track_descriptor
        descriptor.name = name
        descriptor.uuid = track_uuid
        self._apply_color(descriptor, hex_color)
        return track_uuid

    def add_slice(
        self,
        track_uuid: int,
        name: str,
        category: Optional[str],
        start_ms: float,
        duration_ms: float,
        hex_color: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ):
        if not self.available or self.trace is None:
            return
        event = self._new_event(track_uuid, start_ms)
        slice_type = self._get_type(["TYPE_SLICE", "TYPE_COMPLETE", "TYPE_SLICE_BEGIN"])
        if slice_type is not None:
            event.type = slice_type
        if hasattr(event, "name"):
            event.name = name
        self._append_category(event, category)
        self._set_slice_info(event, name, category, duration_ms)
        self._apply_duration(event, duration_ms)
        self._apply_color(event, hex_color)
        self._attach_annotations(event, annotations)

    def add_instant(
        self,
        track_uuid: int,
        name: str,
        category: Optional[str],
        timestamp_ms: float,
        hex_color: Optional[str] = None,
        annotations: Optional[Dict[str, Any]] = None,
    ):
        if not self.available or self.trace is None:
            return
        event = self._new_event(track_uuid, timestamp_ms)
        instant_type = self._get_type(["TYPE_INSTANT", "TYPE_SLICE_BEGIN"])
        if instant_type is not None:
            event.type = instant_type
        if hasattr(event, "name"):
            event.name = name
        self._append_category(event, category)
        self._apply_color(event, hex_color)
        self._attach_annotations(event, annotations)

    def serialize(self, output_path: Path):
        if not self.available or self.trace is None:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("wb") as fp:
            fp.write(self.trace.SerializeToString())

    def _new_event(self, track_uuid: int, timestamp_ms: float):
        if not self.available or self.trace is None:
            raise RuntimeError("Perfetto builder is not available")
        packet = self.trace.packet.add()
        event = packet.track_event
        event.track_uuid = track_uuid
        event.timestamp = _ms_to_ns(timestamp_ms)
        return event

    def _generate_uuid(self) -> int:
        return uuid.uuid4().int & ((1 << 63) - 1)

    def _apply_color(self, message_obj: Any, hex_color: Optional[str]):
        color_value = _hex_to_argb(hex_color)
        if color_value is None or message_obj is None:
            return

        for attr_name in ("color", "track_color", "background_color_argb"):
            if hasattr(message_obj, attr_name):
                try:
                    setattr(message_obj, attr_name, color_value)
                    return
                except Exception:
                    continue

        nested = getattr(message_obj, "track", None)
        if nested is not None:
            self._apply_color(nested, hex_color)

    def _append_category(self, event: Any, category: Optional[str]):
        if not category or event is None:
            return

        if hasattr(event, "categories"):
            try:
                event.categories.append(category)
                return
            except Exception:
                try:
                    event.categories.extend([category])
                    return
                except Exception:
                    pass

        if hasattr(event, "category"):
            try:
                event.category = category
            except Exception:
                pass

    def _set_slice_info(self, event: Any, name: str, category: Optional[str], duration_ms: float):
        slice_field = getattr(event, "slice", None)
        if slice_field is None:
            return
        try:
            if hasattr(slice_field, "name"):
                slice_field.name = name
            if category and hasattr(slice_field, "category"):
                slice_field.category = category
            if duration_ms is not None and hasattr(slice_field, "duration"):
                slice_field.duration = _ms_to_ns(duration_ms)
        except Exception:
            return

    def _apply_duration(self, event: Any, duration_ms: float):
        if duration_ms is None:
            return
        for attr_name in ("duration", "duration_ns"):
            if hasattr(event, attr_name):
                try:
                    setattr(event, attr_name, _ms_to_ns(duration_ms))
                    return
                except Exception:
                    continue

    def _attach_annotations(self, event: Any, annotations: Optional[Dict[str, Any]]):
        if not annotations:
            return
        if hasattr(event, "legacy_event"):
            legacy = event.legacy_event
            if hasattr(legacy, "args"):
                try:
                    args = legacy.args
                    for key, value in annotations.items():
                        args[key] = str(value)
                    return
                except Exception:
                    pass
        if hasattr(event, "debug_annotations"):
            try:
                for key, value in annotations.items():
                    annotation = event.debug_annotations.add()
                    annotation.name = key
                    if hasattr(annotation, "string_value"):
                        annotation.string_value = str(value)
            except Exception:
                pass

    def _get_type(self, candidates: List[str]) -> Optional[int]:
        if not self.track_event_type_values:
            return None
        for name in candidates:
            if name in self.track_event_type_values:
                return self.track_event_type_values[name]
        return None
