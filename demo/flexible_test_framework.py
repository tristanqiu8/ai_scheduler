#!/usr/bin/env python3
"""
灵活的调度测试框架 - 支持多种配置场景
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import os
import sys

from core.resource_queue import ResourceQueueManager
from core.schedule_tracer import ScheduleTracer
from core.enhanced_launcher import EnhancedTaskLauncher
from core.executor import ScheduleExecutor
from core.evaluator import PerformanceEvaluator
from core.enums import ResourceType
from viz.schedule_visualizer import ScheduleVisualizer

# 导入配置类
if 'core.scheduling_config' in sys.modules:
    from core.scheduling_config import SchedulingConfig
else:
    from scheduling_config import SchedulingConfig


@dataclass
class TestResult:
    """测试结果"""
    config: 'SchedulingConfig'
    stats: Dict
    metrics: 'PerformanceMetrics'
    utilization: Dict[str, float]
    system_utilization: float
    tracer: ScheduleTracer
    
    def summary(self) -> str:
        """生成结果摘要"""
        lines = [
            f"配置: {self.config.get_resource_summary()}",
            f"完成实例: {self.stats['completed_instances']}",
            f"System利用率: {self.system_utilization:.1f}%",
            f"FPS满足率: {self.metrics.fps_satisfaction_rate:.1f}%"
        ]
        return "\n".join(lines)


class SchedulingTestFramework:
    """调度测试框架"""
    
    def __init__(self, tasks: List):
        """
        初始化测试框架
        
        Args:
            tasks: 任务列表
        """
        self.tasks = tasks
        self.results: Dict[str, TestResult] = {}
    
    def calculate_theory_demand(self, tasks: List, config: SchedulingConfig) -> Dict:
        """
        计算理论资源需求
        
        Args:
            tasks: 任务列表
            config: 调度配置
            
        Returns:
            理论需求分析结果
        """
        # 获取平均带宽
        npu_bandwidth = config.get_npu_bandwidth()
        dsp_bandwidth = config.get_dsp_bandwidth()
        time_window = config.analysis_window
        
        npu_total_time = 0.0
        dsp_total_time = 0.0
        
        for task in tasks:
            # 计算任务在时间窗口内需要执行的次数
            instances_needed = task.fps_requirement * (time_window / 1000.0)
            
            # 应用分段策略获取实际执行的段
            segments = task.apply_segmentation()
            if not segments:
                segments = task.segments
            
            # 计算每个段的执行时间
            for seg in segments:
                if seg.resource_type.value == "NPU":
                    duration = seg.get_duration(npu_bandwidth)
                    npu_total_time += duration * instances_needed
                elif seg.resource_type.value == "DSP":
                    duration = seg.get_duration(dsp_bandwidth)
                    dsp_total_time += duration * instances_needed
        
        # 计算利用率
        npu_utilization = (npu_total_time / time_window) * 100
        dsp_utilization = (dsp_total_time / time_window) * 100
        
        return {
            'npu_demand_ms': npu_total_time,
            'dsp_demand_ms': dsp_total_time,
            'npu_utilization': npu_utilization,
            'dsp_utilization': dsp_utilization,
            'feasible': npu_utilization <= 100 and dsp_utilization <= 100
        }
    
    def run_test(self, config: SchedulingConfig, verbose: bool = True) -> TestResult:
        """
        运行单个测试
        
        Args:
            config: 调度配置
            verbose: 是否打印详细信息
            
        Returns:
            测试结果
        """
        if verbose:
            config.print_config()
        
        # 1. 创建资源
        queue_manager = ResourceQueueManager()
        for resource in config.resources:
            queue_manager.add_resource(
                resource.resource_id,
                resource.resource_type,
                resource.bandwidth
            )
        
        # 2. 创建追踪器和启动器
        tracer = ScheduleTracer(queue_manager)
        launcher = EnhancedTaskLauncher(queue_manager, tracer)
        
        # 3. 注册任务
        for task in self.tasks:
            launcher.register_task(task)
        
        # 4. 创建和执行计划
        plan = launcher.create_launch_plan(
            config.simulation_duration,
            config.launch_strategy
        )
        
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(
            plan,
            config.simulation_duration,
            segment_mode=config.segment_mode
        )
        
        # 5. 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(config.simulation_duration, plan.events)
        
        # 6. 计算利用率
        resource_utilization = tracer.get_resource_utilization(
            time_window=config.analysis_window
        )
        system_util = self._calculate_system_utilization(
            tracer, config.analysis_window
        )
        
        # 7. 创建结果
        result = TestResult(
            config=config,
            stats=stats,
            metrics=metrics,
            utilization=resource_utilization,
            system_utilization=system_util,
            tracer=tracer
        )
        
        # 8. 保存结果
        self.results[config.scenario_name] = result
        
        if verbose:
            self._print_test_result(result)
        
        return result
    
    def run_comparison_tests(self, configs: List[SchedulingConfig]) -> Dict[str, TestResult]:
        """
        运行多个配置的对比测试
        
        Args:
            configs: 配置列表
            
        Returns:
            结果字典
        """
        print("\n🔬 开始对比测试")
        print("="*80)
        
        for config in configs:
            print(f"\n▶ 测试场景: {config.scenario_name}")
            self.run_test(config, verbose=False)
            print(f"  ✓ 完成")
        
        # 打印对比结果
        self._print_comparison_results()
        
        return self.results
    
    def generate_visualizations(self, output_dir: str = "results"):
        """
        为所有测试结果生成可视化
        
        Args:
            output_dir: 输出目录
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n📊 生成可视化文件到 {output_dir}/")
        
        for name, result in self.results.items():
            # 生成文件名（去除特殊字符）
            safe_name = name.replace(" ", "_").replace("×", "x").replace("+", "_")
            
            visualizer = ScheduleVisualizer(result.tracer)
            
            # 生成PNG
            png_file = os.path.join(output_dir, f"{safe_name}.png")
            visualizer.plot_resource_timeline(png_file)
            
            # 生成Chrome Trace
            json_file = os.path.join(output_dir, f"{safe_name}.json")
            visualizer.export_chrome_tracing(json_file)
            
            print(f"  ✓ {name}: {safe_name}.png, {safe_name}.json")
    
    def export_comparison_report(self, filename: str = "comparison_report.txt"):
        """导出对比报告"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("调度系统对比测试报告\n")
            f.write("="*80 + "\n\n")
            
            # 写入详细结果
            for name, result in self.results.items():
                f.write(f"\n{name}\n")
                f.write("-"*40 + "\n")
                f.write(result.summary() + "\n")
                
                # 资源利用率详情
                f.write("\n资源利用率:\n")
                for res_id, util in sorted(result.utilization.items()):
                    f.write(f"  {res_id}: {util:.1f}%\n")
        
        print(f"\n📄 对比报告已保存到: {filename}")
    
    def _calculate_system_utilization(self, tracer, window_size):
        """计算系统利用率"""
        busy_intervals = []
        
        for exec in tracer.executions:
            if exec.start_time is not None and exec.end_time is not None:
                busy_intervals.append((exec.start_time, exec.end_time))
        
        if not busy_intervals:
            return 0.0
        
        # 合并重叠的时间段
        busy_intervals.sort()
        merged_intervals = []
        
        for start, end in busy_intervals:
            if merged_intervals and start <= merged_intervals[-1][1]:
                merged_intervals[-1] = (merged_intervals[-1][0], max(merged_intervals[-1][1], end))
            else:
                merged_intervals.append((start, end))
        
        total_busy_time = sum(end - start for start, end in merged_intervals)
        return (total_busy_time / window_size) * 100.0
    
    def _print_test_result(self, result: TestResult):
        """打印单个测试结果"""
        print(f"\n📈 测试结果:")
        print(f"  完成实例: {result.stats['completed_instances']}")
        print(f"  执行段数: {result.stats['total_segments_executed']}")
        print(f"  System利用率: {result.system_utilization:.1f}%")
        print(f"  平均等待时间: {result.metrics.avg_wait_time:.2f}ms")
        print(f"  FPS满足率: {result.metrics.fps_satisfaction_rate*100:.1f}%")
        
        # 打印主要资源利用率
        npu_utils = [(k, v) for k, v in result.utilization.items() if 'NPU' in k]
        dsp_utils = [(k, v) for k, v in result.utilization.items() if 'DSP' in k]
        
        if npu_utils:
            print(f"\n  NPU利用率:")
            for res_id, util in sorted(npu_utils):
                print(f"    {res_id}: {util:.1f}%")
        
        if dsp_utils:
            print(f"\n  DSP利用率:")
            for res_id, util in sorted(dsp_utils):
                print(f"    {res_id}: {util:.1f}%")
    
    def _print_comparison_results(self):
        """打印对比结果表格"""
        if not self.results:
            return
        
        print("\n\n📊 对比结果汇总")
        print("="*100)
        
        # 表头
        headers = ["配置", "完成实例", "System利用率", "平均NPU利用率", "平均DSP利用率", "FPS满足率"]
        col_widths = [25, 10, 15, 15, 15, 12]
        
        # 打印表头
        header_line = ""
        for header, width in zip(headers, col_widths):
            header_line += f"{header:<{width}}"
        print(header_line)
        print("-"*100)
        
        # 打印数据行
        for name, result in self.results.items():
            # 计算平均利用率
            npu_utils = [v for k, v in result.utilization.items() if 'NPU' in k]
            dsp_utils = [v for k, v in result.utilization.items() if 'DSP' in k]
            
            avg_npu = sum(npu_utils) / len(npu_utils) if npu_utils else 0
            avg_dsp = sum(dsp_utils) / len(dsp_utils) if dsp_utils else 0
            
            row = [
                name[:24],  # 截断过长的名称
                str(result.stats['completed_instances']),
                f"{result.system_utilization:.1f}%",
                f"{avg_npu:.1f}%",
                f"{avg_dsp:.1f}%",
                f"{result.metrics.fps_satisfaction_rate:.1f}%"
            ]
            
            row_line = ""
            for cell, width in zip(row, col_widths):
                row_line += f"{cell:<{width}}"
            print(row_line)
