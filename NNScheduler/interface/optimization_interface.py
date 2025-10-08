#!/usr/bin/env python3
"""
优化接口层 - 提供与test_cam_auto_priority_optimization.py相同效果的JSON API
"""

import json
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.executor import ScheduleExecutor, set_execution_log_enabled
from NNScheduler.core.enums import ResourceType, TaskPriority, SegmentationStrategy
from NNScheduler.core.evaluator import PerformanceEvaluator
from NNScheduler.core.task import NNTask
from NNScheduler.scenario.camera_task import create_real_tasks
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
from .json_interface import JsonInterface


@dataclass
class OptimizationResult:
    """优化结果"""
    iteration: int
    priority_config: Dict[str, str]  # 使用字符串存储优先级名称
    fps_satisfaction: Dict[str, bool]
    latency_satisfaction: Dict[str, bool]
    total_satisfaction_rate: float
    avg_latency: float
    resource_utilization: Dict[str, float]
    fps_analysis: Dict[str, float]  # 新增：FPS分析
    power_analysis: Dict[str, float]  # 新增：功耗分析
    ddr_analysis: Dict[str, float]    # 新增：DDR分析
    system_utilization: float        # 新增：系统利用率


class OptimizationInterface:
    """优化接口处理器 - 提供JSON格式的优化功能"""

    def __init__(self):
        self.optimization_history = []
        self.task_features = {}
        self.priority_levels = [
            TaskPriority.LOW,
            TaskPriority.NORMAL,
            TaskPriority.HIGH,
            TaskPriority.CRITICAL
        ]

    def optimize_from_json(self, config_file: str, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        从JSON配置文件运行优化

        Args:
            config_file: 包含优化配置的JSON文件路径
            output_file: 可选的输出文件路径，如果不指定则自动生成

        Returns:
            优化结果字典
        """
        # 加载配置
        config = JsonInterface.load_from_file(config_file)

        # 运行优化
        return self.optimize_from_config(config, output_file)

    def optimize_from_config(self, config: Dict[str, Any], output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        从配置字典运行优化

        Args:
            config: 优化配置字典
            output_file: 可选的输出文件路径

        Returns:
            优化结果字典
        """
        # 解析配置
        optimization_config = config.get("optimization", {})
        scenario_config = config.get("scenario", {})
        resource_config = config.get("resources", {})

        # 从配置中读取search_priority设置，默认为true
        search_priority = optimization_config.get("search_priority", True)

        # 从配置中读取log_level设置，默认为"normal"
        log_level = optimization_config.get("log_level", "normal")

        # 发射策略（eager|lazy|balanced），默认balanced
        launch_strategy = str(optimization_config.get("launch_strategy", "balanced")).strip().lower()
        if launch_strategy not in {"eager", "lazy", "balanced"}:
            print(f"[WARN] 无效的launch_strategy: {launch_strategy}，已回退为balanced")
            launch_strategy = "balanced"

        # 创建任务
        if "use_camera_tasks" in scenario_config and scenario_config["use_camera_tasks"]:
            # 使用预定义的相机任务
            tasks = create_real_tasks()
            # 对于预定义任务，如果不进行搜索，需要从任务本身获取优先级
            user_priority_config = {}
            if not search_priority:
                for task in tasks:
                    user_priority_config[task.task_id] = task.priority.name
        else:
            # 从JSON配置创建任务
            tasks = JsonInterface.parse_scenario_config(scenario_config)
            # 提取用户配置的优先级信息
            user_priority_config = {}
            for task_config in scenario_config.get("tasks", []):
                task_id = task_config.get("task_id")
                priority = task_config.get("priority", "NORMAL")
                user_priority_config[task_id] = priority

        # 创建优化器
        optimizer = JsonPriorityOptimizer(
            tasks=tasks,
            time_window=optimization_config.get("time_window", 1000.0),
            segment_mode=optimization_config.get("segment_mode", True),
            resources=resource_config,
            search_priority=search_priority,
            user_priority_config=user_priority_config,
            launch_strategy=launch_strategy
        )

        # 设置日志级别
        if log_level == "detailed":
            set_execution_log_enabled(True)
            print(f"[INFO] 启用详细日志模式，将显示任务入队、出队和执行详情")
        else:
            set_execution_log_enabled(False)

        # 执行优化
        best_config, best_result = optimizer.optimize(
            max_iterations=optimization_config.get("max_iterations", 50),
            max_time_seconds=optimization_config.get("max_time_seconds", 300),
            target_satisfaction=optimization_config.get("target_satisfaction", 0.95)
        )

        # 优化完成后恢复默认日志设置
        if log_level == "detailed":
            set_execution_log_enabled(False)

        # 准备结果
        result = {
            "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
            "optimization_config": optimization_config,
            "best_configuration": {
                "priority_config": {k: v.name for k, v in best_config.items()},
                "satisfaction_rate": best_result.total_satisfaction_rate,
                "avg_latency": best_result.avg_latency,
                "resource_utilization": best_result.resource_utilization,
                "fps_analysis": best_result.fps_analysis,
                "power_analysis": best_result.power_analysis,
                "ddr_analysis": best_result.ddr_analysis,
                "system_utilization": best_result.system_utilization,
                "fps_satisfaction": best_result.fps_satisfaction,
                "latency_satisfaction": best_result.latency_satisfaction
            },
            "optimization_history": [asdict(r) for r in optimizer.optimization_history],
            "task_features": optimizer.task_features
        }

        # 生成可视化文件
        visualization_files = self._generate_visualizations(
            tasks, best_config, resource_config, optimization_config, log_level
        )
        result["visualization_files"] = visualization_files

        # 保存结果
        if output_file is None:
            output_file = f"optimization_result_{time.strftime('%Y%m%d_%H%M%S')}.json"

        JsonInterface.save_to_file(result, output_file)
        result["output_file"] = output_file

        return result

    def _generate_visualizations(self, tasks: List[NNTask], best_config: Dict[str, TaskPriority],
                               resource_config: Dict[str, Any], optimization_config: Dict[str, Any], log_level: str = "normal") -> Dict[str, str]:
        """生成可视化文件"""
        # 应用最佳配置
        for task in tasks:
            task.priority = best_config[task.task_id]

        # 创建资源
        resources = resource_config.get("resources", [
            {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 160.0},
            {"resource_id": "DSP_0", "resource_type": "DSP", "bandwidth": 160.0}
        ])

        queue_manager = ResourceQueueManager()
        for res in resources:
            queue_manager.add_resource(res["resource_id"], ResourceType[res["resource_type"]], res["bandwidth"])

        # 创建调度器
        tracer = ScheduleTracer(queue_manager)
        segment_mode = optimization_config.get("segment_mode", True)

        if segment_mode:
            launcher = EnhancedTaskLauncher(queue_manager, tracer)
        else:
            launcher = TaskLauncher(queue_manager, tracer)

        for task in tasks:
            launcher.register_task(task)

        # 设置日志级别
        if log_level == "detailed":
            set_execution_log_enabled(True)

        # 执行调度
        time_window = optimization_config.get("time_window", 1000.0)
        # 读取并规范化发射策略
        launch_strategy = str(optimization_config.get("launch_strategy", "balanced")).strip().lower()
        if launch_strategy not in {"eager", "lazy", "balanced"}:
            print(f"[WARN] 无效的launch_strategy: {launch_strategy}，已回退为balanced")
            launch_strategy = "balanced"

        plan = launcher.create_launch_plan(time_window, launch_strategy)
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        executor.execute_plan(plan, time_window, segment_mode=segment_mode)

        # 恢复默认日志设置
        if log_level == "detailed":
            set_execution_log_enabled(False)

        # 生成可视化
        visualizer = ScheduleVisualizer(tracer)
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        files = {}

        # Chrome Tracing
        chrome_trace_file = f"optimized_schedule_chrome_trace_{timestamp}.json"
        visualizer.export_chrome_tracing(chrome_trace_file)
        files["chrome_trace"] = chrome_trace_file

        # Timeline PNG
        png_file = f"optimized_schedule_timeline_{timestamp}.png"
        visualizer.plot_resource_timeline(png_file)
        files["timeline_png"] = png_file

        return files

    def create_optimization_template(self) -> Dict[str, Any]:
        """创建优化配置模板"""
        return {
            "optimization": {
                "max_iterations": 50,
                "max_time_seconds": 300,
                "target_satisfaction": 0.95,
                "time_window": 1000.0,
                "segment_mode": True,
                "enable_detailed_analysis": True,
                "launch_strategy": "balanced"
            },
            "resources": {
                "resources": [
                    {
                        "resource_id": "NPU_0",
                        "resource_type": "NPU",
                        "bandwidth": 160.0
                    },
                    {
                        "resource_id": "DSP_0",
                        "resource_type": "DSP",
                        "bandwidth": 160.0
                    }
                ]
            },
            "scenario": {
                "use_camera_tasks": True,  # 使用预定义相机任务
                # 或者自定义任务
                "scenario_name": "Custom Optimization Scenario",
                "description": "自定义优化场景",
                "tasks": [
                    # 任务配置...
                ]
            }
        }


class JsonPriorityOptimizer:
    """JSON版本的优先级优化器 - 与test_cam_auto_priority_optimization.py功能相同"""

    def __init__(self, tasks: List[NNTask], time_window=1000.0, segment_mode=True, resources=None,
                 search_priority=True, user_priority_config=None, launch_strategy: str = "balanced"):
        self.tasks = tasks
        self.time_window = time_window
        self.segment_mode = segment_mode
        self.resources = resources or {}
        self.search_priority = search_priority
        self.user_priority_config = user_priority_config or {}
        # 规范化发射策略
        self.launch_strategy = str(launch_strategy).strip().lower()
        if self.launch_strategy not in {"eager", "lazy", "balanced"}:
            self.launch_strategy = "balanced"

        # 分析任务特征
        self.task_features = self._analyze_task_features()

        # 优先级选项
        self.priority_levels = [
            TaskPriority.LOW,
            TaskPriority.NORMAL,
            TaskPriority.HIGH,
            TaskPriority.CRITICAL
        ]

        # 优化历史
        self.optimization_history = []

    def _analyze_task_features(self) -> Dict[str, dict]:
        """分析任务特征用于初始优先级分配"""
        features = {}

        # 计算被依赖次数
        dependency_count = defaultdict(int)
        for task in self.tasks:
            for dep in task.dependencies:
                dependency_count[dep] += 1

        for task in self.tasks:
            features[task.task_id] = {
                'name': task.name,
                'fps_requirement': task.fps_requirement,
                'latency_requirement': task.latency_requirement,
                'dependency_count': dependency_count[task.task_id],
                'has_dependencies': len(task.dependencies) > 0,
                'num_segments': len(task.segments),
                'uses_npu': task.uses_npu,
                'uses_dsp': task.uses_dsp,
                'latency_strictness': self._calculate_latency_strictness(task),
                'fps_strictness': task.fps_requirement
            }

        return features

    def _calculate_latency_strictness(self, task) -> float:
        """计算延迟严格度"""
        bandwidth_map = {ResourceType.NPU: 160.0, ResourceType.DSP: 160.0}
        estimated_duration = task.estimate_duration(bandwidth_map)

        if task.latency_requirement > 0:
            return estimated_duration / task.latency_requirement
        return 0.0

    def _calculate_priority_score(self, task_id: str) -> float:
        """计算任务优先级分数"""
        features = self.task_features[task_id]

        score = 0.0

        # 1. 被依赖次数（权重：40%）
        score += features['dependency_count'] * 40

        # 2. FPS要求（权重：20%）
        max_fps = max(f['fps_requirement'] for f in self.task_features.values())
        if max_fps > 0:
            score += (features['fps_requirement'] / max_fps) * 20

        # 3. 延迟严格度（权重：30%）
        score += features['latency_strictness'] * 30

        # 4. 资源复杂度（权重：10%）
        if features['uses_npu'] and features['uses_dsp']:
            score += 10
        elif features['num_segments'] > 5:
            score += 5

        return score

    def generate_initial_priorities(self) -> Dict[str, TaskPriority]:
        """生成初始优先级配置"""
        priority_scores = {}
        for task in self.tasks:
            priority_scores[task.task_id] = self._calculate_priority_score(task.task_id)

        # 根据分数排序
        sorted_tasks = sorted(priority_scores.items(), key=lambda x: x[1], reverse=True)

        # 分配优先级
        priority_config = {}
        num_tasks = len(sorted_tasks)

        for i, (task_id, score) in enumerate(sorted_tasks):
            if i < num_tasks * 0.1:
                priority_config[task_id] = TaskPriority.CRITICAL
            elif i < num_tasks * 0.3:
                priority_config[task_id] = TaskPriority.HIGH
            elif i < num_tasks * 0.7:
                priority_config[task_id] = TaskPriority.NORMAL
            else:
                priority_config[task_id] = TaskPriority.LOW

        return priority_config

    def parse_user_priorities(self) -> Dict[str, TaskPriority]:
        """解析用户配置的优先级"""
        priority_config = {}

        for task in self.tasks:
            if task.task_id in self.user_priority_config:
                # 从用户配置中获取优先级
                priority_str = self.user_priority_config[task.task_id]
                if priority_str.upper() == "CRITICAL":
                    priority_config[task.task_id] = TaskPriority.CRITICAL
                elif priority_str.upper() == "HIGH":
                    priority_config[task.task_id] = TaskPriority.HIGH
                elif priority_str.upper() == "NORMAL":
                    priority_config[task.task_id] = TaskPriority.NORMAL
                elif priority_str.upper() == "LOW":
                    priority_config[task.task_id] = TaskPriority.LOW
                else:
                    # 如果用户配置的优先级无效，使用默认值
                    priority_config[task.task_id] = TaskPriority.NORMAL
            else:
                # 如果用户没有配置这个任务的优先级，使用默认值
                priority_config[task.task_id] = TaskPriority.NORMAL

        return priority_config

    def evaluate_configuration(self, priority_config: Dict[str, TaskPriority]) -> OptimizationResult:
        """评估一个优先级配置"""
        # 应用优先级配置
        for task in self.tasks:
            task.priority = priority_config[task.task_id]

        # 创建资源
        queue_manager = ResourceQueueManager()
        resources = self.resources.get("resources", [
            {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 160.0},
            {"resource_id": "DSP_0", "resource_type": "DSP", "bandwidth": 160.0}
        ])

        for res in resources:
            queue_manager.add_resource(res["resource_id"], ResourceType[res["resource_type"]], res["bandwidth"])

        tracer = ScheduleTracer(queue_manager)

        if self.segment_mode:
            launcher = EnhancedTaskLauncher(queue_manager, tracer)
        else:
            launcher = TaskLauncher(queue_manager, tracer)

        # 注册任务
        for task in self.tasks:
            launcher.register_task(task)

        # 创建并执行计划
        plan = launcher.create_launch_plan(self.time_window, self.launch_strategy)
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, self.time_window, segment_mode=self.segment_mode)

        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(self.time_window, plan.events)

        # 收集详细分析数据
        fps_analysis, power_analysis, ddr_analysis = self._collect_detailed_analysis(evaluator)

        # 收集满足情况
        fps_satisfaction = {}
        latency_satisfaction = {}
        total_satisfied = 0

        for task_id, task_metrics in evaluator.task_metrics.items():
            fps_satisfaction[task_id] = task_metrics.fps_satisfaction
            latency_satisfaction[task_id] = task_metrics.latency_satisfaction_rate > 0.9

            if fps_satisfaction[task_id] and latency_satisfaction[task_id]:
                total_satisfied += 1

        satisfaction_rate = total_satisfied / len(evaluator.task_metrics)

        # 计算系统利用率
        npu_utilization = metrics.avg_npu_utilization / 100.0
        dsp_utilization = metrics.avg_dsp_utilization / 100.0
        system_utilization = (1 - (1 - npu_utilization) * (1 - dsp_utilization)) * 100.0

        return OptimizationResult(
            iteration=len(self.optimization_history),
            priority_config={k: v.name for k, v in priority_config.items()},
            fps_satisfaction=fps_satisfaction,
            latency_satisfaction=latency_satisfaction,
            total_satisfaction_rate=satisfaction_rate,
            avg_latency=metrics.avg_latency,
            resource_utilization={
                'NPU': metrics.avg_npu_utilization,
                'DSP': metrics.avg_dsp_utilization
            },
            fps_analysis=fps_analysis,
            power_analysis=power_analysis,
            ddr_analysis=ddr_analysis,
            system_utilization=system_utilization
        )

    def _collect_detailed_analysis(self, evaluator) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """收集详细分析数据"""
        fps_analysis = {}
        power_analysis = {}
        ddr_analysis = {}

        total_fps = 0.0
        total_power = 0.0  # mW
        total_ddr = 0.0    # MB
        total_segment_executions = 0

        # 分析每个任务
        for task_id, task_metrics in evaluator.task_metrics.items():
            task = next((t for t in self.tasks if t.task_id == task_id), None)
            if not task:
                continue

            frames_per_second = task_metrics.achieved_fps
            total_fps += frames_per_second

            # 功耗和DDR分析
            task_power_per_frame = sum(segment.power for segment in task.segments)
            task_ddr_per_frame = sum(segment.ddr for segment in task.segments)
            task_total_power = task_power_per_frame * frames_per_second
            task_total_ddr = task_ddr_per_frame * frames_per_second

            total_power += task_total_power
            total_ddr += task_total_ddr

            fps_analysis[task_id] = frames_per_second
            power_analysis[task_id] = task_total_power
            ddr_analysis[task_id] = task_total_ddr

        # 计算段执行总数
        for resource_metrics in evaluator.resource_metrics.values():
            total_segment_executions += resource_metrics.segment_executions

        # 添加总计
        fps_analysis["total_fps"] = total_fps
        power_analysis["total_power_mw"] = total_power
        power_analysis["total_power_w"] = total_power / 1000.0
        ddr_analysis["total_ddr_mb"] = total_ddr
        ddr_analysis["total_ddr_gb"] = total_ddr / 1024.0
        fps_analysis["total_segment_executions"] = total_segment_executions

        return fps_analysis, power_analysis, ddr_analysis

    def adjust_priorities(self, current_config: Dict[str, TaskPriority],
                         result: OptimizationResult) -> Dict[str, TaskPriority]:
        """根据评估结果调整优先级"""
        new_config = current_config.copy()

        # 收集不满足要求的任务
        unsatisfied_tasks = []
        for task_id in current_config:
            fps_ok = result.fps_satisfaction.get(task_id, False)
            latency_ok = result.latency_satisfaction.get(task_id, False)

            if not fps_ok or not latency_ok:
                unsatisfied_tasks.append({
                    'task_id': task_id,
                    'fps_ok': fps_ok,
                    'latency_ok': latency_ok,
                    'current_priority': current_config[task_id]
                })

        # 调整策略
        for task_info in unsatisfied_tasks:
            task_id = task_info['task_id']
            current_priority = task_info['current_priority']

            # 尝试提升优先级
            current_index = self.priority_levels.index(current_priority)
            if current_index < len(self.priority_levels) - 1:
                new_config[task_id] = self.priority_levels[current_index + 1]
            else:
                # 已经是最高优先级，尝试降低其他任务优先级
                for other_id, other_priority in current_config.items():
                    if other_id != task_id:
                        other_fps_ok = result.fps_satisfaction.get(other_id, False)
                        other_latency_ok = result.latency_satisfaction.get(other_id, False)

                        if other_fps_ok and other_latency_ok:
                            other_index = self.priority_levels.index(other_priority)
                            if other_index > 0 and other_index >= current_index:
                                new_config[other_id] = self.priority_levels[other_index - 1]
                                break

        # 添加随机性避免局部最优
        if random.random() < 0.1:
            random_task = random.choice(list(new_config.keys()))
            new_config[random_task] = random.choice(self.priority_levels)

        return new_config

    def optimize(self, max_iterations=50, max_time_seconds=300, target_satisfaction=1.0) -> Tuple[Dict[str, TaskPriority], OptimizationResult]:
        """执行优化过程"""
        start_time = time.time()

        # 根据search_priority设置决定是否进行优先级搜索
        if not self.search_priority:
            # 如果不进行优先级搜索，直接使用用户配置的优先级
            current_config = self.parse_user_priorities()
            result = self.evaluate_configuration(current_config)
            self.optimization_history.append(result)
            print(f"[INFO] 使用用户配置的优先级，跳过优化搜索")
            return current_config, result

        # 进行优先级搜索优化
        print(f"[INFO] 启用优先级搜索优化，最大迭代次数: {max_iterations}")

        # 生成初始配置
        current_config = self.generate_initial_priorities()
        best_result = None
        best_config = current_config.copy()

        iteration = 0
        while iteration < max_iterations:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time_seconds:
                break

            # 评估当前配置
            result = self.evaluate_configuration(current_config)
            self.optimization_history.append(result)

            # 更新最佳结果
            if best_result is None or result.total_satisfaction_rate > best_result.total_satisfaction_rate:
                best_result = result
                best_config = current_config.copy()

            # 检查是否达到目标
            if result.total_satisfaction_rate >= target_satisfaction:
                break

            # 调整优先级
            current_config = self.adjust_priorities(current_config, result)
            iteration += 1

        return best_config, best_result
