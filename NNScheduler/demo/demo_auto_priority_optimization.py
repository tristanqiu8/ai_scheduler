#!/usr/bin/env python3
"""
自动化优先级配置优化器
通过迭代调整任务优先级，直到满足所有任务的FPS和延迟要求
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NNScheduler.core.resource_queue import ResourceQueueManager
from NNScheduler.core.schedule_tracer import ScheduleTracer
from NNScheduler.core.launcher import TaskLauncher
from NNScheduler.core.enhanced_launcher import EnhancedTaskLauncher
from NNScheduler.core.executor import ScheduleExecutor
from NNScheduler.core.enums import ResourceType, TaskPriority, SegmentationStrategy
from NNScheduler.core.evaluator import PerformanceEvaluator
from NNScheduler.scenario.hybrid_task import create_real_tasks
from NNScheduler.viz.schedule_visualizer import ScheduleVisualizer
import numpy as np
import random
import time
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json
from collections import defaultdict


@dataclass
class OptimizationResult:
    """优化结果"""
    iteration: int
    priority_config: Dict[str, TaskPriority]
    fps_satisfaction: Dict[str, bool]
    latency_satisfaction: Dict[str, bool]
    total_satisfaction_rate: float
    avg_latency: float
    resource_utilization: Dict[str, float]


class PriorityOptimizer:
    """任务优先级自动优化器"""
    
    def __init__(self, tasks, time_window=1000.0, segment_mode=True):
        self.tasks = tasks
        self.time_window = time_window
        self.segment_mode = segment_mode
        
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
                # 计算延迟严格度（延迟要求相对于理论执行时间的比例）
                'latency_strictness': self._calculate_latency_strictness(task),
                # 计算FPS严格度（相对于其他任务的FPS要求）
                'fps_strictness': task.fps_requirement
            }
        
        return features
    
    def _calculate_latency_strictness(self, task) -> float:
        """计算延迟严格度"""
        # 估算任务在40GB/s带宽下的执行时间
        bandwidth_map = {ResourceType.NPU: 40.0, ResourceType.DSP: 40.0}
        estimated_duration = task.estimate_duration(bandwidth_map)
        
        # 延迟要求与执行时间的比例，越小越严格
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
            score += 10  # 混合任务优先级更高
        elif features['num_segments'] > 5:
            score += 5   # 多段任务优先级较高
        
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
            if i < num_tasks * 0.1:  # 前10%设为CRITICAL
                priority_config[task_id] = TaskPriority.CRITICAL
            elif i < num_tasks * 0.3:  # 前30%设为HIGH
                priority_config[task_id] = TaskPriority.HIGH
            elif i < num_tasks * 0.7:  # 前70%设为NORMAL
                priority_config[task_id] = TaskPriority.NORMAL
            else:  # 其余设为LOW
                priority_config[task_id] = TaskPriority.LOW
        
        print("\n[SUCCESS] 初始优先级配置（基于任务特征）:")
        self._print_priority_config(priority_config)
        
        return priority_config
    
    def _print_priority_config(self, config: Dict[str, TaskPriority]):
        """打印优先级配置"""
        print("-" * 80)
        print(f"{'任务ID':<10} {'任务名':<15} {'优先级':<10} {'被依赖':<8} {'FPS要求':<10} {'延迟要求':<12}")
        print("-" * 80)
        
        for task_id, priority in sorted(config.items()):
            features = self.task_features[task_id]
            print(f"{task_id:<10} {features['name']:<15} {priority.name:<10} "
                  f"{features['dependency_count']:<8} {features['fps_requirement']:<10} "
                  f"{features['latency_requirement']:<12.1f}")
    
    def evaluate_configuration(self, priority_config: Dict[str, TaskPriority]) -> OptimizationResult:
        """评估一个优先级配置"""
        # 应用优先级配置
        for task in self.tasks:
            task.priority = priority_config[task.task_id]
        
        # 创建资源和调度器
        queue_manager = ResourceQueueManager()
        queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
        queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
        
        tracer = ScheduleTracer(queue_manager)
        
        if self.segment_mode:
            launcher = EnhancedTaskLauncher(queue_manager, tracer)
        else:
            launcher = TaskLauncher(queue_manager, tracer)
        
        # 注册任务
        for task in self.tasks:
            launcher.register_task(task)
        
        # 创建并执行计划
        plan = launcher.create_launch_plan(self.time_window, "balanced")
        executor = ScheduleExecutor(queue_manager, tracer, launcher.tasks)
        stats = executor.execute_plan(plan, self.time_window, segment_mode=self.segment_mode)
        
        # 评估性能
        evaluator = PerformanceEvaluator(tracer, launcher.tasks, queue_manager)
        metrics = evaluator.evaluate(self.time_window, plan.events)
        
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
        
        return OptimizationResult(
            iteration=len(self.optimization_history),
            priority_config=priority_config.copy(),
            fps_satisfaction=fps_satisfaction,
            latency_satisfaction=latency_satisfaction,
            total_satisfaction_rate=satisfaction_rate,
            avg_latency=metrics.avg_latency,
            resource_utilization={
                'NPU': metrics.avg_npu_utilization,
                'DSP': metrics.avg_dsp_utilization
            }
        )
    
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
                # 提升一级
                new_config[task_id] = self.priority_levels[current_index + 1]
            else:
                # 已经是最高优先级，尝试降低其他任务优先级
                # 找到满足要求且优先级高的任务
                for other_id, other_priority in current_config.items():
                    if other_id != task_id:
                        other_fps_ok = result.fps_satisfaction.get(other_id, False)
                        other_latency_ok = result.latency_satisfaction.get(other_id, False)
                        
                        if other_fps_ok and other_latency_ok:
                            other_index = self.priority_levels.index(other_priority)
                            if other_index > 0 and other_index >= current_index:
                                # 降低优先级
                                new_config[other_id] = self.priority_levels[other_index - 1]
                                break
        
        # 添加一些随机性避免局部最优
        if random.random() < 0.1:  # 10%概率随机调整
            random_task = random.choice(list(new_config.keys()))
            new_config[random_task] = random.choice(self.priority_levels)
        
        return new_config
    
    def optimize(self, max_iterations=50, max_time_seconds=300, target_satisfaction=1.0):
        """执行优化过程"""
        print(f"\n[DEMO] 开始自动优先级优化")
        print(f"  最大迭代次数: {max_iterations}")
        print(f"  最大运行时间: {max_time_seconds}秒")
        print(f"  目标满足率: {target_satisfaction*100}%")
        
        start_time = time.time()
        
        # 生成初始配置
        current_config = self.generate_initial_priorities()
        best_result = None
        best_config = current_config.copy()
        
        iteration = 0
        while iteration < max_iterations:
            elapsed_time = time.time() - start_time
            if elapsed_time > max_time_seconds:
                print(f"\n⏰ 达到时间限制 ({max_time_seconds}秒)")
                break
            
            # 评估当前配置
            print(f"\n[ANALYSIS] 迭代 {iteration + 1}:")
            result = self.evaluate_configuration(current_config)
            self.optimization_history.append(result)
            
            # 打印进度
            print(f"  满足率: {result.total_satisfaction_rate:.1%}")
            print(f"  平均延迟: {result.avg_latency:.1f}ms")
            print(f"  资源利用率: NPU={result.resource_utilization['NPU']:.1f}%, "
                  f"DSP={result.resource_utilization['DSP']:.1f}%")
            
            # 更新最佳结果
            if best_result is None or result.total_satisfaction_rate > best_result.total_satisfaction_rate:
                best_result = result
                best_config = current_config.copy()
                print(f"  [COMPLETE] 发现更好的配置！")
            
            # 检查是否达到目标
            if result.total_satisfaction_rate >= target_satisfaction:
                print(f"\n🎉 达到目标满足率！")
                break
            
            # 调整优先级
            current_config = self.adjust_priorities(current_config, result)
            iteration += 1
        
        print(f"\n✅ 优化完成！共迭代 {iteration + 1} 次，耗时 {time.time() - start_time:.1f}秒")
        
        return best_config, best_result
    
    def print_optimization_summary(self, best_config: Dict[str, TaskPriority], 
                                 best_result: OptimizationResult):
        """打印优化结果摘要"""
        print("\n" + "=" * 100)
        print("[ANALYSIS] 优化结果摘要")
        print("=" * 100)
        
        print(f"\n最佳配置（满足率: {best_result.total_satisfaction_rate:.1%}）:")
        print("-" * 100)
        print(f"{'任务ID':<10} {'任务名':<15} {'优先级':<10} {'FPS满足':<10} {'延迟满足':<10}")
        print("-" * 100)
        
        for task_id, priority in sorted(best_config.items()):
            features = self.task_features[task_id]
            fps_ok = "[OK]" if best_result.fps_satisfaction.get(task_id, False) else "[FAIL]"
            latency_ok = "[OK]" if best_result.latency_satisfaction.get(task_id, False) else "[FAIL]"
            
            print(f"{task_id:<10} {features['name']:<15} {priority.name:<10} "
                  f"{fps_ok:<10} {latency_ok:<10}")
        
        # 打印优化历史
        print(f"\n优化历史（共{len(self.optimization_history)}次迭代）:")
        print("-" * 60)
        print(f"{'迭代':<6} {'满足率':<10} {'平均延迟':<12} {'NPU利用率':<12} {'DSP利用率':<12}")
        print("-" * 60)
        
        for i, result in enumerate(self.optimization_history[-10:]):  # 只显示最后10次
            print(f"{i+1:<6} {result.total_satisfaction_rate:<10.1%} "
                  f"{result.avg_latency:<12.1f} "
                  f"{result.resource_utilization['NPU']:<12.1f} "
                  f"{result.resource_utilization['DSP']:<12.1f}")
        
        # 保存最佳配置
        self.save_best_configuration(best_config, best_result)
    
    def save_best_configuration(self, config: Dict[str, TaskPriority], 
                               result: OptimizationResult):
        """保存最佳配置到文件"""
        output = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'satisfaction_rate': result.total_satisfaction_rate,
            'avg_latency': result.avg_latency,
            'resource_utilization': result.resource_utilization,
            'priority_config': {k: v.name for k, v in config.items()},
            'fps_satisfaction': result.fps_satisfaction,
            'latency_satisfaction': result.latency_satisfaction
        }
        
        filename = f"optimized_priority_config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 最佳配置已保存到: {filename}")


def main():
    """主函数"""
    print("=" * 100)
    print("自动化优先级配置优化")
    print("=" * 100)
    
    # 创建任务
    tasks = create_real_tasks()
    
    # 创建优化器
    optimizer = PriorityOptimizer(tasks, time_window=1000.0, segment_mode=True)
    
    # 执行优化
    best_config, best_result = optimizer.optimize(
        max_iterations=50,      # 最多迭代50次
        max_time_seconds=300,   # 最多运行5分钟
        target_satisfaction=0.95  # 目标95%任务满足要求
    )
    
    # 打印结果
    optimizer.print_optimization_summary(best_config, best_result)
    
    # 可选：使用最佳配置运行详细分析
    print("\n\n[DETAIL] 使用最佳配置运行详细分析...")
    
    # 应用最佳配置
    for task in tasks:
        task.priority = best_config[task.task_id]
    
    # 运行详细测试
    from demo_hybrid_task import test_scheduling_modes, analyze_latency_performance
    results = test_scheduling_modes(1000.0)
    analyze_latency_performance(results)


if __name__ == "__main__":
    main()