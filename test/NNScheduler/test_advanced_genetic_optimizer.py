#!/usr/bin/env python3
"""
高级遗传算法优化器 - 不仅优化发射时间，还优化任务优先级和分段策略
"""

import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
import time
import copy

from NNScheduler.core import (
    ResourceType, TaskPriority, SegmentationStrategy,
    ResourceQueueManager, ScheduleTracer,
    TaskLauncher, ScheduleExecutor,
    PerformanceEvaluator, LaunchPlan,
    NNTask
)
from NNScheduler.core.artifacts import ensure_artifact_path

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


@dataclass
class AdvancedGene:
    """高级基因 - 包含发射时间、优先级和分段配置"""
    task_id: str
    launch_delay: float  # 发射延迟
    priority_adjustment: int  # 优先级调整 (-1, 0, 1)
    segmentation_config: int  # 分段配置索引
    
    def __hash__(self):
        return hash((self.task_id, self.launch_delay, self.priority_adjustment, self.segmentation_config))


@dataclass
class AdvancedIndividual:
    """高级个体 - 完整的优化策略"""
    genes: Dict[str, AdvancedGene] = field(default_factory=dict)
    fitness: float = -float('inf')
    total_idle_time: float = 0.0  # 总空闲时间（不仅末尾）
    tail_idle_time: float = 0.0  # 末尾空闲时间
    npu_idle_time: float = 0.0  # NPU空闲时间
    dsp_idle_time: float = 0.0  # DSP空闲时间
    completion_time: float = 200.0
    fps_satisfaction: float = 0.0
    resource_balance: float = 0.0
    
    def __lt__(self, other):
        return self.fitness < other.fitness


class AdvancedGeneticOptimizer:
    """高级遗传算法优化器"""
    
    def __init__(self,
                 launcher: TaskLauncher,
                 queue_manager: ResourceQueueManager,
                 time_window: float = 200.0):
        self.launcher = launcher
        self.queue_manager = queue_manager
        self.time_window = time_window
        
        # 遗传算法参数
        self.population_size = 100
        self.elite_size = 10
        self.mutation_rate = 0.3
        self.crossover_rate = 0.85
        self.max_generations = 50
        
        # 分析任务特征
        self._analyze_tasks()
        
        # 优化历史
        self.best_individual = None
        self.generation_history = []
        
    def _analyze_tasks(self):
        """分析任务特征，找出可优化的任务"""
        self.segmentable_tasks = {}  # 可分段的任务
        self.priority_adjustable = {}  # 可调整优先级的任务
        
        for task_id, task in self.launcher.tasks.items():
            # 检查是否可分段
            if task.segmentation_strategy != SegmentationStrategy.NO_SEGMENTATION:
                # 检查是否有预设配置
                for seg_id in task.preset_cut_configurations:
                    configs = task.preset_cut_configurations[seg_id]
                    if len(configs) > 1:  # 有多个配置选项
                        self.segmentable_tasks[task_id] = {
                            'segment_id': seg_id,
                            'num_configs': len(configs),
                            'configs': configs
                        }
                        break
            
            # 检查优先级是否可调整（非CRITICAL任务可以调整）
            if task.priority != TaskPriority.CRITICAL:
                self.priority_adjustable[task_id] = True
        
        print(f"\n[ANALYSIS] 任务分析:")
        print(f"  可分段任务: {list(self.segmentable_tasks.keys())}")
        print(f"  可调整优先级任务: {list(self.priority_adjustable.keys())}")
    
    def _create_random_individual(self) -> AdvancedIndividual:
        """创建随机个体"""
        individual = AdvancedIndividual()
        
        for task_id, task in self.launcher.tasks.items():
            # 发射延迟
            if task.priority == TaskPriority.CRITICAL:
                launch_delay = random.uniform(0, 2)
            elif task.priority == TaskPriority.HIGH:
                launch_delay = random.uniform(0, 10)
            else:
                launch_delay = random.uniform(0, 30)
            
            # 优先级调整 - 更保守的策略
            priority_adj = 0
            if task_id in self.priority_adjustable:
                # 10%概率调整优先级，避免过度调整
                if random.random() < 0.1:
                    # 倾向于提升优先级而不是降低
                    priority_adj = random.choice([0, 0, 1, -1])  # 75%概率不变或提升
            
            # 分段配置 - T2和T3不参与随机，其他任务保守策略
            seg_config = 0
            if task_id in self.segmentable_tasks and task_id not in ['T2', 'T3']:
                # 20%概率使用分段
                if random.random() < 0.2:
                    num_configs = self.segmentable_tasks[task_id]['num_configs']
                    seg_config = random.randint(1, min(2, num_configs - 1))
            
            gene = AdvancedGene(
                task_id=task_id,
                launch_delay=launch_delay,
                priority_adjustment=priority_adj,
                segmentation_config=seg_config
            )
            individual.genes[task_id] = gene
        
        return individual
    
    def _apply_individual_to_tasks(self, individual: AdvancedIndividual) -> Dict[str, NNTask]:
        """应用个体的基因到任务，返回修改后的任务副本"""
        modified_tasks = {}
        
        for task_id, gene in individual.genes.items():
            # 深拷贝原始任务
            task_copy = copy.deepcopy(self.launcher.tasks[task_id])
            
            # 应用优先级调整
            if gene.priority_adjustment != 0:
                current_priority_value = task_copy.priority.value
                new_priority_value = max(0, min(3, current_priority_value + gene.priority_adjustment))
                # 转换回优先级枚举
                for priority in TaskPriority:
                    if priority.value == new_priority_value:
                        task_copy.priority = priority
                        break
            
            # 强制T2和T3使用最大分段
            if task_id in ['T2', 'T3']:
                if task_id in self.segmentable_tasks:
                    seg_id = self.segmentable_tasks[task_id]['segment_id']
                    num_configs = self.segmentable_tasks[task_id]['num_configs']
                    # 使用最大分段配置
                    task_copy.select_cut_configuration(seg_id, num_configs - 1)
            else:
                # 其他任务按基因配置
                if task_id in self.segmentable_tasks and gene.segmentation_config > 0:
                    seg_id = self.segmentable_tasks[task_id]['segment_id']
                    task_copy.select_cut_configuration(seg_id, gene.segmentation_config)
            
            modified_tasks[task_id] = task_copy
        
        return modified_tasks
    
    def _calculate_resource_idle_times(self, tracer: ScheduleTracer) -> Tuple[float, float, float]:
        """计算资源空闲时间 - 正确处理并行执行"""
        # 收集每个资源的执行时间段
        npu_segments = []
        dsp_segments = []
        
        for execution in tracer.executions:
            segment = (execution.start_time, execution.end_time)
            if "NPU" in execution.resource_id:
                npu_segments.append(segment)
            elif "DSP" in execution.resource_id:
                dsp_segments.append(segment)
        
        # 合并重叠的时间段
        def merge_segments(segments):
            if not segments:
                return []
            
            # 按开始时间排序
            segments.sort(key=lambda x: x[0])
            merged = [segments[0]]
            
            for start, end in segments[1:]:
                last_start, last_end = merged[-1]
                if start <= last_end:
                    # 重叠，合并
                    merged[-1] = (last_start, max(last_end, end))
                else:
                    # 不重叠，添加新段
                    merged.append((start, end))
            
            return merged
        
        # 合并重叠段
        npu_merged = merge_segments(npu_segments)
        dsp_merged = merge_segments(dsp_segments)
        
        # 计算实际忙碌时间
        npu_busy = sum(end - start for start, end in npu_merged)
        dsp_busy = sum(end - start for start, end in dsp_merged)
        
        # 计算空闲时间
        npu_idle = self.time_window - npu_busy
        dsp_idle = self.time_window - dsp_busy
        total_idle = npu_idle + dsp_idle
        
        return total_idle, npu_idle, dsp_idle
    
    def _evaluate_individual(self, individual: AdvancedIndividual) -> None:
        """评估个体"""
        # 创建独立的执行环境
        eval_queue_manager = ResourceQueueManager()
        eval_queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
        eval_queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
        
        eval_tracer = ScheduleTracer(eval_queue_manager)
        
        # 应用个体的修改到任务
        modified_tasks = self._apply_individual_to_tasks(individual)
        
        # 创建新的launcher，使用修改后的任务
        eval_launcher = TaskLauncher(eval_queue_manager, eval_tracer)
        for task_id, task in modified_tasks.items():
            eval_launcher.register_task(task)
        
        # 创建发射计划
        plan = self._create_launch_plan_with_delays(eval_launcher, individual)
        
        # 执行计划
        executor = ScheduleExecutor(eval_queue_manager, eval_tracer, eval_launcher.tasks)
        stats = executor.execute_plan(plan, self.time_window)
        
        # 评估性能
        evaluator = PerformanceEvaluator(eval_tracer, eval_launcher.tasks, eval_queue_manager)
        metrics = evaluator.evaluate(self.time_window, plan.events)
        
        # 计算各种空闲时间
        total_idle, npu_idle, dsp_idle = self._calculate_resource_idle_times(eval_tracer)
        
        # 计算末尾空闲时间
        last_completion = 0.0
        for execution in eval_tracer.executions:
            last_completion = max(last_completion, execution.end_time)
        tail_idle = max(0, self.time_window - last_completion)
        
        # 更新个体属性
        individual.total_idle_time = total_idle
        individual.npu_idle_time = npu_idle
        individual.dsp_idle_time = dsp_idle
        individual.tail_idle_time = tail_idle
        individual.completion_time = last_completion
        individual.fps_satisfaction = metrics.fps_satisfaction_rate
        individual.resource_balance = 1 - abs(npu_idle - dsp_idle) / self.time_window
        
        # 计算适应度 - 只关注空闲时间
        if metrics.fps_satisfaction_rate < 95:  # 提高FPS要求
            # FPS不满足，严重惩罚
            individual.fitness = -10000 * (95 - metrics.fps_satisfaction_rate)
        else:
            # 唯一目标：最大化NPU和DSP的空闲时间
            # 使用NPU和DSP空闲时间的最小值，确保两者都有充足空闲
            min_idle = min(npu_idle, dsp_idle)
            individual.fitness = (
                min_idle * 100 +  # 最小空闲时间权重最高
                total_idle * 50   # 总空闲时间也重要
            )
    
    def _create_launch_plan_with_delays(self, launcher: TaskLauncher, individual: AdvancedIndividual) -> LaunchPlan:
        """创建带延迟的发射计划"""
        plan = LaunchPlan()
        
        for task_id in launcher.tasks:
            task = launcher.tasks[task_id]
            gene = individual.genes[task_id]
            
            # 基础发射时间加上延迟
            launch_time = gene.launch_delay
            
            # 根据FPS计算实例
            period = 1000.0 / task.fps_requirement
            instance_id = 0
            
            current_time = launch_time
            while current_time < self.time_window:
                plan.add_launch(task_id, current_time, instance_id)
                instance_id += 1
                current_time += period
        
        plan.sort_events()
        return plan
    
    def _crossover(self, parent1: AdvancedIndividual, parent2: AdvancedIndividual) -> AdvancedIndividual:
        """交叉操作"""
        child = AdvancedIndividual()
        
        for task_id in self.launcher.tasks:
            if random.random() < 0.5:
                # 继承parent1
                gene = parent1.genes[task_id]
            else:
                # 继承parent2
                gene = parent2.genes[task_id]
            
            # 深拷贝基因
            child.genes[task_id] = AdvancedGene(
                task_id=task_id,
                launch_delay=gene.launch_delay,
                priority_adjustment=gene.priority_adjustment,
                segmentation_config=gene.segmentation_config
            )
        
        return child
    
    def _mutate(self, individual: AdvancedIndividual) -> None:
        """变异操作"""
        for task_id in self.launcher.tasks:
            if random.random() < self.mutation_rate:
                gene = individual.genes[task_id]
                
                # 变异类型
                mutation_type = random.choice(['delay', 'priority', 'segment', 'all'])
                
                if mutation_type in ['delay', 'all']:
                    # 延迟变异
                    if random.random() < 0.5:
                        gene.launch_delay *= random.uniform(0.5, 0.9)
                    else:
                        gene.launch_delay *= random.uniform(1.1, 1.5)
                    gene.launch_delay = max(0, min(50, gene.launch_delay))
                
                if mutation_type in ['priority', 'all']:
                    # 优先级变异
                    if task_id in self.priority_adjustable:
                        gene.priority_adjustment = random.choice([-1, 0, 1])
                
                if mutation_type in ['segment', 'all']:
                    # 分段配置变异 - T2和T3不变异
                    if task_id in self.segmentable_tasks and task_id not in ['T2', 'T3']:
                        num_configs = self.segmentable_tasks[task_id]['num_configs']
                        gene.segmentation_config = random.randint(0, min(2, num_configs - 1))
    
    def optimize(self) -> AdvancedIndividual:
        """运行优化 - 记录空闲时间最长的个体"""
        print("\n🧬 高级遗传算法优化器启动")
        print(f"  目标: 最大化NPU和DSP的空闲时间")
        print(f"  策略: T2和T3强制最大分段")
        print(f"  种群: {self.population_size}, 代数: {self.max_generations}")
        
        start_time = time.time()
        
        # 初始化种群
        population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual()
            self._evaluate_individual(individual)
            population.append(individual)
        
        # 找出初始最佳（基于总空闲时间）
        best_by_idle = max(population, key=lambda x: x.total_idle_time)
        self.best_individual = best_by_idle
        
        print(f"\n初始最佳: NPU空闲={best_by_idle.npu_idle_time:.1f}ms, "
              f"DSP空闲={best_by_idle.dsp_idle_time:.1f}ms, "
              f"总空闲={best_by_idle.total_idle_time:.1f}ms")
        
        # 进化循环
        for generation in range(self.max_generations):
            # 根据适应度排序
            population.sort(reverse=True)
            
            # 精英保留
            new_population = population[:self.elite_size]
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 锦标赛选择
                tournament = random.sample(population[:self.population_size//2], 5)
                parent1 = max(tournament)
                
                tournament = random.sample(population[:self.population_size//2], 5)
                parent2 = max(tournament)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = self._create_random_individual()
                
                # 变异
                self._mutate(child)
                
                # 评估
                self._evaluate_individual(child)
                new_population.append(child)
            
            # 更新种群
            population = new_population
            
            # 找出本代空闲时间最长的个体
            current_best_by_idle = max(population, key=lambda x: x.total_idle_time)
            
            # 如果本代的最佳个体空闲时间更长，更新全局最佳
            if current_best_by_idle.total_idle_time > self.best_individual.total_idle_time:
                self.best_individual = current_best_by_idle
                print(f"\n代 {generation+1}: 新最佳! "
                      f"NPU空闲={self.best_individual.npu_idle_time:.1f}ms, "
                      f"DSP空闲={self.best_individual.dsp_idle_time:.1f}ms, "
                      f"总空闲={self.best_individual.total_idle_time:.1f}ms")
            
            # 记录历史
            self.generation_history.append({
                'generation': generation,
                'best_fitness': population[0].fitness,
                'best_idle_time': current_best_by_idle.total_idle_time,
                'best_npu_idle': current_best_by_idle.npu_idle_time,
                'best_dsp_idle': current_best_by_idle.dsp_idle_time
            })
        
        elapsed_time = time.time() - start_time
        
        print(f"\n[COMPLETE] 优化完成!")
        print(f"  用时: {elapsed_time:.2f}秒")
        print(f"  最佳个体（基于空闲时间）:")
        print(f"    NPU空闲: {self.best_individual.npu_idle_time:.1f}ms ({self.best_individual.npu_idle_time/200*100:.1f}%)")
        print(f"    DSP空闲: {self.best_individual.dsp_idle_time:.1f}ms ({self.best_individual.dsp_idle_time/200*100:.1f}%)")
        print(f"    总空闲: {self.best_individual.total_idle_time:.1f}ms")
        print(f"    FPS满足率: {self.best_individual.fps_satisfaction:.1f}%")
        
        return self.best_individual
    
    def visualize_optimization(self, baseline_tracer: ScheduleTracer, optimized_tracer: ScheduleTracer):
        """可视化优化效果"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. 基线调度
        self._plot_schedule(axes[0], baseline_tracer, "Baseline Schedule (Eager Strategy)")
        
        # 2. 优化后调度
        self._plot_schedule(axes[1], optimized_tracer, "Optimized Schedule (Advanced GA)")
        
        # 3. 空闲时间对比
        ax3 = axes[2]
        
        # 计算基线空闲时间
        baseline_total, baseline_npu, baseline_dsp = self._calculate_resource_idle_times(baseline_tracer)
        
        # 优化后数据
        opt_total = self.best_individual.total_idle_time
        opt_npu = self.best_individual.npu_idle_time
        opt_dsp = self.best_individual.dsp_idle_time
        
        # 绘制对比柱状图
        x = np.arange(3)
        width = 0.35
        
        baseline_values = [baseline_total, baseline_npu, baseline_dsp]
        optimized_values = [opt_total, opt_npu, opt_dsp]
        
        bars1 = ax3.bar(x - width/2, baseline_values, width, label='Baseline', color='lightcoral')
        bars2 = ax3.bar(x + width/2, optimized_values, width, label='Optimized', color='lightgreen')
        
        # 添加数值标签
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}ms',
                        ha='center', va='bottom')
        
        ax3.set_ylabel('Idle Time (ms)')
        ax3.set_title('Idle Time Comparison')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Total Idle', 'NPU Idle', 'DSP Idle'])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = ensure_artifact_path("advanced_optimization_result.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"\n✅ 可视化已保存: {output_path}")
    
    def _plot_schedule(self, ax, tracer: ScheduleTracer, title: str):
        """绘制调度甘特图"""
        all_resources = ['NPU_0', 'DSP_0']
        timeline = tracer.get_timeline()
        
        # 设置Y轴
        y_positions = {res: i for i, res in enumerate(all_resources)}
        ax.set_yticks(range(len(all_resources)))
        ax.set_yticklabels(all_resources)
        
        # 颜色映射
        priority_colors = {
            TaskPriority.CRITICAL: '#FF4444',
            TaskPriority.HIGH: '#FF8844', 
            TaskPriority.NORMAL: '#4488FF',
            TaskPriority.LOW: '#888888'
        }
        
        # 绘制任务块
        for resource_id, executions in timeline.items():
            y_pos = y_positions.get(resource_id, 0)
            
            for exec in executions:
                color = priority_colors.get(exec.priority, '#4488FF')
                
                # 创建矩形
                rect = Rectangle(
                    (exec.start_time, y_pos - 0.3),
                    exec.duration,
                    0.6,
                    facecolor=color,
                    edgecolor='black',
                    linewidth=1
                )
                ax.add_patch(rect)
                
                # 添加标签（只对较宽的任务）
                if exec.duration > 10:
                    parts = exec.task_id.split('#')
                    if len(parts) > 0:
                        label = parts[0]
                        ax.text(
                            exec.start_time + exec.duration / 2,
                            y_pos,
                            label,
                            ha='center',
                            va='center',
                            fontsize=8,
                            color='white' if exec.priority == TaskPriority.CRITICAL else 'black',
                            weight='bold' if exec.priority == TaskPriority.CRITICAL else 'normal'
                        )
        
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Resource')
        ax.set_title(title)
        ax.grid(True, axis='x', alpha=0.3)
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.5, len(all_resources) - 0.5)
        
        # 添加图例
        legend_elements = [
            patches.Patch(color=color, label=priority.name)
            for priority, color in priority_colors.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right')


def run_advanced_optimizer():
    """测试高级遗传算法优化器"""
    print("[DEMO] 测试高级遗传算法优化器")
    print("="*80)
    
    # 创建资源
    queue_manager = ResourceQueueManager()
    queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    tracer = ScheduleTracer(queue_manager)
    launcher = TaskLauncher(queue_manager, tracer)
    
    # 加载真实任务
    from NNScheduler.scenario.real_task import create_real_tasks
    tasks = create_real_tasks()
    
    for task in tasks:
        launcher.register_task(task)
    
    print(f"\n已加载 {len(tasks)} 个任务")
    
    # 创建高级优化器
    optimizer = AdvancedGeneticOptimizer(launcher, queue_manager, time_window=200.0)
    
    # 先执行基线策略作为对比
    print("\n执行基线策略...")
    baseline_plan = launcher.create_launch_plan(200.0, "eager")
    
    baseline_queue_manager = ResourceQueueManager()
    baseline_queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    baseline_queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    baseline_tracer = ScheduleTracer(baseline_queue_manager)
    baseline_executor = ScheduleExecutor(baseline_queue_manager, baseline_tracer, launcher.tasks)
    baseline_stats = baseline_executor.execute_plan(baseline_plan, 200.0)
    
    # 运行优化
    best_individual = optimizer.optimize()
    
    # 执行优化后的策略
    print("\n执行优化策略...")
    
    # 应用最佳个体的修改
    optimized_tasks = optimizer._apply_individual_to_tasks(best_individual)
    
    opt_queue_manager = ResourceQueueManager()
    opt_queue_manager.add_resource("NPU_0", ResourceType.NPU, 40.0)
    opt_queue_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    
    opt_tracer = ScheduleTracer(opt_queue_manager)
    opt_launcher = TaskLauncher(opt_queue_manager, opt_tracer)
    
    for task_id, task in optimized_tasks.items():
        opt_launcher.register_task(task)
    
    opt_plan = optimizer._create_launch_plan_with_delays(opt_launcher, best_individual)
    opt_executor = ScheduleExecutor(opt_queue_manager, opt_tracer, opt_launcher.tasks)
    opt_stats = opt_executor.execute_plan(opt_plan, 200.0)
    
    # 生成可视化
    optimizer.visualize_optimization(baseline_tracer, opt_tracer)
    
    # 打印优化细节
    print("\n" + "="*80)
    print("[ANALYSIS] 优化细节")
    print("="*80)
    
    print("\n任务优化情况:")
    for task_id, gene in best_individual.genes.items():
        task = launcher.tasks[task_id]
        print(f"\n{task_id} ({task.name}):")
        print(f"  原始优先级: {task.priority.name}")
        
        if gene.priority_adjustment != 0:
            new_priority_value = task.priority.value + gene.priority_adjustment
            for p in TaskPriority:
                if p.value == new_priority_value:
                    print(f"  新优先级: {p.name} (调整: {gene.priority_adjustment:+d})")
                    break
        
        if gene.launch_delay > 0.1:
            print(f"  发射延迟: {gene.launch_delay:.1f}ms")
        
        if task_id in optimizer.segmentable_tasks and gene.segmentation_config > 0:
            configs = optimizer.segmentable_tasks[task_id]['configs']
            config = configs[gene.segmentation_config]
            print(f"  分段配置: {len(config) + 1}段")
    
    return {
        'optimizer': optimizer,
        'best_individual': best_individual
    }


def test_advanced_optimizer():
    """Pytest 包装：确保优化器返回结果"""
    result = run_advanced_optimizer()
    assert result['best_individual'] is not None


if __name__ == "__main__":
    run_advanced_optimizer()
