#!/usr/bin/env python3
"""
修正的遗传算法优化器
- 保证FPS不下降
- 更合理的适应度函数
- 更智能的变异策略
"""
# 先导入枚举和基础类型
from core.enums import TaskPriority, RuntimeType, SegmentationStrategy

# 然后导入其他模块
from core.improved_genetic_optimizer import (
    ImprovedGeneticOptimizer, 
    calculate_detailed_utilization,
    print_detailed_utilization,
    analyze_fps_satisfaction,
    generate_comparison_visualization
)
from core.scheduler import MultiResourceScheduler
from scenario.real_task import create_real_tasks
from core.modular_scheduler_fixes import apply_basic_fixes
from core.genetic_task_optimizer import GeneticIndividual
from core.fixed_validation_and_metrics import validate_schedule_correctly
from collections import defaultdict
from viz.elegant_visualization import ElegantSchedulerVisualizer
import copy
import random

# 导入资源冲突修复
from core.minimal_fifo_fix_corrected import apply_minimal_fifo_fix
from core.strict_resource_conflict_fix import apply_strict_resource_conflict_fix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class FixedGeneticOptimizer(ImprovedGeneticOptimizer):
    """修正的遗传算法优化器"""
    
    def __init__(self, scheduler, tasks, time_window=200.0):
        super().__init__(scheduler, tasks, time_window)
        # 记录基线配置的详细信息
        self.baseline_config = None
        self.baseline_performance = None
        
    def set_baseline_performance(self, baseline_stats, baseline_conflicts):
        """设置基线性能指标"""
        self.baseline_performance = {
            'fps_rates': {tid: info['fps_rate'] 
                         for tid, info in baseline_stats['task_fps'].items()},
            'avg_fps': baseline_stats['total_fps_rate'] / len(self.tasks),
            'conflicts': baseline_conflicts,
            'task_counts': {tid: info['count'] 
                           for tid, info in baseline_stats['task_fps'].items()}
        }
        
        # 保存基线配置
        self.baseline_config = self._save_original_config()
        
    def _evaluate_fitness(self, individual: GeneticIndividual) -> float:
        """修正的适应度函数"""
        # 应用配置
        self._apply_individual_config(individual)
        
        # 清空调度历史
        self.scheduler.schedule_history.clear()
        
        # 运行调度
        try:
            results = self.scheduler.priority_aware_schedule_with_segmentation(self.time_window)
            
            # 验证调度结果
            is_valid, conflicts = validate_schedule_correctly(self.scheduler)
            individual.conflict_count = len(conflicts)
            
            # 计算FPS满足率
            task_counts = defaultdict(int)
            for event in self.scheduler.schedule_history:
                task_counts[event.task_id] += 1
            
            # 详细的FPS分析
            fps_improvement = 0.0
            fps_penalty = 0.0
            critical_task_penalty = 0.0
            
            for task in self.tasks:
                task_id = task.task_id
                count = task_counts[task_id]
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                
                if expected > 0:
                    fps_rate = count / expected
                    
                    # 对比基线
                    if self.baseline_performance and task_id in self.baseline_performance['fps_rates']:
                        baseline_rate = self.baseline_performance['fps_rates'][task_id]
                        baseline_count = self.baseline_performance['task_counts'][task_id]
                        
                        # 如果FPS下降，严重惩罚
                        if count < baseline_count:
                            fps_penalty += (baseline_count - count) * 200
                            
                            # 关键任务下降惩罚更重
                            if task.priority == TaskPriority.CRITICAL:
                                critical_task_penalty += (baseline_count - count) * 500
                        else:
                            # FPS提升奖励
                            fps_improvement += (count - baseline_count) * 50
            
            # 计算总体FPS满足率
            total_fps_rate = 0.0
            satisfied_tasks = 0
            for task in self.tasks:
                count = task_counts[task.task_id]
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                if expected > 0:
                    rate = min(1.0, count / expected)
                    total_fps_rate += rate
                    if rate >= 0.95:
                        satisfied_tasks += 1
            
            individual.fps_satisfaction_rate = total_fps_rate / len(self.tasks)
            
            # 计算资源利用率
            npu_util, dsp_util = self._calculate_separate_utilization()
            individual.resource_utilization = (npu_util + dsp_util) / 2
            
            # 新的适应度计算
            fitness = 0.0
            
            # 1. 基础分数（基于总体FPS满足率）
            fitness += individual.fps_satisfaction_rate * 300
            
            # 2. 无冲突奖励（降低权重）
            if individual.conflict_count == 0:
                fitness += 200  # 从1000降到200
            else:
                fitness -= individual.conflict_count * 50  # 降低惩罚
            
            # 3. FPS改进奖励/惩罚
            fitness += fps_improvement - fps_penalty - critical_task_penalty
            
            # 4. 资源利用率（考虑平衡）
            utilization_score = individual.resource_utilization * 100
            balance_score = 30 * (1.0 - abs(npu_util - dsp_util))
            fitness += utilization_score + balance_score
            
            # 5. 满足所有任务FPS要求的额外奖励
            if satisfied_tasks == len(self.tasks):
                fitness += 100
            
            # 6. 与基线相比的整体表现
            if self.baseline_performance:
                current_avg_fps = individual.fps_satisfaction_rate
                baseline_avg_fps = self.baseline_performance['avg_fps']
                if current_avg_fps >= baseline_avg_fps:
                    fitness += 100
                else:
                    fitness -= (baseline_avg_fps - current_avg_fps) * 500
                    
        except Exception as e:
            print(f"评估失败: {e}")
            fitness = -10000.0
            individual.conflict_count = 999
            
        individual.fitness = fitness
        return fitness
    
    def _create_baseline_individual(self) -> GeneticIndividual:
        """创建基线个体（原始配置）"""
        if self.baseline_config:
            return copy.deepcopy(self.baseline_config)
        return self._save_original_config()
    
    def _mutate_conservative(self, individual: GeneticIndividual):
        """保守的变异策略"""
        for task in self.tasks:
            task_id = task.task_id
            
            # 降低变异概率
            mutation_prob = self.mutation_rate * 0.5
            
            # 优先级变异（更保守）
            if random.random() < mutation_prob:
                current_priority = individual.task_priorities[task_id]
                
                # CRITICAL任务不降级
                if current_priority == TaskPriority.CRITICAL:
                    continue
                    
                # HIGH任务很少降级
                if current_priority == TaskPriority.HIGH and random.random() < 0.8:
                    continue
                
                # 小幅调整
                priorities = list(TaskPriority)
                current_idx = priorities.index(current_priority)
                
                # 80%概率保持或提升优先级
                if random.random() < 0.8:
                    new_idx = max(0, current_idx - random.randint(0, 1))
                else:
                    new_idx = min(len(priorities) - 1, current_idx + 1)
                    
                individual.task_priorities[task_id] = priorities[new_idx]
            
            # 运行时类型变异（仅在必要时）
            if random.random() < mutation_prob * 0.5:
                # 倾向于保持原有类型
                if task.uses_dsp and random.random() < 0.7:
                    individual.task_runtime_types[task_id] = RuntimeType.DSP_RUNTIME
                elif not task.uses_dsp and random.random() < 0.7:
                    individual.task_runtime_types[task_id] = RuntimeType.ACPU_RUNTIME
    
    def optimize_conservative(self):
        """保守的优化策略"""
        print("\n🧬 启动保守遗传算法优化")
        print("=" * 60)
        print(f"种群大小: {self.population_size}")
        print(f"精英个体: {self.elite_size}")
        print(f"变异率: {self.mutation_rate * 0.5} (保守)")
        print(f"迭代代数: {self.generations}")
        
        # 初始化种群
        population = []
        
        # 1. 添加多个基线配置副本
        for _ in range(max(3, self.elite_size)):
            baseline = self._create_baseline_individual()
            self._evaluate_fitness(baseline)
            population.append(baseline)
        
        # 2. 添加轻微变异的个体
        while len(population) < self.population_size // 2:
            individual = copy.deepcopy(self.baseline_config)
            self._mutate_conservative(individual)
            self._evaluate_fitness(individual)
            population.append(individual)
        
        # 3. 添加一些智能个体
        while len(population) < self.population_size:
            individual = self._create_intelligent_individual()
            self._evaluate_fitness(individual)
            population.append(individual)
        
        # 排序
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.best_individual = population[0]
        
        print(f"\n初始最佳适应度: {self.best_individual.fitness:.2f}")
        print(f"  - FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
        print(f"  - 资源冲突: {self.best_individual.conflict_count}")
        
        # 进化过程
        no_improvement_count = 0
        
        for generation in range(self.generations):
            # 保留精英和基线
            new_population = population[:self.elite_size]
            
            # 确保基线配置始终在种群中
            if self.baseline_config not in new_population:
                baseline = copy.deepcopy(self.baseline_config)
                self._evaluate_fitness(baseline)
                new_population.append(baseline)
            
            # 生成新个体
            while len(new_population) < self.population_size:
                # 选择父代（偏向高适应度）
                parent1 = self._tournament_selection(population, tournament_size=2)
                parent2 = self._tournament_selection(population, tournament_size=2)
                
                # 交叉
                if random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1 = copy.deepcopy(parent1)
                    child2 = copy.deepcopy(parent2)
                
                # 保守变异
                self._mutate_conservative(child1)
                self._mutate_conservative(child2)
                
                # 评估
                self._evaluate_fitness(child1)
                self._evaluate_fitness(child2)
                
                new_population.extend([child1, child2])
            
            # 更新种群
            population = new_population[:self.population_size]
            population.sort(key=lambda x: x.fitness, reverse=True)
            
            # 更新最佳个体
            if population[0].fitness > self.best_individual.fitness:
                # 额外检查：确保FPS没有显著下降
                if population[0].fps_satisfaction_rate >= self.baseline_performance['avg_fps'] * 0.98:
                    self.best_individual = population[0]
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
            
            # 记录历史
            avg_fitness = sum(ind.fitness for ind in population) / len(population)
            self.generation_history.append({
                'generation': generation,
                'best_fitness': self.best_individual.fitness,
                'avg_fitness': avg_fitness,
                'best_fps_rate': self.best_individual.fps_satisfaction_rate,
                'best_conflicts': self.best_individual.conflict_count
            })
            
            # 定期报告
            if generation % 10 == 0:
                print(f"\n第 {generation} 代:")
                print(f"  最佳适应度: {self.best_individual.fitness:.2f}")
                print(f"  FPS满足率: {self.best_individual.fps_satisfaction_rate:.1%}")
                print(f"  资源冲突: {self.best_individual.conflict_count}")
            
            # 早停条件
            if no_improvement_count >= 30:
                print(f"\n⚠️ 连续{no_improvement_count}代无改进，停止优化")
                break
        
        # 最终检查：如果最佳个体性能低于基线，返回基线
        if self.best_individual.fps_satisfaction_rate < self.baseline_performance['avg_fps']:
            print("\n⚠️ 优化结果低于基线，返回原始配置")
            self.best_individual = self._create_baseline_individual()
            self._evaluate_fitness(self.best_individual)
        
        # 应用最佳配置
        self._apply_individual_config(self.best_individual)
        
        return self.best_individual


def main():
    """主测试函数"""
    print("=" * 80)
    print("🧬 修正的遗传算法优化测试")
    print("=" * 80)
    
    # 创建系统
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=40)
    scheduler.add_dsp("DSP_0", bandwidth=40)
    
    # 应用基础修复
    fix_manager = apply_basic_fixes(scheduler)
    
    # 创建任务
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    # 应用冲突修复
    apply_minimal_fifo_fix(scheduler)
    if apply_strict_resource_conflict_fix:
        apply_strict_resource_conflict_fix(scheduler)
    
    # 获取基线性能
    print("\n📊 评估基线性能...")
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(200.0)
    
    # 验证基线
    is_valid, baseline_conflicts = validate_schedule_correctly(scheduler)
    baseline_stats = analyze_fps_satisfaction(scheduler, 200.0)
    baseline_util = calculate_detailed_utilization(scheduler, 200.0)
    
    print(f"\n基线结果:")
    print(f"  - 资源冲突: {len(baseline_conflicts)}")
    print(f"  - 平均FPS满足率: {baseline_stats['total_fps_rate'] / len(tasks):.1%}")
    print(f"  - NPU利用率: {baseline_util['NPU']['overall_utilization']:.1%}")
    print(f"  - DSP利用率: {baseline_util['DSP']['overall_utilization']:.1%}")
    
    # 保存基线可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt()
        plt.savefig('fixed_baseline.png', dpi=150, bbox_inches='tight')
        plt.close()
        viz.export_chrome_tracing('fixed_baseline_trace.json')
    except Exception as e:
        print(f"⚠️ 基线可视化失败: {e}")
    
    # 创建优化器
    optimizer = FixedGeneticOptimizer(scheduler, tasks, 200.0)
    optimizer.set_baseline_performance(baseline_stats, len(baseline_conflicts))
    
    # 运行保守优化
    best_individual = optimizer.optimize_conservative()
    
    # 获取优化后结果
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(200.0)
    
    is_valid, optimized_conflicts = validate_schedule_correctly(scheduler)
    optimized_stats = analyze_fps_satisfaction(scheduler, 200.0)
    optimized_util = calculate_detailed_utilization(scheduler, 200.0)
    
    # 保存优化后可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt()
        plt.savefig('fixed_optimized.png', dpi=150, bbox_inches='tight')
        plt.close()
        viz.export_chrome_tracing('fixed_optimized_trace.json')
    except Exception as e:
        print(f"⚠️ 优化可视化失败: {e}")
    
    # 打印最终对比
    print("\n" + "=" * 80)
    print("📊 优化效果总结")
    print("=" * 80)
    
    print("\n指标对比:")
    print(f"{'指标':<20} {'基线':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 65)
    
    print(f"{'资源冲突数':<20} {len(baseline_conflicts):<15} {len(optimized_conflicts):<15} "
          f"{len(baseline_conflicts) - len(optimized_conflicts):<15}")
    
    baseline_avg_fps = baseline_stats['total_fps_rate'] / len(tasks)
    optimized_avg_fps = optimized_stats['total_fps_rate'] / len(tasks)
    print(f"{'平均FPS满足率':<20} {baseline_avg_fps:.1%}{'':12} "
          f"{optimized_avg_fps:.1%}{'':12} "
          f"{(optimized_avg_fps - baseline_avg_fps):.1%}")
    
    print(f"{'NPU总体利用率':<20} {baseline_util['NPU']['overall_utilization']:.1%}{'':12} "
          f"{optimized_util['NPU']['overall_utilization']:.1%}{'':12} "
          f"{(optimized_util['NPU']['overall_utilization'] - baseline_util['NPU']['overall_utilization']):.1%}")
    
    print(f"{'DSP总体利用率':<20} {baseline_util['DSP']['overall_utilization']:.1%}{'':12} "
          f"{optimized_util['DSP']['overall_utilization']:.1%}{'':12} "
          f"{(optimized_util['DSP']['overall_utilization'] - baseline_util['DSP']['overall_utilization']):.1%}")
    
    # 生成对比图
    generate_comparison_visualization(scheduler, baseline_stats, optimized_stats,
                                    baseline_util, optimized_util)
    
    # 执行调度紧凑化
    print("\n" + "=" * 80)
    print("🔧 执行调度紧凑化")
    print("=" * 80)
    
    try:
        # 使用调试版紧凑化器
        from core.debug_compactor import test_debug_compactor
        
        print("使用调试版紧凑化器...")
        compacted_events, idle_time = test_debug_compactor(scheduler)
        
        print(f"\n📊 紧凑化结果:")
        print(f"  - 末尾空闲时间: {idle_time:.1f}ms ({idle_time/200.0*100:.1f}%)")
        
    except ImportError:
        print("⚠️ 尝试其他紧凑化方法...")
        try:
            from simple_compactor import SimpleCompactor, visualize_compaction
            original_events = copy.deepcopy(scheduler.schedule_history)
            compactor = SimpleCompactor(scheduler, 200.0)
            compacted_events, idle_time = compactor.compact()
            
            if compacted_events:
                visualize_compaction(scheduler, original_events, compacted_events, idle_time)
                scheduler.schedule_history = compacted_events
                print(f"\n✨ 紧凑化成功!")
                print(f"  - 末尾空闲时间: {idle_time:.1f}ms ({idle_time/200.0*100:.1f}%)")
                
        except Exception as e:
            print(f"⚠️ 紧凑化失败: {e}")
    except Exception as e:
        print(f"⚠️ 紧凑化失败: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n📁 生成的文件:")
    print("  - fixed_baseline.png / fixed_baseline_trace.json")
    print("  - fixed_optimized.png / fixed_optimized_trace.json")
    print("  - optimization_comparison.png")
    print("  - schedule_compaction_comparison.png (如果紧凑化成功)")
    print("  - compacted_schedule_trace.json (如果紧凑化成功)")
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
