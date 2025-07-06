#!/usr/bin/env python3
"""
改进的遗传算法优化器
- 修复输出重复问题
- 分别统计NPU和DSP利用率
- 改进适应度函数确保FPS不下降
"""


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 先导入枚举
from .enums import TaskPriority, RuntimeType, SegmentationStrategy
# 然后导入其他模块
from .scheduler import MultiResourceScheduler
from .task import NNTask
from .genetic_task_optimizer import GeneticTaskOptimizer, GeneticIndividual
from .fixed_validation_and_metrics import validate_schedule_correctly
from .modular_scheduler_fixes import apply_basic_fixes
from .minimal_fifo_fix_corrected import apply_minimal_fifo_fix
from .strict_resource_conflict_fix import apply_strict_resource_conflict_fix
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ImprovedGeneticOptimizer(GeneticTaskOptimizer):
    """改进的遗传算法优化器"""
    
    def __init__(self, scheduler, tasks, time_window=200.0):
        super().__init__(scheduler, tasks, time_window)
        # 保存基线FPS以确保不下降
        self.baseline_fps_rates = {}
        
    def set_baseline_fps(self, baseline_stats):
        """设置基线FPS作为最低要求"""
        for task_id, task_info in baseline_stats['task_fps'].items():
            self.baseline_fps_rates[task_id] = task_info['fps_rate']
    
    def _evaluate_fitness(self, individual: GeneticIndividual) -> float:
        """改进的适应度函数，确保FPS不下降"""
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
            
            satisfied_tasks = 0
            total_fps_rate = 0.0
            total_latency = 0.0
            fps_penalty = 0.0  # FPS下降惩罚
            
            for task in self.tasks:
                count = task_counts[task.task_id]
                expected = int((self.time_window / 1000.0) * task.fps_requirement)
                
                if expected > 0:
                    fps_rate = min(1.0, count / expected)
                    total_fps_rate += fps_rate
                    
                    # 检查是否低于基线
                    if task.task_id in self.baseline_fps_rates:
                        baseline_rate = self.baseline_fps_rates[task.task_id]
                        if fps_rate < baseline_rate:
                            # 严重惩罚FPS下降
                            fps_penalty += (baseline_rate - fps_rate) * 1000
                    
                    if fps_rate >= 0.95:
                        satisfied_tasks += 1
                
                # 计算平均延迟
                if task.schedule_info:
                    total_latency += task.schedule_info.actual_latency
            
            individual.fps_satisfaction_rate = total_fps_rate / len(self.tasks)
            individual.avg_latency = total_latency / len(self.tasks) if self.tasks else 0
            
            # 分别计算NPU和DSP利用率
            npu_util, dsp_util = self._calculate_separate_utilization()
            individual.resource_utilization = (npu_util + dsp_util) / 2
            
            # 计算适应度
            fitness = 0.0
            
            # 1. 无冲突是最重要的
            if individual.conflict_count == 0:
                fitness += 1000.0
            else:
                fitness -= individual.conflict_count * 100.0
            
            # 2. FPS满足率（减去下降惩罚）
            fitness += individual.fps_satisfaction_rate * 500.0 - fps_penalty
            
            # 3. 资源利用率（平衡NPU和DSP）
            balance_bonus = 50.0 * (1.0 - abs(npu_util - dsp_util))
            fitness += individual.resource_utilization * 200.0 + balance_bonus
            
            # 4. 低延迟奖励
            if individual.avg_latency < 50:
                fitness += 100.0
            
            # 5. 关键任务优先级正确性
            if individual.task_priorities.get("T1") == TaskPriority.CRITICAL:
                fitness += 50.0
                
        except Exception as e:
            print(f"评估失败: {e}")
            fitness = -1000.0
            
        individual.fitness = fitness
        return fitness
    
    def _calculate_separate_utilization(self):
        """分别计算NPU和DSP的利用率"""
        npu_busy_time = 0.0
        dsp_busy_time = 0.0
        npu_count = 0
        dsp_count = 0
        
        for res_type, resources in self.scheduler.resources.items():
            if isinstance(resources, dict):
                resource_items = resources.items()
            elif isinstance(resources, list):
                resource_items = [(f"{res_type.value}_{i}", res) for i, res in enumerate(resources)]
            else:
                continue
            
            for res_id, resource in resource_items:
                busy_time = 0.0
                last_end = 0.0
                
                for event in sorted(self.scheduler.schedule_history, key=lambda x: x.start_time):
                    if event.assigned_resources.get(res_type) == res_id:
                        if event.start_time >= last_end:
                            busy_time += event.end_time - event.start_time
                            last_end = event.end_time
                
                if res_type.value == "NPU":
                    npu_busy_time += busy_time
                    npu_count += 1
                elif res_type.value == "DSP":
                    dsp_busy_time += busy_time
                    dsp_count += 1
        
        npu_util = (npu_busy_time / (self.time_window * npu_count)) if npu_count > 0 else 0
        dsp_util = (dsp_busy_time / (self.time_window * dsp_count)) if dsp_count > 0 else 0
        
        return npu_util, dsp_util


def calculate_detailed_utilization(scheduler, time_window):
    """计算详细的资源利用率"""
    utilization_stats = {
        'NPU': {'total_busy': 0, 'total_capacity': 0, 'per_unit': {}},
        'DSP': {'total_busy': 0, 'total_capacity': 0, 'per_unit': {}}
    }
    
    for res_type, resources in scheduler.resources.items():
        if isinstance(resources, dict):
            resource_items = resources.items()
        elif isinstance(resources, list):
            resource_items = [(f"{res_type.value}_{i}", res) for i, res in enumerate(resources)]
        else:
            continue
        
        for res_id, resource in resource_items:
            busy_time = 0.0
            busy_segments = []
            
            # 收集所有使用该资源的事件
            events = [(e.start_time, e.end_time, e.task_id) 
                     for e in scheduler.schedule_history 
                     if e.assigned_resources.get(res_type) == res_id]
            
            # 按开始时间排序
            events.sort()
            
            # 计算实际占用时间（合并重叠）
            if events:
                current_start = events[0][0]
                current_end = events[0][1]
                
                for start, end, task_id in events[1:]:
                    if start <= current_end:
                        # 重叠，扩展当前段
                        current_end = max(current_end, end)
                    else:
                        # 无重叠，记录当前段
                        busy_time += current_end - current_start
                        busy_segments.append((current_start, current_end))
                        current_start = start
                        current_end = end
                
                # 记录最后一段
                busy_time += current_end - current_start
                busy_segments.append((current_start, current_end))
            
            utilization = busy_time / time_window if time_window > 0 else 0
            
            # 分类统计
            if res_type.value == "NPU":
                utilization_stats['NPU']['total_busy'] += busy_time
                utilization_stats['NPU']['total_capacity'] += time_window
                utilization_stats['NPU']['per_unit'][res_id] = {
                    'utilization': utilization,
                    'busy_time': busy_time,
                    'segments': busy_segments
                }
            elif res_type.value == "DSP":
                utilization_stats['DSP']['total_busy'] += busy_time
                utilization_stats['DSP']['total_capacity'] += time_window
                utilization_stats['DSP']['per_unit'][res_id] = {
                    'utilization': utilization,
                    'busy_time': busy_time,
                    'segments': busy_segments
                }
    
    # 计算总体利用率
    for res_type in ['NPU', 'DSP']:
        stats = utilization_stats[res_type]
        if stats['total_capacity'] > 0:
            stats['overall_utilization'] = stats['total_busy'] / stats['total_capacity']
        else:
            stats['overall_utilization'] = 0
    
    return utilization_stats


def print_detailed_utilization(utilization_stats):
    """打印详细的利用率统计"""
    print("\n" + "=" * 60)
    print("📊 详细资源利用率分析")
    print("=" * 60)
    
    for res_type in ['NPU', 'DSP']:
        stats = utilization_stats[res_type]
        print(f"\n{res_type} 利用率:")
        print(f"  总体利用率: {stats['overall_utilization']:.1%}")
        print(f"  总忙碌时间: {stats['total_busy']:.1f}ms")
        print(f"  总可用时间: {stats['total_capacity']:.1f}ms")
        
        if stats['per_unit']:
            print(f"\n  各单元详情:")
            for unit_id, unit_stats in stats['per_unit'].items():
                print(f"    {unit_id}:")
                print(f"      利用率: {unit_stats['utilization']:.1%}")
                print(f"      忙碌时间: {unit_stats['busy_time']:.1f}ms")
                print(f"      活跃段数: {len(unit_stats['segments'])}")


def run_improved_optimization(scheduler, tasks, time_window=200.0):
    """运行改进的遗传算法优化"""
    
    print("\n" + "=" * 80)
    print("🧬 运行改进的遗传算法优化")
    print("=" * 80)
    
    # 先获取基线性能
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    # 计算基线统计
    baseline_stats = analyze_fps_satisfaction(scheduler, time_window)
    baseline_util = calculate_detailed_utilization(scheduler, time_window)
    
    print("\n📊 基线性能:")
    print(f"  - FPS满足率: {baseline_stats['total_fps_rate'] / len(tasks):.1%}")
    print(f"  - NPU利用率: {baseline_util['NPU']['overall_utilization']:.1%}")
    print(f"  - DSP利用率: {baseline_util['DSP']['overall_utilization']:.1%}")
    
    # 保存基线可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt()
        plt.savefig('baseline_improved.png', dpi=150, bbox_inches='tight')
        plt.close()
        viz.export_chrome_tracing('baseline_improved_trace.json')
        print("\n✅ 基线结果已保存")
    except Exception as e:
        print(f"\n⚠️ 基线可视化失败: {e}")
    
    # 创建改进的优化器
    optimizer = ImprovedGeneticOptimizer(scheduler, tasks, time_window)
    optimizer.set_baseline_fps(baseline_stats)
    
    # 调整参数
    optimizer.population_size = 40
    optimizer.generations = 100
    optimizer.elite_size = 8
    optimizer.mutation_rate = 0.2
    
    # 运行优化
    best_individual = optimizer.optimize()
    
    # 重新运行调度获取最终结果
    scheduler.schedule_history.clear()
    results = scheduler.priority_aware_schedule_with_segmentation(time_window)
    
    # 计算优化后统计
    optimized_stats = analyze_fps_satisfaction(scheduler, time_window)
    optimized_util = calculate_detailed_utilization(scheduler, time_window)
    
    # 打印详细利用率
    print_detailed_utilization(optimized_util)
    
    return optimizer, baseline_stats, optimized_stats, baseline_util, optimized_util


def analyze_fps_satisfaction(scheduler, time_window):
    """分析FPS满足情况（避免重复导入）"""
    stats = {
        'task_fps': {},
        'satisfied_count': 0,
        'total_fps_rate': 0.0,
        'resource_utilization': {}
    }
    
    task_counts = defaultdict(int)
    for event in scheduler.schedule_history:
        task_counts[event.task_id] += 1
    
    for task in scheduler.tasks.values():
        count = task_counts[task.task_id]
        expected = int((time_window / 1000.0) * task.fps_requirement)
        
        if expected > 0:
            fps_rate = min(1.0, count / expected)
            stats['task_fps'][task.task_id] = {
                'name': task.name,
                'count': count,
                'expected': expected,
                'fps_rate': fps_rate,
                'satisfied': fps_rate >= 0.95
            }
            
            stats['total_fps_rate'] += fps_rate
            if fps_rate >= 0.95:
                stats['satisfied_count'] += 1
    
    return stats


def generate_comparison_visualization(scheduler, baseline_stats, optimized_stats, 
                                    baseline_util, optimized_util):
    """生成对比可视化"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. FPS满足率对比
    task_ids = sorted(baseline_stats['task_fps'].keys())
    baseline_fps = [baseline_stats['task_fps'][tid]['fps_rate'] for tid in task_ids]
    optimized_fps = [optimized_stats['task_fps'][tid]['fps_rate'] for tid in task_ids]
    
    x = range(len(task_ids))
    width = 0.35
    
    ax1.bar([i - width/2 for i in x], baseline_fps, width, label='Baseline', alpha=0.8)
    ax1.bar([i + width/2 for i in x], optimized_fps, width, label='Optimized', alpha=0.8)
    ax1.set_xlabel('Task ID')
    ax1.set_ylabel('FPS Satisfaction Rate')
    ax1.set_title('FPS Satisfaction Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(task_ids)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Target (95%)')
    
    # 2. 资源利用率对比
    resources = ['NPU', 'DSP']
    baseline_utils = [baseline_util['NPU']['overall_utilization'], 
                     baseline_util['DSP']['overall_utilization']]
    optimized_utils = [optimized_util['NPU']['overall_utilization'],
                      optimized_util['DSP']['overall_utilization']]
    
    x2 = range(len(resources))
    ax2.bar([i - width/2 for i in x2], baseline_utils, width, label='Baseline', alpha=0.8)
    ax2.bar([i + width/2 for i in x2], optimized_utils, width, label='Optimized', alpha=0.8)
    ax2.set_xlabel('Resource Type')
    ax2.set_ylabel('Utilization Rate')
    ax2.set_title('Resource Utilization Comparison')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(resources)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 单个NPU利用率
    npu_units = sorted(optimized_util['NPU']['per_unit'].keys())
    if npu_units:
        npu_utils = [optimized_util['NPU']['per_unit'][unit]['utilization'] for unit in npu_units]
        ax3.bar(range(len(npu_units)), npu_utils, alpha=0.8, color='green')
        ax3.set_xlabel('NPU Unit')
        ax3.set_ylabel('Utilization Rate')
        ax3.set_title('NPU Units Utilization (Optimized)')
        ax3.set_xticks(range(len(npu_units)))
        ax3.set_xticklabels(npu_units)
        ax3.grid(True, alpha=0.3)
    
    # 4. 单个DSP利用率
    dsp_units = sorted(optimized_util['DSP']['per_unit'].keys())
    if dsp_units:
        dsp_utils = [optimized_util['DSP']['per_unit'][unit]['utilization'] for unit in dsp_units]
        ax4.bar(range(len(dsp_units)), dsp_utils, alpha=0.8, color='orange')
        ax4.set_xlabel('DSP Unit')
        ax4.set_ylabel('Utilization Rate')
        ax4.set_title('DSP Units Utilization (Optimized)')
        ax4.set_xticks(range(len(dsp_units)))
        ax4.set_xticklabels(dsp_units)
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('optimization_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ 对比图表已保存到 optimization_comparison.png")


def main():
    """主测试函数"""
    print("=" * 80)
    print("🧬 改进的遗传算法优化测试")
    print("=" * 80)
    
    # 创建系统
    scheduler = MultiResourceScheduler(enable_segmentation=True)
    scheduler.add_npu("NPU_0", bandwidth=120.0)
    scheduler.add_dsp("DSP_0", bandwidth=40.0)
    
    # 应用修复
    fix_manager = apply_basic_fixes(scheduler)
    
    # 应用额外的冲突解决修复（来自dragon4_with_smart_gap.py）
    from minimal_fifo_fix_corrected import apply_minimal_fifo_fix

    from strict_resource_conflict_fix import apply_strict_resource_conflict_fix
    
    # 创建任务
    tasks = create_real_tasks()
    for task in tasks:
        scheduler.add_task(task)
    
    # 应用FIFO和严格资源冲突修复
    apply_minimal_fifo_fix(scheduler)
    
    apply_strict_resource_conflict_fix(scheduler)
    
    # 运行改进的优化
    optimizer, baseline_stats, optimized_stats, baseline_util, optimized_util = \
        run_improved_optimization(scheduler, tasks, 200.0)
    
    # 生成可视化
    try:
        viz = ElegantSchedulerVisualizer(scheduler)
        viz.plot_elegant_gantt()
        plt.savefig('improved_genetic_schedule.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("\n✅ 优化结果已保存到 improved_genetic_schedule.png")
        
        # 生成Chrome trace文件
        viz.export_chrome_tracing('improved_genetic_trace.json')
        print("✅ Chrome trace已保存到 improved_genetic_trace.json")
        
    except Exception as e:
        print(f"\n⚠️ 可视化失败: {e}")
    
    # 打印最终对比
    print("\n" + "=" * 80)
    print("📊 优化效果总结")
    print("=" * 80)
    
    print("\n指标对比:")
    print(f"{'指标':<20} {'基线':<15} {'优化后':<15} {'改进':<15}")
    print("-" * 65)
    
    # FPS对比
    baseline_avg_fps = baseline_stats['total_fps_rate'] / len(tasks)
    optimized_avg_fps = optimized_stats['total_fps_rate'] / len(tasks)
    print(f"{'平均FPS满足率':<20} {baseline_avg_fps:.1%}{'':12} "
          f"{optimized_avg_fps:.1%}{'':12} "
          f"{(optimized_avg_fps - baseline_avg_fps):.1%}")
    
    # NPU利用率对比
    print(f"{'NPU总体利用率':<20} {baseline_util['NPU']['overall_utilization']:.1%}{'':12} "
          f"{optimized_util['NPU']['overall_utilization']:.1%}{'':12} "
          f"{(optimized_util['NPU']['overall_utilization'] - baseline_util['NPU']['overall_utilization']):.1%}")
    
    # DSP利用率对比
    print(f"{'DSP总体利用率':<20} {baseline_util['DSP']['overall_utilization']:.1%}{'':12} "
          f"{optimized_util['DSP']['overall_utilization']:.1%}{'':12} "
          f"{(optimized_util['DSP']['overall_utilization'] - baseline_util['DSP']['overall_utilization']):.1%}")
    
    # 生成对比可视化
    generate_comparison_visualization(scheduler, baseline_stats, optimized_stats,
                                    baseline_util, optimized_util)
    
    print("\n✅ 测试完成！")
    print("\n📁 生成的文件:")
    print("  - baseline_improved.png: 基线调度甘特图")
    print("  - baseline_improved_trace.json: 基线Chrome追踪文件") 
    print("  - improved_genetic_schedule.png: 优化后调度甘特图")
    print("  - improved_genetic_trace.json: 优化后Chrome追踪文件")
    print("  - optimization_comparison.png: 优化效果对比图表")
    print("\n💡 使用Chrome浏览器打开 chrome://tracing 并加载.json文件查看详细时间线")


if __name__ == "__main__":
    main()
