#!/usr/bin/env python3
"""
多场景测试示例 - 展示如何使用新的测试框架
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.scheduling_config import SchedulingConfig, ScenarioType
from core.enums import ResourceType
from flexible_test_framework import SchedulingTestFramework
from demo_real_task_segmentation import prepare_tasks_with_segmentation


def test_single_scenario():
    """测试单个场景"""
    print("=== 测试单个场景 ===")
    
    # 1. 准备任务
    tasks = prepare_tasks_with_segmentation()
    
    # 2. 创建测试框架
    framework = SchedulingTestFramework(tasks)
    
    # 3. 创建高带宽配置
    config = SchedulingConfig.create_high_bandwidth(npu_bw=120.0, dsp_bw=120.0)
    
    # 4. 运行测试
    result = framework.run_test(config)
    
    # 5. 生成可视化
    framework.generate_visualizations("high_bandwidth_results")


def test_multiple_scenarios():
    """测试多个场景并对比"""
    print("=== 测试多个场景 ===")
    
    # 1. 准备任务
    tasks = prepare_tasks_with_segmentation()
    
    # 2. 创建测试框架
    framework = SchedulingTestFramework(tasks)
    
    # 3. 定义测试场景
    configs = [
        # 基准配置
        SchedulingConfig.create_baseline(),
        
        # 高带宽配置
        SchedulingConfig.create_high_bandwidth(120.0, 120.0),
        
        # 多NPU配置
        SchedulingConfig.create_multi_npu(npu_count=2, npu_bw=60.0),
        SchedulingConfig.create_multi_npu(npu_count=3, npu_bw=40.0),
        
        # 多DSP配置
        SchedulingConfig.create_multi_dsp(dsp_count=2, dsp_bw=60.0),
        
        # 平衡配置
        SchedulingConfig.create_balanced(2, 2, 60.0, 60.0),
    ]
    
    # 4. 运行对比测试
    framework.run_comparison_tests(configs)
    
    # 5. 生成所有可视化
    framework.generate_visualizations("comparison_results")
    
    # 6. 导出对比报告
    framework.export_comparison_report("scenario_comparison.txt")


def test_custom_scenario():
    """测试自定义场景"""
    print("=== 测试自定义场景 ===")
    
    # 1. 准备任务
    tasks = prepare_tasks_with_segmentation()
    
    # 2. 创建测试框架
    framework = SchedulingTestFramework(tasks)
    
    # 3. 创建自定义配置
    config = SchedulingConfig(
        scenario=ScenarioType.CUSTOM,
        scenario_name="异构高性能配置"
    )
    
    # 添加3个高性能NPU
    for i in range(3):
        config.add_resource(f"NPU_{i}", ResourceType.NPU, 100.0)
    
    # 添加2个中等性能DSP
    for i in range(2):
        config.add_resource(f"DSP_{i}", ResourceType.DSP, 60.0)
    
    # 设置更长的仿真时间
    config.simulation_duration = 500.0
    config.analysis_window = 500.0
    
    # 4. 运行测试
    result = framework.run_test(config)
    
    # 5. 分析结果
    print("\n自定义场景分析:")
    print(f"理论NPU需求 vs 实际: {framework.calculate_theory_demand(tasks, config)}")


def test_bandwidth_sweep():
    """带宽扫描测试"""
    print("=== 带宽扫描测试 ===")
    
    tasks = prepare_tasks_with_segmentation()
    framework = SchedulingTestFramework(tasks)
    
    # 定义带宽扫描范围
    bandwidth_values = [20, 40, 60, 80, 100, 120, 160]
    
    configs = []
    for bw in bandwidth_values:
        config = SchedulingConfig(
            scenario=ScenarioType.CUSTOM,
            scenario_name=f"带宽{bw}Gbps"
        )
        config.add_resource("NPU_0", ResourceType.NPU, bw)
        config.add_resource("DSP_0", ResourceType.DSP, bw)
        configs.append(config)
    
    # 运行扫描
    results = framework.run_comparison_tests(configs)
    
    # 绘制带宽-性能曲线
    import matplotlib.pyplot as plt
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 使用黑体
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
    
    bandwidths = []
    fps_rates = []
    utils = []
    
    for config in configs:
        result = results[config.scenario_name]
        bandwidths.append(config.resources[0].bandwidth)
        fps_rates.append(result.metrics.fps_satisfaction_rate * 100)
        utils.append(result.system_utilization)
    
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.plot(bandwidths, fps_rates, 'bo-')
    plt.xlabel('带宽 (Gbps)')
    plt.ylabel('FPS满足率 (%)')
    plt.title('带宽 vs FPS满足率')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(bandwidths, utils, 'ro-')
    plt.xlabel('带宽 (Gbps)')
    plt.ylabel('系统利用率 (%)')
    plt.title('带宽 vs 系统利用率')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bandwidth_sweep_results.png')
    print("\n[OK] 带宽扫描结果已保存到: bandwidth_sweep_results.png")


def main():
    """主函数 - 演示各种测试场景"""
    import argparse
    
    parser = argparse.ArgumentParser(description='调度系统多场景测试')
    parser.add_argument('--scenario', type=str, default='all',
                       choices=['single', 'multiple', 'custom', 'sweep', 'all'],
                       help='选择测试场景')
    
    args = parser.parse_args()
    
    if args.scenario == 'single' or args.scenario == 'all':
        test_single_scenario()
        print("\n" + "="*80 + "\n")
    
    if args.scenario == 'multiple' or args.scenario == 'all':
        test_multiple_scenarios()
        print("\n" + "="*80 + "\n")
    
    if args.scenario == 'custom' or args.scenario == 'all':
        test_custom_scenario()
        print("\n" + "="*80 + "\n")
    
    if args.scenario == 'sweep' or args.scenario == 'all':
        test_bandwidth_sweep()


if __name__ == "__main__":
    main()
