#!/usr/bin/env python3
"""
测试新增的理论vs实际耗时对比和Chrome Tracing导出功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.demo_camera_optimization import CameraTaskOptimizer


def test_timing_analysis():
    """测试时序分析功能"""
    print("=" * 80)
    print("[TEST] 测试理论vs实际耗时分析和Chrome Tracing导出")
    print("=" * 80)
    
    # 创建优化器（使用120带宽）
    optimizer = CameraTaskOptimizer(time_window=125.0, segment_mode=True, bandwidth=120.0)
    
    # 运行少量代数的优化（快速测试）
    print("\n运行少量代数优化以获取测试数据...")
    best_individual = optimizer.evolve(generations=3, target_fitness=0.95)
    
    # 打印结果（包括新的时序分析）
    optimizer.print_results(best_individual)
    
    print("\n[SUCCESS] 时序分析和Chrome Tracing导出功能测试完成！")


if __name__ == "__main__":
    test_timing_analysis()