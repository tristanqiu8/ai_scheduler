#!/usr/bin/env python3
"""
简化的动态带宽测试 - 直接测试 BandwidthManager
"""

import pytest
import sys
import os

# 仅在直接运行时添加路径
if __name__ == "__main__":
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from NNScheduler.core.bandwidth_manager import BandwidthManager
from NNScheduler.core.enums import ResourceType


def test_bandwidth_manager_directly():
    """直接测试带宽管理器的逻辑"""
    print("=== 直接测试 BandwidthManager ===\n")
    
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    
    # 场景1：单个资源使用
    print("场景1: 单个NPU使用带宽")
    bw1 = bw_manager.allocate_bandwidth("NPU_0", ResourceType.NPU, "T1", 0, 10)
    print(f"  NPU_0 (0-10ms): 分配带宽 = {bw1:.1f}")
    
    status = bw_manager.get_system_status(5)
    print(f"  时间5ms的系统状态: 活跃资源={status['active_resources']['total']}, "
          f"每单元带宽={status['bandwidth_per_unit']:.1f}")
    
    # 场景2：两个资源并行
    print("\n场景2: 两个NPU并行使用")
    bw2 = bw_manager.allocate_bandwidth("NPU_1", ResourceType.NPU, "T2", 5, 15)
    print(f"  NPU_1 (5-15ms): 分配带宽 = {bw2:.1f}")
    
    status = bw_manager.get_system_status(7)
    print(f"  时间7ms的系统状态: 活跃资源={status['active_resources']['total']}, "
          f"每单元带宽={status['bandwidth_per_unit']:.1f}")
    
    # 场景3：三个资源重叠
    print("\n场景3: NPU和DSP混合使用")
    bw3 = bw_manager.allocate_bandwidth("DSP_0", ResourceType.DSP, "T3", 8, 12)
    print(f"  DSP_0 (8-12ms): 分配带宽 = {bw3:.1f}")
    
    status = bw_manager.get_system_status(9)
    print(f"  时间9ms的系统状态: 活跃资源={status['active_resources']['total']}, "
          f"每单元带宽={status['bandwidth_per_unit']:.1f}")
    
    # 显示时间线
    print("\n带宽分配时间线:")
    for res_id in ["NPU_0", "NPU_1", "DSP_0"]:
        timeline = bw_manager.get_bandwidth_timeline(res_id, 0, 20)
        if timeline:
            print(f"  {res_id}:")
            for start, end, bw in timeline:
                print(f"    {start:.1f}-{end:.1f}ms: {bw:.1f}带宽")
    
    # 测试不同时间点的系统状态
    print("\n不同时间点的系统状态:")
    test_times = [0, 5, 8, 10, 12, 15, 20]
    for t in test_times:
        status = bw_manager.get_system_status(t)
        active = status['active_resources']['list']
        bw_per_unit = status['bandwidth_per_unit']
        print(f"  时间{t:>2}ms: 活跃={len(active)} {active}, 每单元带宽={bw_per_unit:.1f}")


def test_simultaneous_allocation():
    """测试同时分配的情况"""
    print("\n\n=== 测试同时分配 ===\n")
    
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    
    # 预先计算两个资源同时运行时的带宽
    print("场景：两个NPU同时开始执行")
    
    # 方法1：先查询可用带宽
    available_bw = bw_manager.get_available_bandwidth(10.0)
    print(f"  时间10ms，无活跃资源时的可用带宽: {available_bw:.1f}")
    
    # 方法2：分配第一个资源
    bw1 = bw_manager.allocate_bandwidth("NPU_0", ResourceType.NPU, "T1", 10, 20)
    print(f"  NPU_0分配: {bw1:.1f}")
    
    # 方法3：分配第二个资源（应该考虑第一个）
    bw2 = bw_manager.allocate_bandwidth("NPU_1", ResourceType.NPU, "T2", 10, 20)
    print(f"  NPU_1分配: {bw2:.1f}")
    
    # 检查最终状态
    status = bw_manager.get_system_status(15)
    print(f"\n  时间15ms的系统状态:")
    print(f"    活跃资源: {status['active_resources']['list']}")
    print(f"    每单元带宽: {status['bandwidth_per_unit']:.1f}")


def main():
    """运行测试"""
    print("开始测试动态带宽管理器\n")
    
    test_bandwidth_manager_directly()
    test_simultaneous_allocation()
    
    print("\n✅ 测试完成！")


if __name__ == "__main__":
    main()
