#!/usr/bin/env python3
"""
测试带宽管理器的功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.bandwidth_manager import BandwidthManager
from core.resource_queue import ResourceQueueManager, QueuedTask
from core.enums import ResourceType, TaskPriority
from core.models import SubSegment
from core.task import NNTask


def test_static_vs_dynamic_bandwidth():
    """测试固定带宽和动态带宽的区别"""
    print("=== 测试固定带宽 vs 动态带宽 ===\n")
    
    # 1. 固定带宽模式（旧模式）
    print("1. 固定带宽模式（旧架构）:")
    static_manager = ResourceQueueManager()  # 没有带宽管理器
    
    # 添加资源，各自有固定带宽
    static_manager.add_resource("NPU_0", ResourceType.NPU, 60.0)
    static_manager.add_resource("NPU_1", ResourceType.NPU, 60.0)
    static_manager.add_resource("DSP_0", ResourceType.DSP, 40.0)
    static_manager.add_resource("DSP_1", ResourceType.DSP, 40.0)
    
    print("  NPU_0: 固定带宽 60")
    print("  NPU_1: 固定带宽 60")
    print("  DSP_0: 固定带宽 40")
    print("  DSP_1: 固定带宽 40")
    print("  总计: 200 (但各自独立，不反映实际硬件)")
    
    # 2. 动态带宽模式（新架构）
    print("\n2. 动态带宽模式（共享带宽池）:")
    
    # 创建带宽管理器 - NPU和DSP共享同一个带宽池！
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    
    dynamic_manager = ResourceQueueManager(bandwidth_manager=bw_manager)
    
    # 添加资源，都从共享池中分配带宽
    dynamic_manager.add_resource("NPU_0", ResourceType.NPU)
    dynamic_manager.add_resource("NPU_1", ResourceType.NPU)
    dynamic_manager.add_resource("DSP_0", ResourceType.DSP)
    dynamic_manager.add_resource("DSP_1", ResourceType.DSP)
    
    print("  所有单元共享120带宽池:")
    print("  - 1个单元活跃时: 获得120带宽")
    print("  - 2个单元活跃时: 每个获得60带宽")
    print("  - 4个单元活跃时: 每个获得30带宽")
    print("  - NPU和DSP平等竞争带宽")
    
    return static_manager, dynamic_manager, bw_manager


def test_bandwidth_allocation_scenario():
    """测试带宽分配场景 - NPU和DSP共享"""
    print("\n=== 测试NPU/DSP共享带宽场景 ===\n")
    
    # 创建带宽管理器 - 总共120带宽
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    
    print("系统配置:")
    print("  总带宽池: 120 (NPU和DSP共享)")
    
    # 场景1: 单个NPU运行
    print("\n场景1: 时间0-10ms，只有NPU_0运行")
    bw1 = bw_manager.allocate_bandwidth("NPU_0", ResourceType.NPU, "T1", 0, 10)
    print(f"  NPU_0获得: {bw1:.1f}带宽 (独占全部)")
    
    # 场景2: NPU和DSP并行
    print("\n场景2: 时间5-15ms，DSP_0也开始运行")
    bw2 = bw_manager.allocate_bandwidth("DSP_0", ResourceType.DSP, "T2", 5, 15)
    print(f"  时间5-10ms: NPU_0和DSP_0各获得60带宽")
    print(f"  时间10-15ms: DSP_0独占120带宽")
    
    # 场景3: 多个单元竞争
    print("\n场景3: 时间8-18ms，NPU_1和DSP_1也加入")
    bw3 = bw_manager.allocate_bandwidth("NPU_1", ResourceType.NPU, "T3", 8, 18)
    bw4 = bw_manager.allocate_bandwidth("DSP_1", ResourceType.DSP, "T4", 8, 18)
    print(f"  时间8-10ms: 4个单元(2NPU+2DSP)各获得30带宽")
    print(f"  时间10-15ms: 3个单元各获得40带宽")
    print(f"  时间15-18ms: 2个单元各获得60带宽")
    
    # 显示系统状态
    status = bw_manager.get_system_status(9.0)
    print(f"\n时间9ms时的系统状态:")
    print(f"  活跃资源: {status['active_resources']['list']}")
    print(f"  NPU数量: {status['active_resources']['npus']}")
    print(f"  DSP数量: {status['active_resources']['dsps']}")
    print(f"  每单元带宽: {status['bandwidth_per_unit']:.1f}")


def test_duration_estimation():
    """测试执行时间估算"""
    print("\n=== 测试执行时间估算 ===\n")
    
    # 创建一个子段，有不同带宽下的执行时间
    sub_segment = SubSegment(
        sub_id="conv_0",
        resource_type=ResourceType.NPU,
        duration_table={
            40: 12.0,   # 40带宽时需要12ms
            60: 8.0,    # 60带宽时需要8ms
            120: 4.0    # 120带宽时需要4ms
        },
        original_segment_id="conv"
    )
    
    print("子段duration_table:")
    for bw, dur in sub_segment.duration_table.items():
        print(f"  {bw}带宽: {dur}ms")
    
    # 测试不同带宽下的执行时间
    test_bandwidths = [40, 60, 80, 100, 120]
    print("\n不同带宽下的执行时间:")
    for bw in test_bandwidths:
        duration = sub_segment.get_duration(bw)
        print(f"  {bw}带宽: {duration:.1f}ms")


def test_mixed_task_bandwidth_sharing():
    """测试混合任务（DSP+NPU）的带宽共享"""
    print("\n=== 测试混合任务带宽共享 ===\n")
    
    # 创建带宽管理器
    bw_manager = BandwidthManager(total_system_bandwidth=120.0)
    
    # 创建队列管理器
    queue_manager = ResourceQueueManager(bandwidth_manager=bw_manager)
    npu0_queue = queue_manager.add_resource("NPU_0", ResourceType.NPU)
    dsp0_queue = queue_manager.add_resource("DSP_0", ResourceType.DSP)
    npu1_queue = queue_manager.add_resource("NPU_1", ResourceType.NPU)
    
    print("场景: MOTR任务（DSP+NPU混合）执行")
    print("  时间0-2ms: NPU_0执行MOTR的NPU段（独占120带宽）")
    print("  时间2-4ms: DSP_0执行MOTR的DSP段（独占120带宽）")
    print("  时间1-3ms: NPU_1执行其他任务")
    print("\n预期结果:")
    print("  时间1-2ms: NPU_0和NPU_1共享120带宽（各60）")
    print("  时间2-3ms: DSP_0和NPU_1共享120带宽（各60）")
    
    # 分配带宽
    bw_npu0 = bw_manager.allocate_bandwidth("NPU_0", ResourceType.NPU, "MOTR_npu", 0, 2)
    bw_dsp0 = bw_manager.allocate_bandwidth("DSP_0", ResourceType.DSP, "MOTR_dsp", 2, 4)
    bw_npu1 = bw_manager.allocate_bandwidth("NPU_1", ResourceType.NPU, "Other", 1, 3)
    
    print("\n实际分配结果:")
    print(f"  NPU_0 (0-2ms): 平均{bw_npu0:.1f}带宽")
    print(f"  DSP_0 (2-4ms): 平均{bw_dsp0:.1f}带宽")
    print(f"  NPU_1 (1-3ms): 平均{bw_npu1:.1f}带宽")
    
    # 显示详细时间线
    print("\n详细时间线:")
    for res_id in ["NPU_0", "DSP_0", "NPU_1"]:
        timeline = bw_manager.get_bandwidth_timeline(res_id, 0, 4)
        if timeline:
            print(f"  {res_id}:")
            for start, end, bw in timeline:
                print(f"    {start:.1f}-{end:.1f}ms: {bw:.1f}带宽")


def main():
    """运行所有测试"""
    print("开始测试带宽管理器 - NPU和DSP共享带宽池\n")
    
    test_static_vs_dynamic_bandwidth()
    test_bandwidth_allocation_scenario()
    test_duration_estimation()
    test_mixed_task_bandwidth_sharing()
    
    print("\n所有测试完成！")
    print("\n关键点总结:")
    print("1. NPU和DSP共享同一个带宽池（如120）")
    print("2. 所有活跃的硬件单元平均分配带宽")
    print("3. 不区分资源类型 - NPU和DSP平等竞争")
    print("4. 混合任务的不同段在各自时间片内竞争带宽")


if __name__ == "__main__":
    main()
