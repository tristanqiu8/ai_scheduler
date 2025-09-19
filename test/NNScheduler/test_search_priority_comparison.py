#!/usr/bin/env python3
"""
Search Priority 功能比较测试程序
测试search_priority=true和search_priority=false的区别
"""

import sys
import os
import json

# 动态添加项目根目录到Python路径
def setup_project_path():
    """动态查找并添加项目根目录到Python路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 向上查找包含NNScheduler模块目录的根目录
    project_root = current_dir
    search_count = 0
    while project_root != os.path.dirname(project_root) and search_count < 10:  # 防止无限循环
        nnscheduler_path = os.path.join(project_root, 'NNScheduler')
        interface_path = os.path.join(nnscheduler_path, 'interface')

        # 确保这是真正的NNScheduler模块目录(包含interface子目录)
        if os.path.exists(nnscheduler_path) and os.path.exists(interface_path):
            break
        project_root = os.path.dirname(project_root)
        search_count += 1

    nnscheduler_check = os.path.join(project_root, 'NNScheduler')
    interface_check = os.path.join(nnscheduler_check, 'interface')
    if os.path.exists(nnscheduler_check) and os.path.exists(interface_check):
        # 确保项目根目录在Python路径的最前面
        if project_root in sys.path:
            sys.path.remove(project_root)
        sys.path.insert(0, project_root)
        return project_root
    else:
        raise ImportError(f"无法找到NNScheduler模块。搜索路径: {project_root}")

try:
    setup_project_path()
except Exception as e:
    print(f"[ERROR] 设置Python路径失败: {e}")
    sys.exit(1)

from NNScheduler.interface.optimization_interface import OptimizationInterface


def run_test_comparison():
    """运行search_priority功能比较测试"""
    print("="*80)
    print("Search Priority 功能比较测试")
    print("="*80)

    # 测试文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    enabled_config = os.path.join(script_dir, "test_search_priority_enabled.json")
    disabled_config = os.path.join(script_dir, "test_search_priority_disabled.json")

    # 检查文件是否存在
    if not os.path.exists(enabled_config):
        print(f"[ERROR] 找不到测试文件: {enabled_config}")
        return
    if not os.path.exists(disabled_config):
        print(f"[ERROR] 找不到测试文件: {disabled_config}")
        return

    print(f"[INFO] 读取测试配置...")
    print(f"  - search_priority=true:  {enabled_config}")
    print(f"  - search_priority=false: {disabled_config}")

    # 创建优化接口
    optimizer_interface = OptimizationInterface()

    # 测试 1: search_priority=true (启用优先级搜索)
    print(f"\n" + "="*60)
    print("测试 1: search_priority=true (启用优先级搜索)")
    print("="*60)

    try:
        result_enabled = optimizer_interface.optimize_from_json(enabled_config)
        print("[SUCCESS] search_priority=true 测试完成")
    except Exception as e:
        print(f"[ERROR] search_priority=true 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 测试 2: search_priority=false (禁用优先级搜索)
    print(f"\n" + "="*60)
    print("测试 2: search_priority=false (禁用优先级搜索)")
    print("="*60)

    try:
        result_disabled = optimizer_interface.optimize_from_json(disabled_config)
        print("[SUCCESS] search_priority=false 测试完成")
    except Exception as e:
        print(f"[ERROR] search_priority=false 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 比较结果
    print(f"\n" + "="*80)
    print("结果比较")
    print("="*80)

    # 读取原始配置中的用户优先级设置
    with open(enabled_config, 'r', encoding='utf-8') as f:
        original_config = json.load(f)

    user_priorities = {}
    for task in original_config['scenario']['tasks']:
        user_priorities[task['task_id']] = task['priority']

    print("原始用户配置的优先级:")
    print("-" * 40)
    for task_id, priority in user_priorities.items():
        print(f"  {task_id}: {priority}")

    print(f"\nsearch_priority=true 的最终优先级配置:")
    print("-" * 40)
    enabled_priorities = result_enabled['best_configuration']['priority_config']
    for task_id, priority in enabled_priorities.items():
        print(f"  {task_id}: {priority}")

    print(f"\nsearch_priority=false 的最终优先级配置:")
    print("-" * 40)
    disabled_priorities = result_disabled['best_configuration']['priority_config']
    for task_id, priority in disabled_priorities.items():
        print(f"  {task_id}: {priority}")

    # 检查优先级是否发生变化
    print(f"\n优先级变化分析:")
    print("-" * 40)

    priority_changed = False
    for task_id in user_priorities:
        original = user_priorities[task_id]
        enabled_final = enabled_priorities.get(task_id, "UNKNOWN")
        disabled_final = disabled_priorities.get(task_id, "UNKNOWN")

        if original != disabled_final:
            print(f"  [WARNING] {task_id}: 用户配置 {original} != search_priority=false 结果 {disabled_final}")

        if enabled_final != disabled_final:
            priority_changed = True
            print(f"  [INFO] {task_id}: search_priority=true({enabled_final}) vs search_priority=false({disabled_final})")

    if priority_changed:
        print(f"\n[SUCCESS] 优先级搜索功能正常工作：search_priority=true 和 search_priority=false 产生了不同的优先级配置")
    else:
        print(f"\n[INFO] 两种模式产生了相同的优先级配置")

    # 性能比较
    print(f"\n性能比较:")
    print("-" * 40)
    enabled_satisfaction = result_enabled['best_configuration']['satisfaction_rate']
    disabled_satisfaction = result_disabled['best_configuration']['satisfaction_rate']
    enabled_latency = result_enabled['best_configuration']['avg_latency']
    disabled_latency = result_disabled['best_configuration']['avg_latency']

    print(f"  满足率:")
    print(f"    search_priority=true:  {enabled_satisfaction:.1%}")
    print(f"    search_priority=false: {disabled_satisfaction:.1%}")

    print(f"  平均延迟:")
    print(f"    search_priority=true:  {enabled_latency:.1f}ms")
    print(f"    search_priority=false: {disabled_latency:.1f}ms")

    # 迭代次数比较
    enabled_iterations = len(result_enabled['optimization_history'])
    disabled_iterations = len(result_disabled['optimization_history'])

    print(f"  优化迭代次数:")
    print(f"    search_priority=true:  {enabled_iterations} 次")
    print(f"    search_priority=false: {disabled_iterations} 次")

    if disabled_iterations == 1:
        print(f"    [SUCCESS] search_priority=false 只进行了1次评估，符合预期")
    else:
        print(f"    [WARNING] search_priority=false 进行了{disabled_iterations}次评估，可能有问题")

    print(f"\n[SUCCESS] Search Priority 功能测试完成！")


if __name__ == "__main__":
    try:
        run_test_comparison()
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)