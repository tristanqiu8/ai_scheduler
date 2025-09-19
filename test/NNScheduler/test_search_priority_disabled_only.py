#!/usr/bin/env python3
"""
Search Priority Disabled 独立测试程序
专门测试search_priority=false功能，验证系统直接使用用户配置的优先级
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


def run_disabled_search_priority_test():
    """运行search_priority=false的独立测试"""
    print("="*80)
    print("Search Priority Disabled 独立测试")
    print("="*80)

    # 测试文件路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_config = os.path.join(script_dir, "test_search_priority_disabled.json")

    # 检查文件是否存在
    if not os.path.exists(test_config):
        print(f"[ERROR] 找不到测试文件: {test_config}")
        return

    print(f"[INFO] 读取测试配置: {test_config}")

    # 读取并显示原始配置
    with open(test_config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    search_priority = config['optimization'].get('search_priority', True)
    print(f"[INFO] search_priority 设置: {search_priority}")

    if search_priority:
        print(f"[WARNING] 配置文件中的search_priority应该为false，当前为true")
        return

    print(f"\n原始用户配置的优先级:")
    print("-" * 40)
    user_priorities = {}
    for task in config['scenario']['tasks']:
        task_id = task['task_id']
        priority = task['priority']
        user_priorities[task_id] = priority
        print(f"  {task_id}: {priority}")

    # 创建优化接口并运行测试
    print(f"\n[INFO] 开始运行优化（search_priority=false）...")
    optimizer_interface = OptimizationInterface()

    try:
        result = optimizer_interface.optimize_from_json(test_config)
        print("[SUCCESS] 优化完成")
    except Exception as e:
        print(f"[ERROR] 优化失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # 分析结果
    print(f"\n" + "="*60)
    print("结果分析")
    print("="*60)

    # 检查最终优先级配置
    final_priorities = result['best_configuration']['priority_config']
    print(f"\n最终优先级配置:")
    print("-" * 40)
    for task_id, priority in final_priorities.items():
        print(f"  {task_id}: {priority}")

    # 验证优先级是否保持用户配置
    print(f"\n优先级验证:")
    print("-" * 40)
    all_match = True
    for task_id in user_priorities:
        original = user_priorities[task_id]
        final = final_priorities.get(task_id, "UNKNOWN")

        if original == final:
            print(f"  [OK] {task_id}: {original} == {final}")
        else:
            print(f"  [FAIL] {task_id}: {original} != {final}")
            all_match = False

    if all_match:
        print(f"\n[SUCCESS] 所有优先级都保持了用户配置，search_priority=false 功能正常")
    else:
        print(f"\n[FAIL] 有优先级没有保持用户配置，search_priority=false 功能异常")

    # 检查优化迭代次数
    optimization_history = result.get('optimization_history', [])
    iterations = len(optimization_history)
    print(f"\n优化过程分析:")
    print("-" * 40)
    print(f"  优化迭代次数: {iterations}")

    if iterations == 1:
        print(f"  [SUCCESS] 只进行了1次评估，符合search_priority=false的预期行为")
    else:
        print(f"  [WARNING] 进行了{iterations}次评估，可能存在问题")

    # 性能结果
    satisfaction_rate = result['best_configuration']['satisfaction_rate']
    avg_latency = result['best_configuration']['avg_latency']
    npu_utilization = result['best_configuration']['resource_utilization']['NPU']
    dsp_utilization = result['best_configuration']['resource_utilization']['DSP']
    system_utilization = result['best_configuration']['system_utilization']

    print(f"\n性能结果:")
    print("-" * 40)
    print(f"  满足率: {satisfaction_rate:.1%}")
    print(f"  平均延迟: {avg_latency:.1f}ms")
    print(f"  NPU利用率: {npu_utilization:.1f}%")
    print(f"  DSP利用率: {dsp_utilization:.1f}%")
    print(f"  系统利用率: {system_utilization:.1f}%")

    # FPS和延迟满足情况
    fps_satisfaction = result['best_configuration']['fps_satisfaction']
    latency_satisfaction = result['best_configuration']['latency_satisfaction']

    print(f"\n任务满足情况:")
    print("-" * 40)
    for task_id in user_priorities:
        fps_ok = fps_satisfaction.get(task_id, False)
        latency_ok = latency_satisfaction.get(task_id, False)
        fps_status = "[OK]" if fps_ok else "[FAIL]"
        latency_status = "[OK]" if latency_ok else "[FAIL]"
        print(f"  {task_id}: FPS {fps_status}, 延迟 {latency_status}")

    print(f"\n[SUCCESS] Search Priority Disabled 测试完成！")


if __name__ == "__main__":
    try:
        run_disabled_search_priority_test()
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)