#!/usr/bin/env python3
"""
最小化JSON API测试程序
输入: JSON配置文件
输出: 甘特图、Chrome trace、优先级配置和满足情况总结
"""

import sys
import os

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

import json
import time
import argparse
from pathlib import Path
from NNScheduler.interface.optimization_interface import OptimizationInterface
from NNScheduler.core.artifacts import get_artifacts_root, resolve_artifact_path


def run_optimization_from_json(config_file: str, output_dir: str = "./artifacts_sim"):
    """从JSON文件运行优化并生成所有输出"""

    if output_dir:
        artifacts_dir = resolve_artifact_path(output_dir)
    else:
        artifacts_dir = get_artifacts_root()

    artifacts_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] 使用输出目录: {artifacts_dir}")

    # 处理配置文件路径，确保使用绝对路径
    if not os.path.isabs(config_file):
        # 相对路径：首先尝试当前工作目录，然后尝试脚本所在目录
        if not os.path.exists(config_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_config_file = os.path.join(script_dir, config_file)
            if os.path.exists(alt_config_file):
                config_file = alt_config_file

    # 转换为绝对路径，避免切换目录后找不到文件
    config_file = os.path.abspath(config_file)
    print(f"[INFO] 读取配置文件: {config_file}")

    # 创建优化接口
    optimizer_interface = OptimizationInterface()

    # 切换到输出目录执行优化，确保所有文件生成在指定目录
    original_cwd = Path.cwd()
    try:
        os.chdir(artifacts_dir)
        print(f"[INFO] 开始优化...")
        start_time = time.time()

        result = optimizer_interface.optimize_from_json(config_file)
    finally:
        os.chdir(original_cwd)

    elapsed_time = time.time() - start_time
    print(f"[INFO] 优化完成，耗时: {elapsed_time:.1f}秒")

    # 提取关键结果
    best_config = result['best_configuration']

    # 打印优化结果总结
    print("\n" + "="*80)
    print("优化结果总结")
    print("="*80)

    print(f"满足率: {best_config['satisfaction_rate']:.1%}")
    print(f"平均延迟: {best_config['avg_latency']:.1f}ms")

    resource_utilization = best_config.get('resource_utilization', {})
    display_items = [
        (res_type, util) for res_type, util in sorted(resource_utilization.items())
        if util is not None and abs(util) > 1e-3
    ]

    if display_items:
        print("资源利用率:")
        print("-"*40)
        for res_type, util in display_items:
            print(f"{res_type}利用率: {util:.1f}%")
    else:
        print("资源利用率: 暂无数据")

    print(f"系统利用率: {best_config['system_utilization']:.1f}%")

    # 优先级配置
    print(f"\n任务优先级配置:")
    print("-"*40)
    for task_id, priority in best_config['priority_config'].items():
        print(f"{task_id}: {priority}")

    # FPS满足情况
    print(f"\nFPS满足情况:")
    print("-"*40)
    fps_satisfied = sum(1 for satisfied in best_config['fps_satisfaction'].values() if satisfied)
    total_tasks = len(best_config['fps_satisfaction'])
    print(f"满足FPS要求的任务: {fps_satisfied}/{total_tasks}")
    for task_id, satisfied in best_config['fps_satisfaction'].items():
        status = "[OK]" if satisfied else "[FAIL]"
        print(f"  {task_id}: {status}")

    # 延迟满足情况
    print(f"\n延迟满足情况:")
    print("-"*40)
    latency_satisfied = sum(1 for satisfied in best_config['latency_satisfaction'].values() if satisfied)
    print(f"满足延迟要求的任务: {latency_satisfied}/{total_tasks}")
    for task_id, satisfied in best_config['latency_satisfaction'].items():
        status = "[OK]" if satisfied else "[FAIL]"
        print(f"  {task_id}: {status}")

    # 输出文件信息
    print(f"\n生成的输出文件:")
    print("-"*40)

    viz_files = result.get('visualization_files', {})

    timeline_rel = viz_files.get('timeline_png')
    if timeline_rel:
        timeline_path = Path(timeline_rel)
        if not timeline_path.is_absolute():
            timeline_path = artifacts_dir / timeline_path
        print(f"甘特图: {timeline_path}")

    trace_rel = viz_files.get('chrome_trace')
    if trace_rel:
        trace_path = Path(trace_rel)
        if not trace_path.is_absolute():
            trace_path = artifacts_dir / trace_path
        print(f"Chrome trace: {trace_path}")

    # 最优配置文件（若存在）
    import glob
    current_time = time.strftime('%Y%m%d')
    config_files = sorted(glob.glob(str(artifacts_dir / f"optimized_priority_config*_{current_time}_*.json")))
    if config_files:
        print(f"最优配置: {config_files[-1]}")

    output_file_path = Path(result.get('output_file', ''))
    if output_file_path:
        if not output_file_path.is_absolute():
            output_file_path = artifacts_dir / output_file_path
        print(f"详细结果: {output_file_path}")

    print("\n[SUCCESS] 优化完成！")

    return result


if __name__ == "__main__":
    # 命令行参数解析
    parser = argparse.ArgumentParser(
        description="最小化JSON API优化程序",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python test_minimal_json_api.py config.json
  python test_minimal_json_api.py config.json --out ./my_results
  python test_minimal_json_api.py config.json --out /absolute/path/to/results
        """
    )

    parser.add_argument(
        'config_file',
        nargs='?',
        default='optimization_config_custom_example.json',
        help='JSON配置文件路径 (默认: optimization_config_custom_example.json)'
    )

    parser.add_argument(
        '--out',
        default='./artifacts_sim',
        help='输出文件目录路径 (默认: ./artifacts_sim)'
    )

    args = parser.parse_args()

    # 检查配置文件是否存在
    config_file = args.config_file
    if not os.path.isabs(config_file):
        # 相对路径：首先尝试当前工作目录，然后尝试脚本所在目录
        if not os.path.exists(config_file):
            script_dir = os.path.dirname(os.path.abspath(__file__))
            alt_config_file = os.path.join(script_dir, config_file)
            if os.path.exists(alt_config_file):
                config_file = alt_config_file

    if not os.path.exists(config_file):
        print(f"[ERROR] 配置文件不存在: {args.config_file}")
        print(f"[INFO] 请确保文件存在，或使用以下格式指定:")
        print(f"       python {sys.argv[0]} <config_file> [--out <output_dir>]")
        sys.exit(1)

    try:
        run_optimization_from_json(config_file, args.out)
    except Exception as e:
        print(f"[ERROR] 优化过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
