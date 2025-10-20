#!/usr/bin/env python3
"""
优化效果对比验证程序
真正验证JSON API与原始test_cam_auto_priority_optimization.py的一致性
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import json
import time
import random
from typing import Dict, Any, Tuple

import numpy as np
import pytest

# 导入原始优化器
from test_cam_auto_priority_optimization import PriorityOptimizer
from NNScheduler.scenario.camera_task import create_real_tasks
from NNScheduler.core.executor import set_execution_log_enabled
from NNScheduler.core.artifacts import ensure_artifact_path

# 导入JSON API优化器
from NNScheduler.interface.optimization_interface import OptimizationInterface, JsonPriorityOptimizer


class OptimizationComparator:
    """优化效果对比器"""

    def __init__(self):
        self.tolerance = 0.01  # 1%的容差范围

    def run_original_optimizer(self) -> Dict[str, Any]:
        """运行原始优化器"""
        print("\n[ORIGINAL] 运行原始test_cam_auto_priority_optimization.py...")

        seed_value = 12345
        random.seed(seed_value)
        np.random.seed(seed_value)

        # 关闭执行日志
        set_execution_log_enabled(False)

        # 创建任务
        tasks = create_real_tasks()

        # 创建原始优化器
        optimizer = PriorityOptimizer(tasks, time_window=1000.0, segment_mode=True)

        # 执行优化 - 使用相同参数
        best_config, best_result = optimizer.optimize(
            max_iterations=50,
            max_time_seconds=300,
            target_satisfaction=0.95
        )

        return {
            'best_config': best_config,
            'best_result': best_result,
            'optimizer': optimizer,
            'tasks': tasks
        }

    def run_json_api_optimizer(self) -> Dict[str, Any]:
        """运行JSON API优化器"""
        print("\n[JSON_API] 运行JSON API优化器...")

        # 配置与原始测试完全相同
        config = {
            "optimization": {
                "max_iterations": 50,
                "max_time_seconds": 300,
                "target_satisfaction": 0.95,
                "time_window": 1000.0,
                "segment_mode": True,
                "slack": 0.0,
                "enable_random_slack": False
            },
            "resources": {
                "resources": [
                    {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 160.0},
                    {"resource_id": "DSP_0", "resource_type": "DSP", "bandwidth": 160.0}
                ]
            },
            "scenario": {
                "use_camera_tasks": True
            }
        }

        seed_value = 12345
        random.seed(seed_value)
        np.random.seed(seed_value)

        # 运行优化
        previous_env = os.environ.get("AI_SCHEDULER_DISABLE_RANDOM_SLACK")
        os.environ["AI_SCHEDULER_DISABLE_RANDOM_SLACK"] = "1"
        try:
            optimizer_interface = OptimizationInterface()
            result = optimizer_interface.optimize_from_config(config)
        finally:
            if previous_env is None:
                os.environ.pop("AI_SCHEDULER_DISABLE_RANDOM_SLACK", None)
            else:
                os.environ["AI_SCHEDULER_DISABLE_RANDOM_SLACK"] = previous_env

        return result

    def extract_metrics(self, original_result: Dict[str, Any], json_result: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """提取关键指标进行对比"""

        # 原始结果指标
        original_metrics = {
            'satisfaction_rate': original_result['best_result'].total_satisfaction_rate,
            'avg_latency': original_result['best_result'].avg_latency,
            'npu_utilization': original_result['best_result'].resource_utilization['NPU'],
            'dsp_utilization': original_result['best_result'].resource_utilization['DSP'],
            'priority_config': {k: v.name for k, v in original_result['best_config'].items()},
            'fps_satisfaction': original_result['best_result'].fps_satisfaction,
            'latency_satisfaction': original_result['best_result'].latency_satisfaction
        }

        # 计算系统利用率
        npu_util = original_metrics['npu_utilization'] / 100.0
        dsp_util = original_metrics['dsp_utilization'] / 100.0
        original_metrics['system_utilization'] = (1 - (1 - npu_util) * (1 - dsp_util)) * 100.0

        # JSON API结果指标
        best_config = json_result['best_configuration']
        json_metrics = {
            'satisfaction_rate': best_config['satisfaction_rate'],
            'avg_latency': best_config['avg_latency'],
            'npu_utilization': best_config['resource_utilization']['NPU'],
            'dsp_utilization': best_config['resource_utilization']['DSP'],
            'system_utilization': best_config['system_utilization'],
            'priority_config': best_config['priority_config'],
            'fps_satisfaction': best_config['fps_satisfaction'],
            'latency_satisfaction': best_config['latency_satisfaction'],
            'total_fps': best_config['fps_analysis'].get('total_fps', 0),
            'total_power_w': best_config['power_analysis'].get('total_power_w', 0),
            'total_ddr_gb': best_config['ddr_analysis'].get('total_ddr_gb', 0)
        }

        return original_metrics, json_metrics

    def compare_metrics(self, original: Dict, json_api: Dict) -> Dict[str, Dict]:
        """详细对比关键指标"""

        comparison = {}

        # 数值指标对比
        numerical_metrics = [
            'satisfaction_rate', 'avg_latency', 'npu_utilization',
            'dsp_utilization', 'system_utilization'
        ]

        for metric in numerical_metrics:
            orig_val = original[metric]
            json_val = json_api[metric]
            diff = abs(orig_val - json_val)
            diff_pct = (diff / max(orig_val, 0.001)) * 100  # 避免除零

            comparison[metric] = {
                'original': orig_val,
                'json_api': json_val,
                'difference': diff,
                'difference_pct': diff_pct,
                'within_tolerance': diff_pct <= (self.tolerance * 100)
            }

        # 优先级配置对比
        priority_match = original['priority_config'] == json_api['priority_config']
        comparison['priority_config'] = {
            'original': original['priority_config'],
            'json_api': json_api['priority_config'],
            'exact_match': priority_match
        }

        # FPS和延迟满足情况对比
        fps_match = original['fps_satisfaction'] == json_api['fps_satisfaction']
        latency_match = original['latency_satisfaction'] == json_api['latency_satisfaction']

        comparison['fps_satisfaction'] = {
            'original': original['fps_satisfaction'],
            'json_api': json_api['fps_satisfaction'],
            'exact_match': fps_match
        }

        comparison['latency_satisfaction'] = {
            'original': original['latency_satisfaction'],
            'json_api': json_api['latency_satisfaction'],
            'exact_match': latency_match
        }

        return comparison

    def print_comparison_report(self, comparison: Dict[str, Dict]):
        """打印详细对比报告"""

        print("\n" + "="*100)
        print("优化效果对比验证报告")
        print("="*100)

        # 数值指标对比
        print(f"\n{'指标':<20} {'原始程序':<15} {'JSON API':<15} {'差异':<12} {'差异%':<10} {'通过':<8}")
        print("-"*100)

        numerical_metrics = [
            ('satisfaction_rate', '满足率', '%'),
            ('avg_latency', '平均延迟', 'ms'),
            ('npu_utilization', 'NPU利用率', '%'),
            ('dsp_utilization', 'DSP利用率', '%'),
            ('system_utilization', '系统利用率', '%')
        ]

        all_numerical_pass = True
        for metric_key, metric_name, unit in numerical_metrics:
            if metric_key in comparison:
                comp = comparison[metric_key]
                orig = comp['original']
                json_val = comp['json_api']
                diff = comp['difference']
                diff_pct = comp['difference_pct']
                passed = comp['within_tolerance']

                if unit == '%':
                    orig_str = f"{orig:.1f}%"
                    json_str = f"{json_val:.1f}%"
                    diff_str = f"{diff:.1f}"
                else:
                    orig_str = f"{orig:.1f}{unit}"
                    json_str = f"{json_val:.1f}{unit}"
                    diff_str = f"{diff:.1f}"

                status = "[PASS]" if passed else "[FAIL]"
                print(f"{metric_name:<20} {orig_str:<15} {json_str:<15} {diff_str:<12} {diff_pct:<10.2f} {status:<8}")

                if not passed:
                    all_numerical_pass = False

        # 优先级配置对比
        print(f"\n优先级配置对比:")
        print("-"*100)
        priority_comp = comparison['priority_config']
        priority_match = priority_comp['exact_match']

        if priority_match:
            print("[PASS] 优先级配置完全匹配")
        else:
            print("[PARTIAL] 优先级配置存在差异:")
            orig_config = priority_comp['original']
            json_config = priority_comp['json_api']

            print(f"{'任务ID':<10} {'原始程序':<15} {'JSON API':<15} {'匹配':<8}")
            print("-"*60)

            all_tasks = set(orig_config.keys()) | set(json_config.keys())
            priority_differences = 0

            for task_id in sorted(all_tasks):
                orig_priority = orig_config.get(task_id, 'N/A')
                json_priority = json_config.get(task_id, 'N/A')
                match = orig_priority == json_priority
                match_str = "[OK]" if match else "[DIFF]"

                print(f"{task_id:<10} {orig_priority:<15} {json_priority:<15} {match_str:<8}")

                if not match:
                    priority_differences += 1

            print(f"\n优先级差异数量: {priority_differences}")

        # 满足情况对比
        print(f"\n任务满足情况对比:")
        print("-"*100)

        fps_comp = comparison['fps_satisfaction']
        latency_comp = comparison['latency_satisfaction']

        fps_match = fps_comp['exact_match']
        latency_match = latency_comp['exact_match']

        print(f"FPS满足情况匹配: {'[PASS]' if fps_match else '[FAIL]'}")
        print(f"延迟满足情况匹配: {'[PASS]' if latency_match else '[FAIL]'}")

        # 如果不匹配，显示详细差异
        if not fps_match or not latency_match:
            print(f"\n详细满足情况对比:")
            print(f"{'任务ID':<10} {'FPS满足(原始)':<15} {'FPS满足(JSON)':<15} {'延迟满足(原始)':<15} {'延迟满足(JSON)':<15}")
            print("-"*90)

            all_tasks = set(fps_comp['original'].keys()) | set(fps_comp['json_api'].keys())
            for task_id in sorted(all_tasks):
                fps_orig = fps_comp['original'].get(task_id, False)
                fps_json = fps_comp['json_api'].get(task_id, False)
                lat_orig = latency_comp['original'].get(task_id, False)
                lat_json = latency_comp['json_api'].get(task_id, False)

                print(f"{task_id:<10} {str(fps_orig):<15} {str(fps_json):<15} {str(lat_orig):<15} {str(lat_json):<15}")

        # 总体评估
        print(f"\n总体评估:")
        print("-"*100)

        overall_pass = all_numerical_pass and priority_match and fps_match and latency_match

        if overall_pass:
            print("[SUCCESS] JSON API与原始程序效果完全一致！")
        else:
            print("[PARTIAL] JSON API与原始程序存在部分差异，详情如上")

            # 分析可能的原因
            print(f"\n可能的差异原因:")
            if not all_numerical_pass:
                print("- 数值计算差异：可能由于随机性或算法实现细节差异导致")
            if not priority_match:
                print("- 优先级配置差异：可能由于初始化或调整策略的细微差异")
            if not fps_match or not latency_match:
                print("- 满足情况差异：可能由于调度执行过程的差异")

        return overall_pass


def test_optimization_comparison():
    """主函数 - 执行完整的对比验证"""

    print("="*100)
    print("优化效果对比验证")
    print("="*100)
    print("验证JSON API是否与test_cam_auto_priority_optimization.py产生完全一致的效果")

    start_time = time.time()

    # 创建对比器
    comparator = OptimizationComparator()

    try:
        # 运行原始优化器
        original_result = comparator.run_original_optimizer()

        # 运行JSON API优化器
        json_result = comparator.run_json_api_optimizer()

        # 提取关键指标
        original_metrics, json_metrics = comparator.extract_metrics(original_result, json_result)

        # 执行详细对比
        comparison = comparator.compare_metrics(original_metrics, json_metrics)

        # 打印对比报告
        overall_pass = comparator.print_comparison_report(comparison)

        # 保存对比结果
        comparison_result = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'overall_pass': overall_pass,
            'original_metrics': original_metrics,
            'json_metrics': json_metrics,
            'detailed_comparison': comparison,
            'test_duration': time.time() - start_time
        }

        comparison_file = ensure_artifact_path(
            f"optimization_comparison_result_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison_result, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n[SAVED] 对比结果已保存到: {comparison_file}")

        # 最终结论
        elapsed_time = time.time() - start_time
        print(f"\n[COMPLETE] 对比验证完成，耗时: {elapsed_time:.1f}秒")

        if overall_pass:
            print("\n[CONCLUSION] ✅ JSON API与原始程序效果完全一致，可以放心发布！")
        else:
            print("\n[CONCLUSION] ⚠️ JSON API与原始程序存在差异，需要进一步调整")

        assert overall_pass, "JSON API 与原始程序的关键指标存在差异"

    except Exception as e:
        print(f"\n[ERROR] 对比验证过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"对比验证失败: {e}")


if __name__ == "__main__":
    test_optimization_comparison()
