#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Scheduler 使用示例

这个文件演示了如何使用AI Scheduler包的各种功能，包括：
1. 命令行接口使用
2. Python API编程接口
3. 配置文件验证
4. 样本配置文件使用

安装: pip install ai-scheduler
"""

import os
import json
from pathlib import Path

def example_command_line_usage():
    """演示命令行接口的使用方法"""
    print("=" * 80)
    print("命令行接口使用示例")
    print("=" * 80)

    print("""
# 1. 基本用法 - 使用自定义配置文件
ai-scheduler my_config.json

# 2. 指定输出目录
ai-scheduler my_config.json --output ./results

# 3. 使用内置样本配置
ai-scheduler sample:config_1npu_1dsp.json

# 4. 列出所有可用的样本配置
ai-scheduler --list-samples

# 5. 验证配置文件格式
ai-scheduler --validate my_config.json

# 6. 启用详细输出
ai-scheduler my_config.json --verbose

# 7. 不显示横幅信息
ai-scheduler my_config.json --no-banner

# 8. 查看帮助信息
ai-scheduler --help

# 9. 查看版本信息
ai-scheduler --version
    """)

def example_python_api_basic():
    """演示基本Python API使用"""
    print("=" * 80)
    print("Python API 基本使用示例")
    print("=" * 80)

    print("# 导入AI Scheduler包")
    print("import ai_scheduler")
    print()

    try:
        import ai_scheduler

        # 1. 最简单的使用方式
        print("# 1. 使用便捷函数（最简单的方式）")
        print("result = ai_scheduler.optimize_from_json('config.json')")
        print("print(f\"满足率: {result['best_configuration']['satisfaction_rate']:.1%}\")")
        print()

        # 2. 使用样本配置
        print("# 2. 使用内置样本配置")
        print("sample_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')")
        sample_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')

        if sample_path:
            print(f"# 找到样本配置: {sample_path}")
            print("result = ai_scheduler.optimize_from_json(sample_path, output_dir='./demo_output')")

            # 实际运行一个示例
            print("\n[INFO] 运行样本配置演示...")
            result = ai_scheduler.optimize_from_json(sample_path, output_dir='./demo_output')

            print(f"✅ 优化完成!")
            print(f"   满足率: {result['best_configuration']['satisfaction_rate']:.1%}")
            print(f"   平均延迟: {result['best_configuration']['avg_latency']:.1f}ms")
            print(f"   系统利用率: {result['best_configuration']['system_utilization']:.1f}%")

        else:
            print("# 样本配置文件未找到")
        print()

        # 3. 列出所有样本配置
        print("# 3. 列出所有可用的样本配置")
        print("configs = ai_scheduler.get_sample_configs()")
        configs = ai_scheduler.get_sample_configs()
        print(f"# 找到 {len(configs)} 个样本配置:")
        for config in configs[:3]:  # 只显示前3个
            print(f"#   {config}")
        print()

        # 4. 获取版本信息
        print("# 4. 获取版本信息")
        print("version_info = ai_scheduler.version_info()")
        version_info = ai_scheduler.version_info()
        print(f"# 版本: {version_info['version']}")
        print(f"# 维护者: {version_info['maintainer']}")
        print(f"# 团队: {version_info['team']}")

    except ImportError:
        print("⚠️  AI Scheduler包未安装。请先运行: pip install ai-scheduler")
        print()

def example_python_api_advanced():
    """演示高级Python API使用"""
    print("=" * 80)
    print("Python API 高级使用示例")
    print("=" * 80)

    try:
        import ai_scheduler

        # 1. 创建优化器实例
        print("# 1. 创建优化器实例")
        print("api = ai_scheduler.create_optimizer()")
        api = ai_scheduler.create_optimizer()
        print()

        # 2. 验证配置文件
        print("# 2. 配置文件验证")
        sample_path = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')
        if sample_path:
            print(f"validation = api.validate_config('{sample_path}')")
            validation = api.validate_config(sample_path)

            if validation['valid']:
                print("✅ 配置文件有效")
                print(f"   找到 {len(validation['config']['scenario']['tasks'])} 个任务")
                print(f"   找到 {len(validation['config']['resources']['resources'])} 个资源")
            else:
                print("❌ 配置文件无效:")
                for error in validation['errors']:
                    print(f"   - {error}")
        print()

        # 3. 列出样本配置（带详细信息）
        print("# 3. 获取样本配置详细信息")
        print("sample_configs = api.list_sample_configs()")
        sample_configs = api.list_sample_configs()
        if sample_configs:
            config = sample_configs[0]
            print(f"# 第一个样本配置:")
            print(f"#   名称: {config['name']}")
            print(f"#   场景: {config['scenario_name']}")
            print(f"#   描述: {config['description']}")
            print(f"#   路径: {config['path']}")
        print()

        # 4. 从配置字典运行优化
        print("# 4. 从配置字典运行优化")
        print("""
config_dict = {
    "optimization": {
        "max_iterations": 25,
        "target_satisfaction": 0.9,
        "search_priority": True
    },
    "resources": {
        "resources": [
            {"resource_id": "NPU_0", "resource_type": "NPU", "bandwidth": 160.0}
        ]
    },
    "scenario": {
        "tasks": [
            {
                "task_id": "TEST_TASK",
                "name": "TestTask",
                "priority": "NORMAL",
                "fps": 30.0,
                "latency": 20.0,
                "model": {"segments": [...]}
            }
        ]
    }
}

result = api.optimize_from_dict(config_dict, "output_dir")
        """)

    except ImportError:
        print("⚠️  AI Scheduler包未安装。请先运行: pip install ai-scheduler")
        print()

def example_custom_config():
    """演示如何创建自定义配置"""
    print("=" * 80)
    print("自定义配置示例")
    print("=" * 80)

    # 创建一个简单的配置示例
    config = {
        "optimization": {
            "max_iterations": 30,
            "max_time_seconds": 120,
            "target_satisfaction": 0.95,
            "time_window": 1000.0,
            "segment_mode": True,
            "enable_detailed_analysis": True,
            "search_priority": True,
            "log_level": "normal"
        },
        "resources": {
            "resources": [
                {
                    "resource_id": "NPU_0",
                    "resource_type": "NPU",
                    "bandwidth": 160.0
                },
                {
                    "resource_id": "DSP_0",
                    "resource_type": "DSP",
                    "bandwidth": 160.0
                }
            ]
        },
        "scenario": {
            "scenario_name": "Custom Example Scenario",
            "description": "自定义的简单示例场景",
            "tasks": [
                {
                    "task_id": "CUSTOM_NPU_TASK",
                    "name": "CustomNpuTask",
                    "priority": "HIGH",
                    "runtime_type": "ACPU_RUNTIME",
                    "segmentation_strategy": "NO_SEGMENTATION",
                    "fps": 30.0,
                    "latency": 25.0,
                    "model": {
                        "segments": [
                            {
                                "resource_type": "NPU",
                                "duration_table": {
                                    "80": 8.5,
                                    "120": 7.0,
                                    "160": 5.8
                                },
                                "segment_id": "custom_npu_inference",
                                "power": 500.0,
                                "ddr": 25.0
                            }
                        ]
                    }
                },
                {
                    "task_id": "CUSTOM_DSP_TASK",
                    "name": "CustomDspTask",
                    "priority": "NORMAL",
                    "runtime_type": "ACPU_RUNTIME",
                    "segmentation_strategy": "NO_SEGMENTATION",
                    "fps": 60.0,
                    "latency": 15.0,
                    "model": {
                        "segments": [
                            {
                                "resource_type": "DSP",
                                "duration_table": {
                                    "80": 2.5,
                                    "120": 2.0,
                                    "160": 1.6
                                },
                                "segment_id": "custom_dsp_processing",
                                "power": 200.0,
                                "ddr": 8.0
                            }
                        ]
                    }
                }
            ]
        }
    }

    # 保存配置文件
    config_file = "example_custom_config.json"
    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    print(f"✅ 创建了自定义配置文件: {config_file}")
    print()
    print("# 使用自定义配置文件:")
    print(f"# 命令行方式: ai-scheduler {config_file}")
    print("# Python API方式:")
    print("import ai_scheduler")
    print(f"result = ai_scheduler.optimize_from_json('{config_file}')")

    try:
        import ai_scheduler
        print("\n[INFO] 运行自定义配置演示...")
        result = ai_scheduler.optimize_from_json(config_file, "custom_output")

        print(f"✅ 自定义配置优化完成!")
        print(f"   满足率: {result['best_configuration']['satisfaction_rate']:.1%}")
        print(f"   平均延迟: {result['best_configuration']['avg_latency']:.1f}ms")

    except ImportError:
        print("\n⚠️  AI Scheduler包未安装，无法运行演示")
    except Exception as e:
        print(f"\n❌ 运行失败: {e}")

def example_integration_patterns():
    """演示不同的集成模式"""
    print("=" * 80)
    print("集成模式示例")
    print("=" * 80)

    print("""
# 1. 批处理模式 - 处理多个配置文件
import ai_scheduler
import glob

def batch_optimize(config_pattern, output_base):
    config_files = glob.glob(config_pattern)
    results = []

    for config_file in config_files:
        output_dir = f"{output_base}/{Path(config_file).stem}"
        try:
            result = ai_scheduler.optimize_from_json(config_file, output_dir)
            results.append({
                'config': config_file,
                'success': True,
                'satisfaction_rate': result['best_configuration']['satisfaction_rate']
            })
        except Exception as e:
            results.append({
                'config': config_file,
                'success': False,
                'error': str(e)
            })

    return results

# 使用示例
results = batch_optimize("configs/*.json", "batch_results")

# 2. Web服务集成模式
from flask import Flask, request, jsonify
import ai_scheduler

app = Flask(__name__)

@app.route('/optimize', methods=['POST'])
def optimize_api():
    try:
        config_data = request.json
        result = ai_scheduler.create_optimizer().optimize_from_dict(
            config_data,
            f"./temp_results/{request.remote_addr}"
        )
        return jsonify({
            'success': True,
            'result': result['best_configuration']
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

# 3. 配置参数扫描模式
import itertools

def parameter_sweep():
    base_config = ai_scheduler.get_sample_config_path('config_1npu_1dsp.json')

    # 定义要扫描的参数
    max_iterations_values = [10, 25, 50]
    target_satisfaction_values = [0.8, 0.9, 0.95]

    api = ai_scheduler.create_optimizer()

    for max_iter, target_sat in itertools.product(
        max_iterations_values,
        target_satisfaction_values
    ):
        # 修改配置参数
        validation = api.validate_config(base_config)
        config = validation['config']
        config['optimization']['max_iterations'] = max_iter
        config['optimization']['target_satisfaction'] = target_sat

        # 运行优化
        result = api.optimize_from_dict(
            config,
            f"sweep_results/{max_iter}_{target_sat}"
        )

        print(f"参数组合 {max_iter}, {target_sat}: 满足率 {result['best_configuration']['satisfaction_rate']:.1%}")
    """)

def main():
    """主函数"""
    print("AI Scheduler 使用示例集合")
    print(f"运行环境: {os.getcwd()}")
    print()

    # 运行各种示例
    example_command_line_usage()
    example_python_api_basic()
    example_python_api_advanced()
    example_custom_config()
    example_integration_patterns()

    print("=" * 80)
    print("示例运行完成！")
    print()
    print("安装AI Scheduler:")
    print("  pip install ai-scheduler")
    print()
    print("或者从源码安装:")
    print("  git clone <repository>")
    print("  cd ai-scheduler")
    print("  pip install -e .")
    print()
    print("更多信息请参考文档或使用 ai-scheduler --help")
    print("=" * 80)

if __name__ == "__main__":
    main()