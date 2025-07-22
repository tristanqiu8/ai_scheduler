#!/usr/bin/env python3
"""
测试运行脚本
支持多种运行模式和选项
"""

import os
import sys
import argparse
import subprocess


def run_pytest(args):
    """运行pytest"""
    cmd = ["pytest"]
    
    # 基本选项
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # 测试路径
    if args.test_path:
        cmd.append(args.test_path)
    else:
        cmd.append("test/")
    
    # 标记过滤
    if args.markers:
        cmd.extend(["-m", args.markers])
    
    # 关键字过滤
    if args.keyword:
        cmd.extend(["-k", args.keyword])
    
    # 失败后停止
    if args.failfast:
        cmd.append("-x")
    
    # 显示本地变量
    if args.showlocals:
        cmd.append("-l")
    
    # 覆盖率
    if args.coverage:
        cmd.extend(["--cov=core", "--cov-report=html", "--cov-report=term"])
    
    # 并行执行（需要pytest-xdist）
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
    
    # 自定义参数
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    print(f"执行命令: {' '.join(cmd)}")
    return subprocess.call(cmd)


def run_specific_test(test_name):
    """运行特定的测试"""
    cmd = ["pytest", "-v", f"test/{test_name}"]
    print(f"执行命令: {' '.join(cmd)}")
    return subprocess.call(cmd)


def list_tests():
    """列出所有测试文件"""
    test_dir = "test"
    print("\n可用的测试文件:")
    print("-" * 50)
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.startswith("test_") and filename.endswith(".py"):
            filepath = os.path.join(test_dir, filename)
            # 尝试读取文件描述
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[1:10]:  # 查找文档字符串
                        if '"""' in line or "'''" in line:
                            continue
                        if line.strip() and not line.startswith("#"):
                            desc = line.strip()
                            print(f"  {filename:<40} # {desc}")
                            break
                    else:
                        print(f"  {filename}")
            except:
                print(f"  {filename}")
    
    print("\n测试标记:")
    print("  - unit:          单元测试")
    print("  - integration:   集成测试")
    print("  - slow:          慢速测试")
    print("  - visualization: 生成可视化的测试")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="AI Scheduler测试运行器")
    
    parser.add_argument("test_path", nargs="?", help="测试文件或目录路径")
    parser.add_argument("-v", "--verbose", action="store_true", help="详细输出")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")
    parser.add_argument("-x", "--failfast", action="store_true", help="第一个失败后停止")
    parser.add_argument("-l", "--showlocals", action="store_true", help="显示本地变量")
    parser.add_argument("-m", "--markers", help="只运行指定标记的测试 (如: 'unit', 'not slow')")
    parser.add_argument("-k", "--keyword", help="只运行名称包含关键字的测试")
    parser.add_argument("--coverage", action="store_true", help="生成覆盖率报告")
    parser.add_argument("-n", "--parallel", type=int, help="并行执行测试（需要pytest-xdist）")
    parser.add_argument("--list", action="store_true", help="列出所有测试")
    parser.add_argument("--pytest-args", help="传递给pytest的额外参数")
    
    # 快捷命令
    parser.add_argument("--unit", action="store_true", help="只运行单元测试")
    parser.add_argument("--integration", action="store_true", help="只运行集成测试")
    parser.add_argument("--fast", action="store_true", help="跳过慢速测试")
    
    args = parser.parse_args()
    
    # 列出测试
    if args.list:
        list_tests()
        return 0
    
    # 处理快捷命令
    if args.unit:
        args.markers = "unit"
    elif args.integration:
        args.markers = "integration"
    elif args.fast:
        args.markers = "not slow"
    
    # 运行测试
    return run_pytest(args)


if __name__ == "__main__":
    sys.exit(main())