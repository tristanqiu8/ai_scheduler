#!/usr/bin/env python3
"""
FPS要求分析窗口计算器
基于所有任务的FPS要求，计算最优的分析时间窗口
"""

import math
from typing import List, Dict, Tuple
from functools import reduce


def gcd(a: int, b: int) -> int:
    """计算两个数的最大公约数"""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """计算两个数的最小公倍数"""
    return abs(a * b) // gcd(a, b)


def gcd_multiple(numbers: List[int]) -> int:
    """计算多个数的最大公约数"""
    return reduce(gcd, numbers)


def lcm_multiple(numbers: List[int]) -> int:
    """计算多个数的最小公倍数"""
    return reduce(lcm, numbers)


class FPSWindowCalculator:
    """FPS分析窗口计算器"""
    
    def __init__(self, fps_requirements: List[int]):
        """
        初始化计算器
        
        Args:
            fps_requirements: 所有任务的FPS要求列表
        """
        self.fps_requirements = fps_requirements
        self.fps_gcd = gcd_multiple(fps_requirements)
        self.fps_lcm = lcm_multiple(fps_requirements)
        
    def calculate_optimal_window(self, max_window_ms: float = 1000.0) -> float:
        """
        计算最优分析时间窗口
        
        Args:
            max_window_ms: 最大允许的时间窗口（毫秒）
            
        Returns:
            最优时间窗口（毫秒）
        """
        # 方法1：基于最大公约数
        gcd_window = 1000.0 / self.fps_gcd
        
        # 方法2：基于最小公倍数的因子
        # 找到一个合适的窗口，使得所有FPS在该窗口内都有整数个实例
        lcm_window = 1000.0
        
        # 如果GCD窗口太大，尝试找一个更小的合适窗口
        if gcd_window > max_window_ms:
            # 寻找一个较小的窗口，使得大多数任务都有合理的实例数
            for window in [1000.0, 500.0, 250.0, 200.0, 100.0]:
                if window <= max_window_ms:
                    instances = self.calculate_instances_in_window(window)
                    # 检查是否所有任务都至少有1个实例
                    if all(inst >= 1 for inst in instances.values()):
                        return window
            
            # 如果还是找不到合适的，返回最大允许窗口
            return max_window_ms
        
        return min(gcd_window, max_window_ms)
    
    def calculate_instances_in_window(self, window_ms: float) -> Dict[int, float]:
        """
        计算每个FPS要求在给定窗口内的实例数
        
        Args:
            window_ms: 时间窗口（毫秒）
            
        Returns:
            FPS -> 实例数的映射
        """
        instances = {}
        for fps in self.fps_requirements:
            instances[fps] = fps * (window_ms / 1000.0)
        return instances
    
    def get_window_analysis(self, window_ms: float) -> Dict:
        """
        获取窗口分析报告
        
        Args:
            window_ms: 时间窗口（毫秒）
            
        Returns:
            分析报告字典
        """
        instances = self.calculate_instances_in_window(window_ms)
        
        # 计算整数实例数和小数部分
        integer_instances = {fps: int(inst) for fps, inst in instances.items()}
        fractional_parts = {fps: inst - int(inst) for fps, inst in instances.items()}
        
        # 统计
        total_instances = sum(integer_instances.values())
        has_fractional = any(frac > 0.01 for frac in fractional_parts.values())
        
        return {
            'window_ms': window_ms,
            'fps_gcd': self.fps_gcd,
            'fps_lcm': self.fps_lcm,
            'exact_instances': instances,
            'integer_instances': integer_instances,
            'fractional_parts': fractional_parts,
            'total_instances': total_instances,
            'has_fractional': has_fractional,
            'is_exact': not has_fractional
        }
    
    def recommend_window(self, prefer_exact: bool = True, max_window_ms: float = 1000.0) -> Tuple[float, Dict]:
        """
        推荐最优时间窗口
        
        Args:
            prefer_exact: 是否偏好精确的整数实例数
            max_window_ms: 最大允许的时间窗口
            
        Returns:
            (推荐窗口, 分析报告)
        """
        if prefer_exact:
            # 尝试找到一个窗口，使得所有FPS都有整数实例数
            optimal_window = self.calculate_optimal_window(max_window_ms)
        else:
            # 使用固定的窗口（如200ms）
            optimal_window = min(200.0, max_window_ms)
        
        analysis = self.get_window_analysis(optimal_window)
        
        return optimal_window, analysis


def analyze_fps_requirements(fps_list: List[int], max_window_ms: float = 1000.0) -> Dict:
    """
    分析FPS要求并推荐时间窗口
    
    Args:
        fps_list: FPS要求列表
        max_window_ms: 最大允许时间窗口
        
    Returns:
        完整的分析报告
    """
    calculator = FPSWindowCalculator(fps_list)
    
    # 获取推荐窗口
    recommended_window, analysis = calculator.recommend_window(max_window_ms=max_window_ms)
    
    # 比较几个候选窗口
    candidate_windows = [200.0, 500.0, 1000.0]
    candidate_windows = [w for w in candidate_windows if w <= max_window_ms]
    
    window_comparisons = {}
    for window in candidate_windows:
        window_comparisons[window] = calculator.get_window_analysis(window)
    
    return {
        'fps_requirements': fps_list,
        'fps_gcd': calculator.fps_gcd,
        'fps_lcm': calculator.fps_lcm,
        'recommended_window': recommended_window,
        'recommended_analysis': analysis,
        'window_comparisons': window_comparisons
    }


def print_fps_analysis_report(fps_list: List[int], max_window_ms: float = 1000.0):
    """
    打印FPS分析报告
    
    Args:
        fps_list: FPS要求列表
        max_window_ms: 最大允许时间窗口
    """
    report = analyze_fps_requirements(fps_list, max_window_ms)
    
    print("="*80)
    print("📊 FPS要求与时间窗口分析")
    print("="*80)
    
    print(f"\nFPS要求: {report['fps_requirements']}")
    print(f"最大公约数: {report['fps_gcd']}")
    print(f"最小公倍数: {report['fps_lcm']}")
    
    print(f"\n推荐时间窗口: {report['recommended_window']:.1f}ms")
    
    # 打印推荐窗口的详细信息
    rec_analysis = report['recommended_analysis']
    print(f"\n推荐窗口分析:")
    print(f"  是否精确整数: {'是' if rec_analysis['is_exact'] else '否'}")
    print(f"  总实例数: {rec_analysis['total_instances']}")
    
    print(f"\n各FPS在推荐窗口内的实例数:")
    for fps in sorted(report['fps_requirements']):
        exact = rec_analysis['exact_instances'][fps]
        integer = rec_analysis['integer_instances'][fps]
        fractional = rec_analysis['fractional_parts'][fps]
        
        if fractional > 0.01:
            print(f"  FPS {fps:2d}: {exact:.2f} ({integer} + {fractional:.2f})")
        else:
            print(f"  FPS {fps:2d}: {integer}")
    
    # 比较不同窗口
    print(f"\n候选窗口比较:")
    print(f"{'窗口(ms)':<10} {'精确整数':<10} {'总实例数':<10} {'示例(FPS->实例)'}")
    print("-" * 60)
    
    for window, analysis in sorted(report['window_comparisons'].items()):
        exact_str = "是" if analysis['is_exact'] else "否"
        total = analysis['total_instances']
        
        # 选几个代表性的FPS显示
        examples = []
        sample_fps = sorted(report['fps_requirements'])[:3]  # 取前3个作为示例
        for fps in sample_fps:
            inst = analysis['exact_instances'][fps]
            if inst == int(inst):
                examples.append(f"{fps}->{int(inst)}")
            else:
                examples.append(f"{fps}->{inst:.1f}")
        
        example_str = ", ".join(examples)
        print(f"{window:<10.0f} {exact_str:<10} {total:<10} {example_str}")


if __name__ == "__main__":
    # 测试旧的FPS要求（5的倍数）
    old_fps_requirements = [25, 10, 10, 5, 25, 60, 25, 25, 25]
    print("旧FPS要求测试:")
    print_fps_analysis_report(old_fps_requirements)
    
    print("\n" + "="*80 + "\n")
    
    # 测试新的FPS要求（非5的倍数）
    new_fps_requirements = [33, 13, 13, 7, 33, 80, 33, 33, 33]
    print("新FPS要求测试:")
    print_fps_analysis_report(new_fps_requirements)
