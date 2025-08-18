"""
AI Scheduler Interface Module
提供JSON和Web API两层接口
"""

from interface.json_interface import JsonInterface
from interface.api_client_example import SchedulerAPIClient

__all__ = ['JsonInterface', 'SchedulerAPIClient']