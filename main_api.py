#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Scheduler Main API Entry Point

Version: 1.0
Maintainer: Tristan.Qiu, Xiong.Guo, and Neal.Nie
Team: AIC (Artificial Intelligence Compilation)
Description: A Multi-task Neural Network Task Simulator with Automatic Optimization

This is the main entry point for the AI Scheduler API.
It provides a simple interface to the optimization functionality.
"""

import sys
import os
import argparse
from pathlib import Path

# Version and metadata
__version__ = "1.0"
__maintainer__ = "Tristan.Qiu, Xiong.Guo, Neal.Nie"
__team__ = "AIC"
__description__ = "A Multi-task Neural Network Task Simulator with Automatic Optimization"

def print_banner():
    """Print the application banner with version info."""
    print("=" * 80)
    print(f"  AI Scheduler - {__description__}")
    print("=" * 80)
    print(f"  Version: {__version__}")
    print(f"  Maintainer: {__maintainer__}")
    print(f"  Team: {__team__}")
    print("=" * 80)

def main():
    """Main entry point for the AI Scheduler API."""

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description=f"AI Scheduler v{__version__} - {__description__}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  python main_api.py config.json
  python main_api.py test/sample_config/config_1npu_1dsp.json
  python main_api.py test/sample_config/config_2npu_1dsp.json --output results/

Maintainer: {__maintainer__} (Team: {__team__})
        """
    )

    parser.add_argument(
        "config_file",
        help="Path to the JSON configuration file"
    )

    parser.add_argument(
        "-o", "--output",
        help="Output directory for results (optional)",
        default=None
    )

    parser.add_argument(
        "-v", "--version",
        action="version",
        version=f"AI Scheduler {__version__}"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress banner output"
    )

    args = parser.parse_args()

    # Print banner unless suppressed
    if not args.no_banner:
        print_banner()

    # Validate config file exists
    config_path = Path(args.config_file)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        print("\nAvailable sample configurations:")
        sample_dir = Path("test/sample_config")
        if sample_dir.exists():
            for config in sample_dir.glob("*.json"):
                print(f"  {config}")
        sys.exit(1)

    if args.verbose:
        print(f"[INFO] Loading configuration: {config_path}")
        if args.output:
            print(f"[INFO] Output directory: {args.output}")

    # Import and run the actual API
    try:
        # Get project root directory
        project_root = Path(__file__).parent

        # Change to project root directory to ensure relative paths work
        original_cwd = os.getcwd()
        os.chdir(project_root)

        # Add the project root to the Python path
        sys.path.insert(0, str(project_root))

        # Import the NNScheduler module directly
        import importlib.util

        # Load the minimal json api module directly
        api_module_path = project_root / "test" / "NNScheduler" / "test_minimal_json_api.py"
        spec = importlib.util.spec_from_file_location("test_minimal_json_api", api_module_path)
        api_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(api_module)

        # Convert to absolute path to handle any working directory issues
        abs_config_path = Path(config_path).resolve()

        # Set default output directory if not provided
        output_dir = args.output if args.output else "./artifacts_sim"

        # Run the optimization
        print(f"[INFO] Starting optimization with config: {abs_config_path}")
        api_module.run_optimization_from_json(str(abs_config_path), output_dir)

        # Restore original working directory
        os.chdir(original_cwd)

    except ImportError as e:
        # Restore original working directory on error
        os.chdir(original_cwd)
        print(f"ERROR: Failed to import required modules: {e}")
        print("Please ensure all dependencies are installed and the project structure is correct.")
        sys.exit(1)
    except Exception as e:
        # Restore original working directory on error
        os.chdir(original_cwd)
        print(f"ERROR: Optimization failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
