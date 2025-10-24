"""Command-line interface exposed as the ``ai-scheduler`` console script."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Optional

from NNScheduler import __version__ as ENGINE_VERSION

from .optimization_api import (
    OptimizationAPI,
    copy_sample_config,
    list_sample_configs,
    load_sample_config,
)

APP_DESCRIPTION = "A heterogenous AI tasks simulator and scheduling optimizer"
APP_TEAM = "AIC"
APP_MAINTAINER = "Tristan.Qiu, Xiong.Guo, Neal.Nie"


def _print_banner() -> None:
    line = "=" * 80
    print(line)
    print(f"  AI Scheduler CLI - {APP_DESCRIPTION}")
    print(line)
    print(f"  Version: {ENGINE_VERSION}")
    print(f"  Maintainer: {APP_MAINTAINER}")
    print(f"  Team: {APP_TEAM}")
    print(line)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-scheduler",
        description="AI Scheduler command-line interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usages:
  ai-scheduler run config.json --output ./artifacts
  ai-scheduler run sample:config_1npu_1dsp.json --verbose
  ai-scheduler list-samples
  ai-scheduler copy-sample config_2npu_2dsp.json --dest ./configs/
        """.strip(),
    )

    parser.add_argument(
        "--version",
        action="version",
        version=f"AI Scheduler {ENGINE_VERSION}",
    )

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser(
        "run",
        help="Optimize a schedule from a JSON configuration",
    )
    run_parser.add_argument(
        "config",
        help="Path to JSON configuration or 'sample:<name>.json' to use a packaged sample",
    )
    run_parser.add_argument(
        "-o",
        "--output",
        dest="output_dir",
        help="Directory to store optimization artifacts (default: ./artifacts_sim)",
        default=None,
    )
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    run_parser.add_argument(
        "--no-banner",
        action="store_true",
        help="Suppress the ASCII banner",
    )

    subparsers.add_parser(
        "list-samples",
        help="List packaged sample configuration files",
    )

    copy_parser = subparsers.add_parser(
        "copy-sample",
        help="Copy a packaged sample configuration to the desired destination",
    )
    copy_parser.add_argument(
        "name",
        help="Sample file name (with or without .json)",
    )
    copy_parser.add_argument(
        "--dest",
        default=".",
        help="Destination directory or file path (default: current directory)",
    )
    copy_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists",
    )

    show_parser = subparsers.add_parser(
        "show-sample",
        help="Print a packaged sample configuration to stdout",
    )
    show_parser.add_argument(
        "name",
        help="Sample file name (with or without .json)",
    )

    return parser


def _ensure_command(argv: Iterable[str]) -> list[str]:
    argv_list = list(argv)
    known_commands = {"run", "list-samples", "copy-sample", "show-sample"}
    if argv_list and argv_list[0] not in known_commands and not argv_list[0].startswith("-"):
        return ["run", *argv_list]
    return argv_list


def _run_command(args: argparse.Namespace) -> int:
    config_arg: str = args.config
    output_dir = args.output_dir
    verbose = args.verbose

    if not args.no_banner:
        _print_banner()

    api = OptimizationAPI()

    try:
        if config_arg.startswith("sample:"):
            sample_name = config_arg.split(":", 1)[1]
            config_data = load_sample_config(sample_name)
            if verbose:
                print(f"[ai-scheduler] Loaded packaged sample: {sample_name}")
            result = api.optimize_from_config(config_data, output_dir=output_dir, verbose=verbose)
        else:
            result = api.optimize_from_json(config_arg, output_dir=output_dir, verbose=verbose)
    except FileNotFoundError as exc:
        print(f"[ai-scheduler] ERROR: {exc}")
        return 1
    except Exception as exc:  # pragma: no cover - surface unexpected errors
        print(f"[ai-scheduler] ERROR: {exc}")
        return 1

    best = result.get("best_configuration", {})
    satisfaction = best.get("satisfaction_rate")
    if satisfaction is not None:
        print(f"[ai-scheduler] Satisfaction rate: {satisfaction:.2%}")

    output_file = result.get("output_file")
    if output_file:
        print(f"[ai-scheduler] Optimization result saved to: {output_file}")

    visuals = result.get("visualization_files", {}) or {}
    for label, path in visuals.items():
        print(f"[ai-scheduler] {label.replace('_', ' ').title()} -> {path}")

    return 0


def _list_samples_command(args: argparse.Namespace) -> int:
    samples = list_sample_configs()
    if not samples:
        print("No packaged samples found.")
        return 0

    for name in samples:
        print(name)
    return 0


def _copy_sample_command(args: argparse.Namespace) -> int:
    sample_name = args.name
    destination = Path(args.dest).expanduser()

    api = OptimizationAPI()
    try:
        resolved_target = api.copy_sample_config(sample_name, destination, overwrite=args.overwrite)
    except (FileExistsError, FileNotFoundError) as exc:
        print(exc)
        return 1

    print(f"Sample '{sample_name}' copied to {resolved_target}")
    return 0


def _show_sample_command(args: argparse.Namespace) -> int:
    sample_name = args.name
    config = load_sample_config(sample_name)
    import json as json_module

    print(json_module.dumps(config, indent=2, ensure_ascii=False))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    argv = _ensure_command(argv or sys.argv[1:])
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        return _run_command(args)
    if args.command == "list-samples":
        return _list_samples_command(args)
    if args.command == "copy-sample":
        return _copy_sample_command(args)
    if args.command == "show-sample":
        return _show_sample_command(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
