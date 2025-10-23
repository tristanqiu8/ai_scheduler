"""High-level wrapper utilities around the core NNScheduler interfaces."""

from __future__ import annotations

import json
import os
import shutil
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import importlib.resources as pkg_resources

import ai_scheduler.sample_config as sample_pkg
from NNScheduler.core import artifacts as artifact_utils
from NNScheduler.interface.json_interface import JsonInterface
from NNScheduler.interface.optimization_interface import OptimizationInterface

SampleName = str


def _normalize_sample_name(name: SampleName) -> SampleName:
    if not name:
        raise FileNotFoundError("Sample name cannot be empty")
    name = name.strip()
    if not name.endswith(".json"):
        name += ".json"
    return name


def _get_sample_resource(name: SampleName) -> SampleName:
    normalized = _normalize_sample_name(name)
    if not pkg_resources.is_resource(sample_pkg, normalized):
        raise FileNotFoundError(f"Sample config '{normalized}' not found")
    return normalized


def list_sample_configs() -> List[SampleName]:
    """Return the available packaged sample configuration filenames."""

    files: List[SampleName] = []
    try:
        contents = pkg_resources.contents(sample_pkg)
    except (AttributeError, TypeError):  # pragma: no cover - older Python fallback
        contents = None
    if contents is not None:
        for entry in contents:
            if entry.endswith(".json") and pkg_resources.is_resource(sample_pkg, entry):
                files.append(entry)
    else:  # pragma: no cover - fallback to files() API if available
        package_root = getattr(pkg_resources, "files", lambda _: None)(sample_pkg)
        if package_root is not None:
            for entry in package_root.iterdir():
                if entry.is_file() and entry.name.endswith(".json"):
                    files.append(entry.name)
    return sorted(files)


def load_sample_config(name: SampleName) -> Dict[str, Any]:
    """Load a packaged sample configuration as a Python dictionary."""

    resource_name = _get_sample_resource(name)
    with pkg_resources.open_text(sample_pkg, resource_name, encoding="utf-8") as handle:
        return json.load(handle)


def copy_sample_config(
    name: SampleName,
    destination: Union[str, os.PathLike[str]],
    *,
    overwrite: bool = False,
) -> Path:
    """Copy a packaged sample configuration to *destination* and return the file path."""

    resource_name = _get_sample_resource(name)
    dest_path = Path(destination).expanduser()
    destination_str = str(destination)
    if dest_path.exists() and dest_path.is_dir():
        dest_path = dest_path / resource_name
    elif destination_str.endswith(os.sep) or dest_path.suffix == "":
        dest_path = dest_path / resource_name
    dest_path = dest_path.resolve()
    if dest_path.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dest_path}")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with pkg_resources.open_binary(sample_pkg, resource_name) as src, open(dest_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return dest_path


def _configure_artifacts_dir(path: Optional[Union[str, os.PathLike[str]]]) -> Path:
    """Ensure the artifacts root points to *path* (or default) and return it."""

    if path is not None:
        resolved = Path(path).expanduser().resolve()
        os.environ[artifact_utils.ARTIFACTS_ENV_VAR] = str(resolved)
    else:
        os.environ.pop(artifact_utils.ARTIFACTS_ENV_VAR, None)

    artifact_utils._ARTIFACTS_ROOT = None  # type: ignore[attr-defined]
    return artifact_utils.get_artifacts_root()


def _resolve_output_paths(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert relative artifact filenames in *result* to absolute paths."""

    resolved: Dict[str, Any] = deepcopy(result)

    output_file = resolved.get("output_file")
    if isinstance(output_file, str):
        resolved["output_file"] = str(
            artifact_utils.ensure_artifact_path(output_file, ensure_parent=False)
        )

    visuals = resolved.get("visualization_files")
    if isinstance(visuals, dict):
        resolved_visuals: Dict[str, str] = {}
        for key, value in visuals.items():
            if isinstance(value, str):
                resolved_visuals[key] = str(
                    artifact_utils.ensure_artifact_path(value, ensure_parent=False)
                )
        resolved["visualization_files"] = resolved_visuals

    return resolved


class OptimizationAPI:
    """User-friendly bridge over :class:`OptimizationInterface` with packaging helpers."""

    def __init__(self, artifacts_dir: Optional[Union[str, os.PathLike[str]]] = None):
        self._interface = OptimizationInterface()
        self._artifacts_root: Optional[Path] = None
        if artifacts_dir is not None:
            self.set_artifacts_dir(artifacts_dir)

    # ------------------------------------------------------------------
    # Artifacts management helpers
    # ------------------------------------------------------------------
    def set_artifacts_dir(self, path: Union[str, os.PathLike[str]]) -> Path:
        """Persistently route future artifacts to *path*."""

        root = _configure_artifacts_dir(path)
        self._artifacts_root = root
        return root

    def get_artifacts_dir(self) -> Path:
        """Return the currently active artifacts directory."""

        if self._artifacts_root is None:
            self._artifacts_root = _configure_artifacts_dir(None)
        return self._artifacts_root

    def _prepare_artifacts_dir(self, override: Optional[Union[str, os.PathLike[str]]]) -> Path:
        if override is not None:
            self._artifacts_root = _configure_artifacts_dir(override)
        elif self._artifacts_root is not None:
            _configure_artifacts_dir(self._artifacts_root)
        return self.get_artifacts_dir()

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    def optimize_from_json(
        self,
        config_file: Union[str, os.PathLike[str]],
        output_dir: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # 在 verbose 模式下改为读取配置并走 optimize_from_config，方便覆写 log_level
        if verbose:
            config = JsonInterface.load_from_file(str(config_path))
            return self.optimize_from_config(
                config,
                output_dir=output_dir,
                verbose=verbose,
            )

        artifacts_dir = self._prepare_artifacts_dir(output_dir)
        result = self._interface.optimize_from_json(str(config_path))
        if verbose:
            # 由于前面未打印，仍然补充基础信息
            print(f"[ai-scheduler] Using artifacts directory: {artifacts_dir}")
            print(f"[ai-scheduler] Optimizing configuration: {config_path}")
        return _resolve_output_paths(result)

    def optimize_from_config(
        self,
        config: Dict[str, Any],
        output_dir: Optional[Union[str, os.PathLike[str]]] = None,
        *,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        artifacts_dir = self._prepare_artifacts_dir(output_dir)

        working_config = deepcopy(config)
        if verbose:
            print(f"[ai-scheduler] Using artifacts directory: {artifacts_dir}")
            print("[ai-scheduler] Optimizing provided configuration dictionary")
            optimization_block = working_config.setdefault("optimization", {})
            optimization_block["log_level"] = "detailed"

        result = self._interface.optimize_from_config(working_config)
        return _resolve_output_paths(result)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    def validate_config(
        self,
        config_source: Union[str, os.PathLike[str], Dict[str, Any]],
    ) -> Dict[str, Any]:
        if isinstance(config_source, (str, os.PathLike)):
            config = JsonInterface.load_from_file(str(config_source))
        else:
            config = deepcopy(config_source)

        scenario = config.get("scenario", {})
        tasks = JsonInterface.parse_scenario_config(scenario)
        resources = config.get("resources", {}).get("resources", [])
        optimization = config.get("optimization", {})

        return {
            "task_count": len(tasks),
            "resource_count": len(resources),
            "launch_strategy": optimization.get("launch_strategy", "balanced"),
            "max_iterations": optimization.get("max_iterations"),
            "max_time_seconds": optimization.get("max_time_seconds"),
        }

    # ------------------------------------------------------------------
    # Sample utilities (instance-level proxies)
    # ------------------------------------------------------------------
    def list_sample_configs(self) -> List[SampleName]:
        return list_sample_configs()

    def load_sample_config(self, name: SampleName) -> Dict[str, Any]:
        return load_sample_config(name)

    def copy_sample_config(
        self,
        name: SampleName,
        destination: Union[str, os.PathLike[str]],
        *,
        overwrite: bool = False,
    ) -> Path:
        return copy_sample_config(name, destination, overwrite=overwrite)
