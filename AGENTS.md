# Repository Guidelines

## Project Structure & Module Organization
- `NNScheduler/`: Core scheduling engine grouped into `core/`, `interface/`, `scenario/`, and `viz/`; update this tree for algorithm or API changes.
- `main_api.py`: Lightweight CLI front door that wraps the JSON optimization workflow used in tests.
- `test/`: Pytest suite plus `sample_config/` JSON inputs and a mirrored `NNScheduler/` fixture that exercises full-stack scheduling scenarios.
- `dist/`: Built wheels such as `ai_scheduler-1.0.0rc0-py3-none-any.whl`; inspect contents before publishing to confirm both `ai_scheduler/` and bundled assets are present.
- Docs: `CLI_AND_API_USAGE_GUIDE.md` and `DEVELOP.md` capture CLI usage and packing hints—update alongside feature work.

## Build, Test, and Development Commands
- `pip install -e .`: Editable install that exposes the `ai-scheduler` console script for local iteration.
- `python main_api.py test/sample_config/config_1npu_1dsp.json --output ./artifacts_sim`: Runs the reference pipeline without installing the package.
- `pytest`: Executes the full regression set under `test/`; use `pytest test/NNScheduler/test_simple_optimization.py -k priority` for targeted runs.
- `python setup.py bdist_wheel` or `python -m build --wheel`: Generates distributable wheels; clean `build/`, `dist/`, and `*.egg-info/` first.

## Coding Style & Naming Conventions
- Follow PEP 8 with 4-space indentation and descriptive snake_case identifiers; keep class names in PascalCase.
- Mirror existing module layout when adding optimizers or visualizers (e.g., keep new executors inside `NNScheduler/core/`).
- Prefer type hints in new modules and ensure configuration JSON keys stay lowercase with hyphen-free names.
- Run formatting before review—`ruff` is not bundled, so rely on `black` (`black NNScheduler`) if already installed locally.

## Testing Guidelines
- Pytest drives all scheduling regressions; keep new tests under `test/NNScheduler/` and name files `test_<feature>.py`.
- Validate new JSON samples with `python main_api.py --verbose <config>` prior to committing to avoid runtime surprises.
- Aim for scenario coverage: add minimal smoke tests plus one stress test that hits the newly introduced path.
- Capture large artifacts under `artifacts_sim/` or a temp directory and exclude them from source control.

## Commit & Pull Request Guidelines
- Use imperative, English commit subjects mirroring existing history (e.g., `Add hybrid bandwidth estimator`); include scope tags only when clarifying.
- Reference tracked issues in the body (`Refs #123`) and summarize configuration impacts or required migrations.
- Pull requests need: clear motivation, before/after behavior, validation commands, and screenshots for visualization tweaks.
- Rebase onto `main` before submitting, ensure all tests pass locally, and flag any follow-up work explicitly in the PR notes.
