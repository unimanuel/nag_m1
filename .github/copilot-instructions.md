## Neural Atlas Graphs — Agent Instructions

Purpose: make AI coding agents productive quickly in this repo by capturing architecture, workflows, and project-specific conventions. Keep outputs concise, actionable, and aligned with existing patterns.

### Big Picture
- Core flow: [run script](nag/scripts/run_nag.py) → [runner](nag/run/nag_runner.py) → [model + dataset](nag/model/**, nag/dataset/nag_dataset.py) → [callbacks](nag/callbacks/nag_callback.py) for logging/eval.
- Training is driven by a dataclass config [NAGConfig](nag/config/nag_config.py) parsed from YAML or CLI. Many types are referenced via import-strings (e.g., `nag.sampling.random_timed_uv_grid_sampler.RandomTimedUVGridSampler`).
- The repo depends on an internal “tools” library (git submodule) wired via Poetry editable dependency; code uses it pervasively (logging, trackers, I/O, metrics).

### Dev Workflows
- Environment (Poetry; CUDA 12.4 assumed): see [installation](docs/installation.md). Install with
  - `poetry install --with dev --extras torch --extras post-torch` (add `--extras depth` if needed)
  - Build tinycudann from the maintained float32 fork; follow docs and install torch bindings into the active Poetry env.
- Datasets: download into `data/datasets` and set `DATA_PATH` in `.env`. See [datasets](docs/datasets.md).
- Train: `python nag/scripts/run_nag.py --config-path [path]`. Example configs in [config/davis](config/davis) and [config/waymo](config/waymo/final).
- Logging: default to TensorBoard unless `experiment_logger=wandb` and `WANDB_API_KEY` present (see [training](docs/training.md)).
- Outputs: runner writes under `runs/{date}/...` and saves images/plots during and after training (e.g., `in_training/`, `final/`). Metrics stored via `tools.agent.util.tracker.Tracker` as CSV and logger artifacts.

### Config & CLI Conventions
- Config dataclasses support placeholders and env:
  - `{data_path}` expands inside other paths; `{env:VAR}` reads from environment (see [blackswan.yaml](config/davis/blackswan.yaml)).
- CLI overrides: pass hyphenated names for config fields. Example:
  - `--learn-resolution-factor 0.5 --frame-indices-filter 0 2 4` (to downsample learning res and use a frame subset)
- Types are dynamic by string: e.g., `sampler_type`, `plane_init_strategy`, `model_type`, `background_plane_type` point to importable classes.
- Masks/Images/Depth patterns are configurable via regex fields in config; defaults match our datasets.

### Architecture Highlights
- Runner ([NAGRunner](nag/run/nag_runner.py)) builds/loads: dataset, model, sampler, PL trainer, and orchestrates training/eval, including checkpoint restore (`checkpoint_path`) and bundle loading (`bundle_path`).
- Dataset ([NAGDataset](nag/dataset/nag_dataset.py)) abstracts images/masks/depth and frame/timestamp indexing; supports caching and resizing policies.
- Model is scene-graph-like: planes (foreground/background) + camera as nodes; functional composition in [NAGFunctionalModel](nag/model/nag_functional_model.py). View-dependence and per-plane networks defined by encoding/network configs.
- Sampling: default `RandomTimedUVGridSampler` with `sampler_kwargs` controlling ray count/timestamps.
- Callbacks ([nag_callback.py](nag/callbacks/nag_callback.py)) handle in-training plots, periodic image dumps, object-wise renders, videos, metrics; respects config toggles and intervals.

### Integration Points & Patterns
- Strategies: Plane placement/initialization via `plane_init_strategy` and `plane_position_strategy` (see [strategy](nag/strategy)).
- Hooks: Gradient/alpha-chain rescaling and position sampling via string-resolved hooks (see [model/hooks](nag/model/hooks.py) and references in config).
- Boxes: Optional 3D bbox integration via `boxes_*` fields; Waymo-specific handling converts into world coordinates and validates coverage.
- Logging/Metrics: use `tools.logger.*` and `tools.metric.*`; keep new metrics torch-friendly and batchable like [PSNR/LPIPS](config/davis/blackswan.yaml#L113-L133).

### Practical Examples
- Run a DAVIS experiment: `python nag/scripts/run_nag.py --config-path config/davis/blackswan.yaml`
- Resume from checkpoint: `--checkpoint-path runs/.../checkpoints/epoch=XX.ckpt`
- Switch sampler rays on CLI: `--sampler-kwargs.num-rays 50000 --sampler-kwargs.num-timestamps 10`
- Flip logger to TB: set `experiment_logger=tensorboard` or remove `WANDB_API_KEY`.

### Project-Specific Gotchas
- Clone with submodules: `git clone --recurse-submodules ...` or ensure `tools/` available (Poetry uses it via editable path).
- CUDA/PyTorch index source must match your CUDA version; update `pyproject.toml` sources if not using 12.4.
- Tiny-cuda-nn: use the provided float32 fork to avoid half-precision instabilities.
- `.env` required for `DATA_PATH` (datasets) and optionally `WANDB_API_KEY`.

### When Extending
- New dataset → follow patterns in [nag/dataset](nag/dataset) and wire via config fields.
- New plane/node type → implement under [nag/model](nag/model) and make selectable via config string.
- New sampler/strategy/hook → implement under [nag/sampling](nag/sampling) or [nag/strategy](nag/strategy) or [nag/model/hooks.py](nag/model/hooks.py); expose via config.
- New metrics/plots → integrate via `nag/callbacks/nag_callback.py` or `tools.metric.*`, and register in config.

If any section seems incomplete for your task, tell us what’s missing and propose what to scan next (file paths welcome).
