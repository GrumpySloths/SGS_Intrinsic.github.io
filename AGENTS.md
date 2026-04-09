# AGENTS.md

## Scope
- This repository is a Python research codebase (not a JS/TS monorepo). There is no `package.json`, no `pyproject.toml`, no root `pytest.ini`, and no CI workflow in `.github/workflows/`.
- Prefer executable sources over README prose: `README.md` still references commands/files that are not present in this checkout.

## High-value entrypoints (verified)
- Main training/eval pipeline (compare_methods):
  - `compare_methods/train_r3dg.py` (training entrypoint)
  - `compare_methods/eval_nvs.py` (evaluation/render entrypoint)
- Generic rendering:
  - `render.py`
- Metrics:
  - `metrics.py` (expects `--model_paths`)
- Batch/orchestration scripts:
  - `compare_methods/run_eval_r3dg.sh`
  - `compare_methods/run_myscene.sh`, `compare_methods/run_myscene_v2.sh`
  - `scripts/train_rgbx_nviews.sh`, `scripts/train_rgbx_mipnerf_nviews.sh`

## Architecture that affects edits
- Render mode dispatch is centralized in `gaussian_renderer/__init__.py` via `render_fn_dict`.
  - `compare_methods/train_r3dg.py` and `compare_methods/eval_nvs.py` both select renderers through this mapping.
  - If you add a new render type, update `render_fn_dict` and the CLI `--type` choices together.
- Scene/data wiring is in `scene/__init__.py` (`Scene` class).
  - Dataset recognition is strict (`source_path/sparse` for Colmap-like data, or `transforms_train.json` for Blender).
  - Unknown layouts fail fast (`assert False, "Could not recognize scene type!"`).
- Parameter groups are defined in `arguments/`:
  - `arguments/__init__.py` used by `render.py`.
  - `arguments/config.py` used by `compare_methods/train_r3dg.py` and `compare_methods/eval_nvs.py`.

## Environment and dependency quirks
- Two environment manifests exist and are inconsistent:
  - `environment.yaml` pins Python 3.8 and Torch `1.12.1+cu113`.
  - `compare_methods/requirements.txt` pins Torch `2.1.0+cu118` and many CUDA 11/12 libraries.
- There are local-path dependencies in `compare_methods/requirements.txt` (e.g., `simple-knn @ file:///home/jiahao/...`, `nvdiffrast @ file:///home/jiahao/...`) and editable Git dependencies.
  - Expect installation failures on clean machines unless these paths are replaced or submodules are prepared.
- Some scripts export local proxy settings (`127.0.0.1:7890`), e.g. `scripts/train_rgbx_mipnerf_nviews.sh`, `compare_methods/train_r3dg_rgbx.sh`.

## Command truths (what actually runs)
- Canonical compare-method flow in scripts is:
  1) train 3DGS (`train.py`)
  2) train NeILF/SGS with `-c <3DGS checkpoint>`
  3) evaluate (`eval_nvs.py`)
  - See `compare_methods/run_eval_r3dg.sh` for concrete sequencing and flags.
- `render.py` usage is driven by CLI flags (`--skip_train`, `--skip_test`, `--video`, `--render_all_imgs`, `--render_depth`).
- `metrics.py` requires `-m/--model_paths` (list) and uses CUDA (`torch.cuda.set_device`).

## Known mismatches / gotchas to avoid
- `scripts/script_for_llff.py` calls `python train.py`, but `train.py` is not present in this repository.
- `scripts/train_rgbx_nviews.sh` and `scripts/train_rgbx_mipnerf_nviews.sh` invoke `train_modify_rgbx.py`, which is not present in this repository.
- Several script templates (`script_for_dtu.py`, `script_for_sds.py`) include placeholder paths like `<dataset_scene_dir>` and are not directly runnable without editing.

## Testing reality
- There is no repo-level automated test command discovered.
- Existing tests are under `pbr/renderutils/tests/` and are GPU/CUDA extension checks (not lightweight unit tests):
  - `pbr/renderutils/tests/test_bsdf.py`
  - `pbr/renderutils/tests/test_perf.py`
  - `pbr/renderutils/tests/test_mesh.py`
  - `pbr/renderutils/tests/test_loss.py`

## Practical workflow for agents
- Before changing training behavior, inspect and keep aligned:
  - `compare_methods/train_r3dg.py`
  - `arguments/config.py`
  - `gaussian_renderer/__init__.py`
  - `scene/__init__.py`
- Before running long jobs, verify script-referenced files exist in this checkout (some historical scripts point to missing entrypoints).
- Prefer running one scene first with explicit `-s`/`-m` paths before using the multi-GPU schedulers.
