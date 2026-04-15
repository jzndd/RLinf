# Offline OpenPI GRPO validation

## Static checks

- `python -m py_compile` passed for the new dataset, worker, runner, and train entry files.
- Helper tensor checks passed for observation repetition, reward tensor shape, and synthetic rollout batch packing.
- Hydra compose plus `materialize_offline_dimensions(cfg)` plus `validate_cfg(cfg)` passed.

## Dataset checks

- `OpenPIGRPODataset` built successfully from `/mnt/project_rlinf/jzn/workspace/openpi/data/libero_130_oneshot`.
- Current offline config resolves to `18865` valid samples.
- One sample returns `obs.state.shape == (8,)` and `actions.shape == (10, 7)`.

## Smoke run

Command used:

```bash
source /opt/venv/openpi/bin/activate
export PYTHONPATH=/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf:$PYTHONPATH
export EMBODIED_PATH=/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/examples/embodiment
cd /mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/examples/embodiment
python train_offline_grpo_openpi.py \
  runner.max_steps=1 \
  runner.save_interval=1000 \
  runner.logger.experiment_name=offline_openpi_grpo_smoke \
  data.prompt_batch_size_per_gpu=1 \
  data.num_workers=0 \
  algorithm.group_size=2 \
  algorithm.update_epoch=1 \
  actor.micro_batch_size=2
```

Observed result:

- The actor group launched on 4 H100 GPUs.
- The pi05 checkpoint loaded successfully.
- One offline GRPO step completed successfully.
- A checkpoint for `global_step_1` was written and then cleaned from the repo workspace after validation.

## Fixes discovered during validation

- Offline sampling cannot call `predict_action_batch()` directly on the FSDP wrapper. The fix was to route sampling through `self.model(forward_type=ForwardType.ACTION_SAMPLING, ...)`.
- The offline runner expected a timer for `compute_advantages_and_returns`. The fix was to add a timed override in `OfflineOpenPIGRPOActor`.
