# Offline OpenPI GRPO pipeline

## Goal

Add an actor-only Offline GRPO path for OpenPI pi05 that reuses RLinf embodied GRPO loss code without launching env or rollout workers.

## Batch contract

- Dataset returns one observation frame and one fixed-length future action chunk.
- The actor repeats each observation `group_size` times and samples a group of action chunks.
- Reward is computed offline as `-mse(sampled_action, groundtruth_action)` per action position.
- The sampled logprob and forward inputs are packed into the same rollout batch structure expected by `EmbodiedFSDPActor.run_training()`.
- `algorithm.rollout_epoch` is set to `1` because one offline batch corresponds to one synthetic rollout step.

## Files

- `rlinf/workers/actor/fsdp_offline_openpi_grpo_worker.py`
- `rlinf/runners/offline_embodied_grpo_runner.py`
- `examples/embodiment/train_offline_grpo_openpi.py`
- `examples/embodiment/config/libero_130_offline_grpo_openpi_pi05.yaml`

## MVP checks

1. Import helper functions and verify repeated obs, reward tensor, and synthetic rollout batch shapes with fake tensors.
2. `python -m py_compile` the new worker, runner, and entry script.
3. Compose the new Hydra config, run `materialize_offline_dimensions(cfg)`, then `validate_cfg(cfg)`.
4. Confirm the resolved config points to the offline dataset path and a local pi05 checkpoint.

## Run entry

```bash
source /opt/venv/openpi/bin/activate
export PYTHONPATH=/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf:$PYTHONPATH
cd /mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/examples/embodiment
python train_offline_grpo_openpi.py
```
