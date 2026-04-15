# Libero Spatial Offline GRPO Diagnosis (2026-04-16)

## Scope

- Config under investigation: `/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/examples/embodiment/config/libero_spatial_offline_grpo_openpi_pi05.yaml`.
- Training log inspected: `/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260415-13:18:48-libero_spatial_offline_grpo_openpi_pi05/run_offline_grpo_openpi.log`.
- TensorBoard inspected: `/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/logs/20260415-13:18:48-libero_spatial_offline_grpo_openpi_pi05/tensorboard`.

## Checklist

1. Read the offline spatial config and the matching online spatial GRPO config.
2. Inspect TensorBoard scalars to confirm the failure mode instead of inferring it.
3. Compare offline OpenPI GRPO reward/logprob/advantage handling with the online embodied GRPO path.
4. Probe the spatial checkpoint on real offline samples to see which action horizon actually carries reward signal.
5. Apply the minimal code and hyperparameter changes that address the observed issues.
6. Run a short smoke test with the modified config before launching the formal training job.

## What the TensorBoard Run Showed

- `rollout/reward_mean` stayed essentially flat across 1000 steps, around `0.687 -> 0.686`.
- `rollout/l1_mean` improved only slightly, around `0.104 -> 0.099`.
- `train/actor/ratio` stayed very close to `1.0`, and `train/actor/approx_kl` decayed to a small value.
- This pattern indicates that training was stable but the effective policy update signal was weak.

## Findings

### 1. The offline actor skipped the online rollout post-processing path

The online embodied GRPO path runs `_process_received_rollout_batch()` before computing advantages. That path is where loss masks and optional reward filtering are aligned with the embodied GRPO implementation. The offline actor built `self.rollout_batch` manually and sent it directly into `compute_advantages_and_returns()`, so the offline batch never went through the same post-processing stage.

Change made:
- After building the offline rollout batch, call `_process_received_rollout_batch(self.rollout_batch)`.
- Add an explicit assertion that offline GRPO currently expects `algorithm.rollout_epoch == 1`, because the offline actor materializes exactly one rollout batch per training step.

### 2. The offline spatial config optimized a 10-step chunk, but the spatial reward signal is much stronger in the first 5 steps

A direct probe on spatial-fullshot data with the base checkpoint showed the reward quality drops materially in the second half of the 10-step chunk. On one diagnostic batch:

- mean reward over the first 5 steps: about `0.627`
- mean reward over the last 5 steps: about `0.522`
- per-step reward decayed from about `0.60-0.67` early to about `0.44` by the 10th step

This matches two reference signals in the codebase:

- the online spatial GRPO config uses `num_action_chunks: 5` and `num_steps: 3`
- the reference L1 reward implementation in `LifeLong-RFT/.../plugin.py` uses a `time_horizon` of `5`

Change made:
- Set `actor.model.num_action_chunks: 5`
- Set `actor.model.num_steps: 3`
- Because `data.action_chunk` is tied to `actor.model.openpi.action_chunk`, the offline dataset target horizon now also becomes 5.

### 3. The offline spatial batch size was too small for the available 80G GPUs

The previous config used:

- `prompt_batch_size_per_gpu: 8`
- `micro_batch_size: 8`

On 4 GPUs, that gives:

- runtime global batch size: `8 * 8 * 4 = 256`
- per-rank rollout batch: `8 * 8 = 64`
- gradient accumulation: `64 / 8 = 8`

This is conservative for the available hardware and leaves update statistics relatively noisy.

Change made:
- Set `prompt_batch_size_per_gpu: 16`
- Set `micro_batch_size: 32`

Now the derived quantities are:

- runtime global batch size: `16 * 8 * 4 = 512`
- per-rank rollout batch: `16 * 8 = 128`
- gradient accumulation: `128 / 32 = 4`

### 4. The online `filter_rewards` thresholds do not transfer directly to this offline dense-reward setup

The online spatial config filters prompt groups with summed rewards in `[0.1, 0.9]`. That threshold range is not compatible with the offline dense L1 reward used here, because offline prompt-level scores are the sum of multiple positive per-step rewards and therefore land much higher. A probe on the current offline spatial setup showed prompt scores around `6.7-7.5`, so the online thresholds would drop every prompt group.

Change made:
- Keep `algorithm.filter_rewards: False` in the offline spatial config.
- Add a config comment so this mismatch is explicit and does not get reintroduced by accident.

### 5. `unnorm_key` was inconsistent with the spatial online baseline

The offline spatial config still used `rollout.unnorm_key: libero_130`, while the corresponding online spatial GRPO config uses `libero_10`. This field is not the main cause of the flat offline reward curve, but it was an unnecessary spatial/libero-130 mismatch.

Change made:
- Set `rollout.unnorm_key: libero_10` to match the spatial online baseline.

## Why the Learning Rate Was Not Changed

The actor learning rate was left at `5e-6`.

Reasoning:

- the online spatial GRPO baseline also uses `5e-6`
- the more important issues were horizon mismatch, skipped rollout post-processing, and under-sized batch configuration
- changing several optimizer knobs at once would make it harder to attribute any improvement

If the next run still shows `ratio` tightly pinned to `1.0` with very small `approx_kl`, then the next knob to test should be a moderate learning-rate increase, for example `7.5e-6` or `1.0e-5`, but that was intentionally deferred until after the structural fixes above.

## Files Changed

- `/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/rlinf/workers/actor/fsdp_offline_openpi_grpo_worker.py`
- `/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/examples/embodiment/config/libero_spatial_offline_grpo_openpi_pi05.yaml`
- `/mnt/project_rlinf/jzn/workspace/continual_learning/RLinf/docs/notes/libero_spatial_offline_grpo_diagnosis_20260416.md`

## Expected Outcome After These Changes

- offline GRPO now uses the same rollout-batch post-processing entry point as the online embodied GRPO path
- the offline spatial objective now focuses on the higher-signal first 5 action steps instead of averaging in a much weaker 6-10 step tail
- the effective batch size is larger and should make updates less noisy while using more of the available GPU memory
- the remaining signal quality can now be judged more fairly before considering a learning-rate change
