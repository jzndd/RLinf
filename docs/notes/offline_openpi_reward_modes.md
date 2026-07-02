# Offline OpenPI Reward Modes

- `algorithm.offline_reward_fn` controls the offline GRPO reward path and accepts `mse`, `l1`, `velocity`, or `l1_velocity`.
- `mse` keeps the original behavior: reward is `-mean((pred - gt)^2)` over the full action dimension for each action chunk step.
- `l1` follows `Action_Chunk_L1_V2_Reward`: the first 6 action dims use `mean(abs(pred - gt))`, the reward term is `0.8 * exp(-5 * l1)`, and the gripper term contributes `0.2 * match`.
- `velocity` computes OpenPI's flow-matching SFT loss on each sampled model-space action and uses `exp(-5 * mean(loss))` over the configured action chunk and env action dimensions as the reward.
- `l1_velocity` combines the two bounded similarity rewards as `0.5 * l1_reward + 0.5 * velocity_reward`, where `l1_reward = 0.8 * exp(-5 * l1) + 0.2 * gripper_match` and `velocity_reward = exp(-5 * velocity_mse)`.
- For `libero_130_oneshot`, ground-truth gripper values are discrete in `{-1, 1}`. The pi05 output path does not discretize gripper values in `output_transform()`, so the offline L1 reward binarizes the predicted gripper at threshold `0` before matching.
