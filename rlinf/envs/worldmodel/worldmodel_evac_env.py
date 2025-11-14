# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from copy import deepcopy
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from einops import rearrange
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation as R

from rlinf.envs.worldmodel.evac.lvdm.data.domain_table import DomainTable
from rlinf.envs.worldmodel.evac.lvdm.data.statistics import StatisticInfo
from rlinf.envs.worldmodel.evac.utils.general_utils import (
    instantiate_from_config,
    load_checkpoints,
)

__all__ = ["EvacEnv"]


class EvacEnv:
    def __init__(self, cfg, seed_offset, total_num_processes, record_metrics=True):
        self.cfg = cfg

        # Load basic configuration information
        self.seed = cfg.seed + seed_offset
        self.total_num_processes = total_num_processes
        self.num_envs = cfg.num_envs
        self._is_start = True
        self.record_metrics = record_metrics
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.use_rel_reward = cfg.use_rel_reward

        # Video recording
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []

        # Load model
        self.device = torch.device(cfg.device)
        self.config = self._load_config()
        self.model = self._load_model()
        self.reward_model = self._load_reward_model()
        self.model.eval()
        self.model.to(self.device)

        # Model hyperparameters
        self.chunk = self.config.chunk
        self.n_previous = self.config.n_previous
        self.sample_size = tuple(self.config.data.params.train.params.sample_size)

        # Initialize camera parameters
        self._init_camera_params()

        # Initialize state
        # For multi-env, we maintain separate current_obs for each environment
        # Each current_obs has shape [b, c, v, t, h, w] where b=1, v=1
        self.current_obs = (
            None  # Will be a list for multi-env, or single tensor for single env
        )
        self.prev_step_reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.device
        )
        self.task_descriptions = [""] * self.num_envs
        self.init_ee_poses = [None] * self.num_envs

        # Initialize inference-related state
        self.all_x_samples = None
        self.all_samples = None
        self.all_c2ws_list = None
        self.all_trajs = None
        self.x_samples = None
        self.i_chunk = 0

        # Initialize data preprocessing
        self.trans_resize = transforms.Compose(
            [
                transforms.Resize(self.sample_size),
            ]
        )
        self.trans_norm = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True
                ),
            ]
        )

        # Inference parameters
        self.inference_dtype = torch.float32
        self.ddim_steps = 50
        self.ddim_eta = 1.0
        self.unconditional_guidance_scale = 7.5
        self.guidance_rescale = 0.7
        self.use_cat_mask = True
        self.sparse_memory = True

        # Set model mode
        self.model.rand_cond_frame = False
        self.model.ddim_num_chunk = 1

        if self.record_metrics:
            self._init_metrics()

    @property
    def elapsed_steps(self):
        if not hasattr(self, "_elapsed_steps"):
            self._elapsed_steps = 0
        return self._elapsed_steps

    @elapsed_steps.setter
    def elapsed_steps(self, value):
        self._elapsed_steps = value

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _load_config(self):
        config_file = self.cfg.config_path
        config = OmegaConf.load(config_file)
        if hasattr(self.cfg, "ckp_path") and self.cfg.ckp_path:
            config.model.pretrained_checkpoint = self.cfg.ckp_path
        return config

    def _load_model(self):
        model = instantiate_from_config(self.config.world_model_cfg)
        model = load_checkpoints(
            model, self.config.world_model_cfg, ignore_mismatched_sizes=False
        )
        return model

    def _load_reward_model(self):
        reward_model = instantiate_from_config(self.config.reward_model_cfg)
        reward_model = load_checkpoints(
            reward_model, self.config.reward_model_cfg, ignore_mismatched_sizes=False
        )
        return reward_model

    def _init_camera_params(self):
        """Initialize camera intrinsic and extrinsic parameters"""
        # Fixed intrinsic parameters
        self.intrinsic = torch.tensor(
            [[64, 0, 64], [0, 64, 64], [0, 0, 1]], dtype=torch.float32
        )

        # Fixed extrinsic parameters
        pos = np.array([0.30000001192092896, 0.0, 0.6000000238418579])
        quat_wxyz = np.array([0.0, -0.43318870663642883, 0.0, 0.901303231716156])
        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
        rot = R.from_quat(quat_xyzw).as_matrix()

        self.c2w = torch.eye(4, dtype=torch.float32)
        self.c2w[:3, :3] = torch.from_numpy(rot).float()
        self.c2w[:3, 3] = torch.from_numpy(pos).float()
        self.w2c = torch.linalg.inv(self.c2w)

        # Repeat for temporal dimension
        self.c2w_list = self.c2w.unsqueeze(0).repeat(self.n_previous, 1, 1)
        self.w2c_list = self.w2c.unsqueeze(0).repeat(self.n_previous, 1, 1)

    def _get_action_bias_std(self, domain_name="agibotworld"):
        return torch.tensor(StatisticInfo[domain_name]["mean"]).unsqueeze(
            0
        ), torch.tensor(StatisticInfo[domain_name]["std"]).unsqueeze(0)

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.returns[mask] = 0
            self._elapsed_steps = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.returns[:] = 0.0
            self._elapsed_steps = 0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = torch.tensor(self.elapsed_steps).to(self.device)
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    def _calc_step_reward(self, chunk_rewards):
        """Calculate step reward"""
        reward_diffs = []
        for i in range(self.chunk):
            reward_diff = reward[i] - self.prev_step_reward
            reward_diffs.append(reward_diff)
            self.prev_step_reward = reward[i]

        if self.use_rel_reward:
            return reward_diffs
        else:
            return chunk_rewards

    def reset(
        self,
        *,
        seed: Optional[Union[int, list[int]]] = None,
        options: Optional[dict] = {},
    ):
        self.current_step = 0
        """Reset the environment with initial observation from randomly selected npy files"""
        # Get data directory from options or config
        if "initial_image_path" in options:
            initial_image_path = options["initial_image_path"]
        elif hasattr(self.cfg, "initial_image_path") and self.cfg.initial_image_path:
            initial_image_path = self.cfg.initial_image_path
        else:
            raise ValueError(
                "initial_image_path must be provided for reset (either in options or cfg)"
            )

        # List all npy files in the directory
        npy_files = [f for f in os.listdir(initial_image_path) if f.endswith(".npy")]
        if len(npy_files) == 0:
            raise ValueError(f"No .npy files found in directory: {initial_image_path}")

        # Randomly select self.num_envs files
        num_envs = self.num_envs
        if len(npy_files) < num_envs:
            raise ValueError(
                f"Not enough npy files in directory. Found {len(npy_files)}, need {num_envs}"
            )

        # Set random seed if provided
        if seed is not None:
            if isinstance(seed, list):
                np.random.seed(seed[0])
            else:
                np.random.seed(seed)

        selected_files = np.random.choice(npy_files, size=num_envs, replace=False)

        # Load first frame from each selected npy file
        img_tensors = []
        task_descriptions = []
        init_ee_poses = []
        for npy_file in selected_files:
            npy_path = os.path.join(initial_image_path, npy_file)
            traj_data = np.load(npy_path, allow_pickle=True)

            # Extract first frame (index 0)
            if len(traj_data) == 0:
                raise ValueError(f"Empty trajectory file: {npy_path}")

            first_frame = traj_data[0]
            if not isinstance(first_frame, dict) or "image" not in first_frame:
                raise ValueError(f"Invalid trajectory format in file: {npy_path}")

            # Get image: shape should be (H, W, 3)
            img_array = first_frame["image"]

            # Extract instruction/task description if available
            if "instruction" in first_frame:
                task_desc = first_frame["instruction"]
                if isinstance(task_desc, (bytes, np.bytes_)):
                    task_desc = task_desc.decode("utf-8")
                task_descriptions.append(str(task_desc))
            else:
                task_descriptions.append("")

            # Extract init_ee_pose if available (for state)
            if "init_ee_pose" in first_frame:
                init_ee_poses.append(first_frame["init_ee_pose"])
            else:
                init_ee_poses.append(None)

            # Convert to tensor and normalize: (H, W, 3) -> (3, H, W)
            img_tensor = torch.from_numpy(img_array).float()
            if img_tensor.dim() == 3 and img_tensor.shape[2] == 3:
                # Convert from (H, W, 3) to (3, H, W)
                img_tensor = img_tensor.permute(2, 0, 1)
            elif img_tensor.dim() != 3 or img_tensor.shape[0] != 3:
                raise ValueError(
                    f"Unexpected image shape: {img_tensor.shape}, expected (H, W, 3) or (3, H, W)"
                )

            # Normalize pixel values from [0, 255] to [0, 1] if needed
            if img_tensor.max() > 1.0:
                img_tensor = img_tensor / 255.0

            # Resize if needed using functional resize for tensors
            if img_tensor.shape[1:] != self.sample_size:
                img_tensor = img_tensor.unsqueeze(0)  # [1, 3, H, W] for interpolation
                img_tensor = F.interpolate(
                    img_tensor,
                    size=self.sample_size,
                    mode="bilinear",
                    align_corners=False,
                )
                img_tensor = img_tensor.squeeze(0)  # [3, H, W]

            # Normalize to [-1, 1]
            img_tensor = self.trans_norm(img_tensor)

            # Repeat to fill memory frames: [3, H, W] -> [3, n_previous, H, W]
            img_tensor = img_tensor.unsqueeze(1).repeat(
                1, self.n_previous, 1, 1
            )  # [3, n_previous, H, W]

            img_tensors.append(img_tensor)

        self.current_obs = []
        for img_tensor in img_tensors:
            # [3, n_previous, H, W] -> [1, 3, 1, n_previous, H, W]
            env_obs = img_tensor.unsqueeze(1)
            self.current_obs.append(env_obs)
        self.current_obs = torch.stack(self.current_obs, dim=0)

        self._is_start = False
        self._reset_metrics()

        # Initialize action buffer
        # Action format: [xyz_l, quat_xyzw_l, gripper_l, xyz_r, quat_xyzw_r, gripper_r] = [16]
        init_action = np.array(
            [
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    -1.0,
                ]
            ],
            dtype=np.float32,
        )
        self.action_buffer = (
            torch.from_numpy(init_action)
            .unsqueeze(0)
            .repeat(num_envs, self.n_previous, 1)
        )

        # Reset inference state
        self.all_x_samples = None
        self.all_samples = None
        self.all_c2ws_list = None
        self.all_trajs = None
        self.x_samples = None
        self.i_chunk = 0

        # Store task descriptions and init_ee_poses for later use
        self.task_descriptions = task_descriptions
        self.init_ee_poses = init_ee_poses

        # Wrap observation to match libero_env format
        extracted_obs = self._wrap_obs()
        infos = {}

        return extracted_obs, infos

    def step(self, actions, auto_reset=True):
        """Take a step in the environment"""
        raise NotImplementedError(
            "Step method is not implemented for EvacEnv, because evac is a chunk-step environment"
        )

    def _infer_next_chunk_rewards(self):
        """Predict next reward using the reward model"""
        rewards = self.reward_model.predict_rew(self.current_obs)
        return rewards

    def _infer_next_chunk_frames(self):
        """Predict next frame using the world model"""
        from copy import deepcopy

        # Prepare current state
        # For single env: self.current_obs shape: [b, c, v, t, h, w] where b=1, v=1
        # For multi env: self.current_obs is a list of [b, c, v, t, h, w] tensors
        num_envs = self.num_envs
        current_obs = self.current_obs
        b, c, v, t, h, w = current_obs.shape

        assert num_envs == b, (
            "Number of environments in current_obs does not match num_envs"
        )

        # Prepare video input
        # Multi-env: stack all envs' current_obs along batch dimension
        stacked_obs = deepcopy(self.current_obs)  # [num_envs, 3, 1, t, h, w]
        # when i_chunk=0, T=n_previous ; when i_chunk!=0, T=chunk+n_previous
        # 事实上，i_chunk-0 时就是相同帧的堆叠
        # Reshape to [num_envs, 3, 1, t, h, w] -> treat num_envs as batch
        if self.i_chunk == 0:
            video = torch.cat(
                (
                    stacked_obs,
                    stacked_obs[:, :, :, -1:, :, :].repeat(1, 1, 1, self.chunk, 1, 1),
                ),
                dim=3,
            )  # [num_envs, 3, 1, t+chunk, h, w]
            # Add batch dim: [num_envs, 3, 1, t+chunk, h, w] -> [1, num_envs, 3, 1, t+chunk, h, w]?
            # Actually, we need to process each env separately, so keep as [num_envs, 3, 1, t+chunk, h, w]
        else:
            # Use previous predictions for each env
            # For now, use first env's history (will need to handle all envs)
            n_history = self.all_x_samples.shape[3]
            idx_history = [
                n_history * i // (self.n_previous - 1)
                for i in range(self.n_previous - 1)
            ]
            # This needs to be fixed for multi-env
            video = torch.cat(
                (
                    self.all_x_samples[:, :, :, idx_history, :, :]
                    if hasattr(self, "all_x_samples") and self.all_x_samples is not None
                    else stacked_obs,
                    stacked_obs[:, :, :, -1:, :, :].repeat(
                        1, 1, 1, self.chunk + 1, 1, 1
                    ),
                ),
                dim=3,
            )

        # video shape: for single env [b, c, v, t, h, w] where b=1

        # Prepare trajectory input
        num_envs = self.num_envs
        if self.i_chunk == 0:
            traj_end_idx = self.n_previous + self.chunk

            # Ensure camera pose history is long enough for current trajectory
            if self.c2w_list.shape[0] < traj_end_idx:
                pad_count = traj_end_idx - self.c2w_list.shape[0]
                pad_c2w = self.c2w_list[-1:].repeat(pad_count, 1, 1)
                pad_w2c = self.w2c_list[-1:].repeat(pad_count, 1, 1)
                self.c2w_list = torch.cat([self.c2w_list, pad_c2w], dim=0)
                self.w2c_list = torch.cat([self.w2c_list, pad_w2c], dim=0)

            i_c2w_list = self.c2w_list[:traj_end_idx].unsqueeze(0)

            # Select camera params: w2c_list and c2w_list are (n_previous, 4, 4)
            # We need (V, 4, 4) where V=1 for single view
            # Use the first camera param (or average) for the trajectory
            w2c_for_traj = self.w2c_list[
                0:1
            ]  # (1, 4, 4) -> select first frame's camera
            c2w_for_traj = self.c2w_list[0:1]  # (1, 4, 4)

            # Multiple environments: process each environment separately
            trajs = []
            for env_idx in range(num_envs):
                action_for_traj = self.action_buffer[env_idx]  # (time_steps, 16)
                traj_env = self.model.get_traj_maniskill_new(
                    self.sample_size,
                    action_for_traj.numpy(),
                    w2c_for_traj,  # (1, 4, 4)
                    c2w_for_traj,  # (1, 4, 4)
                    self.intrinsic.unsqueeze(0),  # (1, 3, 3)
                )
                traj_env = rearrange(traj_env, "c v t h w -> (v t) c h w")
                traj_env = self.trans_norm(traj_env)
                traj_env = rearrange(traj_env, "(v t) c h w -> c v t h w", v=1)
                trajs.append(traj_env)
            # Stack: [num_envs, c, v, t, h, w] -> [c, num_envs, t, h, w]
            traj = torch.stack(trajs, dim=0)  # [num_envs, c, v, t, h, w]
            traj = traj.squeeze(2)  # [num_envs, c, t, h, w]
            traj = traj.permute(1, 0, 2, 3, 4)  # [c, num_envs, t, h, w]
            traj = traj.unsqueeze(0)  # [1, c, num_envs, t, h, w] for compatibility

            i_delta_action = self._compute_delta_action(self.action_buffer)
        else:
            # Use previous predictions for trajectory
            # Reference: ddpm3d.py line 1772-1775
            n_history = self.all_trajs.shape[3]
            idx_history = [
                n_history * i // (self.n_previous - 1)
                for i in range(self.n_previous - 1)
            ]

            # Calculate trajectory for current chunk
            # Use the last camera param for trajectory computation
            w2c_for_traj = (
                self.w2c_list[-1:] if len(self.w2c_list) > 0 else self.w2c_list[0:1]
            )  # (1, 4, 4)
            c2w_for_traj = (
                self.c2w_list[-1:] if len(self.c2w_list) > 0 else self.c2w_list[0:1]
            )  # (1, 4, 4)

            # Multiple environments: process each environment separately
            trajs_new = []
            for env_idx in range(num_envs):
                action_for_traj = self.action_buffer[
                    env_idx, -self.chunk - self.n_previous :
                ]
                traj_env_new = self.model.get_traj_maniskill_new(
                    self.sample_size,
                    action_for_traj.numpy(),
                    w2c_for_traj,  # (1, 4, 4)
                    c2w_for_traj,  # (1, 4, 4)
                    self.intrinsic.unsqueeze(0),  # (1, 3, 3)
                )
                traj_env_new = rearrange(traj_env_new, "c v t h w -> (v t) c h w")
                traj_env_new = self.trans_norm(traj_env_new)
                traj_env_new = rearrange(traj_env_new, "(v t) c h w -> c v t h w", v=1)
                trajs_new.append(traj_env_new)

            # Stack and process all environments (moved outside loop)
            # trajs_new: list of [c, v, t, h, w] where v=1
            traj_new = torch.stack(trajs_new, dim=0)  # [num_envs, c, v, t, h, w]
            traj_new = traj_new.squeeze(
                2
            )  # [num_envs, c, t, h, w] - remove v dimension
            traj_new = traj_new.permute(1, 0, 2, 3, 4)  # [c, num_envs, t, h, w]
            traj_new = traj_new.unsqueeze(0)  # [1, c, num_envs, t, h, w]

            # Concatenate history with new trajectory
            # self.all_trajs shape: [1, c, num_envs, t_history, h, w]
            # traj_new shape: [1, c, num_envs, t_new, h, w]
            # Reference: ddpm3d.py line 1772-1775
            traj = torch.cat(
                (
                    self.all_trajs[
                        :, :, :, idx_history, :, :
                    ],  # [1, c, num_envs, n_previous-1, h, w]
                    traj_new[
                        :, :, :, self.n_previous - 1 :, :, :
                    ],  # [1, c, num_envs, chunk+1, h, w]
                ),
                dim=3,
            )  # Result: [1, c, num_envs, chunk+n_previous, h, w]

            # Concatenate camera poses
            # Reference: ddpm3d.py line 1779-1782
            # self.all_c2ws_list shape: [1, t_history, 4, 4]
            # self.c2w_list shape: [t, 4, 4]
            i_c2w_list = torch.cat(
                (
                    self.all_c2ws_list[:, idx_history, :, :],  # [1, n_previous-1, 4, 4]
                    self.c2w_list[-self.chunk - self.n_previous :].unsqueeze(0)[
                        :, self.n_previous - 1 :, :, :
                    ],  # [1, chunk+1, 4, 4]
                ),
                dim=1,
            )  # Result: [1, chunk+n_previous, 4, 4]

            i_delta_action = self._compute_delta_action(
                self.action_buffer[:, -self.chunk - self.n_previous :]
            )

        if traj.shape[3] < self.chunk + self.n_previous:
            # Pad trajectory
            traj = torch.cat(
                (
                    traj,
                    traj[:, :, :, -1:].repeat(
                        1, 1, 1, self.chunk + self.n_previous - traj.shape[3], 1, 1
                    ),
                ),
                dim=3,
            )

        # Clamp values
        video = torch.clamp(video, min=-1, max=1)
        traj = torch.clamp(traj, min=-1, max=1)

        # Prepare batch
        intrinsic_batch = (
            self.intrinsic.unsqueeze(0).unsqueeze(0).repeat(num_envs, 1, 1, 1)
        )
        extrinsic_batch = i_c2w_list.repeat(num_envs, 1, 1, 1)

        fps = torch.tensor([2.0]).to(dtype=torch.float32, device=self.device)
        # pad to num_envs
        fps = fps.repeat(num_envs)
        domain_id = torch.LongTensor([DomainTable["agibotworld"]]).to(
            device=self.device
        )
        # pad to num_envs
        domain_id = domain_id.repeat(num_envs)

        batch = {
            "video": video.to(dtype=self.inference_dtype, device=self.device),
            "traj": traj.to(dtype=self.inference_dtype, device=self.device),
            "delta_action": i_delta_action.to(
                dtype=self.inference_dtype, device=self.device
            ),
            "domain_id": domain_id,
            "intrinsic": intrinsic_batch.to(dtype=torch.float32, device=self.device),
            "extrinsic": extrinsic_batch.to(dtype=torch.float32, device=self.device),
            "caption": [""],
            "cond_id": torch.tensor(
                [-self.n_previous - self.chunk], dtype=torch.int64
            ).to(device=self.device),
            "fps": fps,
        }

        # import pdb; pdb.set_trace()

        # Get batch input
        pre_z = None
        pre_img_emb = None
        if self.i_chunk != 0 and self.all_samples is not None:
            # Reference: ddpm3d.py line 1826-1829
            # self.all_samples shape: [num_envs, c, t, h, w] (latent space)
            n_history = self.all_samples.shape[2]
            idx_history = [
                n_history * i // (self.n_previous - 1)
                for i in range(self.n_previous - 1)
            ]
            # Use the last sample from previous chunk (in latent space, not decoded)
            # Need to get the last sample from the most recent chunk
            # Since we store samples after decoding, we need to use all_samples
            pre_z = torch.cat(
                (
                    self.all_samples[:, :, idx_history].to(
                        self.device
                    ),  # [num_envs, c, n_previous-1, h, w]
                    self.all_samples[:, :, -1:]
                    .repeat(1, 1, self.chunk + 1, 1, 1)
                    .to(self.device),  # [num_envs, c, chunk+1, h, w]
                ),
                dim=2,
            ).to(
                dtype=self.inference_dtype, device=self.device
            )  # [num_envs, c, n_previous+chunk, h, w]

        z, cond, xc, fs, did, img_emb = self.model.get_batch_input(
            batch,
            random_uncond=False,
            return_first_stage_outputs=False,
            return_original_cond=True,
            return_fs=True,
            return_did=True,
            return_traj=False,
            return_img_emb=True,
            pre_z=pre_z,
            pre_img_emb=pre_img_emb,
        )

        # Prepare conditions
        kwargs = {
            "fs": fs.long(),
            "domain_id": did.long(),
            "dtype": self.inference_dtype,
            "timestep_spacing": "uniform_trailing",
            "guidance_rescale": self.guidance_rescale,
            "return_intermediates": False,
        }

        # No unconditional guidance for now
        uc = None

        # Ensure correct dtype
        for _c_cat in range(len(cond["c_concat"])):
            cond["c_concat"][_c_cat] = cond["c_concat"][_c_cat].to(
                dtype=self.inference_dtype
            )
        for _c_cro in range(len(cond["c_crossattn"])):
            cond["c_crossattn"][_c_cro] = cond["c_crossattn"][_c_cro].to(
                dtype=self.inference_dtype
            )

        # Sample
        N = z.shape[0]
        samples, _ = self.model.sample_log(
            cond=cond,
            batch_size=N,
            ddim=True,
            ddim_steps=self.ddim_steps,
            causal=True,
            eta=self.ddim_eta,
            unconditional_guidance_scale=self.unconditional_guidance_scale,
            unconditional_conditioning=uc,
            x0=z.to(self.inference_dtype),
            chunk=self.chunk,
            cat_mask=self.use_cat_mask,
            sparse=self.sparse_memory,
            traj=False,
            ddim_dtype=torch.float16,
            **kwargs,
        )

        # Decode samples
        x_samples = self.model.decode_first_stage(samples.to(z.device)).data.cpu()
        x_samples = rearrange(x_samples, "(b v) c t h w -> b c v t h w", v=v)

        # Store for next iteration
        # x_samples shape: [b, c, v, t, h, w]
        if self.all_x_samples is None:
            self.all_x_samples = x_samples.data.cpu()
            self.all_samples = samples.data.cpu()
            self.all_c2ws_list = i_c2w_list.data.cpu()
            self.all_trajs = traj.data.cpu()
        else:
            # Concatenate along time dimension (dim=3)
            self.all_x_samples = torch.cat(
                (
                    self.all_x_samples,
                    x_samples[:, :, :, self.n_previous :, :, :].data.cpu(),
                ),
                dim=3,
            )
            self.all_samples = torch.cat(
                (self.all_samples, samples[:, :, self.n_previous :, :, :].data.cpu()),
                dim=2,
            )
            self.all_c2ws_list = torch.cat(
                (self.all_c2ws_list, i_c2w_list[:, self.n_previous :].data.cpu()), dim=1
            )
            self.all_trajs = torch.cat(
                (self.all_trajs, traj.data.cpu()[:, :, :, self.n_previous :]), dim=3
            )

        self.x_samples = x_samples

        # Update current observation
        # x_samples shape: [b, c, v, T, h, w], extract from n_previous onwards  T=chunk+n_previous
        assert x_samples.shape[0] == num_envs, (
            f"Unexpected x_samples shape: {x_samples.shape}, expected {num_envs}"
        )
        # self.current_obs shape: [num_envs, c, v, T, h, w] T=chunk
        self.current_obs = x_samples[:, :, :, self.n_previous :, :, :]
        # Increment chunk counter
        self.i_chunk += 1

        # Update camera list for next iteration
        self.c2w_list = torch.cat((self.c2w_list, self.c2w_list[-1:]), dim=0)
        self.w2c_list = torch.cat((self.w2c_list, self.w2c_list[-1:]), dim=0)

        # Wrap observation to match libero_env format
        return self._wrap_obs()

    def _wrap_obs(self):
        """Wrap observation to match libero_env format"""
        num_envs = self.num_envs

        # Extract the last frame (most recent observation) for each environment
        # self.current_obs is [b, c, v, t, h, w]  v=1 for single view
        b, c, v, t, h, w = self.current_obs.shape
        assert b == num_envs, (
            f"Unexpected current_obs shape: {self.current_obs.shape}, expected {num_envs}"
        )

        last_frame = self.current_obs[:, :, :, -1, :, :]  # [b, 3, 1, h, w]
        # Remove batch and view dims: [b, 3, 1, h, w] -> [3, h, w]
        last_frame = last_frame.squeeze(2)  # [b, 3, h, w] remove view dim

        # Convert from [3, num_envs, H, W] to [num_envs, H, W, 3] (HWC format)
        # Permute: [3, num_envs, H, W] -> [num_envs, 3, H, W] -> [num_envs, H, W, 3]
        full_image = last_frame.permute(0, 2, 3, 1)  # [num_envs, H, W, 3]

        # Denormalize from [-1, 1] to [0, 255]
        full_image = (full_image + 1.0) / 2.0 * 255.0

        full_image = torch.clamp(full_image, 0, 255)

        # Resize to 256x256 to match libero_env format
        # full_image shape: [num_envs, H, W, 3]
        # Need to resize to [num_envs, 256, 256, 3]
        target_size = (256, 256)
        if full_image.shape[1:3] != target_size:
            # Reshape for interpolation: [num_envs, H, W, 3] -> [num_envs, 3, H, W]
            full_image = full_image.permute(0, 3, 1, 2)  # [num_envs, 3, H, W]
            # Resize using F.interpolate
            full_image = F.interpolate(
                full_image, size=target_size, mode="bilinear", align_corners=False
            )
            # Convert back: [num_envs, 3, 256, 256] -> [num_envs, 256, 256, 3]
            full_image = full_image.permute(0, 2, 3, 1)  # [num_envs, 256, 256, 3]

        # Convert to numpy
        full_image_np = full_image.cpu().numpy().astype(np.uint8)

        num_envs = self.num_envs
        # Use the last abs action from buffer for this env
        states = self.action_buffer[:, -1].cpu().numpy()  # [num_envs, 16]

        # Get task descriptions
        if hasattr(self, "task_descriptions"):
            task_descriptions = self.task_descriptions
        else:
            raise ValueError("task_descriptions not found")

        # Wrap observation
        obs = {
            "images_and_states": {
                "full_image": full_image_np,  # [num_envs, H, W, 3]
                "state": states.astype(
                    np.float32
                ),  # [num_envs, 16] - padded to match model compatibility
            },
            "task_descriptions": task_descriptions,  # list of strings
        }

        return obs

    def _compute_delta_action(self, action_buffer):
        """Compute delta action from action buffer"""
        from rlinf.envs.worldmodel.evac.sim.simulator import get_action_sim

        num_envs = self.num_envs

        # Handle both single and multiple environments
        if num_envs == 1:
            # Single environment: action_buffer shape is (time_steps, action_dim)
            if len(action_buffer) < 5:
                # Not enough actions, return zero delta action
                return torch.zeros((1, 14), dtype=torch.float32)

            action_np = action_buffer.numpy()
            _, delta_action = get_action_sim(action_np)
            # Ensure we return only one delta action
            if isinstance(delta_action, np.ndarray) and len(delta_action.shape) > 1:
                return torch.from_numpy(delta_action[[-1]])  # Take only the last one
            else:
                return torch.from_numpy(delta_action)
        else:
            # Multiple environments: action_buffer shape is (num_envs, time_steps, action_dim)
            if action_buffer.shape[1] < 5:
                # Not enough actions, return zero delta action
                return torch.zeros((num_envs, 14), dtype=torch.float32)

            # Process each environment separately
            delta_actions = []
            for env_idx in range(num_envs):
                action_np = action_buffer[env_idx].numpy()
                _, delta_action = get_action_sim(action_np)
                if isinstance(delta_action, np.ndarray) and len(delta_action.shape) > 1:
                    delta_action = delta_action[-1]  # Take only the last one
                delta_actions.append(delta_action)

            return torch.from_numpy(np.stack(delta_actions, axis=0))  # [num_envs, 14]

    def _handle_auto_reset(self, dones, extracted_obs, infos):
        """Handle automatic reset on episode termination"""
        final_obs = extracted_obs
        final_info = infos
        # env_idx = torch.arange(0, self.num_envs, device=self.device)[dones]

        extracted_obs, infos = self.reset()

        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones

        return extracted_obs, infos

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions - optimized version that processes chunk actions together"""
        # chunk_actions: [num_envs, chunk_steps, action_dim=8]
        chunk_size = chunk_actions.shape[1]
        num_envs = self.num_envs

        # Duplicate actions from 8 to 16 dimensions by stacking
        if chunk_actions.shape[2] == 8:
            padded_chunk_actions = np.concatenate(
                [chunk_actions, chunk_actions], axis=-1
            )  # [num_envs, chunk_steps, 16]
        elif chunk_actions.shape[2] == 16:
            padded_chunk_actions = chunk_actions
        else:
            raise ValueError(
                f"Unexpected action dimension: {chunk_actions.shape[2]}, expected 8 or 16"
            )

        chunk_actions_tensor = torch.from_numpy(
            padded_chunk_actions
        )  # [num_envs, chunk_steps, 16]
        self.action_buffer = torch.cat(
            [self.action_buffer, chunk_actions_tensor], dim=1
        )

        # Update elapsed steps
        self._elapsed_steps += chunk_size

        # Process inference based on chunk relationship
        # If chunk_size <= self.chunk, we can process it in one inference
        # If chunk_size > self.chunk, we need multiple inferences
        chunk_rewards = []

        with torch.cuda.amp.autocast(dtype=torch.float16):
            # extracted_chunk_obs shape: [num_envs, c, v, chunk_size, h, w]
            extracted_chunk_obs = self._infer_next_chunk_frames()

        self.current_step += chunk_size

        chunk_rewards = self._infer_next_chunk_rewards(extracted_chunk_obs)

        # Calculate reward for this step (dummy for now)
        chunk_rewards = self._calc_step_reward(chunk_rewards)
        chunk_rewards_tensor = torch.stack(
            chunk_rewards, dim=1
        )  # [num_envs, num_inferences]

        # Use the last observation
        extracted_obs = (
            extracted_chunk_obs[:, :, :, -1]
            if extracted_chunk_obs
            else self._wrap_obs()
        )

        # No terminations/truncations for now (could implement based on max steps)
        raw_chunk_terminations = deepcopy(chunk_rewards_tensor)

        raw_chunk_truncations = torch.zeros(
            num_envs, chunk_size, dtype=torch.bool, device=self.device
        )
        truncations = torch.tensor(self.elapsed_steps >= self.cfg.max_episode_steps).to(
            self.device
        )
        if truncations.any():
            raw_chunk_truncations[:, -1] = truncations

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, {}
            )
        else:
            infos = {}

        infos = self._record_metrics(chunk_rewards_tensor.sum(dim=1), infos)

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations

        return (
            extracted_obs,
            chunk_rewards_tensor,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def add_new_frames(self, infos, rewards=None):
        """Add frames for video rendering"""
        # Simplified implementation
        if self.current_obs is not None:
            # Convert tensor to image
            # For single env: self.current_obs shape: [b, c, v, t, h, w] where b=1, v=1
            # For multi env: self.current_obs is a list
            if isinstance(self.current_obs, list):
                # Multi-env: use first env
                img = self.current_obs[0][0, :, 0, -1, :, :].permute(
                    1, 2, 0
                )  # [H, W, C]
            else:
                # Single env: [1, 3, 1, t, h, w] -> [h, w, 3]
                img = self.current_obs[0, :, 0, -1, :, :].permute(1, 2, 0)  # [H, W, C]
            img_np = (img.cpu().numpy() + 1) / 2 * 255  # Denormalize
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)
            self.render_images.append(img_np)

    def flush_video(self, video_sub_dir: Optional[str] = None):
        """Save accumulated video frames"""
        if len(self.render_images) == 0:
            return

        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")

        os.makedirs(output_dir, exist_ok=True)

        from mani_skill.utils.visualization.misc import images_to_video

        images_to_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
            fps=self.cfg.get("fps", 10),
            verbose=False,
        )

        self.video_cnt += 1
        self.render_images = []


if __name__ == "__main__":
    config_path = (
        "/path/home/RLinf/examples/embodiment/config/env/train/libero_spatial_evac.yaml"
    )
    print(f"Loading config from {config_path}")
    cfg = OmegaConf.load(config_path)
    env = EvacEnv(cfg, seed_offset=0, total_num_processes=1)

    print("\n" + "=" * 80)
    print("Testing chunk_step with different chunk sizes")
    print("=" * 80)

    # Reset environment
    obs, info = env.reset(options={"initial_image_path": cfg.initial_image_path})
    print("\nAfter reset:")
    print(f"  obs keys: {list(obs.keys())}")

    # Test 1: chunk_steps = self.chunk
    print("\n" + "-" * 80)
    chunk_steps = 8
    num_envs = cfg.num_envs
    init_action = np.array(
        [[0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 1.0, -1.0]], dtype=np.float32
    )
    chunk_actions = np.tile(init_action, (num_envs, chunk_steps, 1))

    for i in range(10):
        obs, reward, term, trunc, info = env.chunk_step(chunk_actions)
        print(f"  Action buffer after: {env.action_buffer.shape}")
        print(
            f"  Output obs['images_and_states']['full_image'].shape: {obs['images_and_states']['full_image'].shape}"
        )
        print(f"  Output reward shape: {env.current_obs.shape}")
