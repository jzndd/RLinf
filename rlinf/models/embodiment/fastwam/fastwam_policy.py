# Copyright 2026 The RLinf Authors.

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.nn.modules.module import _IncompatibleKeys

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType


class FastWAMPolicy(nn.Module, BasePolicy):
    def __init__(self, cfg, model: nn.Module, processor: FastWAMProcessor):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.processor = processor
        self.num_action_chunks = int(cfg.num_action_chunks)
        self.action_horizon = int(cfg.action_horizon)
        self.video_size = [int(v) for v in cfg.video_size]
        self.concat_multi_camera = str(cfg.get("concat_multi_camera", "horizontal"))
        self.binarize_gripper = bool(cfg.get("binarize_gripper", True))
        self.score_loss_type = str(cfg.get("score_loss_type", "l1_loss")).lower()
        self.score_weighted_by_scheduler = bool(
            cfg.get("score_weighted_by_scheduler", True)
        )
        self.score_video_cache_no_grad = bool(
            cfg.get("score_video_cache_no_grad", True)
        )
        self._frozen_modules: list[nn.Module] = []
        self._configure_trainable_modules()

    def _set_module_trainability(
        self,
        module: nn.Module | None,
        trainable: bool,
    ) -> None:
        if module is None:
            return
        module.requires_grad_(trainable)
        if trainable:
            module.train()
            return
        module.eval()
        self._frozen_modules.append(module)

    def _configure_trainable_modules(self) -> None:
        if bool(self.cfg.get("is_lora", False)):
            self.model.requires_grad_(False)
            for name, param in self.model.named_parameters():
                if "lora_" in name:
                    param.requires_grad_(True)

            proprio_encoder = getattr(self.model, "proprio_encoder", None)
            if bool(self.cfg.get("train_proprio_encoder", False)):
                self._set_module_trainability(proprio_encoder, True)
            else:
                self._set_module_trainability(proprio_encoder, False)

            self._set_module_trainability(self.model.text_encoder, False)
            self._set_module_trainability(self.model.vae, False)
            return

        self.model.requires_grad_(False)
        self._set_module_trainability(
            self.model.mot,
            bool(self.cfg.get("train_mot", True)),
        )
        self._set_module_trainability(
            self.model.action_expert,
            bool(self.cfg.get("train_action_expert", True)),
        )
        self._set_module_trainability(
            self.model.video_expert,
            bool(self.cfg.get("train_video_expert", False)),
        )
        self._set_module_trainability(
            getattr(self.model, "proprio_encoder", None),
            bool(self.cfg.get("train_proprio_encoder", True)),
        )
        self._set_module_trainability(
            self.model.text_encoder,
            bool(self.cfg.get("train_text_encoder", False)),
        )
        self._set_module_trainability(
            self.model.vae,
            bool(self.cfg.get("train_vae", False)),
        )
        self._restore_frozen_eval_mode()

    def _restore_frozen_eval_mode(self) -> None:
        for module in self._frozen_modules:
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._restore_frozen_eval_mode()
        return self

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        del assign

        if not isinstance(state_dict, dict):
            raise TypeError(f"state_dict must be dict-like, got {type(state_dict)}")

        if "mot" in state_dict:
            mot_result = self.model.mot.load_state_dict(state_dict["mot"], strict=strict)
            missing_keys = list(mot_result.missing_keys)
            unexpected_keys = list(mot_result.unexpected_keys)

            proprio_module = getattr(self.model, "proprio_encoder", None)
            proprio_payload = state_dict.get("proprio_encoder")
            if proprio_module is not None:
                if proprio_payload is None:
                    if strict:
                        raise RuntimeError(
                            "FastWAM checkpoint is missing `proprio_encoder` while current model requires it."
                        )
                    missing_keys.extend(
                        [
                            "proprio_encoder.weight",
                            "proprio_encoder.bias",
                        ]
                    )
                else:
                    proprio_result = proprio_module.load_state_dict(
                        proprio_payload,
                        strict=strict,
                    )
                    missing_keys.extend(
                        [
                            f"proprio_encoder.{key}"
                            for key in proprio_result.missing_keys
                        ]
                    )
                    unexpected_keys.extend(
                        [
                            f"proprio_encoder.{key}"
                            for key in proprio_result.unexpected_keys
                        ]
                    )
            elif proprio_payload is not None and strict:
                raise RuntimeError(
                    "FastWAM checkpoint contains `proprio_encoder` but current model does not use it."
                )

            return _IncompatibleKeys(missing_keys, unexpected_keys)

        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.ACTION_SAMPLING:
            return self.predict_action_batch(**kwargs)
        raise NotImplementedError

    @staticmethod
    def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        pil_image = Image.fromarray(image)
        src_w, src_h = pil_image.size
        scale = max(width / src_w, height / src_h)
        resized = pil_image.resize(
            (round(src_w * scale), round(src_h * scale)),
            resample=Image.BILINEAR,
        )
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        cropped = resized.crop((left, top, left + width, top + height))
        return np.asarray(cropped, dtype=np.uint8)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            tensor = value.detach()
            if tensor.is_floating_point():
                tensor = tensor.to(dtype=torch.float32)
            return tensor.cpu().numpy()
        return np.asarray(value)

    @staticmethod
    def _normalize_task_descriptions(
        task_descriptions: Any,
        batch_size: int,
    ) -> list[str]:
        if isinstance(task_descriptions, str):
            return [task_descriptions] * batch_size
        if task_descriptions is None:
            return [""] * batch_size
        task_list = list(task_descriptions)
        if len(task_list) != batch_size:
            raise ValueError(
                f"task_descriptions length mismatch: expected {batch_size}, got {len(task_list)}"
            )
        return [str(task) for task in task_list]

    def _normalize_proprio(self, states: np.ndarray) -> torch.Tensor:
        if states.ndim != 2:
            raise ValueError(f"Expected states to have shape [B, D], got {states.shape}")

        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("FastWAM LIBERO eval expects a single merged state key.")

        state_key = state_meta[0]["key"]
        state_batch = {
            "state": {
                state_key: torch.as_tensor(states, dtype=torch.float32),
            }
        }
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _normalize_action_chunk_shape(self, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 3:
            if action.shape[0] != 1:
                raise ValueError(
                    "Expected a single-sample action tensor with shape [1, T, D], "
                    f"got {tuple(action.shape)}"
                )
            action = action[0]
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.ndim != 2:
            raise ValueError(
                "Expected action tensor with shape [T, D] for a single sample, "
                f"got {tuple(action.shape)}"
            )
        return action

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        action = self._normalize_action_chunk_shape(action)
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(
                f"Expected action tensor with shape [B, T, D], got {tuple(action.shape)}"
            )

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("FastWAM LIBERO eval expects a single merged action key.")

        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _postprocess_action(self, action: torch.Tensor) -> torch.Tensor:
        action_np = self._denormalize_action(action)[0]
        action_np[..., -1] = action_np[..., -1] * 2.0 - 1.0
        action_np[..., -1] = action_np[..., -1] * -1.0
        if self.binarize_gripper:
            action_np[..., -1] = np.sign(action_np[..., -1])
        action_np = action_np[: self.num_action_chunks].astype(np.float32)
        return torch.from_numpy(action_np)

    def _build_input_image(
        self,
        main_image: np.ndarray,
        wrist_image: np.ndarray,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        image_meta = self.processor.shape_meta["images"]
        if len(image_meta) < int(self.processor.num_output_cameras):
            raise ValueError(
                "shape_meta.images has fewer entries than num_output_cameras."
            )

        def _meta_hw(meta: dict) -> tuple[int, int]:
            shape = meta["shape"]
            if len(shape) != 3:
                raise ValueError(f"Image shape metadata must be [C,H,W], got {shape}")
            return int(shape[1]), int(shape[2])

        primary_h, primary_w = _meta_hw(image_meta[0])
        primary = self._center_crop_resize(main_image, width=primary_w, height=primary_h)

        if int(self.processor.num_output_cameras) == 1:
            rgb = primary
        elif int(self.processor.num_output_cameras) == 2:
            wrist_h, wrist_w = _meta_hw(image_meta[1])
            wrist = self._center_crop_resize(wrist_image, width=wrist_w, height=wrist_h)
            if self.concat_multi_camera == "horizontal":
                rgb = np.concatenate([primary, wrist], axis=1)
            elif self.concat_multi_camera == "vertical":
                rgb = np.concatenate([primary, wrist], axis=0)
            else:
                raise ValueError(
                    f"Unsupported concat_multi_camera: {self.concat_multi_camera}"
                )
        else:
            raise ValueError(
                f"Unsupported num_output_cameras: {self.processor.num_output_cameras}"
            )

        actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
        expected_h, expected_w = self.video_size
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise ValueError(
                "Input image size mismatch after camera resize/concat: "
                f"got {(actual_h, actual_w)}, expected {(expected_h, expected_w)}"
            )

        image_tensor = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(device=device, dtype=dtype)
        return image_tensor * (2.0 / 255.0) - 1.0

    def _build_input_image_batch(
        self,
        main_images: np.ndarray,
        wrist_images: np.ndarray,
    ) -> torch.Tensor:
        model_dtype = self.model.torch_dtype
        model_device = self.model.device
        image_batch = []
        for idx in range(int(main_images.shape[0])):
            image_batch.append(
                self._build_input_image(
                    main_image=main_images[idx],
                    wrist_image=wrist_images[idx],
                    device=model_device,
                    dtype=model_dtype,
                )
            )
        return torch.cat(image_batch, dim=0)

    def _encode_input_image_latents_batch(self, input_images: torch.Tensor) -> torch.Tensor:
        if input_images.ndim != 4 or input_images.shape[1] != 3:
            raise ValueError(
                "Expected batched input images with shape [B, 3, H, W], "
                f"got {tuple(input_images.shape)}"
            )

        latent_batch = []
        for idx in range(int(input_images.shape[0])):
            latent_batch.append(
                self.model._encode_input_image_latents_tensor(
                    input_image=input_images[idx : idx + 1],
                    tiled=bool(self.cfg.get("tiled", False)),
                )
            )
        return torch.cat(latent_batch, dim=0)

    @staticmethod
    def _group_sample_indices(condition_group_ids: torch.Tensor) -> list[list[int]]:
        group_ids = condition_group_ids.detach().cpu().tolist()
        grouped_indices: dict[int, list[int]] = {}
        for sample_idx, group_id in enumerate(group_ids):
            grouped_indices.setdefault(int(group_id), []).append(sample_idx)
        return list(grouped_indices.values())

    @staticmethod
    def _expand_video_kv_cache(
        video_kv_cache: list[dict[str, torch.Tensor]],
        batch_size: int,
    ) -> list[dict[str, torch.Tensor]]:
        expanded_cache = []
        for layer_cache in video_kv_cache:
            expanded_cache.append(
                {
                    "k": layer_cache["k"].expand(batch_size, -1, -1),
                    "v": layer_cache["v"].expand(batch_size, -1, -1),
                }
            )
        return expanded_cache

    def _tokenize_prompts(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.tokenizer is None:
            raise ValueError(
                "FastWAM offline GRPO requires a tokenizer to rebuild prompt embeddings."
            )
        prompt_ids, prompt_mask = self.model.tokenizer(
            prompts,
            return_mask=True,
            add_special_tokens=True,
        )
        return prompt_ids.to(torch.long), prompt_mask.to(torch.bool)

    def _encode_prompt_tokens(
        self,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.model.text_encoder is None:
            raise ValueError(
                "FastWAM offline GRPO requires `load_text_encoder=true`."
            )

        prompt_ids = prompt_ids.to(device=self.model.device, dtype=torch.long)
        prompt_mask = prompt_mask.to(device=self.model.device, dtype=torch.bool)
        prompt_emb = self.model.text_encoder(prompt_ids, prompt_mask)
        seq_lens = prompt_mask.gt(0).sum(dim=1).long()
        for idx, seq_len in enumerate(seq_lens):
            prompt_emb[idx, seq_len:] = 0
        full_mask = torch.ones_like(prompt_mask, dtype=torch.bool)
        return prompt_emb.to(device=self.model.device), full_mask

    def _compute_action_scores(
        self,
        *,
        input_images: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        prompt_context: torch.Tensor | None,
        prompt_context_mask: torch.Tensor | None,
        proprio: torch.Tensor | None,
        actions: torch.Tensor,
        action_noise: torch.Tensor,
        action_timesteps: torch.Tensor,
        first_frame_latents: torch.Tensor | None,
        condition_group_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        model_dtype = self.model.torch_dtype
        model_device = self.model.device

        input_images = input_images.to(device=model_device, dtype=model_dtype)
        actions = actions.to(device=model_device, dtype=model_dtype)
        action_noise = action_noise.to(device=model_device, dtype=model_dtype)
        action_timesteps = action_timesteps.to(device=model_device, dtype=model_dtype)

        if prompt_context is None or prompt_context_mask is None:
            context, context_mask = self._encode_prompt_tokens(
                prompt_ids=prompt_input_ids,
                prompt_mask=prompt_attention_mask,
            )
        else:
            context = prompt_context.to(device=model_device, dtype=model_dtype)
            context_mask = prompt_context_mask.to(
                device=model_device,
                dtype=torch.bool,
            )
        if proprio is not None:
            proprio = proprio.to(device=model_device, dtype=model_dtype)
        if first_frame_latents is not None:
            first_frame_latents = first_frame_latents.to(
                device=model_device,
                dtype=model_dtype,
            )
        if condition_group_ids is None:
            condition_group_ids = torch.arange(
                actions.shape[0],
                device=model_device,
                dtype=torch.long,
            )
        else:
            condition_group_ids = condition_group_ids.to(
                device=model_device,
                dtype=torch.long,
            ).reshape(-1)
        if condition_group_ids.shape[0] != actions.shape[0]:
            raise ValueError(
                "condition_group_ids batch size mismatch: "
                f"expected {actions.shape[0]}, got {condition_group_ids.shape[0]}"
            )

        noisy_actions = self.model.train_action_scheduler.add_noise(
            actions,
            action_noise,
            action_timesteps,
        )
        target_actions = self.model.train_action_scheduler.training_target(
            actions,
            action_noise,
            action_timesteps,
        )

        fuse_flag = bool(
            getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False)
        )
        pred_actions_by_index: list[torch.Tensor | None] = [None] * int(actions.shape[0])
        for sample_indices in self._group_sample_indices(condition_group_ids):
            first_idx = int(sample_indices[0])
            group_size = len(sample_indices)
            group_context = context[first_idx : first_idx + 1]
            group_context_mask = context_mask[first_idx : first_idx + 1]
            if proprio is not None:
                group_context, group_context_mask = self.model._append_proprio_to_context(
                    context=group_context,
                    context_mask=group_context_mask,
                    proprio=proprio[first_idx : first_idx + 1],
                )

            if first_frame_latents is None:
                group_first_frame_latents = self._encode_input_image_latents_batch(
                    input_images[first_idx : first_idx + 1]
                )
            else:
                group_first_frame_latents = first_frame_latents[first_idx : first_idx + 1]

            group_noisy_actions = noisy_actions[sample_indices]
            group_action_timesteps = action_timesteps[sample_indices]
            group_context_batch = group_context.expand(group_size, -1, -1)
            group_context_mask_batch = group_context_mask.expand(group_size, -1)
            action_pre = self.model.action_expert.pre_dit(
                action_tokens=group_noisy_actions,
                timestep=group_action_timesteps,
                context=group_context_batch,
                context_mask=group_context_mask_batch,
            )

            group_context_for_video = group_context

            def _build_group_video_cache(
                video_context: torch.Tensor,
            ) -> tuple[
                list[dict[str, torch.Tensor]],
                torch.Tensor,
                int,
            ]:
                timestep_video = torch.zeros(
                    (1,),
                    dtype=group_first_frame_latents.dtype,
                    device=model_device,
                )
                video_pre = self.model.video_expert.pre_dit(
                    x=group_first_frame_latents,
                    timestep=timestep_video,
                    context=video_context,
                    context_mask=group_context_mask,
                    action=None,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                video_seq_len = int(video_pre["tokens"].shape[1])
                attention_mask = self.model._build_mot_attention_mask(
                    video_seq_len=video_seq_len,
                    action_seq_len=action_pre["tokens"].shape[1],
                    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
                    device=model_device,
                )
                video_kv_cache = self.model.mot.prefill_video_cache(
                    video_tokens=video_pre["tokens"],
                    video_freqs=video_pre["freqs"],
                    video_t_mod=video_pre["t_mod"],
                    video_context_payload={
                        "context": video_pre["context"],
                        "mask": video_pre["context_mask"],
                    },
                    video_attention_mask=attention_mask[:video_seq_len, :video_seq_len],
                )
                return video_kv_cache, attention_mask, video_seq_len

            # Treat video K/V as shared conditioning so only the action branch keeps
            # the backward graph for each GRPO sample in the group.
            if self.score_video_cache_no_grad:
                group_first_frame_latents = group_first_frame_latents.detach()
                group_context_for_video = group_context.detach()
                with torch.no_grad():
                    video_kv_cache, attention_mask, video_seq_len = (
                        _build_group_video_cache(group_context_for_video)
                    )
            else:
                video_kv_cache, attention_mask, video_seq_len = (
                    _build_group_video_cache(group_context_for_video)
                )
            action_tokens = self.model.mot.forward_action_with_video_cache(
                action_tokens=action_pre["tokens"],
                action_freqs=action_pre["freqs"],
                action_t_mod=action_pre["t_mod"],
                action_context_payload={
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
                video_kv_cache=self._expand_video_kv_cache(video_kv_cache, group_size),
                attention_mask=attention_mask,
                video_seq_len=video_seq_len,
            )
            group_pred_actions = self.model.action_expert.post_dit(
                action_tokens,
                action_pre,
            )
            for group_offset, sample_idx in enumerate(sample_indices):
                pred_actions_by_index[sample_idx] = group_pred_actions[group_offset]

        if any(pred_action is None for pred_action in pred_actions_by_index):
            raise ValueError("Failed to compute FASTWAM action scores for all samples.")
        pred_actions = torch.stack(
            [pred_action for pred_action in pred_actions_by_index if pred_action is not None],
            dim=0,
        )

        if self.score_loss_type in {"l1", "l1_loss"}:
            per_dim_loss = torch.abs(
                pred_actions.float() - target_actions.float()
            )
        elif self.score_loss_type in {"mse", "mse_loss"}:
            per_dim_loss = (
                pred_actions.float() - target_actions.float()
            ).square()
        else:
            raise ValueError(
                "Unsupported FastWAM score_loss_type: "
                f"{self.cfg.get('score_loss_type')}. Expected one of ['l1_loss', 'mse_loss']."
            )

        score = -per_dim_loss
        if self.score_weighted_by_scheduler:
            action_weight = self.model.train_action_scheduler.training_weight(
                action_timesteps
            ).to(device=score.device, dtype=score.dtype)
            score = score * action_weight.view(-1, 1, 1)

        return score

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        logprobs = self._compute_action_scores(
            input_images=forward_inputs["input_images"],
            prompt_input_ids=forward_inputs["prompt_input_ids"],
            prompt_attention_mask=forward_inputs["prompt_attention_mask"],
            prompt_context=forward_inputs.get("prompt_context"),
            prompt_context_mask=forward_inputs.get("prompt_context_mask"),
            proprio=forward_inputs.get("proprio"),
            actions=forward_inputs["actions"],
            action_noise=forward_inputs["action_noise"],
            action_timesteps=forward_inputs["action_timesteps"],
            first_frame_latents=forward_inputs.get("first_frame_latents"),
            condition_group_ids=forward_inputs.get("condition_group_ids"),
        )

        result: dict[str, Any] = {"logprobs": logprobs}
        if kwargs.get("compute_entropy", False):
            result["entropy"] = torch.zeros(
                (logprobs.shape[0], 1),
                dtype=logprobs.dtype,
                device=logprobs.device,
            )
        if kwargs.get("compute_values", False):
            result["values"] = torch.zeros(
                (logprobs.shape[0], 1),
                dtype=logprobs.dtype,
                device=logprobs.device,
            )
        return result

    def predict_action_batch(self, env_obs, mode="eval", **kwargs) -> np.ndarray:
        del mode
        del kwargs

        main_images = self._to_numpy(env_obs["main_images"])
        wrist_images = self._to_numpy(env_obs["wrist_images"])
        states = self._to_numpy(env_obs["states"]).astype(np.float32)

        if main_images.ndim != 4 or main_images.shape[-1] != 3:
            raise ValueError(
                f"main_images must have shape [B,H,W,C], got {main_images.shape}"
            )
        if wrist_images.ndim != 4 or wrist_images.shape[-1] != 3:
            raise ValueError(
                f"wrist_images must have shape [B,H,W,C], got {wrist_images.shape}"
            )
        if states.ndim != 2:
            raise ValueError(f"states must have shape [B,D], got {states.shape}")

        batch_size = int(main_images.shape[0])
        task_descriptions = self._normalize_task_descriptions(
            env_obs.get("task_descriptions"),
            batch_size,
        )
        prompts = [DEFAULT_PROMPT.format(task=task) for task in task_descriptions]
        normalized_states = self._normalize_proprio(states)
        prompt_input_ids, prompt_attention_mask = self._tokenize_prompts(prompts)
        condition_group_ids = env_obs.get("condition_group_ids")
        if condition_group_ids is None:
            condition_group_ids = torch.arange(batch_size, dtype=torch.long)
        else:
            condition_group_ids = torch.as_tensor(
                self._to_numpy(condition_group_ids),
                dtype=torch.long,
            ).reshape(-1)
        if condition_group_ids.shape[0] != batch_size:
            raise ValueError(
                "condition_group_ids length mismatch: "
                f"expected {batch_size}, got {condition_group_ids.shape[0]}"
            )
        input_images = self._build_input_image_batch(main_images, wrist_images)
        prompt_context, prompt_context_mask = self._encode_prompt_tokens(
            prompt_ids=prompt_input_ids,
            prompt_mask=prompt_attention_mask,
        )
        first_frame_latents = self._encode_input_image_latents_batch(input_images)

        normalized_actions = []
        actions = []
        for idx in range(batch_size):
            prediction = self.model.infer_action(
                prompt=prompts[idx],
                input_image=input_images[idx : idx + 1],
                action_horizon=self.action_horizon,
                proprio=normalized_states[idx : idx + 1].to(
                    device=self.model.device,
                    dtype=self.model.torch_dtype,
                ),
                negative_prompt=str(self.cfg.get("negative_prompt", "")),
                text_cfg_scale=float(self.cfg.get("text_cfg_scale", 1.0)),
                num_inference_steps=int(self.cfg.get("num_inference_steps", 20)),
                sigma_shift=(
                    None
                    if self.cfg.get("sigma_shift") is None
                    else float(self.cfg.get("sigma_shift"))
                ),
                rand_device=str(self.cfg.get("rand_device", "cpu")),
                tiled=bool(self.cfg.get("tiled", False)),
            )
            normalized_action = self._normalize_action_chunk_shape(prediction["action"])[
                : self.num_action_chunks
            ].to(dtype=torch.float32, device="cpu")
            normalized_actions.append(normalized_action)
            actions.append(self._postprocess_action(normalized_action))

        normalized_actions_tensor = torch.stack(normalized_actions, dim=0)
        actions_tensor = torch.stack(actions, dim=0)
        action_noise = torch.randn_like(normalized_actions_tensor, dtype=torch.float32)
        action_timesteps = self.model.train_action_scheduler.sample_training_t(
            batch_size=batch_size,
            device="cpu",
            dtype=torch.float32,
        ).to(dtype=torch.float32, device="cpu")

        forward_inputs = {
            "input_images": input_images.to(dtype=torch.float32, device="cpu"),
            "prompt_input_ids": prompt_input_ids.to(device="cpu"),
            "prompt_attention_mask": prompt_attention_mask.to(device="cpu"),
            "prompt_context": prompt_context.to(dtype=torch.float32, device="cpu"),
            "prompt_context_mask": prompt_context_mask.to(device="cpu"),
            "proprio": normalized_states.to(dtype=torch.float32, device="cpu"),
            "actions": normalized_actions_tensor,
            "action_noise": action_noise,
            "action_timesteps": action_timesteps,
            "condition_group_ids": condition_group_ids.to(device="cpu"),
            "first_frame_latents": first_frame_latents.to(
                dtype=torch.float32,
                device="cpu",
            ),
        }
        prev_logprobs = self.default_forward(
            forward_inputs=forward_inputs,
            compute_values=False,
            compute_entropy=False,
        )["logprobs"]
        prev_logprobs = prev_logprobs.to(dtype=torch.float32).reshape(batch_size, -1)

        result = {
            "prev_logprobs": prev_logprobs,
            "prev_values": torch.zeros((batch_size, 1), dtype=torch.float32),
            "forward_inputs": forward_inputs,
        }
        return actions_tensor.numpy().astype(np.float32), result
