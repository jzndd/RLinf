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
            return value.detach().cpu().numpy()
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

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
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
        proprio: torch.Tensor | None,
        actions: torch.Tensor,
        action_noise: torch.Tensor,
        action_timesteps: torch.Tensor,
    ) -> torch.Tensor:
        model_dtype = self.model.torch_dtype
        model_device = self.model.device

        input_images = input_images.to(device=model_device, dtype=model_dtype)
        actions = actions.to(device=model_device, dtype=model_dtype)
        action_noise = action_noise.to(device=model_device, dtype=model_dtype)
        action_timesteps = action_timesteps.to(device=model_device, dtype=model_dtype)

        context, context_mask = self._encode_prompt_tokens(
            prompt_ids=prompt_input_ids,
            prompt_mask=prompt_attention_mask,
        )
        if proprio is not None:
            proprio = proprio.to(device=model_device, dtype=model_dtype)
            context, context_mask = self.model._append_proprio_to_context(
                context=context,
                context_mask=context_mask,
                proprio=proprio,
            )

        first_frame_latents = self.model._encode_input_image_latents_tensor(
            input_image=input_images,
            tiled=bool(self.cfg.get("tiled", False)),
        )
        fuse_flag = bool(getattr(self.model.video_expert, "fuse_vae_embedding_in_latents", False))
        timestep_video = torch.zeros(
            (actions.shape[0],),
            dtype=first_frame_latents.dtype,
            device=model_device,
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

        video_pre = self.model.video_expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep_video,
            context=context,
            context_mask=context_mask,
            action=None,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        action_pre = self.model.action_expert.pre_dit(
            action_tokens=noisy_actions,
            timestep=action_timesteps,
            context=context,
            context_mask=context_mask,
        )
        attention_mask = self.model._build_mot_attention_mask(
            video_seq_len=video_pre["tokens"].shape[1],
            action_seq_len=action_pre["tokens"].shape[1],
            video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
            device=model_device,
        )
        tokens_out = self.model.mot(
            embeds_all={
                "video": video_pre["tokens"],
                "action": action_pre["tokens"],
            },
            attention_mask=attention_mask,
            freqs_all={
                "video": video_pre["freqs"],
                "action": action_pre["freqs"],
            },
            context_all={
                "video": {
                    "context": video_pre["context"],
                    "mask": video_pre["context_mask"],
                },
                "action": {
                    "context": action_pre["context"],
                    "mask": action_pre["context_mask"],
                },
            },
            t_mod_all={
                "video": video_pre["t_mod"],
                "action": action_pre["t_mod"],
            },
        )
        pred_actions = self.model.action_expert.post_dit(tokens_out["action"], action_pre)

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
            proprio=forward_inputs.get("proprio"),
            actions=forward_inputs["actions"],
            action_noise=forward_inputs["action_noise"],
            action_timesteps=forward_inputs["action_timesteps"],
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
        input_images = self._build_input_image_batch(main_images, wrist_images)

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
            normalized_action = prediction["action"][0, : self.num_action_chunks].to(
                dtype=torch.float32,
                device="cpu",
            )
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
            "proprio": normalized_states.to(dtype=torch.float32, device="cpu"),
            "actions": normalized_actions_tensor,
            "action_noise": action_noise,
            "action_timesteps": action_timesteps,
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
