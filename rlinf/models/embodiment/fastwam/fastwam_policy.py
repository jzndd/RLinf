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
                    missing_keys.extend([
                        "proprio_encoder.weight",
                        "proprio_encoder.bias",
                    ])
                else:
                    proprio_result = proprio_module.load_state_dict(
                        proprio_payload,
                        strict=strict,
                    )
                    missing_keys.extend([
                        f"proprio_encoder.{key}" for key in proprio_result.missing_keys
                    ])
                    unexpected_keys.extend([
                        f"proprio_encoder.{key}" for key in proprio_result.unexpected_keys
                    ])
            elif proprio_payload is not None and strict:
                raise RuntimeError(
                    "FastWAM checkpoint contains `proprio_encoder` but current model does not use it."
                )

            return _IncompatibleKeys(missing_keys, unexpected_keys)

        return super().load_state_dict(state_dict, strict=strict)

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError

    def default_forward(self, **kwargs):
        raise NotImplementedError("FastWAMPolicy only supports evaluation inference in RLinf.")

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

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray:
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

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

    @staticmethod
    def _normalize_task_descriptions(task_descriptions: Any, batch_size: int) -> list[str]:
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
        normalized_states = self._normalize_proprio(states)

        model_dtype = self.model.torch_dtype
        model_device = self.model.device
        all_actions = []
        for idx in range(batch_size):
            input_image = self._build_input_image(
                main_image=main_images[idx],
                wrist_image=wrist_images[idx],
                device=model_device,
                dtype=model_dtype,
            )
            prompt = DEFAULT_PROMPT.format(task=task_descriptions[idx])
            prediction = self.model.infer_action(
                prompt=prompt,
                input_image=input_image,
                action_horizon=self.action_horizon,
                proprio=normalized_states[idx : idx + 1],
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
            action = self._denormalize_action(prediction["action"])[0]
            action[..., -1] = action[..., -1] * 2.0 - 1.0
            action[..., -1] = action[..., -1] * -1.0
            if self.binarize_gripper:
                action[..., -1] = np.sign(action[..., -1])
            all_actions.append(action[: self.num_action_chunks].astype(np.float32))

        actions = np.stack(all_actions, axis=0)
        flat_actions = torch.as_tensor(actions, dtype=torch.float32).reshape(batch_size, -1)
        result = {
            "prev_logprobs": torch.zeros_like(flat_actions, dtype=torch.float32),
            "prev_values": torch.zeros((batch_size, 1), dtype=torch.float32),
            "forward_inputs": {"action": flat_actions},
        }
        return actions, result
