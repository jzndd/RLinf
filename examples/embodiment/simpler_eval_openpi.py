#!/usr/bin/env python3
"""Evaluate an OpenPI websocket policy on SimplerEnv visual matching."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np


WIDOWX_VISUAL_MATCHING_TASKS = (
    "widowx_spoon_on_towel",
    "widowx_carrot_on_plate",
    "widowx_stack_cube",
    "widowx_put_eggplant_in_basket",
)

GOOGLE_ROBOT_VISUAL_MATCHING_TASKS = (
    "google_robot_pick_coke_can",
    "google_robot_move_near",
    "google_robot_open_drawer",
    "google_robot_close_drawer",
    "google_robot_place_apple_in_closed_top_drawer",
)

WIDOWX_BRIDGE_ROTATION = np.asarray(
    [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]],
    dtype=np.float64,
)

TASK_DISPLAY_NAMES = {
    "widowx_spoon_on_towel": "Put Spoon on Towel",
    "widowx_carrot_on_plate": "Put Carrot on Plate",
    "widowx_stack_cube": "Stack Green Block on Yellow Block",
    "widowx_put_eggplant_in_basket": "Put Eggplant in Yellow Basket",
    "google_robot_pick_coke_can": "Pick Coke Can",
    "google_robot_move_near": "Move Near",
    "google_robot_open_drawer": "Open Drawer",
    "google_robot_close_drawer": "Close Drawer",
    "google_robot_place_apple_in_closed_top_drawer": "Place Apple in Closed Top Drawer",
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (set, tuple)):
        return list(value)
    return str(value)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=_json_default) + "\n")


def _save_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(image).save(path)


def _make_policy(host: str, port: int):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    return WebsocketClientPolicy(host=host, port=port)


def _validate_policy_actions(response: dict[str, Any], *, context: str) -> np.ndarray:
    if "actions" not in response:
        raise RuntimeError(f"{context}: policy response is missing 'actions'; keys={list(response.keys())}")
    actions = np.asarray(response["actions"], dtype=np.float32)
    if actions.shape != (16, 7):
        raise RuntimeError(f"{context}: expected actions shape (16, 7), got {actions.shape}")
    if not np.isfinite(actions).all():
        bad_count = int(np.size(actions) - np.isfinite(actions).sum())
        raise RuntimeError(f"{context}: policy returned {bad_count} NaN/Inf action values")
    return actions


def run_policy_smoke(host: str, port: int, image_height: int, image_width: int) -> dict[str, Any]:
    policy = _make_policy(host, port)
    obs = {
        "observation/image": np.zeros((image_height, image_width, 3), dtype=np.uint8),
        "observation/state": np.zeros((8,), dtype=np.float32),
        "prompt": "put spoon on towel",
    }
    started_at = time.time()
    actions = _validate_policy_actions(policy.infer(obs), context="policy smoke")
    elapsed_s = time.time() - started_at
    return {
        "ok": True,
        "elapsed_s": elapsed_s,
        "actions_shape": list(actions.shape),
        "actions_min": float(actions.min()),
        "actions_max": float(actions.max()),
        "actions_mean": float(actions.mean()),
    }


def _maybe_call(value: Any) -> Any:
    if callable(value):
        return value()
    return value


def _iter_controllers(env: Any):
    agent = getattr(env, "agent", None)
    controller = getattr(agent, "controller", None)
    if controller is None:
        return
    controllers = getattr(controller, "controllers", None)
    if isinstance(controllers, dict):
        if "arm" in controllers:
            yield controllers["arm"]
        for name, sub_controller in controllers.items():
            if name != "arm":
                yield sub_controller
    yield controller


def _pose_components_from_any(value: Any) -> tuple[np.ndarray, np.ndarray] | None:
    if value is None:
        return None
    value = _maybe_call(value)
    if value is None:
        return None
    if hasattr(value, "p") and hasattr(value, "q"):
        return np.asarray(value.p, dtype=np.float64), np.asarray(value.q, dtype=np.float64)
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size >= 7:
        return array[:3], array[3:7]
    return None


def _extract_pose_from_controller(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    for controller in _iter_controllers(env) or ():
        for attr in ("ee_pose_at_base", "ee_pose"):
            with suppress(Exception):
                pose = _pose_components_from_any(getattr(controller, attr, None))
                if pose is not None:
                    return pose
        get_state = getattr(controller, "get_state", None)
        if callable(get_state):
            with suppress(Exception):
                state = get_state()
                if isinstance(state, dict):
                    if "target_pose" in state:
                        pose = _pose_components_from_any(state["target_pose"])
                        if pose is not None:
                            return pose
                    for child_state in state.values():
                        if isinstance(child_state, dict) and "target_pose" in child_state:
                            pose = _pose_components_from_any(child_state["target_pose"])
                            if pose is not None:
                                return pose
    return None


def _extract_pose_from_links(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    agent = getattr(env, "agent", None)
    robot = getattr(agent, "robot", None)
    get_links = getattr(robot, "get_links", None)
    if not callable(get_links):
        return None
    preferred_names = ("ee_gripper_link", "ee_arm_link", "gripper_link")
    with suppress(Exception):
        links = list(get_links())
        for name in preferred_names:
            for link in links:
                if getattr(link, "name", None) == name:
                    pose = _pose_components_from_any(getattr(link, "pose", None))
                    if pose is not None:
                        return pose
    return None


def _extract_google_target_pose(env: Any) -> tuple[np.ndarray, np.ndarray] | None:
    for controller in _iter_controllers(env) or ():
        for attr in ("_target_pose", "target_pose"):
            with suppress(Exception):
                pose = _pose_components_from_any(getattr(controller, attr, None))
                if pose is not None:
                    return pose
        get_state = getattr(controller, "get_state", None)
        if callable(get_state):
            with suppress(Exception):
                state = get_state()
                if isinstance(state, dict) and "target_pose" in state:
                    pose = _pose_components_from_any(state["target_pose"])
                    if pose is not None:
                        return pose
    return None


def _quat_wxyz_to_bridge_euler_xyz(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat_wxyz))
    if norm <= 1e-8:
        raise RuntimeError(f"Invalid end-effector quaternion with near-zero norm: {quat_wxyz}")
    quat_wxyz = quat_wxyz / norm
    try:
        from scipy.spatial.transform import Rotation

        quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]], dtype=np.float64)
        pose_rotation = Rotation.from_quat(quat_xyzw).as_matrix()
        bridge_rotation = pose_rotation @ WIDOWX_BRIDGE_ROTATION.T
        return Rotation.from_matrix(bridge_rotation).as_euler("xyz")
    except Exception:
        from transforms3d.euler import mat2euler
        from transforms3d.quaternions import quat2mat

        pose_rotation = quat2mat(quat_wxyz)
        bridge_rotation = pose_rotation @ WIDOWX_BRIDGE_ROTATION.T
        return np.asarray(mat2euler(bridge_rotation, axes="sxyz"), dtype=np.float64)


def _rpy_to_rotvec(rpy: np.ndarray) -> np.ndarray:
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    try:
        from scipy.spatial.transform import Rotation

        return Rotation.from_euler("xyz", rpy).as_rotvec()
    except Exception:
        from transforms3d.euler import euler2mat
        from transforms3d.axangles import mat2axangle

        axis, angle = mat2axangle(euler2mat(*rpy, axes="sxyz"))
        return np.asarray(axis, dtype=np.float64) * float(angle)


def _extract_gripper_open_amount(env: Any, obs: dict[str, Any]) -> float:
    agent = getattr(env, "agent", None)
    get_closedness = getattr(agent, "get_gripper_closedness", None)
    if callable(get_closedness):
        with suppress(Exception):
            closedness = float(get_closedness())
            return float(np.clip(1.0 - closedness, 0.0, 1.0))

    robot = getattr(agent, "robot", None)
    with suppress(Exception):
        qpos = np.asarray(robot.get_qpos(), dtype=np.float64)[-2:]
        qlim = np.asarray(robot.get_qlimits(), dtype=np.float64)[-2:]
        open_amount = (qpos - qlim[:, 0]) / np.maximum(qlim[:, 1] - qlim[:, 0], 1e-8)
        return float(np.clip(np.mean(open_amount), 0.0, 1.0))

    with suppress(Exception):
        agent_obs = obs.get("agent", {})
        qpos = np.asarray(agent_obs.get("qpos"), dtype=np.float64)[-2:]
        return float(np.clip(np.mean(qpos), 0.0, 1.0))

    raise RuntimeError("Unable to extract WidowX gripper open amount from env.agent or observation")


def extract_bridge_state(env: Any, obs: dict[str, Any]) -> np.ndarray:
    pose = _extract_pose_from_controller(env)
    if pose is None:
        pose = _extract_pose_from_links(env)
    if pose is None:
        obs_keys = list(obs.keys()) if isinstance(obs, dict) else type(obs).__name__
        raise RuntimeError(f"Unable to extract end-effector pose for Bridge state; observation keys={obs_keys}")

    position, quat_wxyz = pose
    euler_xyz = _quat_wxyz_to_bridge_euler_xyz(quat_wxyz)
    gripper_open = _extract_gripper_open_amount(env, obs)
    state = np.asarray(
        [
            position[0],
            position[1],
            position[2],
            euler_xyz[0],
            euler_xyz[1],
            euler_xyz[2],
            0.0,
            gripper_open,
        ],
        dtype=np.float32,
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid Bridge state: shape={state.shape}, values={state}")
    return state


def extract_google_robot_state(env: Any, obs: dict[str, Any]) -> np.ndarray:
    pose = _extract_google_target_pose(env)
    if pose is None:
        raise RuntimeError("Unable to extract Google Robot controller target pose")

    position, quat_wxyz = pose
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
    quat_wxyz /= max(float(np.linalg.norm(quat_wxyz)), 1e-8)
    agent = getattr(env, "agent", None)
    get_closedness = getattr(agent, "get_gripper_closedness", None)
    if not callable(get_closedness):
        raise RuntimeError("Unable to extract Google Robot gripper closedness")
    closedness = float(np.clip(get_closedness(), 0.0, 1.0))
    state = np.asarray(
        [
            position[0],
            position[1],
            position[2],
            quat_wxyz[1],
            quat_wxyz[2],
            quat_wxyz[3],
            quat_wxyz[0],
            closedness,
        ],
        dtype=np.float32,
    )
    if state.shape != (8,) or not np.isfinite(state).all():
        raise RuntimeError(f"Invalid Google Robot state: shape={state.shape}, values={state}")
    return state


def extract_policy_state(env: Any, obs: dict[str, Any], robot_setup: str) -> np.ndarray:
    if robot_setup == "widowx_bridge":
        return extract_bridge_state(env, obs)
    if robot_setup == "google_robot":
        return extract_google_robot_state(env, obs)
    raise ValueError(f"Unsupported robot setup: {robot_setup}")


def prepare_env_action(
    model_action: np.ndarray,
    *,
    robot_setup: str,
    rotation_mode: str,
    gripper_mode: str,
) -> np.ndarray:
    action = np.asarray(model_action, dtype=np.float64).reshape(7).copy()
    if robot_setup == "google_robot":
        # Fractal actions use axis-angle rotation and +1=open, 0=close.
        # SimplerEnv's Google gripper controller uses -1=open, +1=close.
        action[6] = 1.0 - 2.0 * float(np.clip(action[6], 0.0, 1.0))
        return action.astype(np.float32)
    if robot_setup != "widowx_bridge":
        raise ValueError(f"Unsupported robot setup: {robot_setup}")

    if rotation_mode == "rpy":
        action[3:6] = _rpy_to_rotvec(action[3:6])
    elif rotation_mode == "axis_angle":
        pass
    else:
        raise ValueError(f"Unsupported rotation mode: {rotation_mode}")

    if gripper_mode == "open01":
        action[6] = 2.0 * float(np.clip(action[6], 0.0, 1.0)) - 1.0
    elif gripper_mode == "env":
        pass
    else:
        raise ValueError(f"Unsupported gripper mode: {gripper_mode}")
    return action.astype(np.float32)


def clip_action_to_env(env: Any, action: np.ndarray) -> tuple[np.ndarray, bool]:
    action_space = getattr(env, "action_space", None)
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return action, False
    low = np.asarray(low, dtype=np.float32).reshape(action.shape)
    high = np.asarray(high, dtype=np.float32).reshape(action.shape)
    clipped = np.clip(action, low, high)
    was_clipped = bool(np.any(np.not_equal(clipped, action)))
    return clipped.astype(np.float32), was_clipped


def _safe_env_reset(env: Any, seed: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    if seed is not None:
        try:
            return env.reset(seed=seed)
        except TypeError:
            pass
    return env.reset()


def _step_env(env: Any, action: np.ndarray) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
    result = env.step(action)
    if len(result) != 5:
        raise RuntimeError(f"Expected env.step to return 5 values, got {len(result)}")
    obs, reward, done_or_success, truncated, info = result
    info = dict(info or {})
    success = bool(done_or_success or info.get("success", False))
    return obs, float(reward), success, bool(truncated), info


def _get_language_instruction(env: Any) -> str:
    get_instruction = getattr(env, "get_language_instruction", None)
    if not callable(get_instruction):
        raise RuntimeError("SimplerEnv task does not expose get_language_instruction()")
    return str(get_instruction())


def _prompt_for_task(task: str, raw_instruction: str, prompt_overrides: dict[str, str] | None) -> str:
    if prompt_overrides and task in prompt_overrides:
        return prompt_overrides[task]
    return raw_instruction


def _save_video(path: Path, frames: list[np.ndarray], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import mediapy as media

    media.write_video(str(path), frames, fps=fps)


def evaluate_task(
    *,
    task: str,
    task_index: int,
    episodes_per_task: int,
    episode_offset: int,
    action_chunk: int,
    policy: Any,
    output_dir: Path,
    seed: int,
    max_episode_steps: int | None,
    camera_name: str | None,
    save_video: bool,
    video_fps: int,
    robot_setup: str,
    rotation_mode: str,
    gripper_mode: str,
    debug_export: bool,
    debug_export_steps: int,
    prompt_overrides: dict[str, str] | None,
) -> list[dict[str, Any]]:
    import simpler_env
    from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict

    logging.info("Creating SimplerEnv task %s (%s)", task, TASK_DISPLAY_NAMES.get(task, task))
    env = simpler_env.make(task)
    records: list[dict[str, Any]] = []
    try:
        for episode_idx in range(episodes_per_task):
            global_episode_idx = episode_offset + episode_idx
            episode_seed = seed + task_index * 100000 + global_episode_idx
            obs, reset_info = _safe_env_reset(env, episode_seed)
            raw_instruction = _get_language_instruction(env)
            instruction = _prompt_for_task(task, raw_instruction, prompt_overrides)
            with suppress(Exception):
                policy.reset()

            chunk: np.ndarray | None = None
            chunk_index = 0
            clipped_actions = 0
            rewards: list[float] = []
            frames: list[np.ndarray] = []
            info: dict[str, Any] = {}
            success = False
            truncated = False
            step_idx = 0
            started_at = time.time()
            debug_dir = output_dir / "debug" / task / f"episode_{global_episode_idx:04d}"
            debug_meta: dict[str, Any] | None = None

            while not truncated:
                if max_episode_steps is not None and step_idx >= max_episode_steps:
                    truncated = True
                    info = {**info, "max_episode_steps_reached": max_episode_steps}
                    break

                image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=camera_name)
                if save_video:
                    frames.append(np.asarray(image))
                if debug_export and step_idx < debug_export_steps:
                    _save_png(debug_dir / f"frame_{step_idx:04d}.png", np.asarray(image))

                if chunk is None or chunk_index >= len(chunk):
                    policy_state = extract_policy_state(env, obs, robot_setup)
                    response = policy.infer(
                        {
                            "observation/image": np.asarray(image, dtype=np.uint8),
                            "observation/state": policy_state,
                            "prompt": instruction,
                        }
                    )
                    chunk = _validate_policy_actions(response, context=f"{task} episode {episode_idx}")[:action_chunk]
                    chunk_index = 0
                    if debug_export and debug_meta is None:
                        debug_meta = {
                            "task": task,
                            "episode_index": global_episode_idx,
                            "seed": episode_seed,
                            "raw_instruction": raw_instruction,
                            "prompt_sent": instruction,
                            "policy_state": policy_state,
                            "action_chunk": chunk,
                            "image_shape": list(np.asarray(image).shape),
                            "image_dtype": str(np.asarray(image).dtype),
                        }
                        if robot_setup == "widowx_bridge":
                            debug_meta["bridge_state"] = policy_state
                        _write_json(debug_dir / "metadata.json", debug_meta)

                raw_action = chunk[chunk_index]
                chunk_index += 1
                env_action = prepare_env_action(
                    raw_action,
                    robot_setup=robot_setup,
                    rotation_mode=rotation_mode,
                    gripper_mode=gripper_mode,
                )
                env_action, was_clipped = clip_action_to_env(env, env_action)
                clipped_actions += int(was_clipped)

                obs, reward, success, truncated, info = _step_env(env, env_action)
                rewards.append(reward)
                step_idx += 1

                new_raw_instruction = _get_language_instruction(env)
                new_instruction = _prompt_for_task(task, new_raw_instruction, prompt_overrides)
                if new_instruction != instruction:
                    logging.info(
                        "Task %s episode %d prompt changed: raw=%s sent=%s",
                        task,
                        episode_idx,
                        new_raw_instruction,
                        new_instruction,
                    )
                    raw_instruction = new_raw_instruction
                    instruction = new_instruction
                    chunk = None
                    chunk_index = 0

                if success:
                    break

            if save_video and frames:
                video_name = f"episode_{global_episode_idx:04d}_success_{int(success)}.mp4"
                _save_video(output_dir / "videos" / task / video_name, frames, fps=video_fps)

            record = {
                "task": task,
                "task_display_name": TASK_DISPLAY_NAMES.get(task, task),
                "episode_index": global_episode_idx,
                "shard_episode_index": episode_idx,
                "seed": episode_seed,
                "success": bool(success),
                "truncated": bool(truncated),
                "steps": int(step_idx),
                "reward_sum": float(np.sum(rewards)) if rewards else 0.0,
                "reward_last": float(rewards[-1]) if rewards else math.nan,
                "clipped_actions": int(clipped_actions),
                "elapsed_s": time.time() - started_at,
                "reset_info": reset_info,
                "final_info": info,
                "raw_instruction": raw_instruction,
                "prompt_sent": instruction,
                "final_instruction": instruction,
            }
            records.append(record)
            logging.info(
                "Task %s shard_episode %d/%d global_episode=%d success=%s steps=%d clipped_actions=%d",
                task,
                episode_idx + 1,
                episodes_per_task,
                global_episode_idx,
                success,
                step_idx,
                clipped_actions,
            )
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            with suppress(Exception):
                close()
    return records


def summarize_records(records: list[dict[str, Any]], tasks: list[str], started_at: float, args: argparse.Namespace) -> dict[str, Any]:
    task_metrics = {}
    for task in tasks:
        task_records = [record for record in records if record["task"] == task]
        successes = sum(int(record["success"]) for record in task_records)
        episodes = len(task_records)
        task_metrics[task] = {
            "display_name": TASK_DISPLAY_NAMES.get(task, task),
            "episodes": episodes,
            "successes": successes,
            "success_rate": float(successes / episodes) if episodes else math.nan,
            "avg_steps": float(np.mean([record["steps"] for record in task_records])) if task_records else math.nan,
            "total_clipped_actions": int(sum(record["clipped_actions"] for record in task_records)),
        }
    rates = [value["success_rate"] for value in task_metrics.values() if not math.isnan(value["success_rate"])]
    return {
        "started_at": started_at,
        "elapsed_s": time.time() - started_at,
        "tasks": task_metrics,
        "average_success_rate": float(np.mean(rates)) if rates else math.nan,
        "total_episodes": len(records),
        "total_successes": int(sum(int(record["success"]) for record in records)),
        "robot_setup": args.robot_setup,
        "rotation_mode": args.rotation_mode,
        "gripper_mode": args.gripper_mode,
        "camera_name": args.camera_name,
        "host": args.host,
        "port": args.port,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=list(WIDOWX_VISUAL_MATCHING_TASKS))
    parser.add_argument("--robot-setup", choices=("widowx_bridge", "google_robot"), default="widowx_bridge")
    parser.add_argument("--episodes-per-task", type=int, default=60)
    parser.add_argument("--episode-offset", type=int, default=0)
    parser.add_argument("--action-chunk", type=int, default=16)
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--max-episode-steps", type=int, default=None)
    parser.add_argument("--camera-name", default=None)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=5)
    parser.add_argument("--rotation-mode", choices=("rpy", "axis_angle"), default="rpy")
    parser.add_argument("--gripper-mode", choices=("open01", "env"), default="open01")
    parser.add_argument("--debug-export", action="store_true")
    parser.add_argument("--debug-export-steps", type=int, default=16)
    parser.add_argument("--prompt-overrides-json", type=Path, default=None)
    parser.add_argument("--policy-smoke-only", action="store_true")
    parser.add_argument("--fake-image-height", type=int, default=256)
    parser.add_argument("--fake-image-width", type=int, default=256)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(args.output_dir / "args.json", vars(args))

    if args.policy_smoke_only:
        result = run_policy_smoke(args.host, args.port, args.fake_image_height, args.fake_image_width)
        _write_json(args.output_dir / "policy_smoke.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    if not 1 <= args.action_chunk <= 16:
        raise ValueError(f"--action-chunk must be in [1, 16], got {args.action_chunk}")
    if args.debug_export_steps < 1:
        raise ValueError(f"--debug-export-steps must be >= 1, got {args.debug_export_steps}")
    prompt_overrides = None
    if args.prompt_overrides_json is not None:
        prompt_overrides = json.loads(args.prompt_overrides_json.read_text())
        if not isinstance(prompt_overrides, dict):
            raise ValueError("--prompt-overrides-json must contain a JSON object mapping task to prompt")

    supported_tasks = (
        WIDOWX_VISUAL_MATCHING_TASKS
        if args.robot_setup == "widowx_bridge"
        else GOOGLE_ROBOT_VISUAL_MATCHING_TASKS
    )
    invalid_tasks = [task for task in args.tasks if task not in supported_tasks]
    if invalid_tasks:
        raise ValueError(f"Unsupported {args.robot_setup} visual matching task(s): {invalid_tasks}")

    import simpler_env
    from simpler_env import ENVIRONMENTS

    missing_tasks = [task for task in args.tasks if task not in ENVIRONMENTS]
    if missing_tasks:
        raise RuntimeError(f"SimplerEnv does not expose task(s): {missing_tasks}")

    policy = _make_policy(args.host, args.port)
    started_at = time.time()
    records: list[dict[str, Any]] = []
    episodes_path = args.output_dir / "episodes.jsonl"
    with episodes_path.open("w", encoding="utf-8") as episodes_file:
        for task_index, task in enumerate(args.tasks):
            task_records = evaluate_task(
                task=task,
                task_index=task_index,
                episodes_per_task=args.episodes_per_task,
                episode_offset=args.episode_offset,
                action_chunk=args.action_chunk,
                policy=policy,
                output_dir=args.output_dir,
                seed=args.seed,
                max_episode_steps=args.max_episode_steps,
                camera_name=args.camera_name,
                save_video=args.save_video,
                video_fps=args.video_fps,
                robot_setup=args.robot_setup,
                rotation_mode=args.rotation_mode,
                gripper_mode=args.gripper_mode,
                debug_export=args.debug_export,
                debug_export_steps=args.debug_export_steps,
                prompt_overrides=prompt_overrides,
            )
            for record in task_records:
                episodes_file.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")
                episodes_file.flush()
            records.extend(task_records)
            metrics = summarize_records(records, list(args.tasks), started_at, args)
            _write_json(args.output_dir / "metrics.json", metrics)

    metrics = summarize_records(records, list(args.tasks), started_at, args)
    _write_json(args.output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logging.exception("SimplerEnv OpenPI evaluation failed")
        sys.exit(1)
