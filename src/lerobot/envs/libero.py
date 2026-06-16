#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv
import libero.libero.envs as libero_envs

from lerobot.processor import RobotObservation

from robosuite.utils import camera_utils as CU
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
if not hasattr(libero_envs, "ControlEnv"):
    libero_envs.ControlEnv = ControlEnv
import third_party.LIBERO.xyg_scripts.rotate_recolor_dataset as rotate_recolor_dataset
from PIL import Image

def _parse_camera_names(camera_name: str | Sequence[str]) -> list[str]:
    """Normalize camera_name into a non-empty list of strings."""
    if isinstance(camera_name, str):
        cams = [c.strip() for c in camera_name.split(",") if c.strip()]
    elif isinstance(camera_name, (list | tuple)):
        cams = [str(c).strip() for c in camera_name if str(c).strip()]
    else:
        raise TypeError(f"camera_name must be str or sequence[str], got {type(camera_name).__name__}")
    if not cams:
        raise ValueError("camera_name resolved to an empty list.")
    return cams


def _get_suite(name: str) -> benchmark.Benchmark:
    """Instantiate a LIBERO suite by name with clear validation."""
    bench = benchmark.get_benchmark_dict()
    if name not in bench:
        raise ValueError(f"Unknown LIBERO suite '{name}'. Available: {', '.join(sorted(bench.keys()))}")
    suite = bench[name]()
    if not getattr(suite, "tasks", None):
        raise ValueError(f"Suite '{name}' has no tasks.")
    return suite


def _select_task_ids(total_tasks: int, task_ids: Iterable[int] | None) -> list[int]:
    """Validate/normalize task ids. If None → all tasks."""
    if task_ids is None:
        return list(range(total_tasks))
    ids = sorted({int(t) for t in task_ids})
    for t in ids:
        if t < 0 or t >= total_tasks:
            raise ValueError(f"task_id {t} out of range [0, {total_tasks - 1}].")
    return ids


def _normalize_task_ids(task_ids: int | str | Iterable[int] | None) -> list[int] | None:
    """Normalize CLI-friendly task id inputs into a list of ints."""
    if task_ids is None:
        return None
    if isinstance(task_ids, int):
        return [task_ids]
    if isinstance(task_ids, str):
        stripped = task_ids.strip()
        if not stripped:
            return None
        stripped = stripped.strip("[]()")
        return [int(t.strip()) for t in stripped.split(",") if t.strip()]
    return [int(t) for t in task_ids]


def _obs_camera_name_to_sim_name(camera_name: str) -> str:
    """Convert LIBERO observation camera keys to robosuite camera names."""
    return camera_name.removesuffix("_image")


def get_task_init_states(task_suite: Any, i: int) -> np.ndarray:
    init_states_path = (
        Path(get_libero_path("init_states"))
        / task_suite.tasks[i].problem_folder
        / task_suite.tasks[i].init_states_file
    )
    init_states = torch.load(init_states_path, weights_only=False)  # nosec B614
    return init_states


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


ACTION_DIM = 7
ACTION_LOW = -1.0
ACTION_HIGH = 1.0
TASK_SUITE_MAX_STEPS: dict[str, int] = {
    "libero_spatial": 280,  # longest training demo has 193 steps
    "libero_object": 280,  # longest training demo has 254 steps
    "libero_goal": 300,  # longest training demo has 270 steps
    "libero_10": 520,  # longest training demo has 505 steps
    "libero_90": 400,  # longest training demo has 373 steps
}


class LiberoEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        task_suite: Any,
        task_id: int,
        task_suite_name: str,
        episode_length: int | None = None,
        camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
        obs_type: str = "pixels",
        render_mode: str = "rgb_array",
        observation_width: int = 256,
        observation_height: int = 256,
        visualization_width: int = 640,
        visualization_height: int = 480,
        init_states: bool = True,
        episode_index: int = 0,
        camera_name_mapping: dict[str, str] | None = None,
        num_steps_wait: int = 10,
        control_mode: str = "relative",
        viewpoint_rotate: float = 0.0,
        viewpoint_debug: bool = False,
        viewpoint_debug_dir: str | None = None,
    ):
        super().__init__()
        self.task_id = task_id
        self.task_suite_name = task_suite_name
        self.obs_type = obs_type
        self.render_mode = render_mode
        self.observation_width = observation_width
        self.observation_height = observation_height
        self.visualization_width = visualization_width
        self.visualization_height = visualization_height
        self.init_states = init_states
        self.camera_name = _parse_camera_names(
            camera_name
        )  # agentview_image (main) or robot0_eye_in_hand_image (wrist)
        self.viewpoint_rotate = viewpoint_rotate
        self.viewpoint_debug = viewpoint_debug
        self.viewpoint_debug_dir = Path(viewpoint_debug_dir) if viewpoint_debug_dir else None
        self._viewpoint_debug_saved = False
        
        # Map raw camera names to "image1" and "image2".
        # The preprocessing step `preprocess_observation` will then prefix these with `.images.*`,
        # following the LeRobot convention (e.g., `observation.images.image`, `observation.images.image2`).
        # This ensures the policy consistently receives observations in the
        # expected format regardless of the original camera naming.
        if camera_name_mapping is None:
            camera_name_mapping = {
                "agentview_image": "image",
                "robot0_eye_in_hand_image": "image2",
            }
        self.camera_name_mapping = camera_name_mapping
        self.num_steps_wait = num_steps_wait
        self.episode_index = episode_index
        self.episode_length = episode_length
        # Load once and keep
        self._init_states = get_task_init_states(task_suite, self.task_id) if self.init_states else None
        self._init_state_id = self.episode_index  # tie each sub-env to a fixed init state

        self._env = self._make_envs_task(task_suite, self.task_id)
        default_steps = 500
        self._max_episode_steps = (
            TASK_SUITE_MAX_STEPS.get(task_suite_name, default_steps)
            if self.episode_length is None
            else self.episode_length
        )
        self.control_mode = control_mode
        images = {}
        for cam in self.camera_name:
            images[self.camera_name_mapping[cam]] = spaces.Box(
                low=0,
                high=255,
                shape=(self.observation_height, self.observation_width, 3),
                dtype=np.uint8,
            )

        if self.obs_type == "state":
            raise NotImplementedError(
                "The 'state' observation type is not supported in LiberoEnv. "
                "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
            )

        elif self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Dict(images),
                    "robot_state": spaces.Dict(
                        {
                            "eef": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float64),
                                    "quat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64
                                    ),
                                    "mat": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64
                                    ),
                                }
                            ),
                            "gripper": spaces.Dict(
                                {
                                    "qpos": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                    "qvel": spaces.Box(
                                        low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64
                                    ),
                                }
                            ),
                            "joints": spaces.Dict(
                                {
                                    "pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                    "vel": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float64),
                                }
                            ),
                        }
                    ),
                    "cam_info": spaces.Dict(
                        {
                            "intrinsic_matrix": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64),
                            "extrinsic_matrix": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float64),
                            "wrist_intrinsic_matrix": spaces.Box(low=-np.inf, high=np.inf, shape=(3, 3), dtype=np.float64),
                            "wrist_extrinsic_matrix": spaces.Box(low=-np.inf, high=np.inf, shape=(4, 4), dtype=np.float64),
                        }
                    )
                }
            )

        self.action_space = spaces.Box(
            low=ACTION_LOW, high=ACTION_HIGH, shape=(ACTION_DIM,), dtype=np.float32
        )

    def render(self):
        raw_obs = self._env.env._get_observations()
        image = self._format_raw_obs(raw_obs)["pixels"]["image"]
        image = image[::-1, ::-1]  # flip both H and W for visualization
        return image

    def _make_envs_task(self, task_suite: Any, task_id: int = 0):
        task = task_suite.get_task(task_id)
        self.task = task.name
        self.task_description = task.language
        task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": self.observation_height,
            "camera_widths": self.observation_width,
        }
        env = OffScreenRenderEnv(**env_args)
        env.reset()
        return env

    def _format_raw_obs(self, raw_obs: RobotObservation) -> RobotObservation:
        images = {}
        cam_info = {}
        for camera_name in self.camera_name:
            image = raw_obs[camera_name]
            images[self.camera_name_mapping[camera_name]] = image

            intrinsic_matrix, extrinsic_matrix = self._get_extrinsic_intrinsic(camera_name)
            # Hard coded ... SRY
            cam_info_prefix = "wrist_" if "eye_in_hand" in camera_name else ""
            cam_info[f"{cam_info_prefix}intrinsic_matrix"] = intrinsic_matrix
            cam_info[f"{cam_info_prefix}extrinsic_matrix"] = extrinsic_matrix

        eef_pos = raw_obs.get("robot0_eef_pos")
        eef_quat = raw_obs.get("robot0_eef_quat")

        # rotation matrix from controller
        eef_mat = self._env.robots[0].controller.ee_ori_mat if eef_pos is not None else None
        gripper_qpos = raw_obs.get("robot0_gripper_qpos")
        gripper_qvel = raw_obs.get("robot0_gripper_qvel")
        joint_pos = raw_obs.get("robot0_joint_pos")
        joint_vel = raw_obs.get("robot0_joint_vel")
        obs = {
            "pixels": images,
            "robot_state": {
                "eef": {
                    "pos": eef_pos,  # (3,)
                    "quat": eef_quat,  # (4,)
                    "mat": eef_mat,  # (3, 3)
                },
                "gripper": {
                    "qpos": gripper_qpos,  # (2,)
                    "qvel": gripper_qvel,  # (2,)
                },
                "joints": {
                    "pos": joint_pos,  # (7,)
                    "vel": joint_vel,  # (7,)
                },
            },
            "cam_info": cam_info
        }
        if self.obs_type == "pixels":
            return {"pixels": images.copy()}

        if self.obs_type == "pixels_agent_pos":
            # Validate required fields are present
            if eef_pos is None or eef_quat is None or gripper_qpos is None:
                raise ValueError(
                    f"Missing required robot state fields in raw observation. "
                    f"Got eef_pos={eef_pos is not None}, eef_quat={eef_quat is not None}, "
                    f"gripper_qpos={gripper_qpos is not None}"
                )
            return obs

        raise NotImplementedError(
            f"The observation type '{self.obs_type}' is not supported in LiberoEnv. "
            "Please switch to an image-based obs_type (e.g. 'pixels', 'pixels_agent_pos')."
        )

    def _get_extrinsic_intrinsic(self, camera_name: str):
        sim = self._env.env.sim
        env_cam_name = _obs_camera_name_to_sim_name(camera_name)
        intrinsic_matrix = CU.get_camera_intrinsic_matrix(sim, env_cam_name, self.observation_height, self.observation_width)
        extrinsic_matrix = CU.get_camera_extrinsic_matrix(sim, env_cam_name)

        return intrinsic_matrix, extrinsic_matrix

    def _save_viewpoint_debug_image(self, camera_name: str, label: str):
        if not self.viewpoint_debug or self.viewpoint_debug_dir is None:
            return

        self.viewpoint_debug_dir.mkdir(parents=True, exist_ok=True)
        sim_camera_name = _obs_camera_name_to_sim_name(camera_name)
        image = self._env.env.sim.render(
            camera_name=sim_camera_name,
            width=self.observation_width,
            height=self.observation_height,
            depth=False,
            mode="offscreen",
        )
        safe_angle = f"{self.viewpoint_rotate:g}".replace("-", "m").replace(".", "p")
        filename = (
            f"{self.task_suite_name}_task{self.task_id}_episode{self.episode_index}_"
            f"angle{safe_angle}_{label}.png"
        )
        Image.fromarray(image[::-1]).save(self.viewpoint_debug_dir / filename)

    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._env.seed(seed)
        raw_obs = self._env.reset()
        if self.init_states and self._init_states is not None:
            raw_obs = self._env.set_init_state(self._init_states[self._init_state_id])
        robot_base_name = "robot0_link0"
        camera_name = "agentview_image"  # hard coded main camera for rotation
        sim_camera_name = _obs_camera_name_to_sim_name(camera_name)
        if self.viewpoint_debug and not self._viewpoint_debug_saved:
            self._save_viewpoint_debug_image(camera_name, "before")
        camera_id = self._env.env.sim.model.camera_name2id(sim_camera_name)
        self._env = rotate_recolor_dataset.rotate_camera(
            self._env,
            camera_id=camera_id,
            camera_name=sim_camera_name,
            robot_base_name=robot_base_name,
            theta=self.viewpoint_rotate,
            debug=False,
        )
        if self.viewpoint_debug and not self._viewpoint_debug_saved:
            self._save_viewpoint_debug_image(camera_name, "after")
            self._viewpoint_debug_saved = True
        # After reset, objects may be unstable (slightly floating, intersecting, etc.).
        # Step the simulator with a no-op action for a few frames so everything settles.
        # Increasing this value can improve determinism and reproducibility across resets.
        for _ in range(self.num_steps_wait):
            raw_obs, _, _, _ = self._env.step(get_libero_dummy_action())

        if self.control_mode == "absolute":
            for robot in self._env.robots:
                robot.controller.use_delta = False
        elif self.control_mode == "relative":
            for robot in self._env.robots:
                robot.controller.use_delta = True
        else:
            raise ValueError(f"Invalid control mode: {self.control_mode}")
        observation = self._format_raw_obs(raw_obs)
        info = {"is_success": False}
        return observation, info

    def step(self, action: np.ndarray) -> tuple[RobotObservation, float, bool, bool, dict[str, Any]]:
        if action.ndim != 1:
            raise ValueError(
                f"Expected action to be 1-D (shape (action_dim,)), "
                f"but got shape {action.shape} with ndim={action.ndim}"
            )
        raw_obs, reward, done, info = self._env.step(action)

        is_success = self._env.check_success()
        terminated = done or is_success
        info.update(
            {
                "task": self.task,
                "task_id": self.task_id,
                "done": done,
                "is_success": is_success,
            }
        )
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "task_id": self.task_id,
                "done": bool(done),
                "is_success": bool(is_success),
            }
            self.reset()
        truncated = False
        return observation, reward, terminated, truncated, info

    def close(self):
        self._env.close()


def _make_env_fns(
    *,
    suite,
    suite_name: str,
    task_id: int,
    n_envs: int,
    camera_names: list[str],
    episode_length: int | None,
    init_states: bool,
    gym_kwargs: Mapping[str, Any],
    control_mode: str,
) -> list[Callable[[], LiberoEnv]]:
    """Build n_envs factory callables for a single (suite, task_id)."""

    def _make_env(episode_index: int, **kwargs) -> LiberoEnv:
        local_kwargs = dict(kwargs)
        return LiberoEnv(
            task_suite=suite,
            task_id=task_id,
            task_suite_name=suite_name,
            camera_name=camera_names,
            init_states=init_states,
            episode_length=episode_length,
            episode_index=episode_index,
            control_mode=control_mode,
            **local_kwargs,
        )

    fns: list[Callable[[], LiberoEnv]] = []
    for episode_index in range(n_envs):
        fns.append(partial(_make_env, episode_index, **gym_kwargs))
    return fns


# ---- Main API ----------------------------------------------------------------


def create_libero_envs(
    task: str,
    n_envs: int,
    gym_kwargs: dict[str, Any] | None = None,
    camera_name: str | Sequence[str] = "agentview_image,robot0_eye_in_hand_image",
    init_states: bool = True,
    env_cls: Callable[[Sequence[Callable[[], Any]]], Any] | None = None,
    control_mode: str = "relative",
    episode_length: int | None = None,
) -> dict[str, dict[int, Any]]:
    """
    Create vectorized LIBERO environments with a consistent return shape.

    Returns:
        dict[suite_name][task_id] -> vec_env (env_cls([...]) with exactly n_envs factories)
    Notes:
        - n_envs is the number of rollouts *per task* (episode_index = 0..n_envs-1).
        - `task` can be a single suite or a comma-separated list of suites.
        - You may pass `task_ids` (list[int]) inside `gym_kwargs` to restrict tasks per suite.
    """
    if env_cls is None or not callable(env_cls):
        raise ValueError("env_cls must be a callable that wraps a list of environment factory callables.")
    if not isinstance(n_envs, int) or n_envs <= 0:
        raise ValueError(f"n_envs must be a positive int; got {n_envs}.")

    gym_kwargs = dict(gym_kwargs or {})
    task_ids_filter = gym_kwargs.pop("task_ids", None)  # optional: limit to specific tasks
    task_ids_filter = _normalize_task_ids(task_ids_filter)

    camera_names = _parse_camera_names(camera_name)
    suite_names = [s.strip() for s in str(task).split(",") if s.strip()]
    if not suite_names:
        raise ValueError("`task` must contain at least one LIBERO suite name.")

    print(
        f"Creating LIBERO envs | suites={suite_names} | n_envs(per task)={n_envs} | init_states={init_states}"
    )
    if task_ids_filter is not None:
        print(f"Restricting to task_ids={task_ids_filter}")

    out: dict[str, dict[int, Any]] = defaultdict(dict)
    for suite_name in suite_names:
        suite = _get_suite(suite_name)
        total = len(suite.tasks)
        selected = _select_task_ids(total, task_ids_filter)
        if not selected:
            raise ValueError(f"No tasks selected for suite '{suite_name}' (available: {total}).")

        for tid in selected:
            fns = _make_env_fns(
                suite=suite,
                episode_length=episode_length,
                suite_name=suite_name,
                task_id=tid,
                n_envs=n_envs,
                camera_names=camera_names,
                init_states=init_states,
                gym_kwargs=gym_kwargs,
                control_mode=control_mode,
            )
            out[suite_name][tid] = env_cls(fns)
            print(f"Built vec env | suite={suite_name} | task_id={tid} | n_envs={n_envs}")

    # return plain dicts for predictability
    return {suite: dict(task_map) for suite, task_map in out.items()}
