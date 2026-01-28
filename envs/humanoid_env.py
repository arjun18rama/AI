"""Simplified single-humanoid MuJoCo environment.

This module defines the body structure and observation/action conventions that are
shared with the two-agent arena environment. The humanoid is intentionally abstract
(capsules + hinge joints) to keep simulation fast and stable.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class HumanoidSpec:
    """Definition of the humanoid joint layout and actuator ordering."""

    joint_names: Tuple[str, ...]
    actuator_names: Tuple[str, ...]


HUMANOID_SPEC = HumanoidSpec(
    joint_names=(
        "root",
        "hip_l",
        "knee_l",
        "ankle_l",
        "hip_r",
        "knee_r",
        "ankle_r",
        "shoulder_l",
        "elbow_l",
        "shoulder_r",
        "elbow_r",
    ),
    actuator_names=(
        "hip_l",
        "knee_l",
        "ankle_l",
        "hip_r",
        "knee_r",
        "ankle_r",
        "shoulder_l",
        "elbow_l",
        "shoulder_r",
        "elbow_r",
    ),
)


def build_humanoid_xml(name: str, x_pos: float, rgba: str) -> str:
    """Build a humanoid body XML snippet with a unique name prefix.

    The geometry is intentionally simple: a torso capsule, two legs, and two arms
    attached with hinge joints. The root uses a free joint so the agent can fall
    and recover balance.
    """

    return f"""
    <body name=\"{name}_torso\" pos=\"{x_pos} 0 1.2\">
      <freejoint name=\"{name}_root\"/>
      <geom name=\"{name}_torso_geom\" type=\"capsule\" size=\"0.15 0.25\" rgba=\"{rgba}\"/>

      <body name=\"{name}_hip_l\" pos=\"0 0.12 -0.25\">
        <joint name=\"{name}_hip_l\" type=\"hinge\" axis=\"1 0 0\" range=\"-45 45\"/>
        <geom type=\"capsule\" size=\"0.08 0.18\"/>
        <body name=\"{name}_knee_l\" pos=\"0 0 -0.35\">
          <joint name=\"{name}_knee_l\" type=\"hinge\" axis=\"1 0 0\" range=\"-80 20\"/>
          <geom type=\"capsule\" size=\"0.07 0.18\"/>
          <body name=\"{name}_ankle_l\" pos=\"0 0 -0.35\">
            <joint name=\"{name}_ankle_l\" type=\"hinge\" axis=\"1 0 0\" range=\"-30 30\"/>
            <geom name=\"{name}_foot_l\" type=\"capsule\" size=\"0.06 0.1\" pos=\"0 0 -0.1\"/>
          </body>
        </body>
      </body>

      <body name=\"{name}_hip_r\" pos=\"0 -0.12 -0.25\">
        <joint name=\"{name}_hip_r\" type=\"hinge\" axis=\"1 0 0\" range=\"-45 45\"/>
        <geom type=\"capsule\" size=\"0.08 0.18\"/>
        <body name=\"{name}_knee_r\" pos=\"0 0 -0.35\">
          <joint name=\"{name}_knee_r\" type=\"hinge\" axis=\"1 0 0\" range=\"-80 20\"/>
          <geom type=\"capsule\" size=\"0.07 0.18\"/>
          <body name=\"{name}_ankle_r\" pos=\"0 0 -0.35\">
            <joint name=\"{name}_ankle_r\" type=\"hinge\" axis=\"1 0 0\" range=\"-30 30\"/>
            <geom name=\"{name}_foot_r\" type=\"capsule\" size=\"0.06 0.1\" pos=\"0 0 -0.1\"/>
          </body>
        </body>
      </body>

      <body name=\"{name}_shoulder_l\" pos=\"0 0.2 0.1\">
        <joint name=\"{name}_shoulder_l\" type=\"hinge\" axis=\"0 1 0\" range=\"-45 45\"/>
        <geom type=\"capsule\" size=\"0.05 0.15\"/>
        <body name=\"{name}_elbow_l\" pos=\"0 0 -0.25\">
          <joint name=\"{name}_elbow_l\" type=\"hinge\" axis=\"0 1 0\" range=\"-90 10\"/>
          <geom type=\"capsule\" size=\"0.04 0.12\"/>
        </body>
      </body>

      <body name=\"{name}_shoulder_r\" pos=\"0 -0.2 0.1\">
        <joint name=\"{name}_shoulder_r\" type=\"hinge\" axis=\"0 1 0\" range=\"-45 45\"/>
        <geom type=\"capsule\" size=\"0.05 0.15\"/>
        <body name=\"{name}_elbow_r\" pos=\"0 0 -0.25\">
          <joint name=\"{name}_elbow_r\" type=\"hinge\" axis=\"0 1 0\" range=\"-90 10\"/>
          <geom type=\"capsule\" size=\"0.04 0.12\"/>
        </body>
      </body>
    </body>
    """


def build_single_humanoid_model() -> mujoco.MjModel:
    """Create a MuJoCo model containing a single humanoid and flat floor."""

    xml = f"""
    <mujoco model=\"humanoid_single\">
      <option timestep=\"0.002\" gravity=\"0 0 -9.81\"/>
      <default>
        <geom friction=\"1.0 0.8 0.1\" density=\"800\"/>
        <motor ctrlrange=\"-1 1\"/>
      </default>
      <worldbody>
        <geom type=\"plane\" size=\"5 5 0.1\" rgba=\"0.8 0.8 0.8 1\"/>
        {build_humanoid_xml("agent", 0.0, "0.2 0.6 0.9 1")}
      </worldbody>
      <actuator>
        <motor name=\"agent_hip_l\" joint=\"agent_hip_l\" gear=\"60\"/>
        <motor name=\"agent_knee_l\" joint=\"agent_knee_l\" gear=\"80\"/>
        <motor name=\"agent_ankle_l\" joint=\"agent_ankle_l\" gear=\"40\"/>
        <motor name=\"agent_hip_r\" joint=\"agent_hip_r\" gear=\"60\"/>
        <motor name=\"agent_knee_r\" joint=\"agent_knee_r\" gear=\"80\"/>
        <motor name=\"agent_ankle_r\" joint=\"agent_ankle_r\" gear=\"40\"/>
        <motor name=\"agent_shoulder_l\" joint=\"agent_shoulder_l\" gear=\"30\"/>
        <motor name=\"agent_elbow_l\" joint=\"agent_elbow_l\" gear=\"20\"/>
        <motor name=\"agent_shoulder_r\" joint=\"agent_shoulder_r\" gear=\"30\"/>
        <motor name=\"agent_elbow_r\" joint=\"agent_elbow_r\" gear=\"20\"/>
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


class HumanoidEnv(gym.Env):
    """Single-humanoid environment used for debugging and unit experiments."""

    metadata = {"render_modes": []}

    def __init__(self, frame_skip: int = 10, seed: int | None = None) -> None:
        super().__init__()
        self.model = build_single_humanoid_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._actuator_ids = [self.model.actuator(name).id for name in self._actuator_names()]
        action_size = len(self._actuator_ids)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)

        obs_size = self._obs_dim()
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _actuator_names(self) -> List[str]:
        return [f"agent_{name}" for name in HUMANOID_SPEC.actuator_names]

    def _obs_dim(self) -> int:
        # root qpos (7) + root qvel (6) + joint qpos/qvel for remaining joints
        joint_count = len(HUMANOID_SPEC.joint_names) - 1
        return 7 + 6 + joint_count * 2

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos.copy()
        qvel = self.data.qvel.copy()
        # Keep the full root pose for balance and velocity, but no absolute x/y in reward.
        obs = np.concatenate([qpos[:7], qvel[:6], qpos[7:], qvel[6:]])
        return obs.astype(np.float32)

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        mujoco.mj_resetData(self.model, self.data)
        noise = self.np_random.uniform(low=-0.02, high=0.02, size=self.data.qpos.shape)
        self.data.qpos[:] += noise
        self.data.qpos[2] = max(self.data.qpos[2], 1.0)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[self._actuator_ids] = action
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        # Simple upright reward for diagnostics.
        torso_height = self.data.qpos[2]
        upright_reward = 1.0 if torso_height > 0.9 else -1.0
        terminated = torso_height < 0.5
        return obs, upright_reward, terminated, False, {}
