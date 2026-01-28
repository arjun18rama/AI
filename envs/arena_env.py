"""Two-humanoid self-play arena environment."""
from __future__ import annotations

from typing import Dict

import gymnasium as gym
import mujoco
import numpy as np

from envs.humanoid_env import HUMANOID_SPEC, build_humanoid_xml


def build_arena_model() -> mujoco.MjModel:
    """Create the shared arena model with two humanoids."""
    motors = "\n".join(
        f"<motor name=\"agent_a_{name}\" joint=\"agent_a_{name}\" gear=\"60\"/>"
        for name in HUMANOID_SPEC.actuator_names
    )
    motors += "\n" + "\n".join(
        f"<motor name=\"agent_b_{name}\" joint=\"agent_b_{name}\" gear=\"60\"/>"
        for name in HUMANOID_SPEC.actuator_names
    )
    xml = f"""
    <mujoco model=\"humanoid_arena\">
      <option timestep=\"0.002\" gravity=\"0 0 -9.81\"/>
      <default>
        <geom friction=\"1.0 0.8 0.1\" density=\"800\"/>
        <motor ctrlrange=\"-1 1\"/>
      </default>
      <worldbody>
        <geom type=\"plane\" size=\"6 6 0.1\" rgba=\"0.9 0.9 0.9 1\"/>
        {build_humanoid_xml("agent_a", -0.5, "0.2 0.6 0.9 1")}
        {build_humanoid_xml("agent_b", 0.5, "0.9 0.4 0.2 1")}
      </worldbody>
      <actuator>
        {motors}
      </actuator>
    </mujoco>
    """
    return mujoco.MjModel.from_xml_string(xml)


def quat_to_up_vector(quat: np.ndarray) -> np.ndarray:
    """Return world-space up direction for a given quaternion."""

    w, x, y, z = quat
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    rotation = np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]])
    return rotation[:, 2]


class ArenaEnv(gym.Env):
    """Two-agent arena with a shared policy acting on both humanoids."""

    metadata = {"render_modes": []}

    def __init__(self, frame_skip: int = 10, seed: int | None = None) -> None:
        super().__init__()
        self.model = build_arena_model()
        self.data = mujoco.MjData(self.model)
        self.frame_skip = frame_skip
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        self._agent_a_actuators = self._actuator_ids("agent_a")
        self._agent_b_actuators = self._actuator_ids("agent_b")
        self._agent_a_joints = self._joint_ids("agent_a")
        self._agent_b_joints = self._joint_ids("agent_b")
        self._agent_a_root_addr = self.model.jnt_qposadr[self._agent_a_joints[0]]
        self._agent_b_root_addr = self.model.jnt_qposadr[self._agent_b_joints[0]]
        self._agent_a_vel_addr = self.model.jnt_dofadr[self._agent_a_joints[0]]
        self._agent_b_vel_addr = self.model.jnt_dofadr[self._agent_b_joints[0]]
        self._agent_a_body_ids = self._body_ids("agent_a")
        self._agent_b_body_ids = self._body_ids("agent_b")
        self._agent_a_geom_ids = self._geom_ids(self._agent_a_body_ids)
        self._agent_b_geom_ids = self._geom_ids(self._agent_b_body_ids)
        self._base_body_mass = self.model.body_mass.copy()
        self._base_geom_friction = self.model.geom_friction.copy()

        action_size = len(self._agent_a_actuators) + len(self._agent_b_actuators)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_size,), dtype=np.float32)

        obs_size = self._obs_dim_per_agent() * 2 + 7
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

    def _actuator_ids(self, prefix: str):
        return [self.model.actuator(f"{prefix}_{name}").id for name in HUMANOID_SPEC.actuator_names]

    def _joint_ids(self, prefix: str):
        joint_names = [f"{prefix}_{name}" for name in HUMANOID_SPEC.joint_names]
        return [self.model.joint(name).id for name in joint_names]

    def _body_ids(self, prefix: str) -> list[int]:
        return [
            body_id
            for body_id in range(self.model.nbody)
            if self.model.body(body_id).name.startswith(prefix)
        ]

    def _geom_ids(self, body_ids: list[int]) -> list[int]:
        body_ids_set = set(body_ids)
        return [
            geom_id
            for geom_id in range(self.model.ngeom)
            if int(self.model.geom_bodyid[geom_id]) in body_ids_set
        ]

    def _obs_dim_per_agent(self) -> int:
        # root qpos (7) + root qvel (6) + joint qpos/qvel for remaining joints
        joint_count = len(HUMANOID_SPEC.joint_names) - 1
        return 7 + 6 + joint_count * 2

    def _get_agent_state(self, prefix: str) -> np.ndarray:
        if prefix == "agent_a":
            joint_ids = self._agent_a_joints
            root_qpos_addr = self._agent_a_root_addr
            root_qvel_addr = self._agent_a_vel_addr
        else:
            joint_ids = self._agent_b_joints
            root_qpos_addr = self._agent_b_root_addr
            root_qvel_addr = self._agent_b_vel_addr

        qpos = self.data.qpos[root_qpos_addr:root_qpos_addr + 7]
        qvel = self.data.qvel[root_qvel_addr:root_qvel_addr + 6]

        other_qpos = []
        other_qvel = []
        for joint_id in joint_ids[1:]:
            qpos_addr = self.model.jnt_qposadr[joint_id]
            qvel_addr = self.model.jnt_dofadr[joint_id]
            other_qpos.append(self.data.qpos[qpos_addr:qpos_addr + 1])
            other_qvel.append(self.data.qvel[qvel_addr:qvel_addr + 1])

        obs = np.concatenate([qpos, qvel, np.concatenate(other_qpos), np.concatenate(other_qvel)])
        return obs.astype(np.float32)

    def _relative_features(self) -> np.ndarray:
        a_root = self._get_agent_state("agent_a")[:7]
        b_root = self._get_agent_state("agent_b")[:7]
        relative_pos = b_root[:3] - a_root[:3]
        relative_quat = b_root[3:7] - a_root[3:7]
        return np.concatenate([relative_pos, relative_quat]).astype(np.float32)

    def _upright_score(self, prefix: str) -> float:
        if prefix == "agent_a":
            qpos_addr = self._agent_a_root_addr
        else:
            qpos_addr = self._agent_b_root_addr
        quat = self.data.qpos[qpos_addr + 3:qpos_addr + 7]
        up_vector = quat_to_up_vector(quat)
        return float(up_vector[2])

    def reset(self, *, seed: int | None = None, options: Dict | None = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        mujoco.mj_resetData(self.model, self.data)
        side_swap = bool(self.np_random.random() < 0.5)
        if side_swap:
            a_x = float(self.data.qpos[self._agent_a_root_addr])
            b_x = float(self.data.qpos[self._agent_b_root_addr])
            self.data.qpos[self._agent_a_root_addr] = b_x
            self.data.qpos[self._agent_b_root_addr] = a_x

        noise = self.np_random.uniform(low=-0.02, high=0.02, size=self.data.qpos.shape)
        self.data.qpos[:] += noise

        a_offset = self.np_random.uniform(low=-0.03, high=0.03, size=2)
        b_offset = self.np_random.uniform(low=-0.03, high=0.03, size=2)
        self.data.qpos[self._agent_a_root_addr:self._agent_a_root_addr + 2] += a_offset
        self.data.qpos[self._agent_b_root_addr:self._agent_b_root_addr + 2] += b_offset

        a_mass_scale = float(self.np_random.uniform(0.98, 1.02))
        b_mass_scale = float(self.np_random.uniform(0.98, 1.02))
        self.model.body_mass[self._agent_a_body_ids] = (
            self._base_body_mass[self._agent_a_body_ids] * a_mass_scale
        )
        self.model.body_mass[self._agent_b_body_ids] = (
            self._base_body_mass[self._agent_b_body_ids] * b_mass_scale
        )

        a_friction_scale = float(self.np_random.uniform(0.97, 1.03))
        b_friction_scale = float(self.np_random.uniform(0.97, 1.03))
        self.model.geom_friction[self._agent_a_geom_ids] = (
            self._base_geom_friction[self._agent_a_geom_ids] * a_friction_scale
        )
        self.model.geom_friction[self._agent_b_geom_ids] = (
            self._base_geom_friction[self._agent_b_geom_ids] * b_friction_scale
        )

        self.data.qpos[self._agent_a_root_addr + 2] = max(self.data.qpos[self._agent_a_root_addr + 2], 1.0)
        self.data.qpos[self._agent_b_root_addr + 2] = max(self.data.qpos[self._agent_b_root_addr + 2], 1.0)
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)
        info = {
            "side_swap": side_swap,
            "agent_a_offset": a_offset,
            "agent_b_offset": b_offset,
            "agent_a_mass_scale": a_mass_scale,
            "agent_b_mass_scale": b_mass_scale,
            "agent_a_friction_scale": a_friction_scale,
            "agent_b_friction_scale": b_friction_scale,
        }
        return self._get_obs(), info

    def _get_obs(self) -> np.ndarray:
        obs_a = self._get_agent_state("agent_a")
        obs_b = self._get_agent_state("agent_b")
        relative = self._relative_features()
        return np.concatenate([obs_a, obs_b, relative]).astype(np.float32)

    def _torso_height(self, prefix: str) -> float:
        if prefix == "agent_a":
            qpos_addr = self._agent_a_root_addr
        else:
            qpos_addr = self._agent_b_root_addr
        return float(self.data.qpos[qpos_addr + 2])

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        split = len(self._agent_a_actuators)
        self.data.ctrl[self._agent_a_actuators] = action[:split]
        self.data.ctrl[self._agent_b_actuators] = action[split:]

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()

        a_height = self._torso_height("agent_a")
        b_height = self._torso_height("agent_b")
        a_upright = self._upright_score("agent_a")
        b_upright = self._upright_score("agent_b")

        # Reward encourages balance plus creating imbalance in the opponent.
        a_reward = 1.5 * a_upright + 0.5 * (a_height - 0.9)
        b_reward = 1.5 * b_upright + 0.5 * (b_height - 0.9)
        push_bonus = 0.5 * abs(a_height - b_height)
        reward = a_reward + b_reward + push_bonus

        terminated = a_height < 0.5 or b_height < 0.5
        info = {
            "agent_a_reward": a_reward,
            "agent_b_reward": b_reward,
            "agent_a_upright": a_upright,
            "agent_b_upright": b_upright,
        }
        return obs, reward, terminated, False, info
