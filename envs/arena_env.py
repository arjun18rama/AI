"""Two-humanoid self-play arena environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym
import mujoco
import numpy as np

from envs.humanoid_env import HumanoidSpec, build_actuator_xml, build_humanoid_xml


@dataclass
class ArenaConfig:
    episode_length: int = 1000
    control_timestep: float = 0.02
    physics_timestep: float = 0.002
    opponent_distance: float = 1.2
    fall_height: float = 0.7
    action_scale: float = 0.7
    healthy_height: float = 0.95


class ArenaEnv(gym.Env[np.ndarray, np.ndarray]):
    """Gymnasium environment with two simplified humanoids.

    The learning agent controls humanoid_0. The opponent action is produced
    by a callable policy that can be swapped for self-play.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: Optional[ArenaConfig] = None,
        opponent_policy: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        seed: Optional[int] = None,
    ) -> None:
        self.config = config or ArenaConfig()
        self._opponent_policy = opponent_policy
        self._rng = np.random.default_rng(seed)
        self._episode_step = 0

        self._model = mujoco.MjModel.from_xml_string(self._build_xml())
        self._model.opt.timestep = self.config.physics_timestep
        self._data = mujoco.MjData(self._model)

        self._agent0_joints = self._collect_joint_info("agent0")
        self._agent1_joints = self._collect_joint_info("agent1")

        action_dim = len(self._agent0_joints["actuator_indices"])
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(action_dim,),
            dtype=np.float32,
        )

        obs_dim = self._observation_dimension()
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )

    def set_opponent_policy(self, policy: Optional[Callable[[np.ndarray], np.ndarray]]) -> None:
        self._opponent_policy = policy

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._episode_step = 0
        self._data.qpos[:] = self._model.qpos0
        self._data.qvel[:] = 0
        mujoco.mj_forward(self._model, self._data)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray):
        self._episode_step += 1
        action = np.clip(action, -1.0, 1.0) * self.config.action_scale

        opponent_obs = self._get_obs(agent_index=1)
        opponent_action = self._opponent_action(opponent_obs)

        self._apply_action(action, opponent_action)

        steps = int(self.config.control_timestep / self.config.physics_timestep)
        for _ in range(steps):
            mujoco.mj_step(self._model, self._data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_terminated()
        truncated = self._episode_step >= self.config.episode_length

        return obs, reward, terminated, truncated, {}

    def _build_xml(self) -> str:
        humanoid0 = build_humanoid_xml(HumanoidSpec(name="agent0"), -self.config.opponent_distance / 2)
        humanoid1 = build_humanoid_xml(HumanoidSpec(name="agent1"), self.config.opponent_distance / 2)
        actuators = "\n".join([
            build_actuator_xml(HumanoidSpec(name="agent0")),
            build_actuator_xml(HumanoidSpec(name="agent1")),
        ])

        return f"""
        <mujoco model=\"humanoid_arena\">
          <option gravity=\"0 0 -9.81\"/>
          <size nconmax=\"200\" nstack=\"20000\"/>
          <worldbody>
            <geom name=\"floor\" type=\"plane\" size=\"10 10 0.1\" rgba=\"0.2 0.2 0.2 1\"/>
            {humanoid0}
            {humanoid1}
          </worldbody>
          <actuator>
            {actuators}
          </actuator>
        </mujoco>
        """

    def _collect_joint_info(self, prefix: str) -> dict:
        joints = [name for name in self._model.joint_names if name.startswith(prefix)]
        qpos_indices = []
        qvel_indices = []
        for joint in joints:
            jid = self._model.joint(joint).id
            qpos_start = self._model.jnt_qposadr[jid]
            qvel_start = self._model.jnt_dofadr[jid]
            qpos_size = 7 if self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            qvel_size = 6 if self._model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE else 1
            qpos_indices.extend(range(qpos_start, qpos_start + qpos_size))
            qvel_indices.extend(range(qvel_start, qvel_start + qvel_size))

        actuators = [
            i
            for i, name in enumerate(self._model.actuator_names)
            if name.startswith(prefix)
        ]

        torso_body_id = self._model.body(f"{prefix}_torso").id

        return {
            "joints": joints,
            "qpos_indices": np.array(qpos_indices, dtype=np.int32),
            "qvel_indices": np.array(qvel_indices, dtype=np.int32),
            "actuator_indices": np.array(actuators, dtype=np.int32),
            "torso_body_id": torso_body_id,
        }

    def _observation_dimension(self) -> int:
        agent_state = len(self._agent0_joints["qpos_indices"]) + len(self._agent0_joints["qvel_indices"])
        opponent_state = len(self._agent1_joints["qpos_indices"]) + len(self._agent1_joints["qvel_indices"])
        relative = 6
        return agent_state + opponent_state + relative

    def _get_obs(self, agent_index: int = 0) -> np.ndarray:
        if agent_index == 0:
            agent = self._agent0_joints
            opponent = self._agent1_joints
        else:
            agent = self._agent1_joints
            opponent = self._agent0_joints

        agent_qpos = self._data.qpos[agent["qpos_indices"]].copy()
        agent_qvel = self._data.qvel[agent["qvel_indices"]].copy()
        opponent_qpos = self._data.qpos[opponent["qpos_indices"]].copy()
        opponent_qvel = self._data.qvel[opponent["qvel_indices"]].copy()

        agent_root_xy = agent_qpos[0:2].copy()
        agent_qpos[0:2] = 0.0
        opponent_qpos[0:2] -= agent_root_xy

        relative = np.concatenate([
            opponent_qpos[0:3],
            opponent_qvel[0:3],
        ])

        obs = np.concatenate([agent_qpos, agent_qvel, opponent_qpos, opponent_qvel, relative])
        return obs.astype(np.float32)

    def _opponent_action(self, opponent_obs: np.ndarray) -> np.ndarray:
        if self._opponent_policy is None:
            return self._rng.uniform(-1.0, 1.0, size=self.action_space.shape).astype(np.float32)
        action = self._opponent_policy(opponent_obs)
        return np.asarray(action, dtype=np.float32)

    def _apply_action(self, action: np.ndarray, opponent_action: np.ndarray) -> None:
        controls = self._data.ctrl
        controls[self._agent0_joints["actuator_indices"]] = action
        controls[self._agent1_joints["actuator_indices"]] = opponent_action

    def _compute_reward(self) -> float:
        agent_height = self._torso_height(self._agent0_joints)
        opponent_height = self._torso_height(self._agent1_joints)

        upright = np.clip((agent_height - self.config.fall_height) / 0.5, 0.0, 1.0)
        balance = self._upright_alignment(self._agent0_joints)
        opponent_offbalance = np.clip((self.config.healthy_height - opponent_height) / 0.4, 0.0, 1.0)

        action_cost = np.square(self._data.ctrl[self._agent0_joints["actuator_indices"]]).mean()

        reward = 0.6 * upright + 0.3 * balance + 0.3 * opponent_offbalance - 0.05 * action_cost
        return float(reward)

    def _check_terminated(self) -> bool:
        agent_height = self._torso_height(self._agent0_joints)
        opponent_height = self._torso_height(self._agent1_joints)
        return agent_height < self.config.fall_height or opponent_height < self.config.fall_height

    def _torso_height(self, agent: dict) -> float:
        torso_id = agent["torso_body_id"]
        return float(self._data.xpos[torso_id][2])

    def _upright_alignment(self, agent: dict) -> float:
        torso_id = agent["torso_body_id"]
        torso_mat = self._data.xmat[torso_id].reshape(3, 3)
        up_vec = torso_mat[:, 2]
        return float(np.clip(up_vec[2], 0.0, 1.0))
