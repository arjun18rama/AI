"""Simplified humanoid definitions for MuJoCo environments.

This module provides a helper to build a minimal humanoid body composed of
capsules and hinge joints. The structure is intentionally simple for fast
simulation and stable learning.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HumanoidSpec:
    """Configuration for a simplified humanoid body."""

    name: str
    torso_height: float = 1.1
    torso_radius: float = 0.12
    thigh_length: float = 0.45
    shin_length: float = 0.45
    limb_radius: float = 0.06


def build_humanoid_xml(spec: HumanoidSpec, x_offset: float) -> str:
    """Return MuJoCo XML snippet for a simplified humanoid.

    The humanoid has a free root joint for translation/rotation and four
    actuated hinge joints (two hips, two knees). The body is deliberately
    minimal to speed up simulation.
    """

    torso_z = spec.torso_height
    thigh_z = torso_z - spec.thigh_length * 0.5
    shin_z = thigh_z - spec.thigh_length * 0.5 - spec.shin_length * 0.5

    return f"""
    <body name=\"{spec.name}_torso\" pos=\"{x_offset} 0 {torso_z}\">
      <freejoint name=\"{spec.name}_root\"/>
      <geom name=\"{spec.name}_torso_geom\" type=\"capsule\" size=\"{spec.torso_radius} {spec.torso_height * 0.5}\" fromto=\"0 0 {spec.torso_height * 0.5} 0 0 {-spec.torso_height * 0.5}\" density=\"300\"/>

      <body name=\"{spec.name}_left_thigh\" pos=\"0 0 {thigh_z}\">
        <joint name=\"{spec.name}_left_hip\" type=\"hinge\" axis=\"1 0 0\" range=\"-60 60\"/>
        <geom name=\"{spec.name}_left_thigh_geom\" type=\"capsule\" size=\"{spec.limb_radius} {spec.thigh_length * 0.5}\" fromto=\"0 0 0 0 0 {-spec.thigh_length}\" density=\"200\"/>
        <body name=\"{spec.name}_left_shin\" pos=\"0 0 {-spec.thigh_length}\">
          <joint name=\"{spec.name}_left_knee\" type=\"hinge\" axis=\"1 0 0\" range=\"-90 0\"/>
          <geom name=\"{spec.name}_left_shin_geom\" type=\"capsule\" size=\"{spec.limb_radius} {spec.shin_length * 0.5}\" fromto=\"0 0 0 0 0 {-spec.shin_length}\" density=\"150\"/>
        </body>
      </body>

      <body name=\"{spec.name}_right_thigh\" pos=\"0 0 {thigh_z}\">
        <joint name=\"{spec.name}_right_hip\" type=\"hinge\" axis=\"1 0 0\" range=\"-60 60\"/>
        <geom name=\"{spec.name}_right_thigh_geom\" type=\"capsule\" size=\"{spec.limb_radius} {spec.thigh_length * 0.5}\" fromto=\"0 0 0 0 0 {-spec.thigh_length}\" density=\"200\"/>
        <body name=\"{spec.name}_right_shin\" pos=\"0 0 {-spec.thigh_length}\">
          <joint name=\"{spec.name}_right_knee\" type=\"hinge\" axis=\"1 0 0\" range=\"-90 0\"/>
          <geom name=\"{spec.name}_right_shin_geom\" type=\"capsule\" size=\"{spec.limb_radius} {spec.shin_length * 0.5}\" fromto=\"0 0 0 0 0 {-spec.shin_length}\" density=\"150\"/>
        </body>
      </body>
    </body>
    """


def build_actuator_xml(spec: HumanoidSpec) -> str:
    """Return MuJoCo XML snippet for actuators for the humanoid joints."""

    joints = [
        f"{spec.name}_left_hip",
        f"{spec.name}_left_knee",
        f"{spec.name}_right_hip",
        f"{spec.name}_right_knee",
    ]

    motors = "\n".join(
        f"<motor name=\"{joint}_motor\" joint=\"{joint}\" ctrlrange=\"-1 1\" gear=\"80\"/>"
        for joint in joints
    )
    return motors
