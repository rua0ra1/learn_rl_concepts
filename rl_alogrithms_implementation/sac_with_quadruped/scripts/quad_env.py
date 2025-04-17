import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import time
from pathlib import Path

script_Dir = Path(__file__).resolve().parent
project_root = script_Dir.parent
model_dir = project_root / "unitree_ros/robots/a1_description"
# urdf and mesh file
urdf_file = model_dir/"urdf/a1.urdf"
mesh_path = model_dir/"meshes"
assert urdf_file.exists(), f"urdf file not found at {urdf_file}"


class CustomQuadEnv (gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 240}

    def __init__(self, render_mode=None):
        super()().__init__()
        self.render_mode = render_mode
        self.dt = 1/240

        # action space one torque for each joint
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(12,), dtype=np.float32)
        # observation space: joint pos, joint vel, base vel and opeitnaiton
        obs_dim = 12 + 12 + 6 + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        # connect the Bulet
        mode = p.GUI if render_mode == "human" else p.DIRECT
        self.cid = p.connect(mode)
        p.setTimeStep(self.dt)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # load the plane
        self.urdf = urdf_file
        self.max_torque = 35.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.dt)
        # load the urdf
        p.loadURDF("plane.urdf")
        flags = flags = p.URDF_USE_INERTIA_FROM_FILE | p.URDF_MAINTAIN_LINK_ORDER

        self.robot = p.loadURDF(self.urdf, [0, 0, 0.4], p.getQuaternionFromEuler([0, 0, 0]),
                                flags=flags, useFixedBase=False)

        for _ in range(20):
            p.stepSimulation()  # let it settle
        return self._get_obs(), {}

    def step(self, action):
        # scale action → torque
        torques = np.clip(action, -1, 1) * self.max_torque
        for j, τ in enumerate(torques):
            p.setJointMotorControl2(
                self.robot, j, p.TORQUE_CONTROL, force=float(τ))
        p.stepSimulation()
        obs = self._get_obs()
        reward = self._compute_reward(obs, torques)
        done = self._check_termination(obs)
        if self.render_mode == "human":
            time.sleep(self.dt)
        return obs, reward, done, False, {}

    def _get_obs(self):
        q, qd = [], []
        for j in range(p.getNumJoints(self.robot)):
            js = p.getJointState(self.robot, j)
            q.append(js[0])
            qd.append(js[1])
        lin, ang = p.getBaseVelocity(self.robot)
        _, quat = p.getBasePositionAndOrientation(self.robot)
        return np.concatenate([q, qd, lin, ang, quat]).astype(np.float32)

    def _compute_reward(self, obs, torques):
        vx = obs[12+12]  # forward velocity
        alive = 0.3
        effort = 0.0002 * np.square(torques).sum()
        return vx + alive - effort

    def _check_termination(self, obs):
        roll, pitch, _ = p.getEulerFromQuaternion(obs[-4:])
        z = p.getBasePositionAndOrientation(self.robot)[0][2]
        return abs(roll) > 0.6 or abs(pitch) > 0.6 or z < 0.2

    def close(self):
        if p.isConnected(self.cid):
            p.disconnect()
