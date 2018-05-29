import numpy as np
from physics_sim import PhysicsSim
import math

class Task():
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, init_pose=None, init_velocities=None,
        init_angle_velocities=None, runtime=5., target_pos=None):
        """Initialize a Task object.
        Params
        ======
            init_pose: initial position of the quadcopter in (x,y,z) dimensions and the Euler angles
            init_velocities: initial velocity of the quadcopter in (x,y,z) dimensions
            init_angle_velocities: initial radians/second for each of the three Euler angles
            runtime: time limit for each episode
            target_pos: target/goal (x,y,z) position for the agent
        """
        # Simulation
        self.sim = PhysicsSim(init_pose, init_velocities, init_angle_velocities, runtime)
        self.action_repeat = 3

        self.state_size = self.action_repeat * 6
        self.action_low = 0
        self.action_high = 900
        self.action_size = 4

        # Goal
        self.target_pos = target_pos if target_pos is not None else np.array([0., 0., 10.])

        self.get_reward = self.get_reward

    def get_reward(self):
        """Uses current pose of sim to return reward."""
        reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        return reward

    def get_reward_v2(self):
        # reaward staying on target Z coordinate
        reward = (10 - abs(self.sim.pose[2] - self.target_pos[2]))
        # penalty for being far from target
        reward += -.3 * (abs(self.sim.pose[:3] - self.target_pos)).sum()
        # penalty for non 0 velocities
        reward += -.3 * abs(self.sim.v).sum()

        return reward

    def get_reward_v3(self):
        # penalize any velocity
        penalty_1 = 0.3 * abs(self.sim.v).sum()
        # heavy penalty for Z velocity
        penalty_2 = 3*abs(self.sim.v[2])
        # penalty for being off target coordinate in general
        penalty_3 = 0.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()

        total_penalty = np.array([penalty_1, penalty_2, penalty_3]).sum()

        return 10 - total_penalty

    def get_reward_v4(self):
        #reward = 1.-.3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        reward = 0
        reward += 1 if np.average(abs(self.sim.v)) < 0.1 else -1
        #reward += 1 if abs(self.sim.v[2]) < 0.05 else -1
        return np.clip(reward, -1, 1)

    def get_reward_v5(self):
        reward = 55
        # quadcopter should be horizontaly oriented
        punish_euler_angles = np.average(abs(self.sim.pose[3:]))
        # absolute difference between target position and actual should be minival
        punish_being_of_target = .3*(abs(self.sim.pose[:3] - self.target_pos)).sum()
        # ensure absolute axis velocities are minimal
        punish_non_zero_velocities = np.average(abs(self.sim.v))
        # extra punishment of Z velocitiy
        punish_extra_z_velocity = abs(self.sim.v[2])

        punish_non_zero_amgular_velocities = np.average(abs(self.sim.angular_v))

        punishment = np.array([punish_euler_angles,
                  punish_being_of_target,
                  punish_non_zero_velocities,
                  punish_extra_z_velocity,
                  punish_non_zero_amgular_velocities]).sum()

        # normalize to [-1, 1] range for simplicity
        # -1 => action was bad
        # 1 => action was good
        return np.clip(reward-punishment, -1, 1)

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        pose_all = []
        for _ in range(self.action_repeat):
            done = self.sim.next_timestep(rotor_speeds) # update the sim pose and velocities
            reward += self.get_reward()
            pose_all.append(self.sim.pose)
            # punish an early crash
            reward = -1 if done and self.sim.time < self.sim.runtime else reward
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset the sim to start a new episode."""
        self.sim.reset()
        state = np.concatenate([self.sim.pose] * self.action_repeat)

        random_pose = np.copy(self.sim.pose)
        random_pose += np.random.normal(0.0, 0.05, 6)
        self.sim.pose = np.copy(random_pose)

        return state