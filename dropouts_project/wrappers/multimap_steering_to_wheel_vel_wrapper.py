import gym
import numpy as np
from gym_duckietown.envs.multimap_env import MultiMapEnv


class MultiMapSteeringToWheelVelWrapper(gym.ActionWrapper):
    """
    NOTE: Almost the exact copy of gym_duckietown.wrappers.SteeringToWheelVelWrapper

    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self, env, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0):
        gym.ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def action(self, action):
        vel, angle = action

        # Distance between the wheels -- NOTE: THIS IS THE LINE WHERE THERE IS A CHANGE
        if isinstance(self.unwrapped, MultiMapEnv):
            env = self.unwrapped.env_list[self.unwrapped.cur_env_idx]
        else:
            env = self.unwrapped
        
        baseline = env.wheel_dist

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

    def reverse_action(self, action):
        raise NotImplementedError()
