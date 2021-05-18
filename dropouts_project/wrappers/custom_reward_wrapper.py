import gym
from gym_duckietown.envs import MultiMapEnv
from gym_duckietown.simulator import REWARD_INVALID_POSE, NotInLane
import numpy as np


# Overrides Invalid Pose Loss. This enables us
SUBTRACT_FOR_INVALID_POSE = 50
SUBTRACT_FOR_INVALID_LANE = 10
DIST_MULTIPLIER = 5.0
PROXIMITY_MULTIPLIER = 2.0
ANGLE_MULTIPLIER = 0.1
SPEED_MULTIPLIER = 0.5
DIST_TRAVELLED_MULTIPLIER = 2.0

# TURN TO True if you want to debug/see logging
DO_PRINTS = True


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(self.__class__, self).__init__(env)
        # Too lazy to do logging
        self.print = print if DO_PRINTS else lambda _: None
        self.total_episode_reward = 0
        self.last_timestamp = np.inf
        if isinstance(self.unwrapped, MultiMapEnv):
            self.start_pos = np.copy(env.env_list[env.cur_env_idx].cur_pos)
        else:
            self.start_pos = np.copy(env.cur_pos)

    def reward(self, reward):
        # Get DuckieTown Environment.
        if isinstance(self.unwrapped, MultiMapEnv):
            env = self.unwrapped.env_list[self.unwrapped.cur_env_idx]
        else:
            env = self.unwrapped

        # Reset Reward if env was .reset()
        if env.timestamp < self.last_timestamp:
            self.total_episode_reward = 0
            self.start_pos = np.copy(env.cur_pos)
        self.last_timestamp = env.timestamp

        self.print(f"Timestamp: {env.timestamp}")

        # Define baseline
        # NOTE(balbok0): We might want to reward longer timestamps, but it might skew reward towards later choices.
        custom_reward = DIST_TRAVELLED_MULTIPLIER * np.linalg.norm(env.cur_pos - self.start_pos)  # env.timestamp

        if reward == REWARD_INVALID_POSE:
            # INVALID POSE LOSS
            # We want to make loss for the whole episode negative, so we subtract the loss of the whole episode.
            custom_reward = -self.total_episode_reward - SUBTRACT_FOR_INVALID_POSE
            self.print("INValid Pose")
        else:
            # Proximity to duckies
            reward -= PROXIMITY_MULTIPLIER * env.proximity_penalty2(env.cur_pos, env.cur_angle)
            try:
                # In Lane
                self.print("Valid Lane")
                lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)

                speed_reward = SPEED_MULTIPLIER * env.speed * lp.dot_dir
                dist_reward = -DIST_MULTIPLIER * abs(lp.dist)
                angle_reward = -ANGLE_MULTIPLIER * abs(lp.angle_rad)

                self.print(f"Travelled Reward: {custom_reward}")
                self.print(f"Speed Reward: {speed_reward}")
                self.print(f"Dist Reward: {dist_reward}")
                self.print(f"Angle Reward: {angle_reward}")

                custom_reward += speed_reward + dist_reward + angle_reward
            except NotInLane:
                # Out of Lane
                self.print("INValid Lane")
                custom_reward -= SUBTRACT_FOR_INVALID_LANE

        self.print(f"Reward: {custom_reward}")
        self.print("")

        # Log reward for a given episode (until next .reset())
        self.total_episode_reward += custom_reward

        return custom_reward
