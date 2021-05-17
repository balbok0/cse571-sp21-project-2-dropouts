import gym
from gym_duckietown.envs import MultiMapEnv
from gym_duckietown.simulator import REWARD_INVALID_POSE, NotInLane


# Overrides Invalid Pose Loss. This enables us
SUBTRACT_FOR_INVALID_POSE = 15
SUBTRACT_FOR_INVALID_LANE = 10
DIST_MULTIPLIER = 5.0
PROXIMITY_MULTIPLIER = 2.0
ANGLE_MULTIPLIER = 2.0
SPEED_MULTIPLIER = 0.5

DO_PRINTS = False


class CustomRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(self.__class__, self).__init__(env)
        # Too lazy to do logging
        self.print = print if DO_PRINTS else lambda _: None
        self.total_episode_reward = 0

    def reward(self, reward):
        if isinstance(self.unwrapped, MultiMapEnv):
            env = self.unwrapped.env_list[self.unwrapped.cur_env_idx]
        else:
            env = self.unwrapped

        if env.timestamp == 0:
            self.total_episode_reward += 0

        custom_reward = 0  # env.timestamp

        if reward == REWARD_INVALID_POSE:
            # INVALID POSE LOSS
            custom_reward = -self.total_episode_reward - SUBTRACT_FOR_INVALID_POSE
            self.print("INValid Pose")
        else:
            reward -= PROXIMITY_MULTIPLIER * env.proximity_penalty2(env.cur_pos, env.cur_angle)
            try:
                lp = env.get_lane_pos2(env.cur_pos, env.cur_angle)

                speed_reward = SPEED_MULTIPLIER * env.speed * lp.dot_dir
                dist_reward = -DIST_MULTIPLIER * abs(lp.dist)
                angle_reward = -ANGLE_MULTIPLIER * abs(lp.angle_rad)

                custom_reward += speed_reward + dist_reward + angle_reward
                self.print("Valid Lane")
            except NotInLane:
                self.print("INValid Lane")
                custom_reward -= SUBTRACT_FOR_INVALID_LANE

        self.print(f"Reward: {custom_reward}")
        self.print("")

        self.total_episode_reward += custom_reward

        return custom_reward