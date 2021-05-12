import argparse
from typing import List

from gym_duckietown import simulator
from gym_duckietown.envs.multimap_env import MultiMapEnv
from gym_duckietown.wrappers import PyTorchObsWrapper, SteeringToWheelVelWrapper, DiscreteWrapper
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import torch
from torch import nn


class ImageCritic(nn.Module):
    def __init__(self):
        """CNN that does the predictions.
        Data from is as follows:

                           > agent(x) -> a (R^256)
                          /
            x -> common(x)
                         |
                         L> critic(x) -> c (R)

        The reason for sharing common part is because images are large so that many Conv's is quite expensive.
        Also predicting value of a state and predicting next action to take is quite similar, and many papers use this approach.
        """
        super(self.__class__, self).__init__()
        self.common = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.Conv2d(8, 16, 3),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.critic = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.agent = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
        )

    def forward(self, x):
        x = x.transpose(3, 1).float()
        x = self.common(x)
        a = self.agent(x)
        v = self.critic(x)
        return a, v


class RLLibDQNCritic(TorchModelV2, nn.Module):
    """
    This is basically a wraper around ImageCritic that satisfies requirements of rllib.
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.base_model = ImageCritic()
        self._value = None

    def forward(self, input_dict, state, seq_lens):
        # Structure of this forward is due to TorchModelV2.
        # Note that we do not immediately return value, but rather save it for `value_function`
        model_out, self._value = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return self._value


def train_model(args):
    # We are using custom model and environment, which need to be registered in ray/rllib
    # Names can be anything.
    register_env("DuckieTown-MultiMap", lambda _: DiscreteWrapper(MultiMapEnv()))

    # Define trainer. Apart from env, config/framework and config/model, which are common among trainers.
    # Here is a list of default config keys/values: https://docs.ray.io/en/master/rllib-training.html#common-parameters
    # For DQN specifically there are also additionally these keys: https://docs.ray.io/en/master/rllib-algorithms.html#dqn
    trainer = DQNTrainer(
        env="DuckieTown-MultiMap",
        config={
            "framework": "torch",
            "model": {
                "custom_model": "image-dqn",
            },
            "learning_starts": 1000,
            # "record_env": True,  # Doing this allows us to record images from the DuckieTown Gym! Might be useful for report.
            "train_batch_size": 16,
            # Use a very small buffer to reduce memory usage, default: 50_000.
            "buffer_size": 2000,
            # Don't save experiences.
            "output": None,
            "compress_observations": True,
            "num_workers": 0,
        }
    )

    for i in range(args.epochs):  # Number of episodes (basically epochs)
        print(f'----------------------- Starting epoch {i} ----------------------- ')
        # train() trains only a single episode
        result = trainer.train()
        print(result)

        # Save model so far.
        checkpoint_path = trainer.save()
        print(f'Epoch {i}, checkpoint saved at: {checkpoint_path}')

        # Cleanup CUDA memory to reduce memory usage.
        torch.cuda.empty_cache()
        # Debug log to monitor memory.
        print(torch.cuda.memory_summary(device=None, abbreviated=False))


def evaluate_model(args):
    if args.model_path == '':
        print('Cannot evaluate model, no --model_path set')
        exit(1)

    def get_env():
        # Simulator env uses a single map, so better for evaluation/testing.
        # DiscreteWrapper just converts wheel velocities to high level discrete actions.
        return DiscreteWrapper(simulator.Simulator(
            map_name=args.map,
            max_steps=2000,
        ))

    # Rather than reuse the env, another one is created later because I can't
    # figure out how to provide register_env with an object, th
    register_env('DuckieTown-Simulator', lambda _: get_env())
    trainer = DQNTrainer(
        env="DuckieTown-Simulator",
        config={
            "framework": "torch",
            "model": {
                "custom_model": "image-dqn",
            },
        },
    )
    trainer.restore(args.model_path)

    sim_env = get_env()

    # Standard OpenAI Gym reset/action/step/render loop.
    # This matches how the `enjoy_reinforcement.py` script works, see: https://git.io/J3js2
    done = False
    observation = sim_env.reset()
    episode_reward = 0
    while not done:
        action = trainer.compute_action(observation)
        observation, reward, done, _ = sim_env.step(action)
        episode_reward += reward
        sim_env.render()

    print(f'Episode complete, total reward: {episode_reward}')


def get_parser():
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument('--epochs', default=10, type=int, help='Num of epochs to train for')

    # Evaluation options
    parser.add_argument('--eval', action='store_true', default=False, help='Evaluate model instead of train')
    parser.add_argument('--map', default='loop_empty', help='Which map to evaluate on, see https://git.io/J3jES')
    # Looks like, e.g. `DQN_DuckieTown-MultiMap_2021-05-10_20-43-32ndgfiq44/checkpoint_000009/checkpoint-9`
    parser.add_argument('--model_path', default='', help='Location to pre-trained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    # Start ray
    ray.init()
    ModelCatalog.register_custom_model("image-dqn", RLLibDQNCritic)

    if args.eval:
        evaluate_model(args)
    else:
        train_model(args)
