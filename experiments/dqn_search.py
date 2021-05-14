import logging
logging.basicConfig(filename='search.log', level=logging.CRITICAL)


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
import numpy as np

import pandas as pd
import os
from tqdm import trange
import time


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
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.MaxPool2d(4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.critic = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.agent = nn.Sequential(
            nn.Linear(384, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        # x = torch.tensor(x.transpose(3, 1), dtype=torch.float16)
        # x -> (N, C, H, W) with range [-1, 1]
        x = (x.transpose(3, 1).float() / 255. - 0.5) * 2
        x = self.common(x)
        a = self.agent(x)
        v = self.critic(x)
        return a, v


# model = ImageCritic().to("cuda")
# print(f"Number elements: {sum([p.numel() for p in model.parameters()])}")
# exit(0)

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


def train_model(args, config):
    # Define trainer. Apart from env, config/framework and config/model, which are common among trainers.
    # Here is a list of default config keys/values: https://docs.ray.io/en/master/rllib-training.html#common-parameters
    # For DQN specifically there are also additionally these keys: https://docs.ray.io/en/master/rllib-algorithms.html#dqn
    trainer = DQNTrainer(
        env="DuckieTown-MultiMap",
        config=config,
    )

    # Start training from a checkpoint, if available.
    if args.model_path:
        trainer.restore(args.model_path)

    best_mean_reward = -np.inf
    epoch_of_best_mean_reward = 0
    path_of_best_mean_reward = None

    for i in trange(args.epochs, desc="Epochs", leave=False):  # Number of episodes (basically epochs)
        # print(f'----------------------- Starting epoch {i} ----------------------- ')
        # train() trains only a single episode
        result = trainer.train()
        # print(result)

        # Save model so far.
        checkpoint_path = trainer.save()
        # print(f'Epoch {i}, checkpoint saved at: {checkpoint_path}')

        if result["episode_reward_mean"] > best_mean_reward:
            best_mean_reward = result["episode_reward_mean"]
            epoch_of_best_mean_reward = i
            path_of_best_mean_reward = checkpoint_path


        # Cleanup CUDA memory to reduce memory usage.
        torch.cuda.empty_cache()
        # Debug log to monitor memory.
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

    return best_mean_reward, epoch_of_best_mean_reward, path_of_best_mean_reward


def get_parser():
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument('--epochs', default=100, type=int, help='Num of epochs to train for')
    parser.add_argument('--n_searches', default=50, type=int, help='Num of epochs to train for')

    # Evaluation options
    parser.add_argument('--map', default='loop_empty', help='Which map to evaluate on, see https://git.io/J3jES')
    # Looks like, e.g. `DQN_DuckieTown-MultiMap_2021-05-10_20-43-32ndgfiq44/checkpoint_000009/checkpoint-9`
    parser.add_argument('--model_path', default='', help='Location to pre-trained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    # Start ray
    ray.init()
    ModelCatalog.register_custom_model("image-dqn", RLLibDQNCritic)

    # We are using custom model and environment, which need to be registered in ray/rllib
    # Names can be anything.
    register_env("DuckieTown-MultiMap", lambda _: DiscreteWrapper(MultiMapEnv()))

    csv_path = "searches/dqn_results.csv"
    starting_idx = 0
    if os.path.exists(csv_path):
        with open(csv_path, mode="r") as f:
            starting_idx = len(f.readlines())

    for search_idx in trange(args.n_searches, desc="Searches"):
        config = {
            "framework": "torch",
            "model": {
                "custom_model": "image-dqn",
            },
            "learning_starts": 500,
            # "record_env": True,  # Doing this allows us to record images from the DuckieTown Gym! Might be useful for report.
            "train_batch_size": 16,
            # Use a very small buffer to reduce memory usage, default: 50_000.
            "buffer_size": 1000,
            # Dueling off
            "dueling": False,
            # No hidden layers
            "hiddens": [],
            # Don't save experiences.
            # "output": None,
            "compress_observations": True,
            "num_workers": 0,
            "num_gpus": 0.66,
            "rollout_fragment_length": 50,
            "log_level": "ERROR",
        }

        additional_params = {}

        lr = 10 ** np.random.uniform(-6, -3.25)
        additional_params["learning_starts"] = np.random.choice(list(range(100, 1001, 100)))
        additional_params["grad_clip"] = np.random.choice([None, 10, 20, 30, 40, 50, 60, 80])
        additional_params["dueling"] = np.random.choice([True, False])
        additional_params["lr"] = lr
        additional_params["lr_schedule"] = [(0, lr), (args.epochs * config["buffer_size"], lr / 1000.)]

        for k, v in additional_params.items():
            config[k] = v

        try:
            best_mean_reward, best_epoch, best_path = train_model(args, config)

            additional_params["best_mean_reward"] = best_mean_reward
            additional_params["best_epoch"] = best_epoch
            additional_params["best_path"] = best_path

            # No need to save lr_schedule, since it's deterministic on lr
            del additional_params["lr_schedule"]

            params_for_df = {}
            for k, v in additional_params.items():
                params_for_df[k] = [v]

            pd.DataFrame.from_dict(params_for_df).to_csv(
                csv_path,
                mode="a" if os.path.exists(csv_path) else "w",
                header=not os.path.exists(csv_path)
            )
        except RuntimeError:
            # Not enough memory on GPU. Might be bad config, or a CUDA not keeping up. Give it few seconds.
            time.sleep(5)
        finally:
            torch.cuda.empty_cache()
