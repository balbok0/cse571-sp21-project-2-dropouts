import argparse
from gym_duckietown import simulator
from gym_duckietown.envs.multimap_env import MultiMapEnv
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models import ModelCatalog
import torch
from tqdm import trange
import time
import os
import numpy as np
from dropouts_project import ImageCritic, RLLibTorchModel, MultiMapSteeringToWheelVelWrapper
import pandas as pd


class DDPGRLLibModel(RLLibTorchModel):
    @staticmethod
    def base_model_generator():
        return ImageCritic(256, 256)


def train_model(args, config):

    # Define trainer. Apart from env, config/framework and config/model, which are common among trainers.
    # Here is a list of default config keys/values:
    #    https://docs.ray.io/en/master/rllib-training.html#common-parameters
    # For DDPG specifically there are also additionally these keys:
    #    https://docs.ray.io/en/master/rllib-algorithms.html#ddpg
    trainer = DDPGTrainer(
        env="DuckieTown-MultiMap",
        config=config,
    )

    # Start training from a checkpoint, if available.
    if args.model_path:
        trainer.restore(args.model_path)

    # TODO(balbok0): Start values from checkpoint, if available.
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


def evaluate_model(args):
    if args.model_path == '':
        print('Cannot evaluate model, no --model_path set')
        exit(1)

    def get_env():
        # Simulator env uses a single map, so better for evaluation/testing.
        # return SteeringToWheelVelWrapper(DuckietownLF(
        # ))
        return MultiMapSteeringToWheelVelWrapper(simulator.Simulator(
            map_name=args.map,
            max_steps=2000,
        ))

    # Rather than reuse the env, another one is created later because I can't
    # figure out how to provide register_env with an object, th
    register_env('DuckieTown-Simulator', lambda _: get_env())
    trainer = DDPGTrainer(
        env="DuckieTown-Simulator",
        config={
            "framework": "torch",
            "model": {
                "custom_model": "image-ddpg",
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
    parser.add_argument('--n_searches', default=50, type=int, help='Num of epochs to train for')
    # Looks like, e.g. `DQN_DuckieTown-MultiMap_2021-05-10_20-43-32ndgfiq44/checkpoint_000009/checkpoint-9`
    parser.add_argument('--model_path', default='', help='Location to pre-trained model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()

    # Start ray
    ray.init()
    # NOTE: We are using DuckietownLF environment because SteeringToWheelVelWrapper does not cooperate with multimap.
    ModelCatalog.register_custom_model(
        "image-ddpg", DDPGRLLibModel,
    )

    register_env("DuckieTown-MultiMap", lambda _: MultiMapSteeringToWheelVelWrapper(MultiMapEnv()))

    csv_path = "searches/ddpg_results.csv"
    starting_idx = 0
    if os.path.exists(csv_path):
        with open(csv_path, mode="r") as f:
            starting_idx = len(f.readlines())

    for search_idx in trange(args.n_searches, desc="Searches"):
        config = {
            "framework": "torch",
            "model": {
                "custom_model": "image-ddpg",
            },
            # "use_state_preprocessor": True,
            "learning_starts": 0,
            # "record_env": True,  # Doing this allows us to record images from the DuckieTown Gym! Might be useful for report.  # noqa: E%01
            "train_batch_size": 16,
            # Use a very small buffer to reduce memory usage, default: 50_000.
            "buffer_size": 1000,
            # No hidden layers
            "actor_hiddens": [],
            "critic_hiddens": [],

            "compress_observations": True,
            "num_workers": 0,
            "num_gpus": 0.66,
            "log_level": "ERROR",
        }

        additional_params = {}

        lr = 10 ** np.random.uniform(-5, -2.5)
        additional_params["learning_starts"] = np.random.choice(list(range(100, 1001, 100)))
        additional_params["grad_clip"] = np.random.choice([None, 10, 20, 40, 50, 60], p=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
        additional_params["lr"] = lr

        for k, v in additional_params.items():
            config[k] = v

        try:
            best_mean_reward, best_epoch, best_path = train_model(args, config)

            additional_params["best_mean_reward"] = best_mean_reward
            additional_params["best_epoch"] = best_epoch
            additional_params["best_path"] = best_path

            # No need to save lr_schedule, since it's deterministic on lr
            additional_params.pop("lr_schedule", None)

            params_for_df = {}
            for k, v in additional_params.items():
                params_for_df[k] = [v]

            pd.DataFrame.from_dict(params_for_df).to_csv(
                csv_path,
                mode="a" if os.path.exists(csv_path) else "w",
                header=not os.path.exists(csv_path)
            )
        except RuntimeError as e:
            if str(e).startswith("RuntimeError: CUDA out of memory."):
                # Not enough memory on GPU. Might be bad config, or a CUDA not keeping up. Give it a minute.
                time.sleep(20)
            else:
                raise e
        finally:
            torch.cuda.empty_cache()
