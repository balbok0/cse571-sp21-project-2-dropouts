import argparse

from gym_duckietown import simulator
from gym_duckietown.envs.multimap_env import MultiMapEnv
from gym_duckietown.wrappers import DiscreteWrapper
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.models import ModelCatalog
import torch

import plotter
from dropouts_project import ImageCritic, RLLibTorchModel


class RLLibDQNCritic(RLLibTorchModel):
    @staticmethod
    def base_model_generator():
        return ImageCritic(3, 1)


def train_model(args):
    # We are using custom model and environment, which need to be registered in ray/rllib
    # Names can be anything.
    register_env("DuckieTown-MultiMap", lambda _: DiscreteWrapper(MultiMapEnv()))

    # Define trainer. Apart from env, config/framework and config/model, which are common among trainers.
    # Here is a list of default config keys/values:
    # https://docs.ray.io/en/master/rllib-training.html#common-parameters
    # For DQN specifically there are also additionally these keys:
    # https://docs.ray.io/en/master/rllib-algorithms.html#dqn
    trainer = DQNTrainer(
        env="DuckieTown-MultiMap",
        config={
            "framework": "torch",
            "model": {
                "custom_model": "image-dqn",
            },
            "learning_starts": 500,
            # Doing this allows us to record images from the DuckieTown Gym! Might be useful for report.
            # "record_env": True,
            "train_batch_size": 16,
            # Use a very small buffer to reduce memory usage, default: 50_000.
            "buffer_size": 1000,
            # Dueling off
            "dueling": False,
            # No hidden layers
            "hiddens": [],
            # Don't save experiences.
            # "output": None,
            # "compress_observations": True,
            "num_workers": 0,
            "num_gpus": 0.5,
            "rollout_fragment_length": 50,
        }
    )

    # Start training from a checkpoint, if available.
    if args.model_path:
        trainer.restore(args.model_path)

    plot = plotter.Plotter('dqn_agent')
    for i in range(args.epochs):  # Number of episodes (basically epochs)
        print(f'----------------------- Starting epoch {i} ----------------------- ')
        # train() trains only a single episode
        result = trainer.train()
        print(result)
        plot.add_results(result)

        # Save model so far.
        checkpoint_path = trainer.save()
        print(f'Epoch {i}, checkpoint saved at: {checkpoint_path}')

        # Cleanup CUDA memory to reduce memory usage.
        torch.cuda.empty_cache()
        # Debug log to monitor memory.
        print(torch.cuda.memory_summary(device=None, abbreviated=False))

    plot.plot('DQN DuckieTown-MultiMap')


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
            # Dueling off
            "dueling": False,
            # No hidden layers
            "hiddens": [],
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
