import argparse

from gym_duckietown import simulator
from gym_duckietown.envs.multimap_env import MultiMapEnv
import ray
from ray.tune.registry import register_env
from ray.rllib.agents.ddpg import DDPGTrainer
from ray.rllib.models import ModelCatalog
import torch

from dropouts_project import (
    RLLibTorchModel,
    DuckieTownGymModel,
    MotionBlurWrapper,
    ResizeWrapper,
    DtRewardWrapper,
    ImgWrapper,
    NormalizeWrapper,
    ActionWrapper,
)
import plotter


class DDPGRLLibModel(RLLibTorchModel):
    @staticmethod
    def base_model_generator():
        return DuckieTownGymModel(256, 256, 1.0)


def train_model(args):
    # Define trainer. Apart from env, config/framework and config/model, which are common among trainers.
    # Here is a list of default config keys/values:
    #    https://docs.ray.io/en/master/rllib-training.html#common-parameters
    # For DDPG specifically there are also additionally these keys:
    #    https://docs.ray.io/en/master/rllib-algorithms.html#ddpg
    trainer = DDPGTrainer(
        env="DuckieTown-MultiMap",
        config={
            "framework": "torch",
            "model": {
                "custom_model": "image-ddpg",
            },
            "learning_starts": 0,
            "train_batch_size": 16,
        }
    )

    # Start training from a checkpoint, if available.
    if args.model_path:
        trainer.restore(args.model_path)

    plot = plotter.Plotter('ddpg_agent')
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

    plot.plot('DDPG DuckieTown-MultiMap')


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
            "num_gpus": args.gpu_use,
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
    # NOTE: We are using DuckietownLF environment because SteeringToWheelVelWrapper does not cooperate with multimap.
    ModelCatalog.register_custom_model("image-ddpg", DDPGRLLibModel)
    register_env(
        "DuckieTown-MultiMap",
        lambda _: DtRewardWrapper(ActionWrapper(NormalizeWrapper(ResizeWrapper(MultiMapEnv())))),
    )

    if args.eval:
        evaluate_model(args)
    else:
        train_model(args)
