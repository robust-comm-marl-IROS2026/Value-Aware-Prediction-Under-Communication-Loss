"""
Visualization script for trained MAPPO agents in SimpleSpreadXY environment.
Loads a trained MAPPO model and its perception model, then runs episodes
with visualization and a configurable communication loss rate.
"""

import argparse
import logging
import numpy as np
import os
import sys
import time
import torch as th
import yaml
from types import SimpleNamespace as SN
from os.path import dirname, abspath

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from perception.models import REGISTRY as perc_model_REGISTRY
from envs.obs_processors import REGISTRY as obs_processors_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from perception.trainers.online_trainer import OnlineTrainer
from utils.logging import Logger


class SilentLogger:
    """Minimal logger that suppresses output during visualization."""
    def log_stat(self, *args, **kwargs):
        pass
    def print_recent_stats(self):
        pass
    def setup_file_logging(self, *args, **kwargs):
        pass
    def setup_tb(self, *args):
        pass
    def setup_sacred(self, *args):
        pass
    def close(self):
        pass
    @property
    def console_logger(self):
        return logging.getLogger(__name__)


def build_config(rl_checkpoint_path):
    """Build configuration for MAPPO visualization in SimpleSpreadXY."""
    config = {
        # Algorithm: MAPPO_NS
        'name': 'mappo_ns',
        'learner': 'ppo_learner',
        'mac': 'non_shared_mac',
        'runner': 'visualization',
        'action_selector': 'soft_policies',
        'mask_before_softmax': True,

        # Environment: SimpleSpreadXY
        'env': 'gymma',
        'env_args': {
            'key': 'SimpleSpreadBlind-v0',
            'time_limit': 25,
            'pretrained_wrapper': None,
            'render_mode': 'human',
            'state_last_action': False,
            'seed': 42
        },

        # Agent parameters
        'agent': 'rnn_ns',
        'hidden_dim': 256,
        'obs_agent_id': False,
        'obs_last_action': False,
        'obs_individual_obs': False,
        'agent_output_type': 'pi_logits',
        'use_rnn': True,

        # Critic
        'critic_type': 'cv_critic_ns',

        # Training params (not used in eval but needed for setup)
        'batch_size_run': 1,
        'batch_size': 10,
        'buffer_size': 10,
        'lr': 0.0003,
        'gamma': 0.99,
        'optim_alpha': 0.99,
        'optim_eps': 0.00001,
        'grad_norm_clip': 10,
        'entropy_coef': 0.01,
        'standardise_rewards': True,
        'q_nstep': 5,
        'epochs': 4,
        'eps_clip': 0.2,

        # Logging intervals
        'test_interval': 20_000,
        'log_interval': 20_000,
        'runner_log_interval': 20_000,
        'learner_log_interval': 20_000,
        't_max': 20_050_000,
        'target_update_interval_or_tau': 200,

        # Evaluation
        'test_greedy': True,
        'test_nepisode': 1,
        'use_cuda': False,
        'buffer_cpu_only': True,
        'seed': 42,

        # Paths
        'checkpoint_path': rl_checkpoint_path,
        'load_step': 0,
        'local_results_path': 'results',

        # Perception model
        'perception_args': {
            'perception': True,
            'model_type': 'maro',
            'train_comm_p': 1.0,
            'comm_at_t0': True,
            'append_masks_to_rl_input': False,
            'accumulate_masks': False,
            'hidden_dim': 128,
            'teacher_forcing': False,
            'learning_rate': 0.001,
            'grad_clip': 1.0,
            'batch_size': 32,
            'buffer_size': 5000,
            'checkpoint_path': '',
            'load_step': 0,
            'save_model': True,
            'save_model_interval': 1000000,
            'trainer_log_interval': 10000
        }
    }
    return config


def load_checkpoint(path, loader_fn, label):
    """Load model checkpoint from a directory, selecting the latest timestep if needed."""
    if not os.path.isdir(path):
        print(f"Error: {label} checkpoint path does not exist: {path}")
        return False

    model_files = ['network.th', 'opt.th'] if label == "Perception" else ['agent.th', 'critic.th']
    has_model_files = all(os.path.exists(os.path.join(path, f)) for f in model_files)

    if has_model_files:
        print(f"Loading {label} model from: {path}")
        loader_fn(path)
        return True

    timesteps = [int(name) for name in os.listdir(path)
                 if os.path.isdir(os.path.join(path, name)) and name.isdigit()]

    if not timesteps:
        print(f"Error: No {label} model checkpoints found in {path}")
        return False

    timestep_to_load = max(timesteps)
    model_path = os.path.join(path, str(timestep_to_load))
    print(f"Loading {label} model from: {model_path}")
    loader_fn(model_path)
    return True


def visualize_trained_agent(rl_checkpoint_path, perc_checkpoint_path,
                            comm_loss_rate=0.0, num_episodes=3):
    """
    Visualize a trained MAPPO agent with perception model.

    Args:
        rl_checkpoint_path: Path to RL model checkpoint directory.
        perc_checkpoint_path: Path to perception model checkpoint directory.
        comm_loss_rate: Communication loss probability (0.0 = no loss, 1.0 = complete loss).
        num_episodes: Number of episodes to visualize.
    """
    print(f"{'=' * 80}")
    print("MAPPO VISUALIZATION — SimpleSpreadXY Environment")
    print(f"{'=' * 80}")
    print(f"  RL Model:             {rl_checkpoint_path}")
    print(f"  Perception Model:     {perc_checkpoint_path}")
    print(f"  Communication Loss:   {comm_loss_rate}")
    print(f"  Episodes:             {num_episodes}")
    print(f"{'=' * 80}")

    config = build_config(rl_checkpoint_path)
    config['perception_args']['checkpoint_path'] = perc_checkpoint_path

    args = SN(**config)
    args.device = "cuda" if args.use_cuda and th.cuda.is_available() else "cpu"

    logger = SilentLogger()

    # Initialize runner
    runner = r_REGISTRY[args.runner](args=args, logger=logger)
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]

    print(f"\nEnvironment: {args.n_agents} agents, {args.n_actions} actions, "
          f"state shape {args.state_shape}, obs shape {env_info['obs_shape']}\n")

    # Setup scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": args.n_agents}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])}

    # Setup perception model
    obs_processor = None
    if args.env == 'gymma' and args.env_args["key"] in obs_processors_REGISTRY.keys():
        obs_processor = obs_processors_REGISTRY[args.env_args["key"]]()

    perc_model = perc_model_REGISTRY[args.perception_args["model_type"]](scheme, args, obs_processor)

    # Load perception model
    if perc_model.is_trainable and perc_checkpoint_path:
        perc_trainer = OnlineTrainer(perc_model=perc_model, logger=logger, args=args)
        load_checkpoint(perc_checkpoint_path, perc_trainer.load_models, "Perception")

    # Setup RL scheme
    rl_scheme = scheme.copy()
    if args.perception_args["perception"]:
        rl_scheme["obs"] = {"vshape": perc_model.get_rl_input_dim(), "group": "agents"}

    # Setup controller and learner
    mac = mac_REGISTRY[args.mac](rl_scheme, groups, args)
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess,
                mac=mac, perception_model=perc_model, rl_scheme=rl_scheme)

    learner = le_REGISTRY[args.learner](mac, rl_scheme, logger, args)
    if args.use_cuda:
        learner.cuda()

    # Load RL model
    if not load_checkpoint(rl_checkpoint_path, learner.load_models, "RL"):
        runner.close_env()
        return

    # Run visualization
    print(f"\n{'=' * 80}")
    print(f"STARTING VISUALIZATION (comm_loss={comm_loss_rate})")
    print(f"{'=' * 80}\n")

    save_dir = os.path.join(dirname(abspath(__file__)), '..', 'results', 'env')
    save_steps = [0]

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep + 1}/{num_episodes} ---\n")
        runner.run(test_mode=True, comm_p=comm_loss_rate, save_dir=save_dir,
                   episode_num=ep+1, save_steps=save_steps)

    runner.close_env()
    print(f"\n{'=' * 80}")
    print("VISUALIZATION COMPLETED")
    print(f"{'=' * 80}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize trained MAPPO agents.")
    parser.add_argument("--rl-checkpoint", required=True,
                        help="Path to RL model checkpoint directory.")
    parser.add_argument("--perc-checkpoint", required=True,
                        help="Path to perception model checkpoint directory.")
    parser.add_argument("--comm-loss", type=float, default=0.0,
                        help="Communication loss rate (0.0=none, 1.0=full). Default: 0.0")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Number of episodes to visualize. Default: 3")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualize_trained_agent(
        rl_checkpoint_path=args.rl_checkpoint,
        perc_checkpoint_path=args.perc_checkpoint,
        comm_loss_rate=args.comm_loss,
        num_episodes=args.episodes
    )
