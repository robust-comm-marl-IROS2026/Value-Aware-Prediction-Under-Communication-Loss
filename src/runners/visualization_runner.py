import time
import numpy as np
import os
from functools import partial
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving

from envs import REGISTRY as env_REGISTRY
from components.episode_buffer import EpisodeBatch


class VisualizationRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run
        assert self.batch_size == 1

        self.env = env_REGISTRY[self.args.env](**self.args.env_args)
        self.episode_limit = self.env.episode_limit
        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.test_returns_comm_p = {}
        self.train_stats = {}
        self.test_stats = {}

        # Log the first run
        self.log_train_stats_t = -1000000

    def setup(self, scheme, groups, preprocess, mac, perception_model=None, rl_scheme=None):
        # Batch to store in replay buffer.
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        # Batch containing data for action selection.
        self.new_action_selection_batch = partial(EpisodeBatch, rl_scheme, groups, self.batch_size,
                            self.episode_limit + 1, preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.perc_model = perception_model

    def get_env_info(self):
        return self.env.get_env_info()

    def save_replay(self):
        self.env.save_replay()

    def close_env(self):
        self.env.close()

    def reset(self):
        self.batch = self.new_batch()
        self.action_selection_batch = self.new_action_selection_batch()
        self.env.reset()
        self.t = 0

    def run(self, test_mode=False, comm_p=None, save_dir=None, episode_num=0, save_steps=None):
        """
        Run an episode with visualization.
        
        Args:
            test_mode: Whether in test mode
            comm_p: Communication loss probability
            save_dir: Directory to save visualizations
            episode_num: Episode number for filename
            save_steps: int, list of ints, or None. Which timesteps to save as PDF.
                       e.g., 0 for initial state, 2 for second step, [0, 2, 5] for multiple steps
        """
        self.reset()

        terminated = False
        episode_return = 0
        self.mac.init_hidden(batch_size=self.batch_size)
        if self.perc_model:
            self.perc_model.init_perception_model(batch_size=self.batch_size)

        # Convert save_steps to a list for easier checking
        if save_steps is None:
            steps_to_save = []
        elif isinstance(save_steps, int):
            steps_to_save = [save_steps]
        else:
            steps_to_save = list(save_steps)

        # Save state visualization if current step is in steps_to_save
        if save_dir is not None and self.t in steps_to_save:
            os.makedirs(save_dir, exist_ok=True)
            
            try:
                # Access the underlying environment
                env = self.env
                while hasattr(env, '_env'):
                    env = env._env
                
                # Get RGB array by rendering with mode='rgb_array'
                rgb_array = env.render(mode='rgb_array')
                
                if rgb_array is not None:
                    # Save as PDF
                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.imshow(rgb_array)
                    ax.axis('off')
                    plt.tight_layout()
                    
                    # Create filename with episode number, timestep, and comm_p
                    comm_p_str = f"comm_p_{comm_p:.2f}" if comm_p is not None else "comm_p_default"
                    filename = f"state_ep{episode_num}_t{self.t}_{comm_p_str}.pdf"
                    filepath = os.path.join(save_dir, filename)
                    
                    plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=150)
                    plt.close(fig)
                    print(f"\n[SAVED] State at t={self.t} saved to: {filepath}")
                else:
                    print("\n[WARNING] render() returned None")
            except Exception as e:
                print(f"\n[WARNING] Could not save initial state: {e}")

        while not terminated:
            print(f'\n{"="*70}')
            print(f'STEP t={self.t}')
            print("="*70)

            pre_transition_data = {
                "state": [self.env.get_state()],
                "avail_actions": [self.env.get_avail_actions()],
                "obs": [self.env.get_obs()]
            }

            # Print raw environment observations
            raw_obs = self.env.get_obs()
            print("\n[RAW ENV OBSERVATIONS]")
            
            # Try to access world through potential wrappers
            env = self.env
            while hasattr(env, 'env'):
                env = env.env
            
            if hasattr(env, 'world'):
                agent_0_pos = env.world.agents[0].state.p_pos
                agent_1_pos = env.world.agents[1].state.p_pos
                lm0_pos = env.world.landmarks[0].state.p_pos
                lm1_pos = env.world.landmarks[1].state.p_pos
                
                print(f"  Agent 0 pos: [{agent_0_pos[0]:+.3f}, {agent_0_pos[1]:+.3f}]")
                print(f"  Agent 1 pos: [{agent_1_pos[0]:+.3f}, {agent_1_pos[1]:+.3f}]")
                print(f"  Landmark 0:  [{lm0_pos[0]:+.3f}, {lm0_pos[1]:+.3f}]")
                print(f"  Landmark 1:  [{lm1_pos[0]:+.3f}, {lm1_pos[1]:+.3f}]")
                print(f"\n  Agent 0 obs (x0, x1, vx0, vx1, lm0_x, lm0_y, lm1_x, lm1_y):")
                print(f"    {raw_obs[0]}")
                print(f"  Agent 1 obs (y0, y1, vy0, vy1, lm0_x, lm0_y, lm1_x, lm1_y):")
                print(f"    {raw_obs[1]}")
            else:
                # Fallback: just print observations without world positions
                print(f"  Agent 0 obs (x0, x1, vx0, vx1, lm0_x, lm0_y, lm1_x, lm1_y):")
                print(f"    {raw_obs[0]}")
                print(f"  Agent 1 obs (y0, y1, vy0, vy1, lm0_x, lm0_y, lm1_x, lm1_y):")
                print(f"    {raw_obs[1]}")

            self.batch.update(pre_transition_data, ts=self.t)
            if self.perc_model:
                perc_model_out = self.perc_model.encode(self.batch, t=self.t,
                            test_mode=test_mode, comm_p=comm_p) # [1,num_agents,latent_dim]
                
                # Print MARO-processed policy inputs
                print("\n[MARO POLICY INPUTS] (after perception model processing)")
                policy_inputs = perc_model_out.numpy()[0]
                print(f"  Agent 0 policy input (x0, x1, vx0, vx1, y0, y1, vy0, vy1, lm0_x, lm0_y, lm1_x, lm1_y):")
                print(f"    {policy_inputs[0]}")
                print(f"  Agent 1 policy input (x0, x1, vx0, vx1, y0, y1, vy0, vy1, lm0_x, lm0_y, lm1_x, lm1_y):")
                print(f"    {policy_inputs[1]}")
                
                pre_transition_data["obs"] = [[s for s in policy_inputs]]
            self.action_selection_batch.update(pre_transition_data, ts=self.t)

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch of size 1
            actions = self.mac.select_actions(self.action_selection_batch, t_ep=self.t,
                                            t_env=self.t_env, test_mode=test_mode)

            reward, terminated, env_info = self.env.step(actions[0])
            episode_return += reward

            post_transition_data = {
                "actions": actions,
                "reward": [(reward,)],
                "terminated": [(terminated != env_info.get("episode_limit", False),)],
            }

            self.batch.update(post_transition_data, ts=self.t)
            self.action_selection_batch.update(post_transition_data, ts=self.t)

            self.t += 1

            # Save state visualization if current step is in steps_to_save
            if save_dir is not None and self.t in steps_to_save:
                try:
                    # Access the underlying environment
                    env = self.env
                    while hasattr(env, '_env'):
                        env = env._env
                    
                    # Get RGB array by rendering with mode='rgb_array'
                    rgb_array = env.render(mode='rgb_array')
                    
                    if rgb_array is not None:
                        # Save as PDF
                        fig, ax = plt.subplots(figsize=(8, 8))
                        ax.imshow(rgb_array)
                        ax.axis('off')
                        plt.tight_layout()
                        
                        # Create filename with episode number, timestep, and comm_p
                        comm_p_str = f"comm_p_{comm_p:.2f}" if comm_p is not None else "comm_p_default"
                        filename = f"state_ep{episode_num}_t{self.t}_{comm_p_str}.pdf"
                        filepath = os.path.join(save_dir, filename)
                        
                        plt.savefig(filepath, format='pdf', bbox_inches='tight', dpi=150)
                        plt.close(fig)
                        print(f"\n[SAVED] State at t={self.t} saved to: {filepath}")
                    else:
                        print(f"\n[WARNING] render() returned None at t={self.t}")
                except Exception as e:
                    print(f"\n[WARNING] Could not save state at t={self.t}: {e}")

            self.env.render()
            time.sleep(0.2)

        print('-'*20)
        print('Episode finished.')
        print("Episode return:", episode_return)
        input("Press any key to skip to the next episode.")

        last_data = {
            "state": [self.env.get_state()],
            "avail_actions": [self.env.get_avail_actions()],
            "obs": [self.env.get_obs()]
        }
        self.batch.update(last_data, ts=self.t)
        if self.perc_model:
            perc_model_out = self.perc_model.encode(self.batch, t=self.t,
                        test_mode=test_mode, comm_p=comm_p) # [1,num_agents,latent_dim]
            last_data["obs"] = [[s for s in perc_model_out.cpu().numpy()[0]]]
        self.action_selection_batch.update(last_data, ts=self.t)

        # Select actions in the last stored state
        actions = self.mac.select_actions(self.action_selection_batch, t_ep=self.t,
                                            t_env=self.t_env, test_mode=test_mode)
        self.batch.update({"actions": actions}, ts=self.t)
        self.action_selection_batch.update({"actions": actions}, ts=self.t)

        return self.batch

