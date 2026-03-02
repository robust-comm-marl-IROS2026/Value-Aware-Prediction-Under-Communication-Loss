import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
import torch as th
from torch.optim import Adam
import torch.nn.functional as F
import time


class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())
        self.last_target_update_episode = 0

        self.mixer = None
        if args.mixer is not None:
            if args.mixer == "vdn":
                self.mixer = VDNMixer()
            elif args.mixer == "qmix":
                self.mixer = QMixer(args)
            else:
                raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        self.optimiser = Adam(params=self.params, lr=args.lr)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)

        self.training_steps = 0
        self.last_target_update_step = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        # Calculate estimated Q-Values
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim

        # Calculate the Q-Values necessary for the target
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # We don't need the first timesteps Q-Value estimate for calculating targets
        target_mac_out = th.stack(target_mac_out[1:], dim=1)  # Concat across time

        # Mask out unavailable actions
        target_mac_out[avail_actions[:, 1:] == 0] = -9999999

        # Max over target Q-Values
        if self.args.double_q:
            # Get actions that maximise live Q (for double q-learning)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach[:, 1:].max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_qvals = target_mac_out.max(dim=3)[0]


        #FOR NOVEL PREDICTION SIGNAL
        individual_targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals
        # Mix
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"][:, 1:])

        # Calculate 1-step Q-Learning targets
        targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals

        # Td-error
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error)

        # 0-out the targets that came from padded data
        masked_td_error = td_error * mask

        # Normal L2 loss, take mean over actual data
        loss = (masked_td_error ** 2).sum() / mask.sum()

        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (self.training_steps - self.last_target_update_step) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_step = self.training_steps
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
        
        # =================================================================================
        # [NEW] VIRTUAL ADVANTAGE CALCULATION (N-Step Reality + Normalization)
        # =================================================================================
        
        # 1. Get Expectation (V_soft) for ALL steps (0 to 25 + 1 bootstrap)
        # We need the full trajectory (26 steps) to bootstrap the N-step targets correctly.
        mac_out_full = mac_out.detach() # Shape: [Batch, 26, Agents, Actions]
        tau = 0.1
        pi_soft_full = F.softmax(mac_out_full / tau, dim=-1)
        v_soft_full = (pi_soft_full * mac_out_full).sum(dim=-1) # Shape: [Batch, 26, Agents]

        # 2. Prepare Data for N-Step Calculation
        # We need rewards/masks aligned to [Batch, 25, Agents]
        rewards = batch["reward"][:, :-1] # [32, 25, 1]
        if rewards.shape[-1] == 1:
             rewards = rewards.repeat(1, 1, self.args.n_agents) # Expand to [32, 25, 4]
             
        terminated = batch["terminated"][:, :-1].float() # [32, 25, 1]
        mask = batch["filled"][:, :-1].float() # [32, 25, 1]
        
        # Combined mask: Stop if episode ends OR if data is padded
        # We expand to match agents so we can multiply directly
        active_mask = (mask * (1 - terminated)).repeat(1, 1, self.args.n_agents)

        # 3. Calculate N-Step Targets (Using the Helper Function)
        # This returns shape [32, 25, 4]
        # We use a horizon of 5 steps to capture real rewards
        n_step_targets = self.calculate_n_step_targets(
            rewards, 
            active_mask, 
            v_soft_full, 
            n_steps=5 
        )

        # 4. Calculate Advantage (Reality - Expectation)
        # We compare the N-step target against the expectation at the current step
        v_soft_current = v_soft_full[:, :-1, :] # Slice back to [32, 25, 4]
        a_n_step = n_step_targets.detach() - v_soft_current

        # 5. [CRITICAL] Normalize & Clip
        # This fixes the "Scale Trap" by forcing the signal into a standard range.
        mean = a_n_step.mean()
        std = a_n_step.std()
        a_norm = (a_n_step - mean) / (std + 1e-8)
        
        # Clip to range [-5, 5] to prevent outliers
        a_final = th.clamp(a_norm, min=-5.0, max=5.0)

        return a_final.detach()

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path, save_mongo=False):
        self.mac.save_models(path, self.logger, save_mongo)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

        if save_mongo:
            if self.mixer is not None:
                self.logger.log_model(filepath= "{}/mixer.th".format(path), name= "mixer.th")
            self.logger.log_model(filepath="{}/opt.th".format(path), name="opt.th")

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
    



    def calculate_n_step_targets(self, rewards, mask, v_soft_full, n_steps=3):
        """
        Calculates N-step returns using the full value trajectory.
        
        Args:
            rewards: [Batch, 25, Agents] (Actual rewards 0..24)
            mask: [Batch, 25, Agents] (1 if active, 0 if terminated/padded)
            v_soft_full: [Batch, 26, Agents] (Value estimates 0..25)
            n_steps: int (Lookahead horizon)
        """
        batch_size, T, n_agents = rewards.shape
        n_step_targets = th.zeros_like(rewards) # Shape [Batch, 25, Agents]

        # Expand mask if needed to match v_soft shape for convenience
        # But we primarily rely on the loop logic
        
        for t in range(T):
            t_return = th.zeros([batch_size, n_agents], device=rewards.device)
            
            # The Loop: Gather rewards for N steps
            for step in range(n_steps):
                current_idx = t + step
                
                # Discount: gamma^0, gamma^1, ...
                discount = self.args.gamma ** step
                
                if current_idx >= T:
                    # We went past the known rewards (t=25+).
                    # We MUST bootstrap using the Value of the last known step.
                    # We use v_soft_full at the index corresponding to the limit.
                    # v_soft_full has T+1 steps (indices 0..25).
                    
                    # If we are at index 25, that's the bootstrap step.
                    bootstrap_value = v_soft_full[:, T, :] # Get V(s_25)
                    t_return += discount * bootstrap_value
                    break
                
                # Check termination
                # If terminated at this step, we stop accumulating future rewards/values
                term_mask = mask[:, current_idx, :]
                
                if step == n_steps - 1:
                    # Final step of the window: Use Value (Bootstrap)
                    # We use v_soft_full at t+n
                    # Note: v_soft_full has data at index t+step if t+step <= 25
                    t_val = v_soft_full[:, current_idx + 1, :] # V(s_{t+n})
                    t_return += discount * t_val * term_mask
                else:
                    # Intermediate step: Use Real Reward
                    t_reward = rewards[:, current_idx, :]
                    t_return += discount * t_reward * term_mask
                    
            n_step_targets[:, t, :] = t_return
            
        return n_step_targets
