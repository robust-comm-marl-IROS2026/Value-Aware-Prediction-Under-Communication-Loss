import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

from perception.models.sampling_schemes import sampling_registry


class MARONetwork(nn.Module):

    def __init__(self, n_agents, input_dim, output_obs_dim, hidden_dim, obs_processor=None):
        super(MARONetwork, self).__init__()

        # Args.
        self.n_agents = n_agents
        self.input_dim = input_dim
        self.output_obs_dim = output_obs_dim
        self.hidden_dim = hidden_dim
        self.obs_processor = obs_processor

        # Layers.
        self.linear_1 = nn.Linear(self.input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.linear_obs = nn.Linear(hidden_dim, 2 * self.output_obs_dim * n_agents) # mus and sigmas.

    def encode(self, latents, hidden):
        # (single-step).
        batchsize = latents.shape[0]
        in_latents = latents.unsqueeze(1) 

        outs = F.relu(self.linear_1(in_latents))
        outs, hidden = self.lstm(outs, hidden)
        outs_obs = self.linear_obs(outs)

        stride = self.output_obs_dim * self.n_agents
        obs_mus = outs_obs[:, :, :stride]
        obs_mus = obs_mus.view(batchsize, self.n_agents, self.output_obs_dim) 
        obs_sigmas = outs_obs[:, :, stride:2 * stride]
        obs_sigmas = obs_sigmas.view(batchsize, self.n_agents, self.output_obs_dim)
        obs_sigmas = th.exp(obs_sigmas) 

        return obs_mus, obs_sigmas, hidden

    def training_step(self, data, mask, train_params):

        # data shape: [bs, n_timesteps, n_agents, obs_dim]
        loss_info = {}

        if self.obs_processor:
            unique_obs, common_obs = self.obs_processor.split_obs(data)
            unique_obs = unique_obs.reshape(unique_obs.shape[0],unique_obs.shape[1], -1) 
            net_input = th.cat([unique_obs, common_obs[:,:,0,:]], dim=-1) 
        else:
            net_input = data.view(data.shape[0], data.shape[1], -1) 

        # Forward Pass.
        pred_mus, pred_sigmas = self.forward(net_input) 

        # Model Loss.
        x_next = th.roll(data.detach(), -1,  dims=1) # Roll. 

        if self.obs_processor:
            unique_obs, _ = self.obs_processor.split_obs(data) 
            unique_obs_next, _ = self.obs_processor.split_obs(x_next) 
            deltas = unique_obs_next - unique_obs
        else:
            deltas = x_next - data # Compute deltas.
        
        deltas = deltas[:, :-1, :, :].clone().detach() # Drop last element

        # === EXTRACT WEIGHTS ===
        weights = train_params.get("weights", None)
        # =======================

        obs_loss = self.training_loss(x_next=deltas,
                                      pred_mus=pred_mus,
                                      pred_sigmas=pred_sigmas,
                                      mask=mask,
                                      weights=weights, 
                                      reduce=True)

        # Compute total_loss.
        loss_info['predictor_obs_loss'] = th.mean(obs_loss).cpu().item()

        return obs_loss, loss_info

    def forward(self, x):
        # Only for training purposes (multi-steps).
        batchsize, seq_len = x.shape[0],x.shape[1]

        outs = F.relu(self.linear_1(x.clone().detach()))
        outs, hidden = self.lstm(outs)
        outs_obs = self.linear_obs(outs)

        stride = self.output_obs_dim * self.n_agents

        pred_mus = outs_obs[:, :, :stride]
        pred_mus = pred_mus.view(batchsize, seq_len, self.n_agents, self.output_obs_dim)
        pred_sigmas = outs_obs[:, :, stride:2 * stride]
        pred_sigmas = pred_sigmas.view(batchsize, seq_len, self.n_agents, self.output_obs_dim)
        pred_sigmas = th.exp(pred_sigmas)

        # Drop last element
        pred_mus    = pred_mus[:, :-1, :, :] 
        pred_sigmas = pred_sigmas[:, :-1, :, :] 

        return pred_mus, pred_sigmas

    def training_loss(self, x_next, pred_mus, pred_sigmas, mask, weights=None, reduce=True):
        """ Computes loss with optional Importance Weighting. """
        
        # x_next: [bs, t, agents, obs_dim]
        normal_dist = Normal(pred_mus, pred_sigmas)
        g_log_probs = normal_dist.log_prob(x_next) 
        g_log_probs = g_log_probs.sum(-1) # Sum over feature dim -> [bs, t, agents]
        
        # === CRITICAL FIX: Support reduce=False for Logging ===
        if not reduce:
            # Return raw Negative Log Likelihood per agent/timestep
            # This allows the logger to inspect "Error on Good Moves" vs "Error on Bad Moves"
            return -g_log_probs
        # ======================================================

        prob_mask = mask.expand_as(g_log_probs).detach()
        
        # === APPLY WEIGHTS ===
        if weights is not None:
            # Ensure proper broadcasting/device match
            if weights.device != prob_mask.device:
                weights = weights.to(prob_mask.device)
                
            active_weights = prob_mask * weights
            
            # Weighted Average: Sum(Weighted_Errors) / Sum(Weights)
            # We add 1e-8 to denominator to prevent division by zero
            # Note: g_log_probs is negative (log likelihood), so we put a minus sign to minimize Loss
            obs_loss = - (active_weights * g_log_probs).sum() / (active_weights.sum() + 1e-8)
        else:
            # Standard Average
            obs_loss = - (prob_mask * g_log_probs).sum() / (prob_mask.sum() + 1e-8)
        # =====================

        return obs_loss


class MARONetworkTeacherForcing(MARONetwork):

    def __init__(self, n_agents, input_dim, output_obs_dim, hidden_dim, obs_processor=None, train_comm_p=None):
        super(MARONetworkTeacherForcing, self).__init__(n_agents, input_dim, output_obs_dim, hidden_dim, obs_processor)
        self.train_comm_p = train_comm_p

    def training_step(self, data, mask, train_params):

        bs, n_timesteps, n_agents, _ = data.shape

        if isinstance(self.train_comm_p, float):
            comm_p = self.train_comm_p
        elif isinstance(self.train_comm_p, str):
            comm_p = sampling_registry[self.train_comm_p]()
        else:
            raise ValueError("Incorrect sampling scheme selected:" + str(self.train_comm_p))

        loss_info = {}

        hidden_states = (th.zeros((1, bs, self.hidden_dim)), th.zeros((1, bs, self.hidden_dim)))
        
        if data.is_cuda:
            hidden_states = (hidden_states[0].cuda(), hidden_states[1].cuda())

        pred_mus, pred_sigmas = [], []
        # First timestep (always communicate).
        if self.obs_processor:
            unique_obs, common_obs = self.obs_processor.split_obs(data[:,0,:,:])
            unique_obs = unique_obs.reshape(bs, -1) 
            net_input = th.cat([unique_obs, common_obs[:,0,:]], dim=-1) 
        else:
            net_input = data[:,0,:,:].view(bs, -1) 
        
        pred_delta_mus, pred_delta_sigmas, hidden_states = self.encode(net_input, hidden=hidden_states) 
        pred_mus.append(pred_delta_mus)
        pred_sigmas.append(pred_delta_sigmas)

        if self.obs_processor:
            unique_observations, _ = self.obs_processor.split_obs(data[:,0,:,:]) 
            last_pred_obs = unique_observations + pred_delta_mus 
        else:
            last_pred_obs = data[:,0,:,:] + pred_delta_mus 
        last_pred_obs = last_pred_obs.detach()

        # For the other timesteps.
        for t in range(1, n_timesteps-1):

            # Generate mask given `comm_p` probability.
            comm_mask = th.rand(bs,n_agents) 
            comm_mask = (comm_mask >= (1.0 - comm_p)) 
            comm_mask = comm_mask.unsqueeze(2) 

            if data.is_cuda:
                comm_mask = comm_mask.cuda()

            if self.obs_processor:
                unique_obs, common_obs = self.obs_processor.split_obs(data[:,t,:,:])
                comm_mask = th.repeat_interleave(comm_mask, unique_obs.shape[-1], axis=-1) 
                mixed_unique_obs = th.where(comm_mask, unique_obs, last_pred_obs) 
                mixed_unique_obs = mixed_unique_obs.reshape(mixed_unique_obs.shape[0], -1) 
                net_input = th.cat([mixed_unique_obs, common_obs[:,0,:]], dim=-1) 
            else:
                comm_mask = th.repeat_interleave(comm_mask, data.shape[-1], axis=-1) 
                mixed_obs = th.where(comm_mask, data[:,t,:,:], last_pred_obs) 
                net_input = mixed_obs.view(bs, -1) 

            pred_delta_mus, pred_delta_sigmas, hidden_states = self.encode(net_input, hidden=hidden_states) 

            if self.obs_processor:
                unique_observations, _ = self.obs_processor.split_obs(data[:,t,:,:]) 
                last_pred_obs = unique_observations + pred_delta_mus 
            else:
                last_pred_obs = data[:,t,:,:] + pred_delta_mus 
            last_pred_obs = last_pred_obs.detach()

            pred_mus.append(pred_delta_mus) 
            pred_sigmas.append(pred_delta_sigmas)

        pred_mus = th.stack(pred_mus, dim=1) 
        pred_sigmas = th.stack(pred_sigmas, dim=1) 

        # Model Loss.
        x_next = th.roll(data.detach(), -1,  dims=1) 

        if self.obs_processor:
            unique_obs, _ = self.obs_processor.split_obs(data) 
            unique_obs_next, _ = self.obs_processor.split_obs(x_next) 
            deltas = unique_obs_next - unique_obs
        else:
            deltas = x_next - data 
        
        deltas = deltas[:, :-1, :, :].clone().detach() 

        # === EXTRACT WEIGHTS ===
        weights = train_params.get("weights", None)
        # =======================

        obs_loss = self.training_loss(x_next=deltas,
                                      pred_mus=pred_mus,
                                      pred_sigmas=pred_sigmas,
                                      mask=mask,
                                      weights=weights, 
                                      reduce=True)

        loss_info['predictor_obs_loss'] = th.mean(obs_loss).cpu().item()

        return obs_loss, loss_info