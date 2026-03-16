import torch as th
import numpy as np

from perception.models.model import PerceptionModel
from perception.models.sampling_schemes import sampling_registry


class MaskedJointObsModel(PerceptionModel):

    def __init__(self, scheme, args, obs_processor=None):
        super(MaskedJointObsModel, self).__init__(scheme,args,obs_processor)
        obs_dim = scheme["obs"]["vshape"]
        action_dim = args.n_actions
        self.data_dim = obs_dim + action_dim
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    @property
    def is_trainable(self):
        return False

    @property
    def is_evaluated_with_different_comm_levels(self):
        return True

    def get_network(self):
        pass

    def init_perception_model(self, batch_size):
        if isinstance(self.args.perception_args["train_comm_p"], float):
            self.train_comm_p = self.args.perception_args["train_comm_p"]
        elif isinstance(self.args.perception_args["train_comm_p"], str):
            self.train_comm_p = sampling_registry[self.args.perception_args["train_comm_p"]]()
        else:
            raise ValueError("Incorrect sampling scheme selected:" + str(self.args.perception_args["train_comm_p"]))

        if self.args.perception_args["accumulate_masks"]:
            self.masks = th.zeros((batch_size, self.args.n_agents, self.args.n_agents))

    def get_rl_input_dim(self):
        obs_dim = self.scheme["obs"]["vshape"]
        if self.obs_processor:
            unique_obs_size = self.obs_processor.obs_split_idx
            common_obs_size = obs_dim - self.obs_processor.obs_split_idx
            rl_input_dim = self.args.n_agents * unique_obs_size + common_obs_size
        else:
            rl_input_dim = self.args.n_agents * obs_dim
        if self.args.perception_args["append_masks_to_rl_input"]:
            rl_input_dim += self.args.n_agents # array encoding the which observations are valid.
        print("rl_input_dim=", rl_input_dim)
        return rl_input_dim

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        """
            Encodes the observations. Can be called at:
                (i) train time for action selection (batch size can be greater than 1 if
                    multiple envs are running in parallel);
                (ii) test time for action selection (batch size can be greater than 1 if
                    multiple envs are running in parallel);
                (iii) train time to process the batch sampled from the replay buffer
                    before passing it to the RL module (batch size equals the replay
                    buffer batch size).
        """
        if not test_mode:
            # Train mode.
            comm_p = self.train_comm_p

        if self.args.perception_args["comm_at_t0"] and t == 0:
            comm_p = 1.0

        agent_inputs = self._build_inputs(ep_batch, t) # [bs,1,num_agent,obs_dim]

        # Generate mask given `comm_p` probability.
        bsize, n_agents = agent_inputs.shape[0], agent_inputs.shape[2]
        agent_comm_mask = th.rand(bsize,n_agents,n_agents) # [bs,num_agents,num_agents]
        agent_comm_mask = agent_comm_mask + th.eye(n_agents) # (agent always communicates with itself).
        if isinstance(comm_p, float):
            agent_comm_mask = (agent_comm_mask >= (1.0 - comm_p)) # Mask out entries.
        else:
            comm_p = th.from_numpy(comm_p)
            agent_comm_mask = (agent_comm_mask >= (1.0 - comm_p)) # Mask out entries.
        inverted_masks = 1.0 - agent_comm_mask.type(th.float32) # Invert masks (zero = agent communicated; one = agent did not communicate).

        if self.obs_processor:
            unique_obs, common_obs = self.obs_processor.split_obs(agent_inputs)
            # unique_obs: [bs,1,num_agent,unique_obs_dim]
            # common_obs: [bs,1,num_agent,common_obs_dim]
            common_obs = common_obs[:,:,0,:] # [bs,1,common_obs_dim] - info is the same, so just take any index.
            common_obs = common_obs.repeat(1, self.args.n_agents, 1)  # [bs,num_agents,common_obs_dim]

            latent = unique_obs.reshape(agent_inputs.shape[0],agent_inputs.shape[1],-1) # [bs,1,unique_obs_dim*n_agents]
            latent = latent.repeat(1, self.args.n_agents, 1)  # [bs,num_agents,unique_obs_dim*n_agents]

            # Mask the entries of the latent vector using the agent_comm_mask_repeated variable.
            unique_obs_size = self.obs_processor.obs_split_idx
            agent_comm_mask_repeated = th.repeat_interleave(agent_comm_mask, unique_obs_size, axis=-1) # [bs,num_agents,unique_obs_dim*n_agents]

            latent = th.where(agent_comm_mask_repeated, latent, th.zeros_like(latent))
            latent = th.cat([latent,common_obs], dim=-1) # [bs,num_agents,unique_obs_dim*n_agents + common_obs_dim]

        else:
            latent = agent_inputs.reshape(agent_inputs.shape[0],agent_inputs.shape[1],-1) # [bs,1,obs_dim*num_agents]
            latent = latent.repeat(1, self.args.n_agents, 1)  # [bs,num_agents,obs_dim*num_agents]

            # Mask the entries of the latent vector using the agent_comm_mask_repeated variable.
            agent_comm_mask_repeated = th.repeat_interleave(agent_comm_mask, self.obs_dim, axis=-1)
            latent = th.where(agent_comm_mask_repeated, latent, th.zeros_like(latent))

        # Append to the observations an array encoding which entries are valid.
        if self.args.perception_args["append_masks_to_rl_input"]:

            if self.args.perception_args["accumulate_masks"]:
                self.masks += inverted_masks
                self.masks = th.where(agent_comm_mask, th.zeros_like(self.masks), self.masks)
                latent = th.concat([latent,self.masks], dim=-1)
            else:
                latent = th.concat([latent,agent_comm_mask.type(th.float32)], dim=-1)

        return latent.detach()
