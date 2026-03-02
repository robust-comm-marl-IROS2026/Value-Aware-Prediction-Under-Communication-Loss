import torch as th
from perception.models.model import PerceptionModel


class JointObsModel(PerceptionModel):

    def __init__(self, scheme, args, obs_processor=None):
        super(JointObsModel, self).__init__(scheme,args,obs_processor)
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
        return False

    def get_network(self):
        pass

    def init_perception_model(self, batch_size):
        pass

    def get_rl_input_dim(self):
        obs_dim = self.scheme["obs"]["vshape"]
        if self.obs_processor:
            unique_obs_size = self.obs_processor.obs_split_idx
            common_obs_size = obs_dim - self.obs_processor.obs_split_idx
            rl_input_dim = self.args.n_agents * unique_obs_size + common_obs_size
        else:
            rl_input_dim = self.args.n_agents * obs_dim
        print("rl_input_dim=", rl_input_dim)
        return rl_input_dim

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        agent_inputs = self._build_inputs(ep_batch, t) # [bs,1,num_agent,obs_dim]

        if self.obs_processor:
            unique_obs, common_obs = self.obs_processor.split_obs(agent_inputs)
            # unique_obs: [bs,1,num_agent,unique_obs_dim]
            # common_obs: [bs,1,num_agent,common_obs_dim]
            common_obs = common_obs[:,:,0,:] # [bs,1,common_obs_dim] - info is the same, so just take any index.
            latent = unique_obs.reshape(agent_inputs.shape[0],agent_inputs.shape[1],-1) # [bs,1,unique_obs_dim*n_agents]
            latent = th.cat([latent,common_obs], dim=-1) # [bs,1,unique_obs_dim*n_agents + common_obs_dim]
        else:
            latent = agent_inputs.reshape(agent_inputs.shape[0],agent_inputs.shape[1],-1) # [bs,1,obs_dim*n_agents]
        
        latent = latent.repeat(1, self.args.n_agents, 1)  # [bs,num_agents,data_dim*n_agents]

        return latent.detach()
