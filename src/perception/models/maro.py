import numpy as np
import torch as th

from perception.models.model import PerceptionModel
from perception.models.sampling_schemes import sampling_registry
from perception.models.nets import MARONetwork, MARONetworkTeacherForcing


class MARO(PerceptionModel):

    """
        Maro's predictive model (deltas prediction).

        Uses a [Linear,LSTM, Linear] (shared) architecture
        to output a distribution over deltas.
    """

    def __init__(self, scheme, args, obs_processor=None):
        super(MARO, self).__init__(scheme,args,obs_processor)

        # Instantiate network.
        if self.obs_processor:
            # If we are using the observation pre-processor then the input
            # will have shape (obs_dim * unique_obs_dim + common_obs_dim)
            # and the output shape will be n_agents * unique_obs_dim.
            # In this case output_obs_dim = unique_obs_dim.
            unique_obs_dim = self.obs_processor.obs_split_idx
            common_obs_dim = scheme["obs"]["vshape"] - self.obs_processor.obs_split_idx
            input_dim = unique_obs_dim * self.args.n_agents + common_obs_dim
            output_obs_dim = unique_obs_dim
        else:
            # If we are not using the observation pre-processor then both
            # the input and output shapes are n_agents * obs_dim. In this
            # case output_obs_dim = obs_dim.
            input_dim = scheme["obs"]["vshape"] * self.args.n_agents
            output_obs_dim = scheme["obs"]["vshape"]

        if self.args.perception_args["teacher_forcing"]:
            self.network = MARONetworkTeacherForcing(
                n_agents=args.n_agents,
                input_dim=input_dim,
                output_obs_dim=output_obs_dim,
                hidden_dim=args.perception_args["hidden_dim"],
                obs_processor=self.obs_processor,
                train_comm_p=self.args.perception_args["train_comm_p"],
            )
        else:
            # Do not use teacher forcing.
            self.network = MARONetwork(
                n_agents=args.n_agents,
                input_dim=input_dim,
                output_obs_dim=output_obs_dim,
                hidden_dim=args.perception_args["hidden_dim"],
                obs_processor=self.obs_processor
            )

        self.hidden_state_agents = None
        self.estimated_agents_obs = None
        self.output_obs_dim = output_obs_dim

        if args.use_cuda:
            self.network.cuda()

    @property
    def is_trainable(self):
        return True

    @property
    def is_evaluated_with_different_comm_levels(self):
        return True

    def get_network(self):
       return self.network

    def init_perception_model(self, batch_size=1):
        if not self.args.use_cuda:
            self.hidden_state_agents = [ (th.zeros((1, batch_size, self.network.hidden_dim)),
                                        th.zeros((1, batch_size, self.network.hidden_dim)))
                                        for _ in range(self.args.n_agents) ]
        else:
            self.hidden_state_agents = [(th.zeros((1, batch_size, self.network.hidden_dim)).cuda(),
                                         th.zeros((1, batch_size, self.network.hidden_dim)).cuda())
                                        for _ in range(self.args.n_agents)]

        self.estimated_agents_obs = [[None] * self.args.n_agents
                                        for _ in range(self.args.n_agents)]

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
            rl_input_dim +=  self.args.n_agents
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


        latents = []
        for agent_id in range(self.args.n_agents):

            # Generate random agent id list given `comm_p` probability.
            if isinstance(comm_p, float):
                agent_com = np.random.choice([0, 1], size=self.args.n_agents, p=[(1 - comm_p), comm_p]) # [n_agents]
            else:
                agent_com = np.array([np.random.choice([0, 1], p=[(1-prob), prob]) for prob in comm_p[agent_id]])

            agent_com[agent_id] = 1  # Agent always communicates with itself.
            agent_com_list = np.where(agent_com == 1)[0].tolist()

            agent_comm_mask = (th.FloatTensor(agent_com) > 0)
            agent_comm_mask = agent_comm_mask.unsqueeze(0)
            agent_comm_mask = th.repeat_interleave(agent_comm_mask, agent_inputs.shape[0], axis=0) # [bs,n_agents]

            inverted_masks = 1.0 - th.FloatTensor(agent_com).unsqueeze(0) # Invert masks (zero = agent communicated; one = agent did not communicate).
            inverted_masks = th.repeat_interleave(inverted_masks, agent_inputs.shape[0], axis=0).to(agent_inputs.device) # [bs,n_agents]

            latent_agent, \
            self.hidden_state_agents[agent_id], \
            self.estimated_agents_obs[agent_id] = self.agent_encode(agent_inputs,
                                                    hidden=self.hidden_state_agents[agent_id],
                                                    agent_estimated_obs=self.estimated_agents_obs[agent_id],
                                                    agent_com=agent_com_list) # [bs, data_dim*n_agents]

            if self.args.perception_args["append_masks_to_rl_input"]:

                if self.args.perception_args["accumulate_masks"]:
                    self.masks[:,agent_id,:] += inverted_masks # [bs,n_agents]
                    self.masks[:,agent_id,:] = th.where(agent_comm_mask, th.zeros_like(self.masks[:,agent_id,:]), self.masks[:,agent_id,:]) # [bs,n_agents]
                    latent_agent = th.cat([latent_agent, self.masks[:,agent_id,:]], dim=-1)  # [bs, obs_dim*n_agents + n_agents]
                else:
                    latent_agent = th.cat([latent_agent, inverted_masks], dim=-1)  # [bs, obs_dim*n_agents + n_agents]

            latents.append(latent_agent.unsqueeze(1))  # [batch_size,1, obs_dim*n_agents (+n_agents)]

        latent = th.concat(latents, dim=1)  # [batch_size,num_agents, obs_dim*n_agents (+n_agents)]
        return latent.detach()

    def agent_encode(self, data, hidden=None, agent_estimated_obs=None, agent_com=None):

        # data shape = [bs, 1, num_agent, obs_dim]

        latents = []
        for i_ag in range(self.args.n_agents):

            # If we communicate the observations
            if i_ag in agent_com:

                agent_obs = data[:, :, i_ag, :].squeeze(1) # [bs, obs_dim]

                if self.obs_processor:
                    agent_obs, _ = self.obs_processor.split_obs(agent_obs) # Keep only the part of the observation that is unique to the agent.
                
                latents.append(agent_obs) # [bs, obs_dim] or [bs,unique_obs_dim]
            else:
                # If we don't communicate the observations.
                # If we don't have any estimate (only for first timestep).
                if agent_estimated_obs[i_ag] is None:
                    raise ValueError('The code must not enter here as we assume the first observations are shared among the agents.')

                # If we have an estimate (from a previous timestep)
                else:
                    latents.append(agent_estimated_obs[i_ag])

        # Prepare data for RL input.
        rl_input_data = th.cat(latents, dim=-1) # [bs, obs_dim*n_agents] or [bs, unique_obs_dim*n_agents]

        if self.obs_processor:
            # Append common info.
            _, common_data = self.obs_processor.split_obs(data[:,:,0,:].squeeze(1))
            rl_input_data = th.cat([rl_input_data, common_data], dim=-1) # [bs, unique_obs_dim*n_agents + unique_obs_dim]

        # Update Predictive model
        pred_delta_mus, _, hidden = self.network.encode(rl_input_data, hidden)

        if self.obs_processor:
            unique_observations, _ = self.obs_processor.split_obs(data) # [bs, 1, n_agents, unique_obs_dim]

        # Update estimates of next observations - use the mean of the gaussian (deltas).
        next_agent_estimated_obs = [None] * self.args.n_agents
        for i_ag in range(self.args.n_agents):

            # Append delta to last observation to get next observation.
            if i_ag in agent_com:
                # If we communicate the observations.
                if self.obs_processor:
                    observation = unique_observations[:,:,i_ag,:].squeeze(1) + pred_delta_mus[:, i_ag, :] # [bs,obs_dim] or [bs, unique_obs_dim]
                else:
                    observation = data[:,:,i_ag,:].squeeze(1) + pred_delta_mus[:, i_ag, :] # [bs,obs_dim] or [bs, unique_obs_dim]
            else:
                # If we don't communicate the observations.
                observation = agent_estimated_obs[i_ag] + pred_delta_mus[:, i_ag, :] # [bs,obs_dim] or [bs, unique_obs_dim]

            next_agent_estimated_obs[i_ag] = observation.detach()

        return rl_input_data.detach(), hidden, next_agent_estimated_obs
