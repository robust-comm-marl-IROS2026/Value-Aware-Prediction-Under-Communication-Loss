from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th

from .attention import MultiHeadAttention

NUM_ATT_HEADS = 4
ATT_HEAD_DIM = 8

class NonSharedAttentionMAC:
    def __init__(self, scheme, groups, args):

        # Internalise arguments.
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.masks_dim = self.n_agents
        self.agent_input_dim = int((scheme["obs"]["vshape"] - self.masks_dim) / self.n_agents)

        input_shape = NUM_ATT_HEADS * ATT_HEAD_DIM
        self._build_agents(input_shape)

        self.attention_networks = th.nn.ModuleList([MultiHeadAttention(d_model=self.agent_input_dim, comm_mask_dim=self.n_agents,
                                    d_head=ATT_HEAD_DIM, num_heads=NUM_ATT_HEADS) for _ in range(self.n_agents)])

        self.hidden_states = None

        if self.args.perception_args['model_type'] != "maro_deltas_no_actions":
            raise ValueError('To use the attention MAC controller the maro_deltas_no_actions perception model needs to be used.')

        if not self.args.perception_args['append_masks_to_rl_input']:
            raise ValueError('Masks need to be used (append_masks_to_rl_input).')

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # Only select actions for the selected batch elements in bs
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t]

        agent_inputs = self._build_inputs(ep_batch, t) #[(bs*n_agents), data_dim] (data_dim = n_agents*obs_dim + n_agents)
        att_inputs = agent_inputs[:,:-self.masks_dim]
        att_inputs = att_inputs.view(ep_batch.batch_size,self.n_agents,self.n_agents,self.agent_input_dim)
        comm_masks = agent_inputs[:,-self.masks_dim:] # [(bs*n_agents), n_agents]
        comm_masks = comm_masks.view(ep_batch.batch_size,self.n_agents,self.masks_dim) # [bs, n_agents, n_agents]

        att_contexts = []
        for a_id in range(self.n_agents):

            att_in_a_id = att_inputs[:,a_id,:,:] # [bs, n_agents, obs_dim]
            att_query_a_id = att_inputs[:,a_id,a_id,:].unsqueeze(1) # [bs, 1, obs_dim]
            comm_mask_a_id = comm_masks[:,a_id,:] # [bs, mask_dim]
            att_context, attn = self.attention_networks[a_id](query=att_query_a_id, key=att_in_a_id, value=att_in_a_id, comm_mask=comm_mask_a_id, mask=None) # [bs,1,att_out_dim] 
            att_context = att_context.squeeze(1) # [bs,att_out_dim=(NUM_ATT_HEADS*ATT_HEAD_DIM]
            att_contexts.append(att_context)

        att_contexts = th.stack(att_contexts, dim=1) # [bs,n_agents,att_out_dim]
        att_contexts = att_contexts.view(ep_batch.batch_size*self.n_agents, -1) # [bs*n_agents,att_out_dim]
        
        agent_outs, self.hidden_states = self.agent(att_contexts, self.hidden_states)

        # Softmax the agent outputs if they're policy logits
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # Make the logits for unavailable actions very negative to minimise their affect on the softmax
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10
            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1)

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path, logger=None, save_mongo=False):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

        if logger is not None and save_mongo is True:
            logger.log_model(filepath="{}/agent.th".format(path), name="agent.th")


    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        n_actions = batch["actions_onehot"].shape[-1]

        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.ones_like(batch["actions_onehot"][:, t]) / n_actions)
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))

        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1) # [batch_size*n_agents, obs_dim]

        return inputs

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += self.args.n_actions
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
