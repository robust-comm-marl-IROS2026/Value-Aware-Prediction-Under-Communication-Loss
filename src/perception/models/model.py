import torch as th


class PerceptionModel(object):
    """
        Generic perception model class.
    """

    def __init__(self, scheme, args, obs_processor=None):
        self.args = args
        self.scheme = scheme
        self.obs_processor = obs_processor

    @property
    def is_trainable(self):
        raise NotImplementedError()

    @property
    def is_evaluated_with_different_comm_levels(self):
        raise NotImplementedError()

    def get_network(self):
        raise NotImplementedError()

    def init_perception_model(self, batch_size):
        raise NotImplementedError()

    def get_rl_input_dim(self):
        raise NotImplementedError()

    def encode(self, ep_batch, t, test_mode=False, comm_p=None):
        raise NotImplementedError()

    def build_inputs(self, batch):
        # Used for training purposes.
        return batch["obs"] # [batch_size,num_timesteps,agents,obs_size]

    def _build_inputs(self, batch, t):
        # Used for encoding purposes.
        # Assumes homogenous agents with flat observations.
        # Other MACs might want to e.g. delegate building inputs to each agent
        bs = batch.batch_size
        inputs = batch["obs"][:, t]
        inputs = inputs.reshape(bs,1, self.args.n_agents,-1) # [batch_size,1,num_agents,obs_dim]
        return inputs

    def process_batch(self, ep_batch, test_mode=False):
        self.init_perception_model(batch_size=ep_batch.batch_size)
        outs = []
        for t in range(ep_batch.max_seq_length):
            outs.append(self.encode(ep_batch, t=t))
        outs = th.stack(outs, dim=1)
        ep_batch.data.transition_data["obs"] = outs
        return ep_batch
