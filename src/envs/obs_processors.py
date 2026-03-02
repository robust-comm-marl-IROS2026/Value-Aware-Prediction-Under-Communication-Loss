REGISTRY = {}

class ObservationSplitter(object):

    def __init__(self, obs_split_idx):
        # obs[:obs_split_idx] corresponds to the part of the observation that is unique to the agent.
        # obs[obs_split_idx:] corresponds to the part of the observation that is shared between agents.
        self.obs_split_idx = obs_split_idx

    def split_obs(self, obs):
        return obs[...,:self.obs_split_idx], obs[...,self.obs_split_idx:] # unique, shared/common

def simple_spread_xy():
    return ObservationSplitter(obs_split_idx=4)
REGISTRY["SimpleSpreadXY-v0"] = simple_spread_xy

def simple_spread_xy_4():
    return ObservationSplitter(obs_split_idx=4)
REGISTRY["SimpleSpreadXY4-v0"] = simple_spread_xy_4

def simple_spread_xy_8():
    return ObservationSplitter(obs_split_idx=4)
REGISTRY["SimpleSpreadXY8-v0"] = simple_spread_xy_8

def simple_spread_blind():
    return ObservationSplitter(obs_split_idx=4)
REGISTRY["SimpleSpreadBlind-v0"] = simple_spread_blind

def simple_spread_blind_6():
    return ObservationSplitter(obs_split_idx=4)
REGISTRY["SimpleSpreadBlind6-v0"] = simple_spread_blind_6

def simple_spread_blind_12():
    return ObservationSplitter(obs_split_idx=4)
REGISTRY["SimpleSpreadBlind12-v0"] = simple_spread_blind_12
