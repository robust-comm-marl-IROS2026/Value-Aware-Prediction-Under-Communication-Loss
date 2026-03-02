from components.episode_buffer import EpisodeBatch
import torch as th
from torch.optim import Adam
import time


class OnlineTrainer:
    def __init__(self, perc_model, logger, args):

        # Internalise arguments.
        self.args = args
        self.logger = logger
        self.perc_model = perc_model
        self.network = perc_model.get_network()

        # Optimizer.
        self.optim = Adam(self.network.parameters(),
                    lr=self.args.perception_args['learning_rate'])

        self.log_stats_t = -self.args.perception_args['trainer_log_interval'] - 1

        print ("\n******************************")
        print (f"ADV_LAMBDA: {self.args.perception_args.get('adv_lambda')}")
        print ("******************************\n")
        time.sleep (10)

    def train(self, batch: EpisodeBatch, t_env: int, weights=None):
        # 1. Prepare Inputs
        obs = self.perc_model.build_inputs(batch) # [batch, 26, agents, obs_dim]
        obs = obs[:, :-1] # [batch, 25, ...]
        
        # Prepare Base Mask [batch, 25, agents]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])

        # --- CRITICAL FIX: PRE-ALIGNMENT ---
        # The network consumes 'obs' (Length T) and predicts 'deltas' (Length T-1).
        # We must align the mask and weights to T-1.
        
        # 1. Calculate computable predictions (25 obs -> 24 predictions)
        num_predictions = obs.shape[1] - 1
        
        # 2. Slice Mask: We need validity of TARGETS (indices 1 to 25)
        # Drop the first mask (t=0) because it's an input, not a prediction target.
        mask_aligned = mask[:, 1:] # [batch, 24, ...]

        # -------------------------------------------------------------------
        # 2. Process Weights (Baseline + Advantage Boost)
        # -------------------------------------------------------------------
        processed_weights = None
        if weights is not None:
            # 3. Slice Weights: We need weights for ACTIONS (indices 0 to 24)
            # Drop the last weight because we can't predict the transition after the last observation.
            
            # Ensure we don't exceed available predictions
            slice_len = min(num_predictions, weights.shape[1])
            
            # Slice weights to match the transitions we are actually predicting
            weights_aligned = weights[:, :slice_len]
            
            # If weights are shorter than predictions (rare), align everything to weights
            if weights_aligned.shape[1] < num_predictions:
                obs = obs[:, :weights_aligned.shape[1] + 1] # +1 because obs is input
                mask_aligned = mask_aligned[:, :weights_aligned.shape[1]]
            
            # --- Apply The Formula ---
            lambda_weight = self.args.perception_args.get("adv_lambda", 1.0)
            raw_weights = 1.0 + (lambda_weight * weights_aligned)
            processed_weights = th.nn.functional.relu(raw_weights)
            #processed_weights = th.clamp(raw_weights, min=-1.0)
            processed_weights = processed_weights / (processed_weights.mean() + 1e-8)

        # -------------------------------------------------------------------
        # 3. Pass Weights to the Network
        # -------------------------------------------------------------------
        train_params = {
            "parameter": None,
            "weights": processed_weights
        }

        '''print ("\n*****************************")
        print (f"obs shape: {obs.shape}")
        print (f"mask_aligned shape: {mask_aligned.shape}")
        print (f"processed_weights shape: {processed_weights.shape}")
        print ("*****************************\n")
        time.sleep (30)'''

        
        # Pass 'mask_aligned' (Size 24) instead of 'mask' (Size 25)
        loss, loss_info = self.network.training_step(obs, mask_aligned, train_params)

        self.optim.zero_grad()
        loss.backward()
        if self.args.perception_args["grad_clip"]:
            th.nn.utils.clip_grad_norm_(self.network.parameters(), self.args.perception_args["grad_clip"])
        self.optim.step()

        # -------------------------------------------------------------------
        # 5. Logging
        # -------------------------------------------------------------------
        if t_env - self.log_stats_t >= self.args.perception_args['trainer_log_interval']:
            for key, val in loss_info.items():
                self.logger.log_stat("perc_" + key, val, t_env)
            
            if weights is not None:
                with th.no_grad():
                    pos_mask = (weights_aligned > 0).float()
                    neg_mask = (weights_aligned <= 0).float()
                    if pos_mask.sum() > 0:
                        self.logger.log_stat("perc_align/avg_pos_advantage", (weights_aligned * pos_mask).sum() / pos_mask.sum(), t_env)
                    if neg_mask.sum() > 0:
                        self.logger.log_stat("perc_align/avg_neg_advantage", (weights_aligned * neg_mask).sum() / neg_mask.sum(), t_env)
            
            self.log_stats_t = t_env

    def save_models(self, path, save_mongo=False):
        th.save(self.network.state_dict(), "{}/network.th".format(path))
        th.save(self.optim.state_dict(), "{}/opt.th".format(path))

        if save_mongo:
            self.logger.log_model(filepath="{}/network.th".format(path), name="perceptual_model.th")
            self.logger.log_model(filepath="{}/opt.th".format(path), name="perceptual_opt.th")

    def load_models(self, path):
        if self.args.use_cuda:
            checkpoint = th.load("{}/network.th".format(path))
            checkpoint_opt = th.load("{}/opt.th".format(path))
        else:
            checkpoint = th.load("{}/network.th".format(path),
                map_location=lambda storage, location: storage)
            checkpoint_opt = th.load("{}/opt.th".format(path),
                map_location=lambda storage, location: storage)
        self.network.load_state_dict(checkpoint)
        self.optim.load_state_dict(checkpoint_opt)
