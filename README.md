# Value-Aware Prediction for Robust Multi-Agent Coordination Under Communication Loss

## Overview

This codebase implements **MARO (Multi-Agent Response Observation)**, a hybrid perception module for multi-agent reinforcement learning that enables centralized training via learned inter-agent communication with configurable dropout. MARO can be combined with any standard MARL algorithm.

The algorithms are derived from [hybrid-marl](https://github.com/PPSantos/hybrid-marl), which extends the [EPyMARL](https://github.com/uoe-agents/epymarl) library.

## Dependencies

This project requires the following external libraries to be installed:

- **Multi-Agent Particle Environments (MPE):** [https://github.com/openai/multiagent-particle-envs](https://github.com/openai/multiagent-particle-envs)
- **Level-Based Foraging (LBF):** [https://github.com/semitable/lb-foraging](https://github.com/semitable/lb-foraging)

## Installation

Tested with Python 3.8.10 on Ubuntu. To install the codebase and its dependencies:

```bash
./install.sh
```

This script will:
1. Create a virtual environment and install Python dependencies from `requirements.txt`
2. Clone and install the [MPE](https://github.com/openai/multiagent-particle-envs) environment
3. Clone and install the [LBF](https://github.com/semitable/lb-foraging) environment (v11 branch)

## Running Experiments

Use `run.sh` to launch experiments. Configure the following variables:

### Environments

| Variable value | Environment name |
|---|---|
| `SimpleSpreadXY-v0` | SpreadXY-2 |
| `SimpleSpreadXY4-v0` | SpreadXY-4 |
| `SimpleSpreadBlind-v0` | SpreadBlindfold |
| `SimpleBlindDeaf-v0` | HearSee |
| `SimpleSpread-v0` | SimpleSpread |
| `SimpleSpeakerListener-v0` | SimpleSpeakerListener |
| `Foraging-2s-15x15-2p-2f-coop-v2` | LBF |

### Algorithms

| MPE | LBF |
|---|---|
| `iql_ns` | `iql_ns_lbf` |
| `qmix_ns` | `qmix_ns_lbf` |
| `ippo_ns` | `ippo_ns_lbf` |
| `mappo_ns` | `mappo_ns_lbf` |

### Perception Models

| Variable value | Description |
|---|---|
| `obs` | Observation only (no perception module) |
| `joint_obs` | Oracle (full joint observation) |
| `joint_obs_drop_test` | Masked joint observation |
| `ablation_no_pred` | MD baseline |
| `ablation_no_pred_masks` | MD with masks baseline |
| `maro_no_training` | MARO |
| `maro` | MARO with dropout |

### Time Limits

- MPE environments: `TIME_LIMIT=25`
- LBF environments: `TIME_LIMIT=30`

### Example

```bash
ENV="SimpleSpreadBlind-v0"
ALGO="qmix_ns"
PERCEPTION="maro"
TIME_LIMIT=25

CONFIG_SOURCE="src/config/custom_configs/${PERCEPTION}.yaml"
cp $CONFIG_SOURCE src/config/perception.yaml

python3 src/main.py --config=$ALGO --env-config=gymma \
    with env_args.key=$ENV env_args.time_limit=$TIME_LIMIT seed=0
```

## Project Structure

```
.
├── README.md
├── requirements.txt
├── install.sh              # Installation script
├── run.sh                  # Experiment runner
└── src/
    ├── main.py             # Entry point (Sacred experiment)
    ├── visualize_mappo.py  # Visualization of trained agents
    ├── run.py              # Training loop
    ├── config/
    │   ├── default.yaml    # Default hyperparameters
    │   ├── algs/           # Algorithm configs
    │   ├── envs/           # Environment configs
    │   └── custom_configs/ # Perception model configs
    ├── components/         # Replay buffer, action selectors
    ├── controllers/        # Multi-agent controllers
    ├── learners/           # RL algorithm implementations
    ├── modules/
    │   ├── agents/         # Agent networks (RNN-based)
    │   ├── critics/        # Critic networks
    │   └── mixers/         # Value mixing networks (QMIX, VDN, QTRAN)
    ├── perception/
    │   ├── models/         # MARO and baseline perception models
    │   └── trainers/       # Perception model training
    ├── envs/               # Environment wrappers and custom envs
    ├── runners/            # Episode and parallel runners
    ├── pretrained/         # Pretrained agent code for Tag env
    └── utils/              # Logging, RL utilities
```

## Acknowledgments

- [EPyMARL](https://github.com/uoe-agents/epymarl) — Extended PyMARL framework
- [hybrid-marl](https://github.com/PPSantos/hybrid-marl) — Base algorithm implementations
- [OpenAI MPE](https://github.com/openai/multiagent-particle-envs) — Multi-agent particle environments
- [LBF](https://github.com/semitable/lb-foraging) — Level-based foraging environment
