#!/bin/bash
# Python version = 3.8.10

ENV="SimpleSpreadBlind-v0"
ALGO="mappo_ns"
PERCEPTION="maro" # check 'src/config/custom_configs/' folder.
TIME_LIMIT=25
ADV_LAMBDA=1.0 # 1.0 = Value-Aware MARO, 0.0 = MARO baseline

# Setup correct config file(s).
CONFIG_SOURCE="src/config/custom_configs/${PERCEPTION}.yaml"
cp $CONFIG_SOURCE src/config/perception.yaml

# Run.
for i in {0..2}
do
   python3 src/main.py --config=$ALGO --env-config=gymma with env_args.key=$ENV env_args.time_limit=$TIME_LIMIT seed=$i perception_args.adv_lambda=$ADV_LAMBDA &
   echo "Running with seed=$i"
   sleep 2s
done
