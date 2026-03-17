#!/bin/bash
# Python version = 3.8.10

# Setup virtual environment and install libraries.
virtualenv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Clone and install MPE.
git clone https://github.com/PPSantos/multiagent-particle-envs.git
cd multiagent-particle-envs
pip3 install -e .
cd ..

# Test installation.
python3 src/main.py --config=iql_ns --env-config=gymma with env_args.key="SimpleSpreadXY-v0"
