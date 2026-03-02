#!/bin/bash
# Python version = 3.8.10

# Setup virtual environment and install libraries.
virtualenv venv
source venv/bin/activate
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Clone and install MPE.
git clone https://github.com/openai/multiagent-particle-envs.git
cd multiagent-particle-envs
pip3 install -e .
cd ..

# Clone and install lbforaging (v11 branch).
git clone https://github.com/semitable/lb-foraging.git
cd lb-foraging
git checkout v11
pip3 install -e .
cd ..

echo "Installation complete."
