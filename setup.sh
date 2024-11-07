#!/bin/bash

# Install PyTorch
pip install torch==2.4.0

# Install wheel, packaging, and ninja
pip install wheel packaging ninja

# Install flash-attn and deepspeed
pip install flash-attn deepspeed

# Install requirements from requirements.txt
pip install -r requirements.txt
