#!/bin/bash

# Install PyTorch
pip install torch

# Install wheel, packaging, and ninja
pip install wheel packaging ninja

# Install flash-attn and deepspeed
pip install flash-attn deepspeed

# Install requirements from requirements.txt
pip install -r requirements.txt
