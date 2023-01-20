#%%
import os.path as osp
from yacs.config import CfgNode as CN

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torch

# We need to grab the checkpoint, and its corresponding config. We can skip on the config
# TODO figure out how to store the config in checkpoint and just load checkpoints
# config = './config/base.yaml'
# variant = osp.split(config)[1].split('.')[0]
# config = get_config(config)
# seed = 0
# version = 1
# root = Path(f'runs/{variant}-{seed}/lightning_logs/')
# # * This is the default output, if you want to play around with a different checkpoint load it here.
# model_ckpt = list(root.joinpath(f"version_{version}").joinpath('checkpoints').glob("*"))[0]

# weights = torch.load(model_ckpt, map_location='cpu')
# model = SeqSeqRNN(config)
# model.load_state_dict(weights['state_dict'])
# model.eval()
