import argparse
import numpy as np
import torch
import os
import yaml
import random
from diffusers.utils.import_utils import is_accelerate_available
from transformers import CLIPTextModel, CLIPTokenizer
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
from diffusers import EulerDiscreteScheduler
import cv2
if is_accelerate_available():
    from accelerate import init_empty_weights
from contextlib import nullcontext


def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


import torch
from typing import Callable, Dict, List, Optional, Union
from collections import defaultdict

LORA_WEIGHT_NAME = "pytorch_lora_weights.bin"

# We need to set Attention Processors for the following keys.
all_processor_keys = [
    'down_blocks.0.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.0.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor',
    'down_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'down_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'down_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.1.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.1.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.2.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.2.attentions.2.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.0.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.1.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.1.transformer_blocks.0.attn2.processor',
    'up_blocks.3.attentions.2.transformer_blocks.0.attn1.processor', 'up_blocks.3.attentions.2.transformer_blocks.0.attn2.processor',
    'mid_block.attentions.0.transformer_blocks.0.attn1.processor', 'mid_block.attentions.0.transformer_blocks.0.attn2.processor'
]

def load_migc_plus(unet, attention_store, pretrained_MIGC_path: Union[str, Dict[str, torch.Tensor]], attn_processor, strict=True,
                        **kwargs):

    state_dict = torch.load(pretrained_MIGC_path, map_location="cpu")

    # fill attn processors
    attn_processors = {}
    # state_dict = state_dict['state_dict']


    adapter_grouped_dict = defaultdict(dict)

    # change the key of MIGC.ckpt as the form of diffusers unet 
    for key, value in state_dict.items():
        attn_processor_key, sub_key = key.split('.attn2.processor.')
        adapter_grouped_dict[attn_processor_key][sub_key] = value

    # Create MIGC Processor
    config = {'not_use_migc': False}
    for key, value_dict in adapter_grouped_dict.items():
        dim = value_dict['migc.norm.bias'].shape[0]
        config['C'] = dim
        key_final = key + '.attn2.processor'
        if key_final.startswith("mid_block"):
            place_in_unet = "mid"
        elif key_final.startswith("up_blocks"):
            place_in_unet = "up"
        elif key_final.startswith("down_blocks"):
            place_in_unet = "down"
        attn_processors[key_final] = attn_processor(config, attention_store, place_in_unet)
        attn_processors[key_final].load_state_dict(value_dict, strict=strict)
        attn_processors[key_final].to(device=unet.device, dtype=unet.dtype)

    # Create CrossAttention/SelfAttention Processor
    config = {'not_use_migc': True}
    for key in all_processor_keys:
        if key not in attn_processors.keys():
            if key.startswith("mid_block"):
                place_in_unet = "mid"
            elif key.startswith("up_blocks"):
                place_in_unet = "up"
            elif key.startswith("down_blocks"):
                place_in_unet = "down"
            attn_processors[key] = attn_processor(config, attention_store, place_in_unet)
    unet.set_attn_processor(attn_processors)
    attention_store.num_att_layers = 32



def change_bbox_to_mask(bbox, height, width):
    mask = np.zeros((height, width))
    w_min = int(width * bbox[0])
    w_max = int(width * bbox[2])
    h_min = int(height * bbox[1])
    h_max = int(height * bbox[3])
    mask[h_min: h_max, w_min: w_max] = 1.0
    return mask


def change_mask_to_bbox(mask):
    assert mask.sum() != 0
    h, w = mask.shape
    mask_1 = mask.sum(axis=0)
    w_indices = np.where(mask_1 != 0)[0]
    w_min, w_max = w_indices[0], w_indices[-1]

    mask_2 = mask.sum(axis=1)

    h_indices = np.where(mask_2 != 0)[0]
    h_min, h_max = h_indices[0], h_indices[-1]
    return [w_min / w, h_min / h, w_max / w, h_max / h]