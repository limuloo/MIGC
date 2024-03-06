import argparse
import numpy as np
import torch
import os
import yaml
import random


def seed_everything(seed):
    # np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


from diffusers.utils import (
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    _get_model_file,
)
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

def load_migc(unet, attention_store, pretrained_model_name_or_path_or_dict: Union[str, Dict[str, torch.Tensor]], attn_processor,
                        **kwargs):
    cache_dir = kwargs.pop("cache_dir", DIFFUSERS_CACHE)
    force_download = kwargs.pop("force_download", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    local_files_only = kwargs.pop("local_files_only", HF_HUB_OFFLINE)
    use_auth_token = kwargs.pop("use_auth_token", None)
    revision = kwargs.pop("revision", None)
    subfolder = kwargs.pop("subfolder", None)
    weight_name = kwargs.pop("weight_name", None)
    use_safetensors = kwargs.pop("use_safetensors", None)
    # This value has the same meaning as the `--network_alpha` option in the kohya-ss trainer script.
    # See https://github.com/darkstorm2150/sd-scripts/blob/main/docs/train_network_README-en.md#execute-learning
    network_alpha = kwargs.pop("network_alpha", None)
    construct_fuser_with_time_threshold = kwargs.pop(
        "construct_fuser_with_time_threshold",
        False)

    user_agent = {
        "file_type": "attn_procs_weights",
        "framework": "pytorch",
    }

    model_file = None
    if not isinstance(pretrained_model_name_or_path_or_dict, dict):
        if model_file is None:
            model_file = _get_model_file(
                pretrained_model_name_or_path_or_dict,
                weights_name=weight_name or LORA_WEIGHT_NAME,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=subfolder,
                user_agent=user_agent,
            )
            state_dict = torch.load(model_file, map_location="cpu")
    else:
        state_dict = pretrained_model_name_or_path_or_dict

    # fill attn processors
    attn_processors = {}
    state_dict = state_dict['state_dict']


    adapter_grouped_dict = defaultdict(dict)

    # change the key of MIGC.ckpt as the form of diffusers unet 
    for key, value in state_dict.items():
        key_list = key.split(".")
        assert 'migc' in key_list
        if 'input_blocks' in key_list:
            model_type = 'down_blocks'
        elif 'middle_block' in key_list:
            model_type = 'mid_block'
        else:
            model_type = 'up_blocks'
        index_number = int(key_list[3])
        if model_type == 'down_blocks':
            input_num1 = str(index_number//3)
            input_num2 = str((index_number%3)-1)
        elif model_type == 'mid_block':
            input_num1 = '0'
            input_num2 = '0'
        else:
            input_num1 = str(index_number//3)
            input_num2 = str(index_number%3)
        attn_key_list = [model_type,input_num1,'attentions',input_num2,'transformer_blocks','0']
        if model_type == 'mid_block':
            attn_key_list = [model_type,'attentions',input_num2,'transformer_blocks','0']
        attn_processor_key = '.'.join(attn_key_list)
        sub_key = '.'.join(key_list[key_list.index('migc'):])
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
        attn_processors[key_final].load_state_dict(value_dict)
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