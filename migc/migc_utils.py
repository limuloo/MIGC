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

def load_migc(unet, attention_store, pretrained_MIGC_path: Union[str, Dict[str, torch.Tensor]], attn_processor,
                        **kwargs):

    state_dict = torch.load(pretrained_MIGC_path, map_location="cpu")

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


def offlinePipelineSetupWithSafeTensor(sd_safetensors_path):
    project_dir = os.path.dirname(os.path.dirname(__file__))
    migc_ckpt_path = os.path.join(project_dir, 'pretrained_weights/MIGC_SD14.ckpt')
    clip_model_path = os.path.join(project_dir, 'migc_gui_weights/clip/text_encoder')
    clip_tokenizer_path = os.path.join(project_dir, 'migc_gui_weights/clip/tokenizer')
    original_config_file = os.path.join(project_dir, 'migc_gui_weights/v1-inference.yaml')
    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        # text_encoder = CLIPTextModel(config)
        text_encoder = CLIPTextModel.from_pretrained(clip_model_path)
        tokenizer = CLIPTokenizer.from_pretrained(clip_tokenizer_path)
    pipe = StableDiffusionMIGCPipeline.from_single_file(sd_safetensors_path,
                                                    original_config_file=original_config_file,
                                                    text_encoder=text_encoder,
                                                    tokenizer=tokenizer,
                                                    load_safety_checker=False)
    print('Initializing pipeline')
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)

    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    return pipe