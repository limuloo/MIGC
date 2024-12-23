import yaml
from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything, offlinePipelineSetupWithSafeTensor
from migc_plus.migc_plus_utils import change_bbox_to_mask, change_mask_to_bbox
from migc_plus.migc_plus_pipeline import StableDiffusionMIGCPlusPipeline, MIGCPlusProcessor, AttentionStore
import os
import cv2
from copy import deepcopy


if __name__ == '__main__':
    migc_plus_ckpt_path = 'pretrained_weights/MIGC++_SD14.ckpt'
    assert os.path.isfile(migc_plus_ckpt_path), "Please download the ckpt of migc++ and put it in the pretrained_weights/ folder!"


    sd1x_path = '/mnt/sda/zdw/ckpt/new_sd14' if os.path.isdir('/mnt/sda/zdw/ckpt/new_sd14') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPlusPipeline.from_pretrained(
        sd1x_path)

    pipe.attention_store = AttentionStore()
    from migc_plus.migc_plus_utils import load_migc_plus
    load_migc_plus(pipe.unet , pipe.attention_store,
            migc_plus_ckpt_path, attn_processor=MIGCPlusProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


    prompt_input = [[]]
    bboxes = [[]]
    masks = [[]]

    image_description = 'masterpiece, best quality, a yellow horse and a black dog are playing on the grass.'
    
    instance_descriptions = [
        {'prompt': 'a yellow horse',     'pos_type': 'mask',   'pos': './migc_plus/horse_mask.png'},
        {'prompt': 'a black dog',        'pos_type': 'box',   'pos': [0.015034, 0.49292, 0.640255, 0.988859]},
                             ]
    
    prompt_input[0].append(image_description)
    for instance_description in instance_descriptions:
        prompt_input[0].append(instance_description['prompt'])
        if instance_description['pos_type'] == 'box':
            bbox = instance_description['pos']
            mask = change_bbox_to_mask(bbox, height=512, width=512)
        elif instance_description['pos_type'] == 'mask':
            mask = cv2.imread(instance_description['pos'], cv2.IMREAD_GRAYSCALE)
            mask = mask / 255.
            bbox = change_mask_to_bbox(mask)
        bboxes[0].append(bbox)
        masks[0].append(mask)



    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    seed = 42
    seed_everything(seed)
    num_inference_steps = 50
    image = pipe(deepcopy(prompt_input), bboxes, num_inference_steps=num_inference_steps, guidance_scale=7.5, 
                    MIGCsteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt,
                    RefinedSteps=num_inference_steps, masks=masks[0]).images[0]
    image.save(f'output_MIGC++.png')