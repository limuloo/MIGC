import yaml
from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
import os


if __name__ == '__main__':
    migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"


    sd1x_path = '/sdb/zdw/weights/stable-diffusion-v1-4' if os.path.isdir('/sdb/zdw/weights/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.
    
    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)

    prompt_final = [['masterpiece, best quality,black colored ball,gray colored cat,white colored  bed,\
                     green colored plant,red colored teddy bear,blue colored wall,brown colored vase,orange colored book,\
                     yellow colored hat', 'black colored ball', 'gray colored cat', 'white colored  bed', 'green colored plant', \
                        'red colored teddy bear', 'blue colored wall', 'brown colored vase', 'orange colored book', 'yellow colored hat']]
    bboxes = [[[0.3125, 0.609375, 0.625, 0.875], [0.5625, 0.171875, 0.984375, 0.6875], \
               [0.0, 0.265625, 0.984375, 0.984375], [0.0, 0.015625, 0.21875, 0.328125], \
                [0.171875, 0.109375, 0.546875, 0.515625], [0.234375, 0.0, 1.0, 0.3125], \
                    [0.71875, 0.625, 0.953125, 0.921875], [0.0625, 0.484375, 0.359375, 0.8125], \
                        [0.609375, 0.09375, 0.90625, 0.28125]]]
    negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
    seed = 7351007268695528845
    seed_everything(seed)
    image = pipe(prompt_final, bboxes, num_inference_steps=50, guidance_scale=7.5, 
                    MIGCsteps=25, aug_phase_with_and=False, negative_prompt=negative_prompt).images[0]
    image.save('output.png')
    image.show()
    image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
    image.save('anno_output.png')
    image.show()