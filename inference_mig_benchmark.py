import yaml
from diffusers import EulerDiscreteScheduler
from migc.migc_utils import seed_everything
from migc.migc_pipeline import StableDiffusionMIGCPipeline, MIGCProcessor, AttentionStore
import os


if __name__ == '__main__':
    bench_file_path = 'bench_file/mig_bench.txt'
    annotation_path = 'bench_file/mig_bench_anno.yaml'
    migc_ckpt_path = 'pretrained_weights/MIGC_SD14.ckpt'
    assert os.path.isfile(migc_ckpt_path), "Please download the ckpt of migc and put it in the pretrained_weighrs/ folder!"
    sd1x_path = '/sdb/zdw/weights/stable-diffusion-v1-4' if os.path.isdir('/sdb/zdw/weights/stable-diffusion-v1-4') else "CompVis/stable-diffusion-v1-4"
    # MIGC is a plug-and-play controller.
    # You can go to https://civitai.com/search/models?baseModel=SD%201.4&baseModel=SD%201.5&sortBy=models_v5 find a base model with better generation ability to achieve better creations.


    with open(annotation_path, 'r') as f:
        cfg = f.read()
        annatation_data = yaml.load(cfg, Loader=yaml.FullLoader)

    # Construct MIGC pipeline
    pipe = StableDiffusionMIGCPipeline.from_pretrained(
        sd1x_path)
    pipe.attention_store = AttentionStore()
    from migc.migc_utils import load_migc
    load_migc(pipe.unet , pipe.attention_store,
            migc_ckpt_path, attn_processor=MIGCProcessor)
    pipe = pipe.to("cuda")
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)


    # Generate Image For COCO-MIG Benchmark, and the results will be saved in mig_bench/MIGC
    seed = 42
    num_iter = 1
    bench_name = os.path.split(bench_file_path)[-1][:-4]
    path_name = f'./{bench_name}/MIGC'
    if not os.path.exists(path_name):
        os.makedirs(path_name)
    with open(bench_file_path, 'r') as f:
        lines = f.readlines()
        for prompt_line in lines:
            prompt_final = [[]]
            bboxes = [[]]
            prompt = prompt_line.split('\n')[0]
            prompt_final[0].append(prompt)
            if prompt in annatation_data:
                for phase in annatation_data[prompt]:
                    if phase == 'coco_id':
                        continue
                    bbox_list = annatation_data[prompt][phase]
                    for bbox in bbox_list:
                        bboxes[0].append(bbox)
                        prompt_final[0].append(phase)
            img_name = prompt
            coco_id = annatation_data[prompt]['coco_id']
            seed_everything(seed)
            for i in range(num_iter):
                image = pipe(prompt_final, bboxes, num_inference_steps=50, guidance_scale=7.5, 
                                MIGCsteps=25, aug_phase_with_and=True).images[0]
                image.save(os.path.join(path_name, f"{img_name}_{seed}{i}_{coco_id}.png"))
                image = pipe.draw_box_desc(image, bboxes[0], prompt_final[0][1:])
                image.save(os.path.join(path_name, f"anno_{img_name}_{seed}{i}_{coco_id}.png"))