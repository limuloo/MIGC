import os
from flask import Flask, request, jsonify, send_from_directory, Response
from flask_cors import CORS  # Import the flask_cors module to handle cross-domain requests
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from migc.migc_utils import offlinePipelineSetupWithSafeTensor, seed_everything
import json, time

last_ckpt_path_lock = threading.Lock()  # Ensure thread-safe updates of last_ckpt_path
last_ckpt_path = None


app = Flask(__name__)  # Initialize Flask application
CORS(app)  # Use the CORS decorator to enable cross-origin resource sharing for all routes
executor = ThreadPoolExecutor(max_workers=1)  # Make sure only one request is processed at a time

# Define the root route, used to serve static files (such as front-end pages)
@app.route('/')
def index():
    # Use the send_from_directory function to send the 'base.html' file from the 'templates' directory
    return send_from_directory('templates', 'base.html')

GUI_progress = [100]  # Global variables are used to track progress bar

pipe = None

@app.route('/GUI_progress')
def progress_updates():
    def generate():
        global GUI_progress
        while GUI_progress[0] < 100:
            print(GUI_progress[0])
            json_data = json.dumps({'GUI_progress': GUI_progress[0]})
            yield f"data:{json_data}\n\n"
            time.sleep(0.1)
        yield "data:{\"GUI_progress\": 100}\n\n"
    GUI_progress[0] = 0
    return Response(generate(), mimetype='text/event-stream')


def process_request(req_data):
    data = req_data['prompt']
    
    InstanceNum = data['InstanceNum']
    width = data['width']
    height = data['height']
    num_inference_steps = int(data["num_inference_steps"])
    prompt_final = [[data['positive_prompt']]]
    negative_prompt = "worst quality, low quality, bad anatomy, " + data['negative_prompt']
    bboxes = [[]]
    ca_scale = []
    ea_scale = []
    sac_scale = []
    for i in range(1, InstanceNum + 1):
        InstanceData = data[f'Instance{i}']['inputs']
        prompt_final[0].append(InstanceData['text'])
        prompt_final[0][0]  += ',' + InstanceData['text']
        l = InstanceData['x'] / width
        u = InstanceData['y'] / height
        r = l + InstanceData['width'] / width
        d = u + InstanceData['height'] / height
        bboxes[0].append([l, u, r, d])
        ca_scale.append(float(InstanceData['ca_scale']))
        ea_scale.append(float(InstanceData['ea_scale']))
        sac_scale.append(float(InstanceData['sac_scale']))
    MIGCsteps = int(data['MIGCsteps'])
    NaiveFuserSteps = int(data['NaiveFuserSteps'])
    global pipe
    project_dir = os.path.dirname(os.path.dirname(__file__))
    sd_safetensors_name = data['sd_checkpoint_name']
    if sd_safetensors_name == "":
        GUI_progress[0] = 100
        return "ckpt_warning"
    sd_safetensors_path = os.path.join(project_dir, 'migc_gui_weights/sd',
                                    sd_safetensors_name)
    
    with last_ckpt_path_lock:
        global last_ckpt_path
        if last_ckpt_path is None or last_ckpt_path != sd_safetensors_path:
            if last_ckpt_path is not None:
                pipe = pipe.to('cpu')
            pipe = offlinePipelineSetupWithSafeTensor(sd_safetensors_path=sd_safetensors_path)
            pipe = pipe.to("cuda")
            last_ckpt_path = sd_safetensors_path
    seed = int(data['seed'])
    cfg = float(data['cfg'])
    seed_everything(seed)
    print(prompt_final)
    print(bboxes)
    print('Generating Image..')
    image = pipe(prompt_final, bboxes, 
                num_inference_steps=num_inference_steps, guidance_scale=cfg,
                width=width, height=height, MIGCsteps=MIGCsteps, NaiveFuserSteps=NaiveFuserSteps,
                ca_scale=ca_scale, ea_scale=ea_scale,
                sac_scale=sac_scale, negative_prompt=negative_prompt, GUI_progress=GUI_progress).images[0]
    app_file_path = __file__
    app_folder = os.path.dirname(app_file_path)
    output_folder = os.path.join(app_folder, 'output_images')
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    image.save(os.path.join(output_folder, "out.png"))
    return "out"


@app.route('/get_image', methods=['POST'])
def get_image():
    data = request.json
    # Add request data to queue
    future = executor.submit(process_request, data)
    fig_name = future.result()  # Block and wait until image processing is completed
    return send_from_directory('output_images', f'{fig_name}.png')


@app.route('/get_sd_ckpts', methods=['POST'])
def get_sd_ckpts():
    directory = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'migc_gui_weights/sd')
    files = [f for f in os.listdir(directory) if f.endswith('.safetensors')]
    print(files)
    return jsonify(files)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=3344)
    args = parser.parse_args()
    # Start the Flask application and enable debugging mode
    app.run(debug=True, port=args.port)