
# [ CVPR2024 Highlight ] MIGC: Multi-Instance Generation Controller for Text-to-Image Synthesis
# [ TPAMI2024 ] MIGC++: Advanced Multi-Instance Generation Controller for Image Synthesis

**COCO-MIG Bench:**  [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/migc-multi-instance-generation-controller-for/conditional-text-to-image-synthesis-on-coco-1)](https://paperswithcode.com/sota/conditional-text-to-image-synthesis-on-coco-1?p=migc-multi-instance-generation-controller-for)


**Online Demo on Colab:** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1rkhi7EylHXACbzfXvWiblM4m1BCGOX5-?usp=sharing)
### [[MIGC Paper]](https://openaccess.thecvf.com/content/CVPR2024/papers/Zhou_MIGC_Multi-Instance_Generation_Controller_for_Text-to-Image_Synthesis_CVPR_2024_paper.pdf)  [[MIGC++ Paper]](https://ieeexplore.ieee.org/document/10794618)    [[Project Page]](https://migcproject.github.io/)    [[ZhiHu(知乎)]](https://zhuanlan.zhihu.com/p/686367982)

## 🔥🔥🔥 News 

- 2024-07-03: Iterative editing mode "Consistent-MIG" in [MIGC++](https://ieeexplore.ieee.org/document/10794618) is **available**!
- 2024-11-24: Our paper ["MIGC++: Advanced Multi-Instance Generation Controller for Image Synthesis"](https://ieeexplore.ieee.org/document/10794618) has been accepted by TPAMI.
- 2024-12-23: We have released the [pretrained weights of MIGC++](https://ieeexplore.ieee.org/document/10794618), which can **simultaneously use masks and boxes to specify instance locations.**

![Demo2](videos/video2.gif)

## To Do List
- [x] Project Page
- [x] Code
- [x] COCO-MIG Benchmark
- [x] Pretrained Weights on SD1.4
- [x] WebUI
- [x] Colab Demo
- [x] Consistent-MIG algorithm in [MIGC++](https://www.computer.org/csdl/journal/tp/5555/01/10794618/22AQoBwTa4U)
- [x] Pretrained Weights of MIGC++ for Simultaneous Box and Mask Control
- [ ] Pretrained Weights of MIGC++ for Simultaneous Image and Text Control (coming soon)
- [ ] Pretrained Weights on SD1.5, SD2, SDXL (Note that MIGC_SD14.ckpt can be used directly for the SD1.5 model.)
<a id="Gallery"></a>
## Gallery
![attr_control](figures/attr_control.png)
![quantity_control](figures/quantity_control.png)
![animation_creation](figures/animation_creation.png)


<a id="Installation"></a>
## Installation

### Conda environment setup
```
conda create -n MIGC_diffusers python=3.9 -y
conda activate MIGC_diffusers
pip install -r requirement.txt
pip install -e .
```
### Checkpoints
Download the [MIGC_SD14.ckpt (219M)](https://drive.google.com/file/d/107fnQ9Fpu5K0UtqnHlKja7hqe-GwjAmz/view?usp=sharing) and put it under the 'pretrained_weights' folder.
```
├── pretrained_weights
│   ├── MIGC_SD14.ckpt
├── migc
│   ├── ...
├── bench_file
│   ├── ...
```
If you want to use MIGC++, please download the [MIGC++_SD14.ckpt (191M)](https://drive.google.com/file/d/1KI8Ih7SHISG9v9zRL1xhDIBsPjDHqPxI/view?usp=drive_link) and put it under the 'pretrained_weights' folder.
Note: Due to our collaborator's request, I can't release the original weights. These are re-implemented weights, trained with a smaller batch size.
```
├── pretrained_weights
│   ├── MIGC++_SD14.ckpt
├── migc
│   ├── ...
├── bench_file
│   ├── ...
```
## Single Image Generation
By using the following command, you can quickly generate an image with **MIGC**.
```
CUDA_VISIBLE_DEVICES=0 python inference_single_image.py
```
The following is an example of the generated image based on stable diffusion v1.4.
 
<p align="center">
  <img src="figures/MIGC_SD14_out.png" alt="example" width="200" height="200"/>
  <img src="figures/MIGC_SD14_out_anno.png" alt="example_annotation" width="200" height="200"/>
</p>

By using the following command, you can quickly generate an image with **MIGC++**, where both the box and mask are used to control the instance location.
```
CUDA_VISIBLE_DEVICES=0 python migc_plus_inference_single_image.py
```
The following are examples of the generated images using MIGC++.

<p align="center">
  <img src="figures/migc++_output.png" alt="example" width="1000" height="300"/>
</p>

🚀 **Enhanced Attribute Control**: For those seeking finer control over attribute management, consider exploring the `python inferencev2_single_image.py` script. This advanced version, `InferenceV2`, offers a significant improvement in mitigating attribute leakage issues. By accepting a slight increase in inference time, it enhances the Instance Success Ratio from 66% to an impressive 68% on COCO-MIG Benchmark. It is worth mentioning that increasing the `NaiveFuserSteps` in `inferencev2_single_image.py` can also gain stronger attribute control.

<p align="center">
  <img src="figures/infer_v2_demo.png" alt="example" width="700" height="300"/>
</p>

💡 **Versatile Image Generation**: MIGC stands out as a plug-and-play controller, enabling the creation of images with unparalleled variety and quality. By simply swapping out different base generator weights, you can achieve results akin to those showcased in our [Gallery](#Gallery). For instance:

- 🌆 **[RV60B1](https://civitai.com/models/4201/realistic-vision-v60-b1)**: Ideal for those seeking lifelike detail, RV60B1 specializes in generating images with stunning realism.
- 🎨 **[Cetus-Mix](https://civitai.com/models/6755/cetus-mix)** and **[Ghost](https://civitai.com/models/36520/ghostmix)**: These robust base models excel in crafting animated content.
<p align="center">
  <img src="figures/diverse_base_model.png" alt="example" width="1000" height="230"/>
</p>

**[New] 🌈 Iterative Editing Mode**: The [Consistent-MIG](https://arxiv.org/pdf/2407.02329) algorithm improves the iterative MIG capabilities of MIGC facilitating modifying certain instances in MIG while preserving consistency in unmodified regions and maximizing the ID consistency of modified instances. You can explore the `python inference_consistent_mig.py` script to know the usage. For instance:

<p align="center">
  <img src="figures/consistent-mig.jpg" alt="example"  />
</p>

## Training
Due to company requirements, we are unable to open the MIGC training code. For now, the best we can do is to provide the community with the script we use to process the COCO dataset data (i.e., obtaining each instance's box and caption). The relevant code is placed in the 'data_preparation' folder. If there are any changes in the future, such as if they grant permission, we will make it open source.


## COCO-MIG Bench

To validate the model's performance in position and attribute control, we designed the [COCO-MIG](https://github.com/LeyRio/MIG_Bench) benchmark for evaluation and validation.

By using the following command, you can quickly run inference on our method on the COCO-MIG bench:
```
CUDA_VISIBLE_DEVICES=0 python inference_mig_benchmark.py
```
We sampled [800 images](https://drive.google.com/drive/folders/1UyhNpZ099OTPy5ILho2cmWkiOH2j-FrB?usp=sharing) and compared MIGC with InstanceDiffusion, GLIGEN, etc. On COCO-MIG Benchmark, the results are shown below.

<table style="text-align: center;">
  <thead>
    <tr>
      <th rowspan="2" style="text-align: center;">Method</th>
      <th colspan="6" style="text-align: center;">MIOU↑</th>
      <th colspan="6" style="text-align: center;">Instance Success Rate↑</th>
	  <th rowspan="2" style="text-align: center;">Model Type</th>
    <th rowspan="2" style="text-align: center;">Publication</th>
    </tr>
	<tr>
      <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
	  <th>L2</th>
      <th>L3</th>
      <th>L4</th>
      <th>L5</th>
      <th>L6</th>
      <th>Avg</th>
    </tr>
  </thead>
  <tbody>
	<tr>
      <td><a href="https://github.com/showlab/BoxDiff">Box-Diffusion</a></td>
      <td>0.37</td>
      <td>0.33</td>
      <td>0.25</td>
      <td>0.23</td>
      <td>0.23</td>
      <td>0.26</td>
	  <td>0.28</td>
      <td>0.24</td>
      <td>0.14</td>
      <td>0.12</td>
      <td>0.13</td>
      <td>0.16</td>
	  <td>Training-free</td>
    <td>ICCV2023</td>
    </tr>
	<tr>
      <td><a href="https://github.com/gligen/GLIGEN">Gligen</a></td>
      <td>0.37</td>
      <td>0.29</td>
      <td>0.253</td>
      <td>0.26</td>
      <td>0.26</td>
      <td>0.27</td>
	<td>0.42</td>
      <td>0.32</td>
      <td>0.27</td>
      <td>0.27</td>
      <td>0.28</td>
      <td>0.30</td>
	  <td>Adapter</td>
    <td>CVPR2023</td>
    </tr>
	<tr>
      <td><a href="https://github.com/microsoft/ReCo">ReCo</a></td>
      <td>0.55</td>
      <td>0.48</td>
      <td>0.49</td>
      <td>0.47</td>
      <td>0.49</td>
      <td>0.49</td>
	  <td>0.63</td>
      <td>0.53</td>
      <td>0.55</td>
      <td>0.52</td>
      <td>0.55</td>
      <td>0.55</td>
	  <td>Full model tuning</td>
    <td>CVPR2023</td>
    </tr>
	<tr>
      <td><a href="https://github.com/frank-xwang/InstanceDiffusion">InstanceDiffusion</a></td>
      <td>0.52</td>
      <td>0.48</td>
      <td>0.50</td>
      <td>0.42</td>
      <td>0.42</td>
      <td>0.46</td>
	  <td>0.58</td>
      <td>0.52</td>
      <td>0.55</td>
      <td>0.47</td>
      <td>0.47</td>
      <td>0.51</td>
	  <td>Adapter</td>
    <td>CVPR2024</td>
    </tr>
	<tr>
      <td><a href="https://github.com/limuloo/MIGC">Ours</a></td>
      <td><b>0.64</b></td>
      <td><b>0.58</b></td>
      <td><b>0.57</b></td>
      <td><b>0.54</b></td>
      <td><b>0.57</b></td>
      <td><b>0.56</b></td>
	  <td><b>0.74</b></td>
      <td><b>0.67</b></td>
      <td><b>0.67</b></td>
      <td><b>0.63</b></td>
      <td><b>0.66</b></td>
      <td><b>0.66</b></td>
	  <td>Adapter</td>
    <td>CVPR2024</td>
    </tr>
  </tbody>
</table>



## MIGC-GUI
We have combined MIGC and [GLIGEN-GUI](https://github.com/mut-ex/gligen-gui) to make art creation more convenient for users. 🔔This GUI is still being optimized. If you have any questions or suggestions, please contact me at zdw1999@zju.edu.cn.

![Demo1](videos/video1.gif)

### Start with MIGC-GUI
**Step 1**: Download the [MIGC_SD14.ckpt](https://drive.google.com/file/d/1v5ik-94qlfKuCx-Cv1EfEkxNBygtsz0T/view?usp=drive_link) and place it in `pretrained_weights/MIGC_SD14.ckpt`. 🚨If you have already completed this step during the [Installation](#Installation) phase, feel free to skip it.

**Step 2**: Download the [CLIPTextModel](https://drive.google.com/file/d/1Z_BFepTXMbe-cib7Lla5A224XXE1mBcS/view?usp=sharing) and place it in `migc_gui_weights/clip/text_encoder/pytorch_model.bin`.

**Step 3**: Download the [CetusMix](https://drive.google.com/file/d/1cmdif24erg3Pph3zIZaUoaSzqVEuEfYM/view?usp=sharing) model and place it in `migc_gui_weights/sd/cetusMix_Whalefall2.safetensors`. Alternatively, you can visit [civitai](https://civitai.com/) to download other models of your preference and place them in `migc_gui_weights/sd/`.

```
├── pretrained_weights
│   ├── MIGC_SD14.ckpt
├── migc_gui_weights
│   ├── sd
│   │   ├── cetusMix_Whalefall2.safetensors
│   ├── clip
│   │   ├── text_encoder
│   │   │   ├── pytorch_model.bin
├── migc_gui
│   ├── app.py
```

**Step 4**: `cd migc_gui`

**Step 5**: Launch the application by running `python app.py --port=3344`. You can now access the MIGC GUI through http://localhost:3344/. Feel free to switch the port as per your convenience.

## Consistent-MIG in MIGC-GUI

![Demo2](videos/video2.gif)

<p align="center">
  <img src="figures/edit_button.jpg" alt="example" style="width: 50%; height: auto;"/>
</p>

Tick the button `EditMode` in area `IMAGE DIMENSIONS` and try it!

## MIGC + LoRA
MIGC can achieve powerful attribute-and-position control capabilities while combining with LoRA. 🚀 We will integrate this function into MIGC-GUI in the future, so stay tuned! 🌟👀
<p align="center">
  <img src="figures/migc_lora_id.png" alt="migc_lora_id" width="190" height="300"/>
  <img src="figures/migc_lora.png" alt="migc_lora" width="190" height="300"/>
  <img src="figures/migc_lora_anno.png" alt="migc_lora_anno" width="190" height="300"/>
  <img src="figures/migc_lora_gui_creation.png" alt="migc_lora_gui_creation" width="580" height="300"/>
</p>

## Ethical Considerations
The broad spectrum of image creation possibilities offered by MIGC might present comparable ethical dilemmas to those encountered with numerous other methods of generating images from text.


## 🏫About us
Thank you for your interest in this project. The project is supervised by the ReLER Lab at Zhejiang University’s College of Computer Science and Technology and [HUAWEI](https://www.huawei.com/cn/). ReLER was established by [Yang Yi](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en), a Qiu Shi Distinguished Professor at Zhejiang University. Our dedicated team of contributors includes [Dewei Zhou](https://scholar.google.com/citations?hl=en&user=4C_OwWMAAAAJ), [You Li](https://scholar.google.com/citations?user=2lRNus0AAAAJ&hl=en&oi=sra), [Ji Xie](https://github.com/HorizonWind2004), [Fan Ma](https://scholar.google.com/citations?hl=en&user=FyglsaAAAAAJ), [Zongxin Yang](https://scholar.google.com/citations?hl=en&user=8IE0CfwAAAAJ), [Yi Yang](https://scholar.google.com/citations?user=RMSuNFwAAAAJ&hl=en).

## Contact us
If you have any questions, feel free to contact me via email zdw1999@zju.edu.cn 

## Acknowledgements
Our work is based on [stable diffusion](https://github.com/Stability-AI/StableDiffusion), [diffusers](https://github.com/huggingface/diffusers), [CLIP](https://github.com/openai/CLIP), and [GLIGEN-GUI](https://github.com/mut-ex/gligen-gui). We appreciate their outstanding contributions.

## Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```
@inproceedings{zhou2024migc,
  title={Migc: Multi-instance generation controller for text-to-image synthesis},
  author={Zhou, Dewei and Li, You and Ma, Fan and Zhang, Xiaoting and Yang, Yi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6818--6828},
  year={2024}
}

@article{zhou2024migc++,
  title={Migc++: Advanced multi-instance generation controller for image synthesis},
  author={Zhou, Dewei and Li, You and Ma, Fan and Yang, Zongxin and Yang, Yi},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
```
