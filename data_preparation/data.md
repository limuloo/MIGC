## Dataset
Our code is heavily based on [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything). We provide a pipeline for annotating data on individual text and images, which you can use to prepare your own data. To use this pipeline, you need to follow these steps:

### Download Grounded-SAM
You need to clone the [Grounded-SAM](https://github.com/IDEA-Research/Grounded-Segment-Anything) project repository.

### Install Environment
```
# Assuming you have already installed PyTorch.

python -m pip install -e segment_anything
pip install --no-build-isolation -e GroundingDINO
pip install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel stanza nltk inflect
```

### Prepare Model Weights
You should download the model weights of Grounding-DINO and SAM model.

Download the GroundingDINO checkpoint:
```
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
```
You shoule also download ViT-H SAM model in [SAM](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

When you automatically download the Stanza model, if you encounter a download error, you can manually download the model and use it offline.
The model can be downloaded offline from the following path, but make sure the current version aligns with the version of Stanza you have installed.
[stanza](https://huggingface.co/stanfordnlp/stanza-en/tree/main), and put the [resources json](https://github.com/stanfordnlp/stanza-resources) in the weight directory.
The weight file path looks like the following:
```
# In stanza-en directory:
├── en (Change the directory name from model to en)
│   ├── backward_charlm
│   ├── constituency
│   ├── coref
│   ├── default.zip
│   ├── depparse
│   ├── forward_charlm
│   ├── lemma
│   ├── mwt
│   ├── ner
│   ├── pos
│   ├── pretrain
│   ├── sentiment
│   └── tokenize
├── README.md
└── resources.json (Download from resources json)
```

### Usage on custom dataset
We provide a simple demo where you only need to provide an image and its corresponding caption.
The demo will segment the instances mentioned in the caption and annotate them with the corresponding bounding box and mask.
You can use this function with your custom dataset by simply building an additional pipeline that iterates through all images and captions. 
Combined with our function, it enables dataset annotation seamlessly.