import cv2
import numpy as np
import supervision as sv

import torch
import torchvision

import os
import json

from groundingdino.util.inference import Model
from segment_anything import sam_model_registry, SamPredictor
from pycocotools import mask as mask_utils
import argparse
import time
from tqdm import tqdm
from PIL import Image
import inflect

engine = inflect.engine()

def plural_to_singular(word):
    singular = engine.singular_noun(word)
    return singular if singular else word

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        # np整数转为内置int
        if isinstance(obj, tuple):
            return list(int(obj[0]),int(obj[1]))
        elif isinstance(obj,np.int64):
            return int(obj)
        else:
            return json.JSONEncoder.default(self, obj)


import stanza
from nltk.tree import Tree


# If automaticly download stanza weights:
# stanza.download('en') 
# nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')

### !!!!!!! Here are some weight paths listed. You need to replace them with the weight paths on your local machine.
# If using offline downloaded Stanza weights
model_dir = '/path/to/stanza-en'  # Path to you stanza-en weights
nlp = stanza.Pipeline(lang='en', model_dir = model_dir, processors='tokenize,pos,constituency',download_method=None)
dino_path = "/path/to/groundingdino_swint_ogc.pth"
sam_path = "/path/to/sam_vit_h_4b8939.pth"

def get_all_nps(tree, full_sent, tokens=None, highest_only=False, lowest_only=False):
    # import pdb;pdb.set_trace()
    start = 0
    end = len(tree.leaves())

    idx_map = get_token_alignment_map(tree, tokens)

    def get_sub_nps(tree, left, right):
        if isinstance(tree, str) or len(tree.leaves()) == 1:
            if not isinstance(tree, str) and tree.label() == 'NP':
                for i,subtree in enumerate(tree):
                    if subtree.label() == 'NN' or subtree.label() =='NNS':
                        return [[tree.leaves()[0],(left,right)]]
                    else:
                        return []
            else:
                return []
        sub_nps = []
        n_leaves = len(tree.leaves())
        n_subtree_leaves = [len(t.leaves()) for t in tree]
        offset = np.cumsum([0] + n_subtree_leaves)[:len(n_subtree_leaves)]
        assert right - left == n_leaves
        # If current node is Noun phrases or PP
        if (tree.label() == 'NP' or tree.label() == 'PP') and n_leaves > 1:
            cc_count = 0
            nn_count = 0
            nn_flag = False
            for i, subtree in enumerate(tree):
                # Find 'and' conjunctions
                if subtree.label() == 'CC':
                    cc_count = cc_count + 1
                    nn_flag = False
                elif (subtree.label() == 'NN' or subtree.label()=='NNS') and nn_flag == False:
                    nn_count = nn_count + 1
                    nn_flag = False
                else:
                    nn_flag = False
            # If has and such as:  a red dog and a blue cat
            if cc_count >0 and nn_count > 1:
                nn_flag = False
                prompt_list = ['a']
                start = 0
                end = 0
                # print(left,right)
                for i,subtree in enumerate(tree):
                    if subtree.label() == 'NN' or subtree.label() == 'NNS':
                        if nn_flag == False:
                            # print(f'subtree: {subtree},{i}')
                            prompt_list.append(plural_to_singular(tree.leaves()[i]))
                            # prompt = "" + 
                            nn_flag = True
                            start = left + i
                        elif nn_flag == True:
                            prompt_list.append(plural_to_singular(tree.leaves()[i]))
                            # prompt = prompt + ' ' + tree.leaves()[i]
                    else:
                        if nn_flag == True:
                            nn_flag = False
                            end = left + i
                            sub_nps.append([' '.join(prompt_list),(start,end)])
                            prompt_list = ['a']
                            
                    if i==len(tree.leaves())-1 and nn_flag == True:
                        end = left + i + 1
                        sub_nps.append([' '.join(prompt_list),(start,end)])
            else:
                prompt_list = ['a']
                start = -1
                end = -1
                for i,subtree in enumerate(tree):
                    if subtree.label() == 'NN' or subtree.label() == 'NNS' or subtree.label() == 'JJ':
                        prompt_list.append(plural_to_singular(tree.leaves()[i]))
                        if start < 0:
                            start = left + i
                    else:
                        if start > 0:
                            end = left + i + 1
                            sub_nps.append([' '.join(prompt_list),(start,end)])
                    if i==len(tree.leaves())-1 and end < 0:
                        end = left + i + 1
                        sub_nps.append([' '.join(prompt_list),(start,end)])
                    
                # sub_nps.append([" ".join(tree.leaves()), (int(min(idx_map[left])), int(min(idx_map[right])))])
            if highest_only and sub_nps[-1][0] != full_sent: return sub_nps
        for i, subtree in enumerate(tree):
            sub_nps += get_sub_nps(subtree, left=left+offset[i], right=left+offset[i]+n_subtree_leaves[i])
        return sub_nps
    
    all_nps = get_sub_nps(tree, left=start, right=end)
    lowest_nps = []
    for i in range(len(all_nps)):
        span = all_nps[i][1]
        lowest = True
        for j in range(len(all_nps)):
            if i == j: continue
            span2 = all_nps[j][1]
            if span2[0] >= span[0] and span2[1] <= span[1]:
                lowest = False
                break
        if lowest:
            lowest_nps.append(all_nps[i])

    if lowest_only:
        all_nps = lowest_nps

    if len(all_nps) == 0:
        all_nps = []
        spans = []
    else:
        all_nps, spans = map(list, zip(*all_nps))
    if full_sent not in all_nps:
        all_nps = [full_sent] + all_nps
        spans = [(min(idx_map[start]), min(idx_map[end]))] + spans

    return all_nps, spans, lowest_nps

def get_token_alignment_map(tree, tokens):
    if tokens is None:
        return {i:[i] for i in range(len(tree.leaves())+1)}
        
    def get_token(token):
        return token[:-4] if token.endswith("</w>") else token

    idx_map = {}
    j = 0
    max_offset = np.abs(len(tokens) - len(tree.leaves()))
    mytree_prev_leaf = ""
    # print(tree)
    for i, w in enumerate(tree.leaves()):
        token = get_token(tokens[j])
        idx_map[i] = [j]
        if token == mytree_prev_leaf+w:
            mytree_prev_leaf = ""
            j += 1
        else:
            if len(token) < len(w):
                prev = ""
                while prev + token != w:
                    prev += token
                    j += 1
                    print(j)
                    token = get_token(tokens[j])
                    idx_map[i].append(j)
                    # assert j - i <= max_offset
            else:
                mytree_prev_leaf += w
                j -= 1
            j += 1
    idx_map[i+1] = [j]
    return idx_map

def load_model(dino_path, sam_path):
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = dino_path

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = sam_path

    # Building GroundingDINO inference model
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

    # Building SAM Model and SAM Predictor
    sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device = 'cuda')
    sam_predictor = SamPredictor(sam)
    
    return grounding_dino_model, sam_predictor

def segment_from_file(jpg_dir = None, caption_path = None,rank_idx = None):
    # jpg_name = [fn for fn in os.listdir(jpg_dir) if fn.endswith('.jpg')]
    model, sam_model = load_model(dino_path, sam_path)
    json_name = caption_path

    with open(json_name,'r') as json_file:
        content = json.load(json_file)
        
    result = {}
    countsss = 0
    image_list = list(content.keys())
    for jpg_name in tqdm(image_list):
        caption = content[jpg_name]['caption']
        image_path = os.path.join(jpg_dir, jpg_name)
        segment_label = {'caption':caption,'annotation':{},'origin_caption':content[jpg_name]['origin_caption']}
        segment_label['annotation'] = annotation_on_caption(caption, image_path, model, sam_model)
        if jpg_name in result.keys():
            result[jpg_name].append(segment_label)
        else:
            result[jpg_name] = []
            result[jpg_name].append(segment_label)

    output_name = f'./result.json'
    with open(output_name,'w') as outs:
        json.dump(result,outs,cls = MyEncoder)

def annotation_on_caption(caption, image_path, grounding_dino_model, sam_predictor):
    
    SOURCE_IMAGE_PATH = image_path
    doc = nlp(caption)
    mytree = Tree.fromstring(str(doc.sentences[0].constituency))
    _,_,nps = get_all_nps(mytree, caption, None)
    
    now_seg = 0
    image = cv2.imread(SOURCE_IMAGE_PATH)
    image = cv2.resize(image,(512,512))
    
    seg_instance = {}
    
    for class_name in nps:
        CLASSES = [class_name[0]]  
        BOX_THRESHOLD = 0.25
        TEXT_THRESHOLD = 0.25
        NMS_THRESHOLD = 0.8

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=CLASSES,
            box_threshold=BOX_THRESHOLD,
            text_threshold=BOX_THRESHOLD
        )
        

        nms_idx = torchvision.ops.nms(
            torch.from_numpy(detections.xyxy), 
            torch.from_numpy(detections.confidence), 
            NMS_THRESHOLD
        ).numpy().tolist()

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]
        # print(detections)

        def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
            sam_predictor.set_image(image)
            result_masks = []
            for box in xyxy:
                masks, scores, logits = sam_predictor.predict(
                    box=box,
                    multimask_output=True
                )
                index = np.argmax(scores)
                maskk = np.asfortranarray(masks[index])
                maskk = mask_utils.encode(maskk)
                maskk['counts'] = maskk['counts'].decode('utf-8')
                result_masks.append(maskk)
            return result_masks

        # convert detections to masks
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        class_id = detections.class_id.tolist()
        labels = [CLASSES[_] for _ in class_id]
        # import pdb;pdb.set_trace()
        mask = mask_utils.decode(detections.mask)

        if len(labels) > 0:
            seg_instance[now_seg] = {'segmentation':detections.mask,'labels':labels,'token_range':class_name[1],'bbox':detections.xyxy.tolist()}
            now_seg = now_seg + 1
    # output:
    """
    {
        '0': {  Range Id of the instance
            'segmentation':  SAM mask
            'labels': instance label, such as: a red boat
            'token_range': index in the origin caption
            'bbox': bounding box of GroundingDINO, [x1, y1, x2, y2], Note that the coordinates are not normalized.
        }
    }
    """        
    
    return seg_instance

if __name__ == '__main__':
    
    model, sam_model = load_model(dino_path, sam_path)
    ###### Examples for annotate on single image #################
    caption = 'Three boads on the lake.'
    image_path = './test_2.jpg'
    annotation_instance = annotation_on_caption(caption, image_path, model, sam_model)
    
    ###### Visualize ###############
    # This part is only for visualizing the annotation results. Comment it out when processing data on a large scale. 
    for _ in annotation_instance.keys():
        mask = mask_utils.decode(annotation_instance[_]['segmentation'])
        label = annotation_instance[_]['labels'][0]
        for idx in range(mask.shape[2]):
            mask_image = Image.fromarray(mask[:,:,idx] * 255, mode='L')
            mask_image.save(f'mask_image_{label}_{idx}.png')

