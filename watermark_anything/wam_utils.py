from watermark_anything.models import Wam, build_embedder, build_extractor
from watermark_anything.augmentation.augmenter import Augmenter
from watermark_anything.data.transforms import default_transform, normalize_img, unnormalize_img
from watermark_anything.modules.jnd import JND

import json
import argparse
import omegaconf
import torch
import os

def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])

def str2msg(str):
    return [True if el=='1' else False for el in str]

def load_model_from_checkpoint(weight_path, num_bits):
    """
    Load a model from a checkpoint file and a JSON file containing the parameters.
    Args:
    - json_path (str): the path to the JSON file containing the parameters
    - ckpt_path (str): the path to the checkpoint file
    """
    json_path = os.path.join(weight_path, "params.json")
    ckpt_path = os.path.join(weight_path, "wam_coco.pth")

    # Load the JSON file
    with open(json_path, 'r') as file:
        params = json.load(file)
    # Create an argparse Namespace object from the parameters
    args = argparse.Namespace(**params)
    # print(args)
    
    # TODO: num_bits is not changed in during inference

    # Load configurations
    embedder_cfg = omegaconf.OmegaConf.load(os.path.join(weight_path, args.embedder_config))
    embedder_params = embedder_cfg[args.embedder_model]
    extractor_cfg = omegaconf.OmegaConf.load(os.path.join(weight_path, args.extractor_config))
    extractor_params = extractor_cfg[args.extractor_model]
    augmenter_cfg = omegaconf.OmegaConf.load(os.path.join(weight_path, args.augmentation_config))
    attenuation_cfg = omegaconf.OmegaConf.load(os.path.join(weight_path, args.attenuation_config))

    # Build models
    embedder = build_embedder(args.embedder_model, embedder_params, args.nbits)
    extractor = build_extractor(extractor_cfg.model, extractor_params, args.img_size, args.nbits)
    augmenter = Augmenter(**augmenter_cfg)
    try:
        attenuation = JND(**attenuation_cfg[args.attenuation], preprocess=unnormalize_img, postprocess=normalize_img)
    except:
        attenuation = None
    
    # Build the complete model
    wam = Wam(embedder, extractor, augmenter, attenuation, args.scaling_w, args.scaling_i)
    
    # Load the model weights
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        wam.load_state_dict(checkpoint)
        print("Model loaded successfully from", ckpt_path)
        print(params)
    else:
        print("Checkpoint path does not exist:", ckpt_path)
    
    return wam