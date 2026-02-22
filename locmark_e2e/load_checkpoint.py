import sys, os
sys.path.insert(0, '/root/watermark-anything')

import torch
import omegaconf
from watermark_anything.models import Wam, build_embedder, build_extractor
from watermark_anything.augmentation.augmenter import Augmenter

CONFIGS_DIR = '/root/watermark-anything/configs'

def load_locmark_checkpoint(weight_path: str, nbits: int = 0, img_size: int = 256) -> Wam:
    """
    Load a trained LocMark (E2E) checkpoint and return a Wam object.

    Args:
        weight_path: directory containing checkpoint.pth
        nbits:       message bits (default 0)
        img_size:    extractor input resolution (default 256)
    Returns:
        Wam object (caller is responsible for .cuda().eval())
    """
    embedder_cfg = omegaconf.OmegaConf.load(os.path.join(CONFIGS_DIR, 'locmark_embedder.yaml'))
    extractor_cfg = omegaconf.OmegaConf.load(os.path.join(CONFIGS_DIR, 'locmark_extractor.yaml'))
    augmenter_cfg = omegaconf.OmegaConf.load(os.path.join(CONFIGS_DIR, 'full_noaugs.yaml'))

    embedder_params = embedder_cfg[embedder_cfg.model]   # embedder_cfg.locmark
    extractor_params = extractor_cfg[extractor_cfg.model]  # extractor_cfg.locmark

    # Build extractor first (creates backbone internally)
    extractor = build_extractor('locmark', extractor_params, img_size, nbits)

    # Build embedder sharing backbone + anchor from extractor
    embedder = build_embedder(
        'locmark', embedder_params, nbits,
        backbone=extractor.backbone,
        anchor_vec=extractor.anchor_vec,
        feat_layer=extractor.feat_layer,
    )

    augmenter = Augmenter(**augmenter_cfg)

    wam = Wam(
        embedder, extractor, augmenter,
        attenuation=None,
        scaling_w=1.0,
        scaling_i=1.0,
        img_size_extractor=img_size,
    )

    ckpt_path = os.path.join(weight_path, 'checkpoint003.pth')
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # train.py saves: {'model': state_dict, 'epoch': ..., ...}
        state = ckpt['model'] if 'model' in ckpt else ckpt
        wam.load_state_dict(state, strict=False)
        print(f"[locmark_e2e] Loaded from {ckpt_path}")
    else:
        print(f"[locmark_e2e] WARNING: checkpoint not found at {ckpt_path}")

    return wam
