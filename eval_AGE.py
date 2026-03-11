import os
import warnings

import piq

from watermark_anything.modules import common
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm
import cv2
import numpy as np
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline, AutoPipelineForInpainting, FluxFillPipeline
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms, ToTensor
from piq import ssim, psnr, LPIPS
import sys
import json
from transformers import T5EncoderModel

from evaluation import PixelF1, PixelAUC, PixelIOU, PixelAccuracy
from dataset import age_collate_fn, AGEDataset
import random
from stableguard.models import MultiplexingWatermarkVAEDecoder, MoEGuidedForensicNet
import yaml
import torch.nn.functional as F
import albumentations as albu
from watermark_anything.wam_utils import load_model_from_checkpoint
from watermark_anything.data.metrics import msg_predict_inference
from watermark_anything.data.transforms import normalize_img, unnormalize_img
from omniguard.model_invert import Model, init_model
from omniguard.modules.Unet_common import DWT, IWT
from omniguard.iml_vit_model import iml_vit_model
from albumentations.pytorch import ToTensorV2
from locmark.locmark import LocMark
from locmark.main import Params
from omegaconf import OmegaConf
from evaluation.augmentation import get_robustness_transform
from diffusers import StableDiffusionControlNetInpaintPipeline, ControlNetModel, DDIMScheduler, StableDiffusionInpaintPipelineLegacy

sys.path.append(os.path.join(os.path.dirname(__file__), "HD-Painter"))

from src import models
from src.methods import rasg, sd, sr
from src.utils import IImage, resize

image_mean = torch.tensor([0.485, 0.456, 0.406])
image_std = torch.tensor([0.229, 0.224, 0.225])
denorm_imagenet = transforms.Normalize(-image_mean / image_std, 1 / image_std)

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def denormalize_tensor(tensor, mode='minmax', value_range=None, scale_each=True):
    """
    Normalize tensor values into [0,1] (in-place style but returns a cloned tensor).
    This helper is used before feeding images to image-quality metrics such as PSNR/SSIM/LPIPS.

    Args:
        tensor (torch.Tensor): input tensor with shape (B, C, H, W) or (C, H, W).
        value_range (tuple or None): (min, max) range to clamp before scaling. If None, uses tensor-specific min/max.
        scale_each (bool): if True, normalize every image in the batch separately; otherwise normalize whole tensor together.

    Returns:
        torch.Tensor: cloned and normalized tensor in range [0,1].
    """
    tensor = tensor.clone()  # avoid modifying caller's tensor

    # Validate value_range argument
    if value_range is not None and not isinstance(value_range, tuple):
        raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

    # In-place clamp/shift/scale helper
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    # Choose whether to use provided range or tensor's min/max
    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    # Normalize each image separately or the whole tensor
    if scale_each is True:
        for t in tensor:  # iterate over batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)

    return tensor

def convert_mask_to_rect(masks, scale_factor=1.0):
    """
    Converts arbitrary masks into bounding box rectangles and scales them up by the given factor.
    
    Args:
        masks: (B, 1, H, W) shape tensor
        scale_factor: Expansion ratio (e.g., 1.1, 1.2)
    """
    B, C, H, W = masks.shape
    rect_masks = torch.zeros_like(masks)

    for i in range(B):
        mask_indices = torch.nonzero(masks[i, 0] > 0.5)

        if mask_indices.numel() == 0:
            continue

        y_min = mask_indices[:, 0].min().item()
        y_max = mask_indices[:, 0].max().item()
        x_min = mask_indices[:, 1].min().item()
        x_max = mask_indices[:, 1].max().item()
        h = y_max - y_min
        w = x_max - x_min
        cy = y_min + h / 2
        cx = x_min + w / 2

        new_h = h * scale_factor
        new_w = w * scale_factor

        y1 = int(max(0, cy - new_h / 2))
        y2 = int(min(H, cy + new_h / 2))
        x1 = int(max(0, cx - new_w / 2))
        x2 = int(min(W, cx + new_w / 2))

        rect_masks[i, 0, y1:y2, x1:x2] = 1.0

    return rect_masks

def make_inpaint_condition(image, image_mask):
    '''
    Args:
        image (torch.Tensor): Input image tensor. [0, 1] or [-1, 1]
        image_mask (torch.Tensor): Binary mask tensor. [0, 1], manipulate: -1
    '''
    image = image.clone()
    # Controlnet: bg [0, 1] and mask=-1
    if image.min() < 0: # if input is in [-1, 1]->[0, 1]
        image = (image + 1.0) / 2.0
    mask_binary = image_mask > 0.5
    image[mask_binary.expand_as(image)] = -1.0 # manipulate regions: -1
    return image

def get_inpainting_function(
    model_id: str,
    method: str,
    negative_prompt: str = '',
    positive_prompt: str = '',
    num_steps: int = 50,
    eta: float = 0.25,
    guidance_scale: float = 7.5,
    cache_dir: str = None
):
    inp_model = models.load_inpainting_model(model_id, device='cuda', cache=True, cache_dir=cache_dir)
    
    if 'rasg' in method:
        runner = rasg
    else:
        runner = sd
    
    def run(image: Image, mask: Image, prompt: str, seed: int = 1) -> Image:
        inpainted_image = runner.run(
            ddim=inp_model,
            method=method,
            prompt=prompt,
            image=IImage(image),
            mask=IImage(mask),
            seed=seed,
            eta=eta,
            negative_prompt=negative_prompt,
            positive_prompt=positive_prompt,
            num_steps=num_steps,
            guidance_scale=guidance_scale
        ).pil()
        w, h = image.size
        inpainted_image = Image.fromarray(np.array(inpainted_image)[:h, :w])
        return inpainted_image
    return run

class Evaluation(object):
    """
    Simple evaluation helper that reads predicted mask images from disk,
    compares them to ground-truth masks and accumulates pixel-level metrics.

    This class relies on PixelF1, PixelAUC, PixelIOU, PixelAccuracy interfaces
    which expose batch_update(predict=..., mask=...) and Cal_FPR(...).
    """

    def __init__(self, pred_path, gt_path, eval_size, end_idx=None) -> None:
        """
        Args:
            pred_path (str): directory containing predicted mask images/files.
            gt_path (str): directory containing ground-truth mask images/files.
            end_idx (int or None): if set, evaluate only the first end_idx sorted files.
        """
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.eval_size = eval_size
        self.end_idx = end_idx
        # instantiate metric calculators
        self.f1 = PixelF1()
        self.auc = PixelAUC()
        self.iou = PixelIOU()
        self.acc = PixelAccuracy()

    def run(self, save_path, tamper_mode):
        """
        Iterate over predicted masks found in self.pred_path, load the corresponding ground-truth,
        compute per-file pixel metrics, accumulate them, print a summary and append to a record file.

        Args:
            save_path (str): directory in which to write record.txt (appends).
        """
        total_f1 = []
        total_auc = []
        total_iou = []
        total_acc = []
        total_fpr = []
        total_pos_acc = []
        total_neg_acc = []
        total_b_acc = []

        # list prediction files in prediction directory (sorted for determinism)
        pred_images_path = sorted(os.listdir(self.pred_path))
        if self.end_idx is not None:
            pred_images_path = pred_images_path[:self.end_idx]

        # iterate through each predicted mask filename
        for pred_image_path in tqdm(pred_images_path):
            try:
                # open predicted mask and corresponding ground-truth mask as grayscale
                pred_image = Image.open(os.path.join(self.pred_path, pred_image_path)).convert("L")

                if tamper_mode == 'zero_mask':
                    gt_image = Image.new("L", pred_image.size, color=256)
                else:
                    gt_image = Image.open(os.path.join(self.gt_path, pred_image_path)).convert("L")

                pred_image = pred_image.resize((self.eval_size, self.eval_size))
                gt_image = gt_image.resize((self.eval_size, self.eval_size))
            except Exception:
                # skip files that cannot be opened / matched
                continue

            # convert PIL images to tensors with shape (1, C, H, W)
            pred_tensor = T.ToTensor()(pred_image).unsqueeze(0)
            gt_tensor = T.ToTensor()(gt_image).unsqueeze(0)

            # resize ground-truth to match predicted mask spatial size if needed
            # invert gt mask: watermarked regions are white (1)
            gt_tensor = torch.nn.functional.interpolate(gt_tensor, (pred_tensor.size(2), pred_tensor.size(3)))

            # call evaluator batch_update for each metric (these update internal state or return stat)
            f1 = self.f1.batch_update(predict=pred_tensor, mask=gt_tensor)
            auc = self.auc.batch_update(predict=pred_tensor, mask=gt_tensor)
            iou = self.iou.batch_update(predict=pred_tensor, mask=gt_tensor)
            acc = self.acc.batch_update(predict=pred_tensor, mask=gt_tensor)
            fpr = self.f1.Cal_FPR(predict=pred_tensor, mask=gt_tensor)

            pred_bin = (pred_tensor > 0.5).float()
            gt_bin = (gt_tensor > 0.5).float()

            tp = (pred_bin * gt_bin).sum()
            tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
            fp = (pred_bin * (1 - gt_bin)).sum()
            fn = ((1 - pred_bin) * gt_bin).sum()

            pos_acc = tp / (tp + fn + 1e-8)
            neg_acc = tn / (tn + fp + 1e-8)
            b_acc = (pos_acc + neg_acc) / 2.0

            # append scalars to lists, skipping NaNs
            if not torch.any(torch.isnan(f1)):
                total_f1.append(f1.item())
            if not torch.any(torch.isnan(auc)):
                total_auc.append(auc.item())
            if not torch.any(torch.isnan(iou)):
                total_iou.append(iou.item())
            if not torch.any(torch.isnan(acc)):
                total_acc.append(acc.item())
            if not torch.any(torch.isnan(fpr)):
                total_fpr.append(fpr.item())

            total_pos_acc.append(pos_acc.item())
            total_neg_acc.append(neg_acc.item())
            total_b_acc.append(b_acc.item())

        # compute aggregated means and print/write a summary line
        n_images = len(total_f1)
        msg = (
            f"N:{n_images}, "
            f"Tampering F1:{np.mean(total_f1):.5f}, AUC:{np.mean(total_auc):.5f}, "
            f"IoU:{np.mean(total_iou):.5f}, Acc:{np.mean(total_acc):.5f}, FPR:{np.mean(total_fpr):.5f}, "
            f"Pos_Acc:{np.mean(total_pos_acc):.5f}, Neg_Acc:{np.mean(total_neg_acc):.5f}, "
            f"B_Acc:{np.mean(total_b_acc):.5f}\n"
        )
        print(msg)

        # append to record file under save_path
        with open(os.path.join(save_path, "record.txt"), "a+") as f:
            f.write(msg)

    def run_blind(self, spliceless_pred_dir: str, save_path: str):
        """
        Blind AUC: pool pixel predictions from self.pred_path (spliced) and
        spliceless_pred_dir (spliceless) with a single global threshold.

        GT for spliced: loaded from self.gt_path (partial mask, 1=tampered).
        GT for spliceless: all-ones (entire image was regenerated).

        Args:
            spliceless_pred_dir: path to pred_mask_{mode}_spliceless directory
            save_path: directory where record.txt lives
        """
        from sklearn.metrics import roc_auc_score

        all_preds, all_labels = [], []

        # --- Spliced: GT from self.gt_path ---
        spliced_files = sorted(os.listdir(self.pred_path))
        if self.end_idx is not None:
            spliced_files = spliced_files[:self.end_idx]
        for fname in tqdm(spliced_files, desc="Blind AUC (spliced)"):
            try:
                pred = Image.open(os.path.join(self.pred_path, fname)).convert("L")
                gt   = Image.open(os.path.join(self.gt_path, fname)).convert("L")
            except Exception:
                continue
            pred = pred.resize((self.eval_size, self.eval_size), resample=Image.BILINEAR)
            gt   = gt.resize((self.eval_size, self.eval_size), resample=Image.NEAREST)
            pred_arr = np.array(pred, dtype=np.float32) / 255.0
            gt_arr   = (np.array(gt,   dtype=np.float32) / 255.0 > 0.5).astype(np.float32)
            all_preds.append(pred_arr.flatten())
            all_labels.append(gt_arr.flatten())

        # --- Spliceless: GT is all-ones (entire image was regenerated) ---
        spliceless_files = sorted(os.listdir(spliceless_pred_dir))
        if self.end_idx is not None:
            spliceless_files = spliceless_files[:self.end_idx]
        for fname in tqdm(spliceless_files, desc="Blind AUC (spliceless)"):
            try:
                pred = Image.open(os.path.join(spliceless_pred_dir, fname)).convert("L")
            except Exception:
                continue
            pred = pred.resize((self.eval_size, self.eval_size))
            pred_arr = np.array(pred, dtype=np.float32) / 255.0
            all_preds.append(pred_arr.flatten())
            all_labels.append(np.ones(self.eval_size * self.eval_size, dtype=np.float32))

        all_preds  = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)

        n_pos = int(all_labels.sum())
        n_neg = len(all_labels) - n_pos
        if n_pos == 0 or n_neg == 0:
            msg = f"Blind AUC: UNDEFINED (pos={n_pos}, neg={n_neg})\n"
        else:
            auc = roc_auc_score(all_labels, all_preds)
            n_images = len(spliced_files) + len(spliceless_files)
            msg = (
                f"Blind AUC: {auc:.5f} "
                f"({n_images} images: {len(spliced_files)} spliced + {len(spliceless_files)} spliceless)\n"
            )

        print(msg)
        with open(os.path.join(save_path, "record_blind.txt"), "a+") as f:
            f.write(msg)

class Evaluation_Fidelity(object):
    """
    Evaluation helper that reads original and watermarked images from disk,
    computes image-similarity metrics (PSNR/SSIM/LPIPS).
    """

    def __init__(self, wm_path, ori_path, eval_size, end_idx=None) -> None:
        """
        Args:
            wm_path (str): directory containing watermarked images.
            ori_path (str): directory containing original images.
            end_idx (int or None): if set, evaluate only the first end_idx sorted files.
        """
        self.wm_path = wm_path
        self.ori_path = ori_path
        self.eval_size = eval_size
        self.end_idx = end_idx

    def run(self, save_path):
        """
        Iterate over watermarked image found in self.wm_path, load the corresponding original,
        compute per-image similarity metrics, accumulate them, print a summary and append to a record file.

        Args:
            save_path (str): directory in which to write record.txt (appends).
        """
        total_psnr = []
        total_ssim = []
        total_lpips = []
        transform = T.Compose([
            T.Resize((self.eval_size, self.eval_size)), # Bilinear
            T.ToTensor()
        ])

        # list prediction files in prediction directory (sorted for determinism)
        wm_images_path = sorted(os.listdir(self.wm_path))
        if self.end_idx is not None:
            wm_images_path = wm_images_path[:self.end_idx]

        # iterate through each predicted mask filename
        for wm_image_path in tqdm(wm_images_path):
            try:
                # open watermarked image and corresponding original image
                # wm_image = Image.open(os.path.join(self.wm_path, wm_image_path)).convert("RGB")
                # ori_image = Image.open(os.path.join(self.ori_path, wm_image_path)).convert("RGB")

                # ori_image = ori_image.resize((self.eval_size, self.eval_size))
                # wm_image = wm_image.resize((self.eval_size, self.eval_size))

                wm_image = Image.open(os.path.join(self.wm_path, wm_image_path)).convert("RGB")
                ori_image = Image.open(os.path.join(self.ori_path, wm_image_path)).convert("RGB")

                wm_tensor = transform(wm_image).unsqueeze(0)
                ori_tensor = transform(ori_image).unsqueeze(0)  

            except Exception:
                # skip files that cannot be opened / matched
                continue

            # convert PIL images to tensors with shape (1, C, H, W)
            # wm_tensor = T.ToTensor()(wm_image).unsqueeze(0)
            # ori_tensor = T.ToTensor()(ori_image).unsqueeze(0)

            # resize ground-truth to match predicted mask spatial size if needed
            # ori_tensor = torch.nn.functional.interpolate(ori_tensor, (wm_tensor.size(2), wm_tensor.size(3)))

            # call evaluator batch_update for each metric (these update internal state or return stat)
            psnr_value = psnr(wm_tensor, ori_tensor).mean()
            ssim_value = ssim(wm_tensor, ori_tensor).mean()
            lpips_value = LPIPS(reduction='none')(wm_tensor, ori_tensor).mean()

            # append scalars to lists, skipping NaNs
            if not torch.any(torch.isnan(psnr_value)):
                total_psnr.append(psnr_value.item())
            if not torch.any(torch.isnan(ssim_value)):
                total_ssim.append(ssim_value.item())
            if not torch.any(torch.isnan(lpips_value)):
                total_lpips.append(lpips_value.item())

        # compute aggregated means and print/write a summary line
        n_images = len(total_psnr)
        msg = f"N:{n_images}, Tampering PSNR:{np.mean(total_psnr):.5f}, SSIM:{np.mean(total_ssim):.5f}, LPIPS: {np.mean(total_lpips):.5f}\n"
        print(msg)

        # append to record file under save_path
        with open(os.path.join(save_path, "record.txt"), "a+") as f:
            f.write(msg)


@torch.no_grad()
def generate_watermark_image(norm, weight_path, target_model, src_image_path, save_path, edit_model_name, seed, num_bits=48, model_size=512, eval_size=256, start_idx=0, end_idx=None, tamper_mode='inpaint', wm_strength=None, brushnet_checkpoint_dir=None, segmentation_mask=False, inverse_mask=False):
    # create output subdirectories
    res = []
    if tamper_mode == 'ldm':
        res = ['cover_images', 'ldm_spliced_images', 'ldm_spliceless_images', 'gt', 'msgs']
    elif tamper_mode == 'controlnet':
        res = ['cover_images', 'control_spliced_images', 'control_spliceless_images', 'gt', 'msgs']
    elif tamper_mode == 'hdpainter':
        res = ['hdpainter_spliced_images', 'hdpainter_spliceless_images']
    elif tamper_mode == 'zero_mask':
        res = ['zero_mask_images', 'zero_mask']
    elif tamper_mode == 'vae_regen':
        res = ['vae_regen_images']
    elif tamper_mode == 'sdxl':
        res = ['sdxl_spliced_images', 'sdxl_spliceless_images']
    elif tamper_mode == 'flux':
        res = ['flux_spliced_images', 'flux_spliceless_images']
    elif tamper_mode == 'brushnet':
        res = ['brushnet_spliced_images', 'brushnet_spliceless_images']
    if target_model == 'clean':
        res = [r for r in res if r != 'msgs']

    segm_suffix = ("_segm" if segmentation_mask else "") + ("_inverse" if inverse_mask else "")
    res = [
        r if r in ('cover_images', 'msgs')
        else (f'gt{segm_suffix}' if r == 'gt' else r.replace('_images', f'{segm_suffix}_images'))
        for r in res
    ]

    for n in res:
        os.makedirs(os.path.join(save_path, '%s' % n), exist_ok=True)

    # load pretrained diffusion VAE (encoder/decoder)
    if tamper_mode == 'controlnet':
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16, cache_dir='/mnt/nas5/suhyeon/caches/', safety_checker=None
        ).to('cuda')
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16, cache_dir='/mnt/nas5/suhyeon/caches/', safety_checker=None
        ).to('cuda')
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    elif tamper_mode == 'hdpainter':
        pipe = get_inpainting_function(
            model_id='ds8_inp',
            method='painta+rasg',
            eta=0.1,
            guidance_scale=7.5,
            num_steps=50,
            negative_prompt="text, bad anatomy, bad proportions, blurry, cropped, deformed, disfigured, duplicate, error, extra limbs, gross proportions, jpeg artifacts, long neck, low quality, lowres, malformed, morbid, mutated, mutilated, out of frame, ugly, worst quality",
            positive_prompt="Full HD, 4K, high quality, high resolution",
            cache_dir='/mnt/nas5/suhyeon/caches/'
        )
    elif tamper_mode == 'sdxl':
        pipe = AutoPipelineForInpainting.from_pretrained(
            "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
            torch_dtype=torch.float16,
            variant="fp16",
            cache_dir='/mnt/nas5/suhyeon/caches/',
            safety_checker=None
        ).to("cuda")
    elif tamper_mode == 'flux':
        text_encoder_2 = T5EncoderModel.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            subfolder="text_encoder_2",
            load_in_8bit=True,
            device_map="auto"
        )
        pipe = FluxFillPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Fill-dev",
            text_encoder_2=text_encoder_2,
            dtype=torch.bfloat16,
            cache_dir='/mnt/nas5/suhyeon/caches/',
        )
        pipe.enable_model_cpu_offload()
    elif tamper_mode == 'brushnet':
        _bn_src = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BrushNet", "src")
        # BrushNet requires its own modified diffusers (UNet with down_block_add_samples).
        # Swap sys.modules so BrushNet's diffusers is active for the ENTIRE load phase,
        # then restore installed diffusers afterwards.
        _saved_mods = {k: v for k, v in sys.modules.items() if k == 'diffusers' or k.startswith('diffusers.')}
        for k in list(_saved_mods.keys()):
            del sys.modules[k]
        sys.path.insert(0, _bn_src)

        from diffusers import StableDiffusionBrushNetPipeline, BrushNetModel, UniPCMultistepScheduler

        _bn_ckpt = os.path.abspath(brushnet_checkpoint_dir)
        _bn_dtype = torch.float16

        brushnet_model = BrushNetModel.from_pretrained(_bn_ckpt, torch_dtype=_bn_dtype)
        pipe = StableDiffusionBrushNetPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            brushnet=brushnet_model,
            torch_dtype=_bn_dtype,
            safety_checker=None,
            cache_dir='/mnt/nas5/suhyeon/caches/',
            low_cpu_mem_usage=False,
        ).to("cuda")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # Restore installed diffusers after all BrushNet models are loaded
        for k in [k for k in list(sys.modules.keys()) if k == 'diffusers' or k.startswith('diffusers.')]:
            del sys.modules[k]
        sys.path.remove(_bn_src)
        sys.modules.update(_saved_mods)
    else:
        original_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae", cache_dir='/mnt/nas5/suhyeon/caches/').to('cuda')
        pipe = StableDiffusionInpaintPipeline.from_pretrained(edit_model_name, cache_dir='/mnt/nas5/suhyeon/caches/', safety_checker=None).to('cuda')

    # load model
    if target_model == "stableguard":
        original_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae", cache_dir='/mnt/nas5/suhyeon/caches/').to('cuda')
        # initialize and load weights for MultiplexingWatermarkVAEDecoder
        mpw_vae_decoder = MultiplexingWatermarkVAEDecoder(num_bits=num_bits)
        mpw_vae_decoder_weight = torch.load(os.path.join(weight_path, "mpw_vae_decoder.bin"), map_location="cpu")
        mpw_vae_decoder.load_state_dict(mpw_vae_decoder_weight)

        # move models to GPU and set eval mode
        original_vae = original_vae.cuda()
        mpw_vae_decoder = mpw_vae_decoder.cuda()
        original_vae.eval()
        mpw_vae_decoder.eval()

    elif target_model == "wam":
        wam = load_model_from_checkpoint(weight_path, num_bits, scaling_w=wm_strength).cuda().eval()

    elif target_model == "omniguard":
        net = Model(checkpoint=weight_path).cuda().eval()
        init_model(net)
        state_dicts = torch.load(os.path.join(weight_path, "model_checkpoint_01500.pt"), map_location="cpu", weights_only=False)
        network_state_dict = {k.removeprefix('module.'):v for k,v in state_dicts['net'].items()}
        net.load_state_dict(network_state_dict)
        print(f"OmniGuard wm_strength: {wm_strength}")

    # prepare dataloader for validation images
    val_dataset = AGEDataset(data_root=src_image_path, norm_type=norm, mode="val", size=eval_size)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=age_collate_fn,
        batch_size=1,
        num_workers=4,
        drop_last=False,
    )

    if start_idx > 0:
        print(f"Resuming from index {start_idx}...")

    # iterate over the validation dataloader
    for i, batch in enumerate(tqdm(val_dataloader)):

        if i < start_idx:
            continue

        if end_idx is not None and i == end_idx:
            break

        # move batch tensors to GPU
        images = batch["images"].cuda()
        # generated_images = batch["generated_images"].cuda()
        masks = batch["masks"].cuda()
        if not segmentation_mask:
            masks = convert_mask_to_rect(masks, scale_factor=1.2)
        if inverse_mask:
            masks = convert_mask_to_rect(masks, scale_factor=2.0)
            masks = 1 - masks
        image_names = batch["image_names"]

        # embed watermark
        if target_model == "stableguard":
            # stableguard is trained on 256x256 images
            # VAE encode -> sample latents -> decode back to image (reconstruction)
            latents = original_vae.encode(images).latent_dist.sample()
            # decode_images = original_vae.decode(latents, return_dict=False)[0]

            # prepare latents for the watermark decoder (post-quant conv if required by model)
            latents = original_vae.post_quant_conv(latents)

            # sample random binary messages for this batch
            phi = torch.empty(latents.size(0), num_bits).uniform_(0, 1).cuda()
            msgs = (torch.bernoulli(phi) + 1e-8)  # small eps to avoid exact zeros if needed

            # produce watermarked cover images from latents+msgs
            cover_images = mpw_vae_decoder(latents, msgs=msgs)

        elif target_model == "wam":
            # normalization: imagenet
            # decode_images = torch.zeros_like(images)
            images_down = F.interpolate(images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            msgs = wam.get_random_msg(1)
            outputs = wam.embed(images_down, msgs)
            cover_images = outputs['imgs_w']
            # cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = denorm_imagenet(cover_images) # [0, 1]
            # cover_images = denormalize_tensor(cover_images, scale_each=True) # [0, 1]
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]
        
        elif target_model == "omniguard":
            # decode_images = torch.zeros_like(images)
            dwt = DWT()
            image = Image.open("./omniguard/bluesky_white2.png").convert("RGB").resize((model_size, model_size))
            result = np.array(image) / 255.
            expanded_matrix = np.expand_dims(result, axis=0) 
            secret = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
            secret = secret.permute(0, 3, 1, 2).cuda()

            # omniguard is trained on 512x512 images
            cover_input = F.interpolate(images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            secret_input = F.interpolate(secret, size=(model_size, model_size), mode="bilinear", align_corners=False)

            cover_input = dwt((cover_input + 1.0) / 2.0) # [-1, 1] to [0, 1], [1, 12, 256, 256]
            secret_input = dwt(secret_input) # [1, 12, 256, 256]
            msgs = torch.randint(2, (1, 64)).to(torch.float32).cuda()

            cover_images, output_z, out_temp, secret_temp = net(cover_input, secret_input, msgs, wm_strength=wm_strength if wm_strength is not None else 1.0)
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]

            # cover_images = F.interpolate(cover_images, size=(size, size), mode="bilinear", align_corners=False)
        
        elif target_model == "ours":
            # save_path will be: /mnt/nas5/suhyeon/projects/eval_spliceless/ours/exp_num
            cover_path = os.path.join(save_path, 'cover_images')
            # TODO: error handling
            cover_images = Image.open(os.path.join(cover_path, image_names[0])).convert("RGB").resize((model_size, model_size))
            cover_images = ToTensor()(cover_images).unsqueeze(0).cuda() # [0, 1]
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]

        elif target_model == "clean":
            # 워터마크 없이 원본 이미지를 그대로 사용 (norm_type="rescale" → [-1, 1])
            cover_images = images

        # per-image generator: same image index → same seed across different target_models
        generator = torch.Generator().manual_seed(seed + i)

        # tamper
        if tamper_mode == 'ldm':
            # inpaint and splice in 512x512
            inpaint_input = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False)
            generated_images = pipe(prompt="", image=inpaint_input, mask_image=masks, generator=generator, num_inference_steps=50).images[0]

            # pil to tensor, normalize to [-1,1], add batch dim
            generated_images = ToTensor()(generated_images).cuda()
            generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0)

            # composite at 512x512, then downsample to original size
            masks = F.interpolate(masks, size=(512, 512), mode="nearest")
            spliced_images = masks * generated_images + (1 - masks) * inpaint_input # operation in [-1, 1]
            spliceless_images = generated_images

            # adjust to each model's training size
            spliced_images = F.interpolate(spliced_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs: cover, tamper, ground-truth mask, and message vector
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                cover_image = cover_images[i]
                spliced_image = spliced_images[i]
                spliceless_image = spliceless_images[i]
                mask = masks[i]
                # msg = msgs[i]

                # make each a single-image tensor and convert to uint8 PIL before saving
                cover_image = cover_image.unsqueeze(0)
                spliced_image = spliced_image.unsqueeze(0)
                spliceless_image = spliceless_image.unsqueeze(0)
                mask = mask.unsqueeze(0)
                # msg = msg.unsqueeze(0)

                # cover image: convert from model range [-1,1] to [0,255] uint8
                cover_image = (cover_image / 2 + 0.5).clamp(0, 1)
                cover_image = cover_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                cover_image = (cover_image * 255).astype(np.uint8)
                cover_image_pil = Image.fromarray(cover_image)

                # spliced image: same conversion
                spliced_image = (spliced_image / 2 + 0.5).clamp(0, 1)
                spliced_image = spliced_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliced_image = (spliced_image * 255).astype(np.uint8)
                spliced_image_pil = Image.fromarray(spliced_image)

                # spliceless image: same conversion
                spliceless_image = (spliceless_image / 2 + 0.5).clamp(0, 1)
                spliceless_image = spliceless_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliceless_image = (spliceless_image * 255).astype(np.uint8)
                spliceless_image_pil = Image.fromarray(spliceless_image)

                # save cover and edited images as PNG (replace .jpg extension if present)
                if target_model != 'ours':
                    cover_image_pil.save(os.path.join(save_path, 'cover_images', save_file_name.replace("jpg", "png")))
                spliced_image_pil.save(os.path.join(save_path, f'ldm_spliced{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                spliceless_image_pil.save(os.path.join(save_path, f'ldm_spliceless{segm_suffix}_images', save_file_name.replace("jpg", "png")))

                # save ground-truth mask as image tensor and message vector as .pt file
                if target_model == 'clean':
                    # clean 모드: gt 마스크를 eval_size(256)로 nearest resize해서 저장
                    mask_gt = F.interpolate(mask, size=(eval_size, eval_size), mode='nearest')
                    save_image(1-mask_gt, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
                else:
                    save_image(1-mask, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
                # torch.save(msg, os.path.join(save_path, 'msgs', save_file_name.split(".")[0] + '.pt'))

        elif tamper_mode == 'controlnet':
            # inpaint and splice in 512x512
            inpaint_input = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False)
            inpaint_mask = F.interpolate(masks, size=(512, 512), mode="nearest")
            control_image = make_inpaint_condition(inpaint_input, inpaint_mask)
            generated_images = pipe(
                "",
                num_inference_steps=50,
                generator=generator,
                eta=1.0,
                image=inpaint_input,
                mask_image=inpaint_mask,
                control_image=control_image,
            ).images[0]

            # pil to tensor, normalize to [-1,1], add batch dim
            generated_images = ToTensor()(generated_images).cuda()
            generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0)

            # composite at 512x512, then downsample to original size
            spliced_images = inpaint_mask * generated_images + (1 - inpaint_mask) * inpaint_input # operation in [-1, 1]
            spliceless_images = generated_images

            # adjust to each model's training size
            spliced_images = F.interpolate(spliced_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs: cover, tamper, ground-truth mask, and message vector
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                cover_image = cover_images[i]
                spliced_image = spliced_images[i]
                spliceless_image = spliceless_images[i]
                mask = masks[i]
                # msg = msgs[i]

                # make each a single-image tensor and convert to uint8 PIL before saving
                cover_image = cover_image.unsqueeze(0)
                spliced_image = spliced_image.unsqueeze(0)
                spliceless_image = spliceless_image.unsqueeze(0)
                mask = mask.unsqueeze(0)
                # msg = msg.unsqueeze(0)

                # cover image: convert from model range [-1,1] to [0,255] uint8
                cover_image = (cover_image / 2 + 0.5).clamp(0, 1)
                cover_image = cover_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                cover_image = (cover_image * 255).astype(np.uint8)
                cover_image_pil = Image.fromarray(cover_image)

                # spliced image: same conversion
                spliced_image = (spliced_image / 2 + 0.5).clamp(0, 1)
                spliced_image = spliced_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliced_image = (spliced_image * 255).astype(np.uint8)
                spliced_image_pil = Image.fromarray(spliced_image)

                # spliceless image: same conversion
                spliceless_image = (spliceless_image / 2 + 0.5).clamp(0, 1)
                spliceless_image = spliceless_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliceless_image = (spliceless_image * 255).astype(np.uint8)
                spliceless_image_pil = Image.fromarray(spliceless_image)

                # save cover and edited images as PNG (replace .jpg extension if present)
                if target_model != 'ours':
                    cover_image_pil.save(os.path.join(save_path, 'cover_images', save_file_name.replace("jpg", "png")))
                spliced_image_pil.save(os.path.join(save_path, f'control_spliced{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                spliceless_image_pil.save(os.path.join(save_path, f'control_spliceless{segm_suffix}_images', save_file_name.replace("jpg", "png")))

                # save ground-truth mask as image tensor and message vector as .pt file
                if target_model == 'clean':
                    # clean 모드: gt 마스크를 eval_size(256)로 nearest resize해서 저장
                    mask_gt = F.interpolate(mask, size=(eval_size, eval_size), mode='nearest')
                    save_image(1-mask_gt, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
                else:
                    save_image(1-mask, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
                # torch.save(msg, os.path.join(save_path, 'msgs', save_file_name.split(".")[0] + '.pt'))
        
        elif tamper_mode == 'hdpainter':
            # inpaint and splice in 512x512
            image_512 = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False)
            mask_512 = F.interpolate(masks, size=(512, 512), mode='nearest')

            # Image: [-1, 1] -> [0, 1] -> [0, 255] -> uint8 -> PIL -> IImage
            inpaint_input = (image_512 / 2 + 0.5).clamp(0, 1)
            inpaint_input = inpaint_input.squeeze(0).cpu().permute(1, 2, 0).numpy()
            inpaint_input = Image.fromarray((inpaint_input*255).astype(np.uint8)).convert('RGB')
            inpaint_input = IImage(inpaint_input)

            inpaint_mask = mask_512.squeeze(0).cpu().permute(1, 2, 0).numpy()
            inpaint_mask = Image.fromarray((inpaint_mask.squeeze() * 255).astype(np.uint8)).convert('RGB')
            inpaint_mask = IImage(inpaint_mask)

            with torch.enable_grad():
                generated_iimage = pipe(
                        inpaint_input, 
                        inpaint_mask, 
                        prompt="", 
                        seed=seed + i
                )

            generated_images = ToTensor()(generated_iimage).cuda()
            generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0) # Scale to [-1, 1]

            # composite at 512x512, then downsample to original size
            spliced_images = mask_512 * generated_images + (1 - mask_512) * image_512 # operation in [-1, 1]
            spliceless_images = generated_images

            # adjust to each model's training size
            spliced_images = F.interpolate(spliced_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs: cover, tamper, ground-truth mask, and message vector
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                cover_image = cover_images[i]
                spliced_image = spliced_images[i]
                spliceless_image = spliceless_images[i]
                mask = masks[i]
                # msg = msgs[i]

                # make each a single-image tensor and convert to uint8 PIL before saving
                cover_image = cover_image.unsqueeze(0)
                spliced_image = spliced_image.unsqueeze(0)
                spliceless_image = spliceless_image.unsqueeze(0)
                mask = mask.unsqueeze(0)
                # msg = msg.unsqueeze(0)

                # cover image: convert from model range [-1,1] to [0,255] uint8
                cover_image = (cover_image / 2 + 0.5).clamp(0, 1)
                cover_image = cover_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                cover_image = (cover_image * 255).astype(np.uint8)
                cover_image_pil = Image.fromarray(cover_image)

                # spliced image: same conversion
                spliced_image = (spliced_image / 2 + 0.5).clamp(0, 1)
                spliced_image = spliced_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliced_image = (spliced_image * 255).astype(np.uint8)
                spliced_image_pil = Image.fromarray(spliced_image)

                # spliceless image: same conversion
                spliceless_image = (spliceless_image / 2 + 0.5).clamp(0, 1)
                spliceless_image = spliceless_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliceless_image = (spliceless_image * 255).astype(np.uint8)
                spliceless_image_pil = Image.fromarray(spliceless_image)

                # save cover and edited images as PNG (replace .jpg extension if present)
                # cover_image_pil.save(os.path.join(save_path, 'cover_images', save_file_name.replace("jpg", "png")))
                spliced_image_pil.save(os.path.join(save_path, f'hdpainter_spliced{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                spliceless_image_pil.save(os.path.join(save_path, f'hdpainter_spliceless{segm_suffix}_images', save_file_name.replace("jpg", "png")))

                # save ground-truth mask as image tensor and message vector as .pt file
                save_image(1-mask, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
                # torch.save(msg, os.path.join(save_path, 'msgs', save_file_name.split(".")[0] + '.pt'))
            
        elif tamper_mode == 'zero_mask':
            inpaint_input = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False)
            zero_mask = torch.zeros((1, 1, 512, 512), dtype=inpaint_input.dtype, device=inpaint_input.device)
            generated_images = pipe(prompt="", image=inpaint_input, mask_image=zero_mask, generator=generator, num_inference_steps=20).images[0]

            # adjust to each model's training size
            generated_images = ToTensor()(generated_images).unsqueeze(0).cuda()
            generated_images = F.interpolate(generated_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs: cover, tamper, ground-truth mask, and message vector
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                generated_image = generated_images[i]

                # make each a single-image tensor and convert to uint8 PIL before saving
                generated_image = generated_image.unsqueeze(0)

                # cover image: convert from model range [-1,1] to [0,255] uint8
                generated_image = generated_image.clamp(0, 1)
                generated_image = generated_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                generated_image = (generated_image * 255).astype(np.uint8)
                generated_image_pil = Image.fromarray(generated_image)

                generated_image_pil.save(os.path.join(save_path, 'zero_mask_images', save_file_name.replace("jpg", "png")))
                # save_image(zero_mask, os.path.join(save_path, 'zero_mask', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)

        elif tamper_mode == 'vae_regen':
            inpaint_input = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False)
            latents = original_vae.encode(inpaint_input).latent_dist.sample()
            generated_images = original_vae.decode(latents, return_dict=False)[0]
            generated_images = F.interpolate(generated_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs: cover, tamper, ground-truth mask, and message vector
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                generated_image = generated_images[i]

                # make each a single-image tensor and convert to uint8 PIL before saving
                generated_image = (generated_image / 2 + 0.5).clamp(0, 1)
                generated_image = generated_image.unsqueeze(0)

                # cover image: convert from model range [-1,1] to [0,255] uint8
                generated_image = generated_image.clamp(0, 1)
                generated_image = generated_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                generated_image = (generated_image * 255).astype(np.uint8)
                generated_image_pil = Image.fromarray(generated_image)

                generated_image_pil.save(os.path.join(save_path, 'vae_regen_images', save_file_name.replace("jpg", "png")))

        elif tamper_mode == 'sdxl':
            # inpaint and splice at 1024x1024 (SDXL native resolution)
            inpaint_input = F.interpolate(cover_images, size=(1024, 1024), mode="bilinear", align_corners=False)
            inpaint_mask = F.interpolate(masks, size=(1024, 1024), mode="nearest")
            generated_images = pipe(
                prompt="",
                image=inpaint_input,
                mask_image=inpaint_mask,
                guidance_scale=8.0,
                num_inference_steps=50,
                strength=0.99,
                generator=generator,
                padding_mask_crop=None
            ).images[0]

            # pil to tensor, normalize to [-1,1], add batch dim
            generated_images = ToTensor()(generated_images).cuda()
            generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0)

            # composite at 1024x1024, then downsample to original size
            spliced_images = inpaint_mask * generated_images + (1 - inpaint_mask) * inpaint_input  # operation in [-1, 1]
            spliceless_images = generated_images

            # adjust to each model's training size
            spliced_images = F.interpolate(spliced_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                spliced_image = spliced_images[i]
                spliceless_image = spliceless_images[i]
                mask = masks[i]

                spliced_image = spliced_image.unsqueeze(0)
                spliceless_image = spliceless_image.unsqueeze(0)
                mask = mask.unsqueeze(0)

                # spliced image: convert from [-1,1] to [0,255] uint8
                spliced_image = (spliced_image / 2 + 0.5).clamp(0, 1)
                spliced_image = spliced_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliced_image = (spliced_image * 255).astype(np.uint8)
                spliced_image_pil = Image.fromarray(spliced_image)

                # spliceless image: same conversion
                spliceless_image = (spliceless_image / 2 + 0.5).clamp(0, 1)
                spliceless_image = spliceless_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliceless_image = (spliceless_image * 255).astype(np.uint8)
                spliceless_image_pil = Image.fromarray(spliceless_image)

                spliced_image_pil.save(os.path.join(save_path, f'sdxl_spliced{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                spliceless_image_pil.save(os.path.join(save_path, f'sdxl_spliceless{segm_suffix}_images', save_file_name.replace("jpg", "png")))

                save_image(1-mask, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)

        elif tamper_mode == 'flux':
            # 입력을 float32로 처리하여 계산 오차와 타입 충돌 방지
            image_512 = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False).float()
            mask_512 = F.interpolate(masks, size=(512, 512), mode='nearest').float()

            # tensor [-1,1] → PIL RGB (float()로 변환하여 numpy 에러 방지)
            inpaint_input_np = (image_512 / 2 + 0.5).clamp(0, 1)
            inpaint_input_np = inpaint_input_np.squeeze(0).cpu().permute(1, 2, 0).numpy()
            inpaint_input_pil = Image.fromarray((inpaint_input_np * 255).astype(np.uint8)).convert('RGB')

            # mask tensor → PIL L
            inpaint_mask_np = mask_512.squeeze(0).squeeze(0).cpu().numpy()
            inpaint_mask_pil = Image.fromarray((inpaint_mask_np * 255).astype(np.uint8)).convert('L')

            generated_images = pipe(
                prompt="",
                image=inpaint_input_pil,
                mask_image=inpaint_mask_pil,
                height=512,
                width=512,
                guidance_scale=30,
                num_inference_steps=50,
                # dtype 충돌을 피하기 위해 generator를 명시적으로 설정
                generator=torch.Generator("cpu").manual_seed(seed + i)
            ).images[0]

            # 4. PIL → tensor [-1,1] 및 타입 통일
            generated_images = ToTensor()(generated_images_pil).cuda().float()
            generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0)

            # 5. 합성 (모두 float32/GPU 상태이므로 안전함)
            spliced_images = mask_512 * generated_images + (1 - mask_512) * image_512
            spliceless_images = generated_images

            # adjust to each model's training size
            spliced_images = F.interpolate(spliced_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)
            cover_images = F.interpolate(cover_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save per-image outputs
            for i in range(images.size(0)):
                save_file_name = image_names[i]
                spliced_image = spliced_images[i]
                spliceless_image = spliceless_images[i]
                mask = masks[i]

                spliced_image = spliced_image.unsqueeze(0)
                spliceless_image = spliceless_image.unsqueeze(0)
                mask = mask.unsqueeze(0)

                # spliced image: convert from [-1,1] to [0,255] uint8
                spliced_image = (spliced_image / 2 + 0.5).clamp(0, 1)
                spliced_image = spliced_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliced_image = (spliced_image * 255).astype(np.uint8)
                spliced_image_pil = Image.fromarray(spliced_image)

                # spliceless image: same conversion
                spliceless_image = (spliceless_image / 2 + 0.5).clamp(0, 1)
                spliceless_image = spliceless_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
                spliceless_image = (spliceless_image * 255).astype(np.uint8)
                spliceless_image_pil = Image.fromarray(spliceless_image)

                spliced_image_pil.save(os.path.join(save_path, f'flux_spliced{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                spliceless_image_pil.save(os.path.join(save_path, f'flux_spliceless{segm_suffix}_images', save_file_name.replace("jpg", "png")))

                save_image(1-mask, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)

        elif tamper_mode == 'brushnet':
            image_512 = F.interpolate(cover_images, size=(512, 512), mode="bilinear", align_corners=False).float()
            mask_512  = F.interpolate(masks, size=(512, 512), mode='nearest').float()

            # tensor [-1,1] → numpy uint8 (H,W,3)
            orig_np = (image_512 / 2 + 0.5).clamp(0, 1).squeeze(0).cpu().permute(1, 2, 0).numpy()
            orig_np_uint8 = (orig_np * 255).astype(np.uint8)

            # mask: (H,W) float [0,1]; 1=tampered=inpaint (matches BrushNet white=inpaint convention)
            mask_np = mask_512.squeeze(0).squeeze(0).cpu().numpy()

            # Masked input: zero out inpaint region (test_brushnet.py pattern)
            masked_np = (orig_np * (1 - mask_np[:, :, np.newaxis]) * 255).astype(np.uint8)
            masked_pil = Image.fromarray(masked_np).convert('RGB')

            # BrushNet mask PIL: white (255)=inpaint, black (0)=keep
            mask_rgb = np.stack([mask_np * 255] * 3, axis=-1).astype(np.uint8)
            mask_pil = Image.fromarray(mask_rgb).convert('RGB')

            result_pil = pipe(
                "",
                masked_pil,
                mask_pil,
                num_inference_steps=50,
                generator=torch.Generator("cuda").manual_seed(seed + i),
                brushnet_conditioning_scale=1.0,
                guidance_scale=7.5,
            ).images[0]

            # spliceless: raw pipe output (before any blending)
            spliceless_np = np.array(result_pil)  # (512,512,3) uint8

            # spliced: Gaussian-blur-blend at mask boundary (test_brushnet.py blended pattern)
            mask_blurred = cv2.GaussianBlur(mask_np * 255, (21, 21), 0) / 255
            blend_mask = 1 - (1 - mask_np[:, :, np.newaxis]) * (1 - mask_blurred[:, :, np.newaxis])
            spliced_np = (orig_np_uint8 * (1 - blend_mask) + spliceless_np * blend_mask).astype(np.uint8)

            def _np_to_tensor(arr_uint8):
                return (torch.from_numpy(arr_uint8).float().permute(2, 0, 1).unsqueeze(0).cuda() / 255.0 * 2.0 - 1.0)

            spliceless_images = _np_to_tensor(spliceless_np)
            spliced_images    = _np_to_tensor(spliced_np)

            spliced_images    = F.interpolate(spliced_images,    size=(model_size, model_size), mode="bilinear", align_corners=False)
            spliceless_images = F.interpolate(spliceless_images, size=(model_size, model_size), mode="bilinear", align_corners=False)

            for j in range(images.size(0)):
                save_file_name = image_names[j]
                mask_j = masks[j].unsqueeze(0)

                def _to_pil(t):
                    arr = (t / 2 + 0.5).clamp(0, 1).squeeze(0).cpu().numpy().transpose(1, 2, 0)
                    return Image.fromarray((arr * 255).astype(np.uint8))

                _to_pil(spliced_images[j].unsqueeze(0)).save(
                    os.path.join(save_path, f'brushnet_spliced{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                _to_pil(spliceless_images[j].unsqueeze(0)).save(
                    os.path.join(save_path, f'brushnet_spliceless{segm_suffix}_images', save_file_name.replace("jpg", "png")))
                save_image(1 - mask_j, os.path.join(save_path, f'gt{segm_suffix}', save_file_name.replace("jpg", "png")),
                           normalize=True, scale_each=True)


@torch.no_grad()
def generate_tamper_mask(weight_path, eval_setting, target_model, save_path, num_bits=48, model_size=512, end_idx=None, aug_type=None, aug_param=None, wm_strength=None, use_refiner=True, save_aug_image=False):
    if aug_type is not None and aug_param is not None:
        exp_suffix = f"{eval_setting}_{aug_type}_{aug_param}"
    else:
        exp_suffix = eval_setting

    refiner_tag = '_refiner' if (target_model == 'ours' and use_refiner) else ''

    # make output folder for predicted masks
    os.makedirs(os.path.join(save_path, f"pred_mask_{exp_suffix}{refiner_tag}"), exist_ok=True)
    os.makedirs(os.path.join(save_path, f"pred_bin_mask_{exp_suffix}{refiner_tag}"), exist_ok=True)
    # os.makedirs(os.path.join(save_path, f"cossim_{exp_suffix}"), exist_ok=True)
    if aug_type is not None and aug_param is not None and save_aug_image:
        os.makedirs(os.path.join(save_path, f"augmented_{exp_suffix}"), exist_ok=True)
    tamper_image_path = os.path.join(save_path, f"{eval_setting}_images")

    valid_exts = (".jpg", ".jpeg", ".png")

    # initialize and load the detector model
    if target_model == "stableguard":
        moe_gfn = MoEGuidedForensicNet(num_bits=num_bits)
        moe_gfn_weight = torch.load(os.path.join(weight_path, "moe_gfn.bin"), map_location="cpu")
        moe_gfn.load_state_dict(moe_gfn_weight)
        moe_gfn = moe_gfn.cuda()
        moe_gfn.eval()
    elif target_model == "wam":
        wam = load_model_from_checkpoint(weight_path, num_bits, scaling_w=wm_strength).cuda().eval()
    elif target_model == "omniguard":
        net = Model(checkpoint=weight_path).cuda().eval()
        init_model(net)
        state_dicts = torch.load(os.path.join(weight_path, "model_checkpoint_01500.pt"), map_location="cpu", weights_only=False)
        network_state_dict = {k.removeprefix('module.'):v for k,v in state_dicts['net'].items()}
        net.load_state_dict(network_state_dict)
        
        extractor = iml_vit_model()
        extractor.load_state_dict(torch.load(os.path.join(weight_path, "checkpoint-175.pth"), weights_only=False)['model'], strict=True)
        extractor = extractor.cuda().eval()
    elif target_model == "ours":
        args = Params()
        locmark = LocMark(args=args)

    # bit_acc = []
    file_paths = os.listdir(tamper_image_path)
    image_paths = [f for f in file_paths if f.lower().endswith(valid_exts)]
    image_paths.sort()
    if end_idx is not None:
        image_paths = image_paths[:end_idx]

    # for cossim dist. figure
    # in_logits_list = []
    # out_logits_list = []
    # logits_list = []

    attack_transform = get_robustness_transform(aug_type, aug_param, image_size=model_size)

    # iterate files and run inference on each image (single-image inference)
    for image_path in tqdm(image_paths):
        # load and resize image to expected input size
        image = Image.open(os.path.join(tamper_image_path, image_path)).resize((model_size, model_size))
        # mask = Image.open(os.path.join(save_path, "gt", image_path)).convert('L').resize((model_size, model_size))
        # clean = Image.open(os.path.join('/mnt/nas5/suhyeon/datasets/valAGE-Set', image_path)).convert('RGB').resize((model_size, model_size))

        if attack_transform is not None:
            image_att = np.array(image)
            augmented = attack_transform(image=image_att)
            image_att = augmented['image']
            image = Image.fromarray(image_att)
            if save_aug_image:
                image.save(os.path.join(save_path, f"augmented_{exp_suffix}", image_path))

        if target_model == "stableguard":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1),  # map [0,1] -> [-1,1] if model expects that
            ])
            image = transform(image).unsqueeze(0)  # shape [1, C, H, W]

            # run detector on GPU
            pred_msgs, pred_mask = moe_gfn(image.cuda())

            # convert mask logits to probabilities
            pred_mask = torch.sigmoid(pred_mask)

            # save predicted mask image to disk
            save_image(1-pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}", image_path), normalize=False, scale_each=True)

            # load ground-truth message that was saved earlier during generation step
            # save_msgs = torch.load(os.path.join(save_path, 'msgs', image_path.split('.')[0] + '.pt'))

            # compute bitwise accuracy between predicted messages and saved messages
            # pred_msgs_bin = torch.round(torch.sigmoid(pred_msgs))
            # msgs_bin = torch.round(torch.sigmoid(save_msgs.squeeze(1)))
            # acc = (((pred_msgs_bin.eq(msgs_bin.data)).sum()) / num_bits).mean().item()
            # bit_acc.append(acc)

        elif target_model == "wam":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            image = transform(image).unsqueeze(0).cuda()  # shape [1, C, H, W]
            image_down = F.interpolate(image, size=(model_size, model_size), mode="bilinear", align_corners=False)

            outputs = wam.detect(image_down)["preds"]
            pred_mask = F.sigmoid(outputs[:, 0, :, :]).unsqueeze(0) # [1, 1, 256, 256]
            pred_bit = outputs[:, 1:, :, :] # [1, 32, 256, 256]
            pred_message = msg_predict_inference(pred_bit, pred_mask).cpu().float()  # [1, 32]
            
            # WAM predicts the watermarked region
            # save_image(denorm_imagenet(image), os.path.join(save_path, f"augmented_image_{exp_suffix}", image_path), normalize=False, scale_each=False)
            save_image(pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}", image_path), normalize=False, scale_each=True)

            # load ground-truth message that was saved earlier during generation step
            # save_msgs = torch.load(os.path.join(save_path, 'msgs', image_path.split('.')[0] + '.pt'))
            # acc = (pred_message == save_msgs).float().mean().item()
            # bit_acc.append(acc)
        
        elif target_model == "omniguard":
            dwt = DWT()
            iwt = IWT()
            transform = transforms.Compose([
                transforms.Resize((model_size, model_size)),
                transforms.ToTensor(),
            ])
            transform_extractor = albu.Compose([
                albu.PadIfNeeded(          
                    min_height=1024,
                    min_width=1024, 
                    border_mode=0, 
                    value=0, 
                    position='top_left',
                    mask_value=0),
                albu.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                albu.Crop(0, 0, 1024, 1024),
                ToTensorV2()
            ])
            image = transform(image).unsqueeze(0)  # shape [1, C, H, W]
            output_steg = dwt(image)
            output_image, bits = net(output_steg.cuda(), rev=True)
            secret_rev = output_image.narrow(1, 0, 12)
            secret_rev = iwt(secret_rev)

            artifact = secret_rev.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255
            fuse = image.permute(0, 2, 3, 1).squeeze().cpu().numpy() * 255

            artifact = transform_extractor(image=artifact)['image'].cuda().unsqueeze(0)
            fuse = transform_extractor(image=fuse)['image'].cuda().unsqueeze(0)

            pred_mask = extractor(artifact, fuse)
            pred_mask = pred_mask[:, :, 0:model_size, 0:model_size]
            pred_mask = F.interpolate(pred_mask, size=(model_size, model_size), mode="bilinear", align_corners=False)

            # save predicted mask image to disk
            save_image(1-pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}", image_path), normalize=False, scale_each=True)

            # load ground-truth message that was saved earlier during generation step
            # 64 bits (training) + zero-padding
            # save_msgs = torch.load(os.path.join(save_path, 'msgs', image_path.split('.')[0] + '.pt'))

            # compute bitwise accuracy between predicted messages and saved messages
            # pred_msgs_bin = (bits > 0).int()
            # msgs_bin = (save_msgs > 0).int()
            # pred_string = ''.join(str(x.item()) for x in pred_msgs_bin.flatten())
            # msgs_string = ''.join(str(x.item()) for x in msgs_bin.flatten())
            
            # remove zero-padding 
            # pred_string = pred_string[:num_bits]
            # correct_bits = sum(1 for a, b in zip(pred_string, msgs_string) if a == b)
    
            # total_bits = len(msgs_string)
            # acc = correct_bits / total_bits
            
            # bit_acc.append(acc)

        elif target_model == "ours":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image).unsqueeze(0)  # shape [1, C, H, W]
            # clean = transform(clean).unsqueeze(0)

            # run detector on GPU
            logits, pred_mask, bin_prediction = locmark.decode_watermark(image.cuda(), use_refiner=use_refiner)

            # save predicted mask image to disk
            save_image(pred_mask, os.path.join(save_path, f"pred_mask_{exp_suffix}{refiner_tag}", image_path), normalize=False, scale_each=False)
            save_image(bin_prediction, os.path.join(save_path, f"pred_bin_mask_{exp_suffix}{refiner_tag}", image_path), normalize=False, scale_each=False)

            # pt_filename = os.path.splitext(image_path)[0] + ".pt"
            # torch.save(logits.cpu(), os.path.join(save_path, f"cossim_{exp_suffix}", pt_filename))

@torch.no_grad()
def eval_cossim_distribution(target_model, save_path, src_image_path, edit_model_name, model_size=256, end_idx=None, seed=42):
    """
    For 'ours' model only: compute and save cosine similarity distribution statistics.

    For each watermarked image in save_path/cover_images:
      1. wm_cossim       - average cossim over the whole watermarked image
      2. clean_cossim    - average cossim over the whole clean image
      3. inp_inside_cossim  - average cossim inside center mask after inpainting
      4. inp_outside_cossim - average cossim outside center mask after inpainting

    Center mask: model_size//2 × model_size//2 square (1/4 area), centered.

    Results saved to <save_path>/dist_results.json and appended to <save_path>/record.txt.
    """
    assert target_model == 'ours', "eval_dist is only supported for 'ours' model"

    args = Params()
    locmark = LocMark(args=args)

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        edit_model_name, cache_dir='/mnt/nas5/suhyeon/caches/', safety_checker=None
    ).to('cuda')

    cover_path = os.path.join(save_path, 'cover_images')
    valid_exts = (".jpg", ".jpeg", ".png")
    image_paths = sorted([f for f in os.listdir(cover_path) if f.lower().endswith(valid_exts)])
    if end_idx is not None:
        image_paths = image_paths[:end_idx]

    transform = transforms.Compose([transforms.ToTensor()])

    # Fixed center mask: model_size//2 × model_size//2 square (1/4 area), centered
    mask_side = model_size // 2
    y0 = model_size // 4
    x0 = model_size // 4
    center_mask = torch.zeros(1, 1, model_size, model_size).cuda()
    center_mask[0, 0, y0:y0 + mask_side, x0:x0 + mask_side] = 1.0

    per_image = {}

    for img_name in tqdm(image_paths, desc="eval_dist"):
        # watermarked image
        wm_tensor = transform(
            Image.open(os.path.join(cover_path, img_name)).convert("RGB").resize((model_size, model_size))
        ).unsqueeze(0).cuda()

        # clean image
        clean_tensor = transform(
            Image.open(os.path.join(src_image_path, img_name)).convert("RGB").resize((model_size, model_size))
        ).unsqueeze(0).cuda()

        # raw cossim grids (use_refiner=False → returns patch-level grid before temperature)
        wm_grid, _, _    = locmark.decode_watermark(wm_tensor,    use_refiner=False)  # [1, 1, H, W]
        clean_grid, _, _ = locmark.decode_watermark(clean_tensor, use_refiner=False)

        wm_cossim    = wm_grid.mean().item()
        clean_cossim = clean_grid.mean().item()

        # center inpainting on watermarked image
        inp_input_512 = F.interpolate(wm_tensor * 2.0 - 1.0, size=(512, 512), mode="bilinear", align_corners=False)
        mask_512      = F.interpolate(center_mask,             size=(512, 512), mode="nearest")

        generator = torch.Generator().manual_seed(seed)
        generated_pil = pipe(
            prompt="", image=inp_input_512, mask_image=mask_512,
            generator=generator, num_inference_steps=50
        ).images[0]

        generated_tensor = F.interpolate(
            ToTensor()(generated_pil).unsqueeze(0).cuda(),
            size=(model_size, model_size), mode="bilinear", align_corners=False
        )
        inpainted = center_mask * generated_tensor + (1 - center_mask) * wm_tensor

        # cossim grid for the inpainted image
        inp_grid, _, _ = locmark.decode_watermark(inpainted, use_refiner=False)

        # downsample center mask to feature grid resolution for per-region averaging
        feat_h, feat_w = inp_grid.shape[2], inp_grid.shape[3]
        feat_mask = F.interpolate(center_mask, size=(feat_h, feat_w), mode="nearest")

        inside_cossim  = inp_grid[feat_mask > 0.5].mean().item()
        outside_cossim = inp_grid[feat_mask <= 0.5].mean().item()

        per_image[img_name] = {
            'wm_cossim':          wm_cossim,
            'clean_cossim':       clean_cossim,
            'inp_inside_cossim':  inside_cossim,
            'inp_outside_cossim': outside_cossim,
        }

    # aggregate
    wm_avg      = float(np.mean([v['wm_cossim']          for v in per_image.values()]))
    clean_avg   = float(np.mean([v['clean_cossim']        for v in per_image.values()]))
    inside_avg  = float(np.mean([v['inp_inside_cossim']   for v in per_image.values()]))
    outside_avg = float(np.mean([v['inp_outside_cossim']  for v in per_image.values()]))

    summary = dict(
        wm_cossim_avg=wm_avg,
        clean_cossim_avg=clean_avg,
        inp_inside_cossim_avg=inside_avg,
        inp_outside_cossim_avg=outside_avg,
        per_image=per_image,
    )

    results_path = os.path.join(save_path, 'dist_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=4)

    msg = (
        f"[eval_dist] N:{len(per_image)}, "
        f"WM: {wm_avg:.4f}, Clean: {clean_avg:.4f}, "
        f"Inp inside: {inside_avg:.4f}, Inp outside: {outside_avg:.4f}\n"
    )
    print(msg)
    with open(os.path.join(save_path, "record.txt"), "a+") as f:
        f.write(msg)


def save_and_print_config(config, save_path):
    """
    Save and print the configuration dictionary to a YAML file.
    """
    os.makedirs(save_path, exist_ok=True)
    
    config_path = os.path.join(save_path, "config_eval.yaml")

    print("-" * 30)
    print(" " * 10 + "Configuration")
    print("-" * 30)
    
    print(yaml.dump(config, allow_unicode=True, default_flow_style=False))
    
    print("-" * 30)

    with open(config_path, 'w') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        
    print(f"Configuration saved to: {config_path}")
    print("-" * 30)

if __name__ == "__main__":
    base_conf = OmegaConf.load('config.yaml')
    cli_conf = OmegaConf.from_cli()
    final_conf = OmegaConf.merge(base_conf, cli_conf)
    c = OmegaConf.to_container(final_conf, resolve=True)
    
    # model-specific settings
    model_name = c['target_model']
    c['weight_path'] = c['weight_paths'][model_name]
    c['normalization'] = c['normalization'][model_name]
    c['num_bits'] = c['num_bits'][model_name]
    c['model_size'] = c['train_img_size'][model_name]
    c['wm_strength'] = c.get('wm_strength', None)  # WAM/OmniGuard watermark strength
    c['use_refiner'] = c.get('use_refiner', True)  # ours: use mask_refiner or direct upsampling
    c['segmentation_mask'] = c.get('segmentation_mask', False)
    c['inverse_mask'] = c.get('inverse_mask', False)

    # evaluation settings based on tamper mode
    eval_setting = []
    if c['tamper_mode'] == 'ldm':
        eval_setting = ["ldm_spliced", "ldm_spliceless"]
    elif c['tamper_mode'] == 'controlnet':
        eval_setting = ["control_spliced", "control_spliceless"]
    elif c['tamper_mode'] == 'hdpainter':
        eval_setting = ["hdpainter_spliced", "hdpainter_spliceless"]
    elif c['tamper_mode'] == 'zero_mask':
        eval_setting = ["zero_mask"]
    elif c['tamper_mode'] == 'vae_regen':
        eval_setting = ["vae_regen"]
    elif c['tamper_mode'] == 'sdxl':
        eval_setting = ["sdxl_spliced", "sdxl_spliceless"]
    elif c['tamper_mode'] == 'flux':
        eval_setting = ["flux_spliced", "flux_spliceless"]
    elif c['tamper_mode'] == 'brushnet':
        eval_setting = ["brushnet_spliced", "brushnet_spliceless"]
    elif c['tamper_mode'] == 'cover':
        eval_setting = ["cover"]

    # clean 모드: 이미지 생성만, detection/evaluation 단계 없음
    if c['target_model'] == 'clean':
        eval_setting = []

    if c['segmentation_mask']:
        eval_setting = [s + "_segm" for s in eval_setting]
    if c['inverse_mask']:
        eval_setting = [s + "_inverse" for s in eval_setting]

    print(f"Eval_setting: {eval_setting}")

    print("-" * 30)
    print("Running Configuration:")
    print(OmegaConf.to_yaml(final_conf))
    print("-" * 30)

    set_seed(c['seed'])
    # 1) generate watermarked/ tampered images and save cover/tamper/gt/msg to disk
    save_and_print_cfg = save_and_print_config(c, c['save_path'])
    # generate_watermark_image(norm=c['normalization'],
    #                          weight_path=c['weight_path'],
    #                          target_model=c['target_model'],
    #                          src_image_path=c['src_image_path'],
    #                          save_path=c['save_path'],
    #                          edit_model_name=c['edit_model_name'],
    #                          seed=c['seed'],
    #                          num_bits=c['num_bits'],
    #                          model_size=c['model_size'],
    #                          eval_size=c['eval_size'],
    #                          start_idx=c['start_idx'],
    #                          end_idx=c['end_idx'],
    #                          tamper_mode=c['tamper_mode'],
    #                          wm_strength=c['wm_strength'],
    #                          brushnet_checkpoint_dir=c.get('brushnet_checkpoint_dir', None),
    #                          segmentation_mask=c['segmentation_mask'],
    #                          inverse_mask=c['inverse_mask'])

    # # 2) run detector over the saved spliced/spliceless images to generate predicted masks and message predictions
    refiner_tag = '_refiner' if (c['target_model'] == 'ours' and c['use_refiner']) else ''
    aug_suffix = f"_{c['aug_type']}_{c['aug_param']}" if (c['aug_type'] is not None and c['aug_param'] is not None) else ""
    for setting in eval_setting:
        generate_tamper_mask(weight_path=c['weight_path'],
                            eval_setting=setting,
                            target_model=c['target_model'],
                            save_path=c['save_path'],
                            num_bits=c['num_bits'],
                            model_size=c['model_size'],
                            end_idx=c['end_idx'],
                            aug_type=c['aug_type'],
                            aug_param=c['aug_param'],
                            wm_strength=c['wm_strength'],
                            use_refiner=c['use_refiner'],
                            save_aug_image=c.get('save_aug_image', False))
        # 3) Evaluate predicted masks against ground-truth masks saved in disk
        pred_mask_dir = f"{c['save_path']}/pred_mask_{setting}{aug_suffix}{refiner_tag}"
        gt_dir_suffix = ("_segm" if c['segmentation_mask'] else "") + ("_inverse" if c['inverse_mask'] else "")
        gt_dir = f"{c['save_path']}/gt{gt_dir_suffix}"
        eva = Evaluation(pred_mask_dir, gt_dir, eval_size=c['eval_size'], end_idx=c['end_idx'])
        eva.run(pred_mask_dir, tamper_mode=c['tamper_mode'])

    # 4) Evaluate fidelity between watermarked and original images
    # eva_fid = Evaluation_Fidelity(f"{c['save_path']}/cover_images", f"{c['src_image_path']}", eval_size=c['eval_size'], end_idx=c['end_idx'])
    # eva_fid.run(f"{c['save_path']}/cover_images")

    # 5) eval_dist: cossim distribution analysis (ours only)
    # if c.get('eval_dist', False) and c['target_model'] == 'ours':
    #     eval_cossim_distribution(
    #         target_model=c['target_model'],
    #         save_path=c['save_path'],
    #         src_image_path=c['src_image_path'],
    #         edit_model_name=c['edit_model_name'],
    #         model_size=c['model_size'],
    #         end_idx=c['end_idx'],
    #         seed=c['seed'],
    #     )