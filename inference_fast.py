import os
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
import numpy as np
from diffusers import AutoencoderKL
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from piq import ssim, psnr, LPIPS
from torchvision.utils import save_image

from evaluation import PixelF1, PixelAUC, PixelIOU, PixelAccuracy
from dataset import ImageDataset, image_collate_fn
from models import MultiplexingWatermarkVAEDecoder, MoEGuidedForensicNet
from utils_img import round_pixel


def denormalize_tensor(tensor, value_range=None, scale_each=True):
    """
    Denormalize a tensor for metrics that expect a [0,1] range.
    - tensor: torch.Tensor, usually with batch dimension first.
    - value_range: optional tuple (min, max) to normalize to [0,1].
    - scale_each: if True, normalize each image in the batch independently.
    Returns a cloned tensor (original not modified).
    """
    tensor = tensor.clone()
    if value_range is not None and not isinstance(value_range, tuple):
        raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

    # in-place normalization helper: clamp, shift, scale
    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    # normalize using provided value_range or the tensor's min/max
    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        # loop over mini-batch dimension and normalize each image separately
        for t in tensor:
            norm_range(t, value_range)
    else:
        # normalize the whole tensor together
        norm_range(tensor, value_range)
    return tensor


# ----------------- streaming pipeline (batch-by-batch online computation) -----------------
@torch.no_grad()
def pipeline_streaming(weight_path,
                       src_image_path,
                       save_path,
                       num_bits=48,
                       size=512,
                       batch_size=8,
                       device="cuda",
                       save_intermediate=False):
    """
    Streaming pipeline that:
      - Generates watermarked / tampered images batch by batch,
      - Runs the detector on each batch immediately,
      - Computes and accumulates metrics for each batch (no intermediate images are saved by default).

    Args:
        weight_path: path to model weights (expects mpw_vae_decoder.bin and moe_gfn.bin inside).
        src_image_path: root path to validation images used by ImageDataset (mode="val").
        save_path: directory where optional records / saved outputs are written.
        num_bits: number of watermark bits.
        size: target image size for dataset and models.
        batch_size: dataloader batch size.
        device: torch device string, e.g., "cuda" or "cpu".
        save_intermediate: if True, predicted masks and message tensors will be saved to disk
                           (default False to reduce I/O).
    """

    device = torch.device(device)

    # Load the pretrained diffusion VAE (encoder/decoder)
    original_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae").to(device)

    # Initialize and load the MultiplexingWatermark VAE decoder
    mpw_vae_decoder = MultiplexingWatermarkVAEDecoder(num_bits=num_bits)
    mpw_vae_decoder_weight = torch.load(os.path.join(weight_path, "mpw_vae_decoder.bin"), map_location="cpu")
    mpw_vae_decoder.load_state_dict(mpw_vae_decoder_weight)
    mpw_vae_decoder = mpw_vae_decoder.to(device)

    # Initialize and load the Mixture-of-Experts Guided Forensic Network (detector)
    moe_gfn = MoEGuidedForensicNet(num_bits=num_bits)
    moe_gfn_weight = torch.load(os.path.join(weight_path, "moe_gfn.bin"), map_location="cpu")
    moe_gfn.load_state_dict(moe_gfn_weight)
    moe_gfn = moe_gfn.to(device)

    # Set models to evaluation mode
    original_vae.eval(); mpw_vae_decoder.eval(); moe_gfn.eval()

    # ---------------- DataLoader ----------------
    val_dataset = ImageDataset(data_root=src_image_path, mode="val", size=size)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=image_collate_fn,
        batch_size=batch_size,
        num_workers=4,
        drop_last=False,
        pin_memory=True
    )

    # ---------------- Initialize metric accumulators and evaluators ----------------
    # Pixel-level evaluators that will be updated per-batch
    f1_eval = PixelF1()
    auc_eval = PixelAUC()
    iou_eval = PixelIOU()
    acc_eval = PixelAccuracy()

    # Lists to accumulate image-similarity metrics across batches (we store per-batch means)
    sd_real_psnr = []
    sd_real_ssim = []
    sd_real_lpips = []

    wm_real_psnr = []
    wm_real_ssim = []
    wm_real_lpips = []

    wm_sd_psnr = []
    wm_sd_ssim = []
    wm_sd_lpips = []

    # Accumulate bit accuracy per batch
    bit_acc_list = []

    # Lists for pixel-level metrics collected per-batch (we append each batch's scalar metrics)
    total_f1 = []
    total_auc = []
    total_iou = []
    total_acc = []
    total_fpr = []

    # Optionally prepare directories for saving intermediate predictions
    if save_intermediate:
        os.makedirs(os.path.join(save_path, "pred_mask"), exist_ok=True)
        os.makedirs(os.path.join(save_path, "msgs"), exist_ok=True)

    # ---------------- Process dataset batch-by-batch ----------------
    for batch in tqdm(val_dataloader):
        # 1) Fetch batch and move tensors to device
        images = batch["images"].to(device)               # expected range [-1, 1]
        generated_images = batch["generated_images"].to(device)
        masks = batch["masks"].to(device)
        image_names = batch["image_names"]                # list of image file names

        B = images.size(0)

        # 2) Encode/decode using the original diffusion VAE to obtain reconstruction
        latents = original_vae.encode(images).latent_dist.sample()
        decode_images = original_vae.decode(latents, return_dict=False)[0]
        latents = original_vae.post_quant_conv(latents)

        # Randomly sample message bits for the batch
        phi = torch.empty(latents.size(0), num_bits, device=device).uniform_(0,1)
        msgs = (torch.bernoulli(phi) + 1e-8)  # shape: [B, num_bits]

        # Generate watermarked cover images using the watermark VAE decoder
        cover_images = mpw_vae_decoder(latents, msgs=msgs)  # assumed output range [-1, 1]

        # Create tampered images by blending generated_images into masked regions
        tamper_images = masks * generated_images + (1-masks) * cover_images
        # quantize pixel values if needed (utility function)
        tamper_images = round_pixel(tamper_images)

        # 3) Compute image-similarity metrics for the batch and accumulate
        # Use denormalize_tensor to bring tensors to [0,1] expected by PIQ metrics
        try:
            # Compute and store batch-mean PSNR/SSIM/LPIPS for SD vs Real
            sd_real_psnr.append(
                psnr(denormalize_tensor(images), denormalize_tensor(decode_images), data_range=1).mean().cpu().item()
            )
            sd_real_ssim.append(
                ssim(denormalize_tensor(images), denormalize_tensor(decode_images), data_range=1).mean().cpu().item()
            )
            sd_real_lpips.append(
                LPIPS(reduction='none')(denormalize_tensor(images), denormalize_tensor(decode_images)).mean().cpu().item()
            )

            # WM vs Real metrics (watermarked vs original)
            wm_real_psnr.append(
                psnr(denormalize_tensor(images), denormalize_tensor(cover_images), data_range=1).mean().cpu().item()
            )
            wm_real_ssim.append(
                ssim(denormalize_tensor(images), denormalize_tensor(cover_images), data_range=1).mean().cpu().item()
            )
            wm_real_lpips.append(
                LPIPS(reduction='none')(denormalize_tensor(images), denormalize_tensor(cover_images)).mean().cpu().item()
            )

            # WM vs SD metrics (watermarked vs VAE reconstruction)
            wm_sd_psnr.append(
                psnr(denormalize_tensor(decode_images), denormalize_tensor(cover_images), data_range=1).mean().cpu().item()
            )
            wm_sd_ssim.append(
                ssim(denormalize_tensor(decode_images), denormalize_tensor(cover_images), data_range=1).mean().cpu().item()
            )
            wm_sd_lpips.append(
                LPIPS(reduction='none')(denormalize_tensor(decode_images), denormalize_tensor(cover_images)).mean().cpu().item()
            )
        except Exception as e:
            # If PIQ metrics fail for some input, log a warning and continue
            print("Warning: similarity metric error:", e)

        # 4) Run the detector on this batch (batch inference)
        # The detector is expected to return predicted messages and predicted masks
        pred_msgs_batch, pred_masks_batch = moe_gfn(tamper_images)   # expected shapes: [B, num_bits], [B, 1, H, W]
        pred_masks_batch = torch.sigmoid(pred_masks_batch)          # convert logits to probabilities in [0,1]

        # 5) Compute bit accuracy for the batch and accumulate
        pred_msgs_sig = torch.sigmoid(pred_msgs_batch)
        gt_msgs_sig = torch.sigmoid(msgs)  # msgs are the ground-truth bits we sampled earlier
        pred_bin = torch.round(pred_msgs_sig.view(B, -1))
        gt_bin = torch.round(gt_msgs_sig.view(B, -1))
        equal_bits = (pred_bin == gt_bin).float().sum().item()
        bit_acc = equal_bits / (B * pred_bin.size(1))
        bit_acc_list.append(bit_acc)

        # 6) Compute pixel-level detection metrics using the evaluators' batch_update
        # Move predictions and ground-truth masks to CPU to be compatible with existing evaluators
        pred_for_eval = pred_masks_batch.detach().cpu()
        # Ensure ground-truth masks take the same channel layout; we select the first channel if masks have extra dims
        gt_for_eval = masks.detach().cpu()[:, 0:1, :, :]
        # If sizes differ, interpolate ground-truth to match predicted mask size
        if pred_for_eval.shape[2:] != gt_for_eval.shape[2:]:
            gt_for_eval = torch.nn.functional.interpolate(
                gt_for_eval, size=pred_for_eval.shape[2:], mode='bilinear', align_corners=False
            )

        # Call batch_update and per-batch metric functions, then append safe scalar values
        try:
            f1 = f1_eval.batch_update(predict=pred_for_eval, mask=gt_for_eval).mean()
            auc = auc_eval.batch_update(predict=pred_for_eval, mask=gt_for_eval).mean()
            iou = iou_eval.batch_update(predict=pred_for_eval, mask=gt_for_eval).mean()
            acc = acc_eval.batch_update(predict=pred_for_eval, mask=gt_for_eval).mean()
            fpr = f1_eval.Cal_FPR(predict=pred_for_eval, mask=gt_for_eval).mean()

            # Append scalars safely (handle torch.Tensor and Python numbers)
            if isinstance(f1, torch.Tensor):
                if not torch.any(torch.isnan(f1)): total_f1.append(f1.item())
            else:
                total_f1.append(float(f1))

            if isinstance(auc, torch.Tensor):
                if not torch.any(torch.isnan(auc)): total_auc.append(auc.item())
            else:
                total_auc.append(float(auc))

            if isinstance(iou, torch.Tensor):
                if not torch.any(torch.isnan(iou)): total_iou.append(iou.item())
            else:
                total_iou.append(float(iou))

            if isinstance(acc, torch.Tensor):
                if not torch.any(torch.isnan(acc)): total_acc.append(acc.item())
            else:
                total_acc.append(float(acc))

            if isinstance(fpr, torch.Tensor):
                if not torch.any(torch.isnan(fpr)): total_fpr.append(fpr.item())
            else:
                total_fpr.append(float(fpr))

        except Exception as e:
            print("Warning: evaluator batch_update error:", e)

        # 7) Optional: save this batch's predicted masks as images to disk (usually disabled)
        if save_intermediate:
            # pred_masks_batch: [B,1,H,W]
            for idx in range(B):
                name = image_names[idx]
                save_image(pred_masks_batch[idx].detach().cpu(), os.path.join(save_path, "pred_mask", name),
                           normalize=False, scale_each=True)

        # 8) Free batch-level intermediate tensors to reduce peak GPU memory usage
        del images, generated_images, masks, latents, decode_images, cover_images, tamper_images, msgs
        del pred_msgs_batch, pred_masks_batch, pred_for_eval, gt_for_eval, pred_msgs_sig, gt_msgs_sig, pred_bin, gt_bin
        torch.cuda.empty_cache()

    # ---------------- After all batches: compute final summaries ----------------
    # Build a readable summary for image-similarity metrics (means across batches)
    summary_msg = []
    if sd_real_psnr:
        summary_msg.append(
            f"SD vs Real | PSNR: {np.mean(sd_real_psnr):.5f}, SSIM: {np.mean(sd_real_ssim):.5f}, LPIPS: {np.mean(sd_real_lpips):.5f}"
        )
    if wm_real_psnr:
        summary_msg.append(
            f"WM vs Real | PSNR: {np.mean(wm_real_psnr):.5f}, SSIM: {np.mean(wm_real_ssim):.5f}, LPIPS: {np.mean(wm_real_lpips):.5f}"
        )
    if wm_sd_psnr:
        summary_msg.append(
            f"WM vs SD   | PSNR: {np.mean(wm_sd_psnr):.5f}, SSIM: {np.mean(wm_sd_ssim):.5f}, LPIPS: {np.mean(wm_sd_lpips):.5f}"
        )
    summary_msg = "\n".join(summary_msg) + "\n" + "-"*100 + "\n"
    print(summary_msg)

    # Helper to format arrays to mean with 5 decimal places (or None if empty)
    def _fmt5(arr):
        """Return mean(arr) rounded to 5 decimal places, or None if arr is empty."""
        return None if not arr else float(f"{np.mean(arr):.5f}")

    # Final detection metrics aggregated across batches (rounded to 5 decimals)
    final_detection = {
        "Tampering F1": _fmt5(total_f1),
        "AUC": _fmt5(total_auc),
        "IoU": _fmt5(total_iou),
        "Acc": _fmt5(total_acc),
        "FPR": _fmt5(total_fpr),
        "Bit Acc": _fmt5(bit_acc_list)
    }

    # Write the summaries to record.txt under save_path (creates directory if needed)
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, "record.txt"), "a+") as f:
        f.write(summary_msg)
        f.write(str(final_detection) + "\n")

    print("Final detection metrics:", final_detection)
    return summary_msg, final_detection


# -------------------- Main entrypoint --------------------
if __name__ == "__main__":
    weight_path = "weights/clean"
    src_image_path = "/mnt/nas5/suhyeon/datasets/coco-2017/val2017"
    save_path = "/mnt/nas5/suhyeon/projects/freq-loc/evaluation_results/stableguard"
    num_bits = 48
    size = 512
    batch_size = 8
    device = "cuda"

    # By default we save intermediate predictions to disk in this run (set save_intermediate=False to disable)
    summary_msg, final_detection = pipeline_streaming(weight_path=weight_path,
                                                      src_image_path=src_image_path,
                                                      save_path=save_path,
                                                      num_bits=num_bits,
                                                      size=size,
                                                      batch_size=batch_size,
                                                      device=device,
                                                      save_intermediate=True)
