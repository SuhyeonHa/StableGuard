import os
import warnings

from watermark_anything.modules import common
# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm
import numpy as np
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms, ToTensor
from piq import ssim, psnr, LPIPS

from evaluation import PixelF1, PixelAUC, PixelIOU, PixelAccuracy
from dataset import age_collate_fn, AGEDataset
import random
from models import MultiplexingWatermarkVAEDecoder, MoEGuidedForensicNet
import yaml
import torch.nn.functional as F

from watermark_anything.wam_utils import load_model_from_checkpoint
from omniguard.model_invert import Model, init_model
from omniguard.modules.Unet_common import DWT, IWT

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


class Evaluation(object):
    """
    Simple evaluation helper that reads predicted mask images from disk,
    compares them to ground-truth masks and accumulates pixel-level metrics.

    This class relies on PixelF1, PixelAUC, PixelIOU, PixelAccuracy interfaces
    which expose batch_update(predict=..., mask=...) and Cal_FPR(...).
    """

    def __init__(self, pred_path, gt_path) -> None:
        """
        Args:
            pred_path (str): directory containing predicted mask images/files.
            gt_path (str): directory containing ground-truth mask images/files.
        """
        self.pred_path = pred_path
        self.gt_path = gt_path
        # instantiate metric calculators
        self.f1 = PixelF1()
        self.auc = PixelAUC()
        self.iou = PixelIOU()
        self.acc = PixelAccuracy()

    def run(self, save_path):
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

        # list prediction files in prediction directory
        pred_images_path = os.listdir(self.pred_path)

        # iterate through each predicted mask filename
        for pred_image_path in tqdm(pred_images_path):
            try:
                # open predicted mask and corresponding ground-truth mask as grayscale
                pred_image = Image.open(os.path.join(self.pred_path, pred_image_path)).convert("L")
                gt_image = Image.open(os.path.join(self.gt_path, pred_image_path)).convert("L")
            except Exception:
                # skip files that cannot be opened / matched
                continue

            # convert PIL images to tensors with shape (1, C, H, W)
            pred_tensor = T.ToTensor()(pred_image).unsqueeze(0)
            gt_tensor = T.ToTensor()(gt_image).unsqueeze(0)

            # resize ground-truth to match predicted mask spatial size if needed
            gt_tensor = torch.nn.functional.interpolate(gt_tensor, (pred_tensor.size(2), pred_tensor.size(3)))

            # call evaluator batch_update for each metric (these update internal state or return stat)
            f1 = self.f1.batch_update(predict=pred_tensor, mask=gt_tensor)
            auc = self.auc.batch_update(predict=pred_tensor, mask=gt_tensor)
            iou = self.iou.batch_update(predict=pred_tensor, mask=gt_tensor)
            acc = self.acc.batch_update(predict=pred_tensor, mask=gt_tensor)
            fpr = self.f1.Cal_FPR(predict=pred_tensor, mask=gt_tensor)

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

        # compute aggregated means and print/write a summary line
        msg = f"Tampering F1:{np.mean(total_f1):.5f}, AUC:{np.mean(total_auc):.5f}, IoU: {np.mean(total_iou):.5f}, Acc: {np.mean(total_acc):.5f}, FPR: {np.mean(total_fpr):.5f}\n"
        print(msg)

        # append to record file under save_path
        with open(os.path.join(save_path, "record.txt"), "a+") as f:
            f.write(msg)


@torch.no_grad()
def generate_watermark_image(norm, weight_path, target_model, src_image_path, save_path, edit_model_name, model_img_size=512, num_bits=48, size=512):
    """
    Generate watermarked and tampered images from a source dataset using:
      - a pretrained diffusion VAE for reconstructing images (AutoencoderKL),
      - a MultiplexingWatermark VAEDecoder to inject watermark bits into latents.

    For each input image the function:
      1) encodes and decodes via the diffusion VAE,
      2) generates a random binary message vector (msgs),
      3) produces a watermarked cover image from VAE latents + msgs,
      4) composes a tampered image by replacing masked regions with generated content,
      5) computes and accumulates image-similarity metrics (PSNR/SSIM/LPIPS),
      6) saves cover/tamper/gt/msg files to disk under save_path.

    Args:
        weight_path (str): directory where model weight files reside (expects mpw_vae_decoder.bin).
        src_image_path (str): data root that ImageDataset reads from (validation set).
        save_path (str): base directory to write outputs ('cover_images', 'tamper_images', 'gt', 'msgs').
        num_bits (int): number of watermark bits to embed.
        size (int): image size used by the dataset (ImageDataset will resize).
    """
    # create output subdirectories
    res = ['cover_images', 'spliced_images', 'spliceless_images', 'gt', 'msgs']
    for n in res:
        os.makedirs(os.path.join(save_path, '%s' % n), exist_ok=True)

    # load pretrained diffusion VAE (encoder/decoder)
    original_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae", cache_dir='/mnt/nas5/suheyon/caches/')
    pipe = StableDiffusionInpaintPipeline.from_pretrained(edit_model_name, cache_dir='/mnt/nas5/suhyeon/caches/')
    generator = torch.Generator().manual_seed(42)

    # load model
    if target_model == "stableguard":
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
        wam = load_model_from_checkpoint(weight_path, num_bits).cuda().eval()

    elif target_model == "omniguard":
        net = Model(checkpoint=weight_path).cuda().eval()
        init_model(net)
        state_dicts = torch.load(os.path.join(weight_path, "model_checkpoint_01500.pt"), map_location="cpu", weights_only=False)
        network_state_dict = {k.removeprefix('module.'):v for k,v in state_dicts['net'].items()}
        net.load_state_dict(network_state_dict)

    # prepare dataloader for validation images
    val_dataset = AGEDataset(data_root=src_image_path, norm_type=norm, mode="val", size=size)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=age_collate_fn,
        batch_size=1,
        num_workers=4,
        drop_last=False,
    )

    # accumulators for image-similarity metrics (per-image appended scalars)
    sd_real_psnr_list = []
    sd_real_ssim_list = []
    sd_real_lpips_list = []

    wm_real_psnr_list = []
    wm_real_ssim_list = []
    wm_real_lpips_list = []

    wm_sd_psnr_list = []
    wm_sd_ssim_list = []
    wm_sd_lpips_list = []

    # iterate over the validation dataloader
    for batch in tqdm(val_dataloader):
        # move batch tensors to GPU
        images = batch["images"].cuda()
        # generated_images = batch["generated_images"].cuda()
        masks = batch["masks"].cuda()
        image_names = batch["image_names"]

        # embed watermark
        if target_model == "stableguard":
            # VAE encode -> sample latents -> decode back to image (reconstruction)
            latents = original_vae.encode(images).latent_dist.sample()
            decode_images = original_vae.decode(latents, return_dict=False)[0]

            # prepare latents for the watermark decoder (post-quant conv if required by model)
            latents = original_vae.post_quant_conv(latents)

            # sample random binary messages for this batch
            phi = torch.empty(latents.size(0), num_bits).uniform_(0, 1).cuda()
            msgs = (torch.bernoulli(phi) + 1e-8)  # small eps to avoid exact zeros if needed

            # produce watermarked cover images from latents+msgs
            cover_images = mpw_vae_decoder(latents, msgs=msgs)

        elif target_model == "wam":
            decode_images = torch.zeros_like(images)
            images_down = F.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
            msgs = wam.get_random_msg(1)
            outputs = wam.embed(images_down, msgs)
            cover_images = outputs['imgs_w']
            cover_images = F.interpolate(cover_images, size=(size, size), mode="bilinear", align_corners=False)
            cover_images = denormalize_tensor(cover_images, scale_each=True) # [0, 1]
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]
        
        elif target_model == "omniguard":
            decode_images = torch.zeros_like(images)
            dwt = DWT()
            image = Image.open("./omniguard/bluesky_white2.png").convert("RGB").resize((size, size))
            result = np.array(image) / 255.
            expanded_matrix = np.expand_dims(result, axis=0) 
            secret = torch.from_numpy(np.ascontiguousarray(expanded_matrix)).float()
            secret = secret.permute(0, 3, 1, 2).cuda()

            cover_input = dwt((images + 1.0) / 2.0) # [-1, 1] to [0, 1]
            secret_input = dwt(secret)
            message = torch.randint(2, (1, 64)).to(torch.float32).cuda()

            cover_images, output_z, out_temp, secret_temp = net(cover_input, secret_input, message)
            cover_images = cover_images * 2.0 - 1.0 # [-1, 1]

        # inpaint
        generated_images = pipe(prompt="", image=cover_images, mask_image=masks, generator=generator).images[0]

        # pil to tensor, normalize to [-1,1], add batch dim
        generated_images = ToTensor()(generated_images).cuda()
        generated_images = (generated_images * 2.0 - 1.0).unsqueeze(0)

        # spliced images: replace regions indicated by mask with generated content
        # spliceless images: just the generated image without splicing
        spliced_images = masks * generated_images + (1 - masks) * cover_images # operation in [-1, 1]
        spliceless_images = generated_images

        # compute image similarity metrics and append their values (per-image scalars)
        sd_real_psnr_list.append(
            psnr(denormalize_tensor(images), denormalize_tensor(decode_images), data_range=1).item()
        )
        sd_real_ssim_list.append(
            ssim(denormalize_tensor(images), denormalize_tensor(decode_images), data_range=1).item()
        )
        sd_real_lpips_list.append(
            LPIPS(reduction='none')(denormalize_tensor(images), denormalize_tensor(decode_images)).mean().item()
        )

        wm_real_psnr_list.append(
            psnr(denormalize_tensor(images), denormalize_tensor(cover_images), data_range=1).item()
        )
        wm_real_ssim_list.append(
            ssim(denormalize_tensor(images), denormalize_tensor(cover_images), data_range=1).item()
        )
        wm_real_lpips_list.append(
            LPIPS(reduction='none')(denormalize_tensor(images), denormalize_tensor(cover_images)).mean().item()
        )

        wm_sd_psnr_list.append(
            psnr(denormalize_tensor(decode_images), denormalize_tensor(cover_images), data_range=1).item()
        )
        wm_sd_ssim_list.append(
            ssim(denormalize_tensor(decode_images), denormalize_tensor(cover_images), data_range=1).item()
        )
        wm_sd_lpips_list.append(
            LPIPS(reduction='none')(denormalize_tensor(decode_images), denormalize_tensor(cover_images)).mean().item()
        )

        # save per-image outputs: cover, tamper, ground-truth mask, and message vector
        for i in range(images.size(0)):
            save_file_name = image_names[i]
            cover_image = cover_images[i]
            spliced_image = spliced_images[i]
            spliceless_image = spliceless_images[i]
            mask = masks[i]
            msg = msgs[i]

            # make each a single-image tensor and convert to uint8 PIL before saving
            cover_image = cover_image.unsqueeze(0)
            spliced_image = spliced_image.unsqueeze(0)
            spliceless_image = spliceless_image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            msg = msg.unsqueeze(0)

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
            cover_image_pil.save(os.path.join(save_path, 'cover_images', save_file_name.replace("jpg", "png")))
            spliced_image_pil.save(os.path.join(save_path, 'spliced_images', save_file_name.replace("jpg", "png")))
            spliceless_image_pil.save(os.path.join(save_path, 'spliceless_images', save_file_name.replace("jpg", "png")))

            # save ground-truth mask as image tensor and message vector as .pt file
            save_image(mask, os.path.join(save_path, 'gt', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
            torch.save(msg, os.path.join(save_path, 'msgs', save_file_name.split(".")[0] + '.pt'))

    # After processing all batches, summarise the image-similarity metrics across the dataset
    # SD vs Real:  similarity between original image and VAE reconstruction
    # WM vs Real:  similarity between original image and watermarked image
    # WM vs SD:    similarity between watermarked image and VAE reconstruction
    # note: main error source often comes from the diffusion VAE reconstruction
    msg = f"SD vs Real | PSNR: {np.mean(sd_real_psnr_list):.5f}, SSIM: {np.mean(sd_real_ssim_list):.5f}, LPIPS: {np.mean(sd_real_lpips_list):.5f} \n"
    msg += f"WM vs Real | PSNR: {np.mean(wm_real_psnr_list):.5f}, SSIM: {np.mean(wm_real_ssim_list):.5f}, LPIPS: {np.mean(wm_real_lpips_list):.5f} \n"
    msg += f"WM vs SD   | PSNR: {np.mean(wm_sd_psnr_list):.5f}, SSIM: {np.mean(wm_sd_ssim_list):.5f}, LPIPS: {np.mean(wm_sd_lpips_list):.5f} \n"
    msg += "-" * 100 + "\n"
    print(msg)

    # write summary into record.txt (overwrite previous content)
    with open(os.path.join(save_path, "record.txt"), "w") as f:
        f.write(msg)


@torch.no_grad()
def generate_tamper_mask(weight_path, eval_setting, target_model, save_path, num_bits=48, size=512):
    """
    Use the trained forensic network (MoEGuidedForensicNet) to predict:
      - the embedded messages for each tampered image
      - the tampering mask (pixel-level prediction).

    The function iterates through image files in tamper_image_path, resizes them,
    runs the detector, saves predicted masks to disk and computes bit accuracy.
    """
    # make output folder for predicted masks
    os.makedirs(os.path.join(save_path, f"pred_mask_{eval_setting}"), exist_ok=True)
    tamper_image_path = os.path.join(save_path, f"{eval_setting}_images")

    # initialize and load the detector model
    if target_model == "stableguard":
        moe_gfn = MoEGuidedForensicNet(num_bits=num_bits)
        moe_gfn_weight = torch.load(os.path.join(weight_path, "moe_gfn.bin"), map_location="cpu")
        moe_gfn.load_state_dict(moe_gfn_weight)
        moe_gfn = moe_gfn.cuda()
        moe_gfn.eval()
    elif target_model == "wam":
        wam = load_model_from_checkpoint(weight_path, num_bits).cuda().eval()
    elif target_model == "omniguard":
        net = Model(checkpoint=weight_path).cuda().eval()
        init_model(net)
        state_dicts = torch.load(os.path.join(weight_path, "model_checkpoint_01500.pt"), map_location="cpu", weights_only=False)
        network_state_dict = {k.removeprefix('module.'):v for k,v in state_dicts['net'].items()}
        net.load_state_dict(network_state_dict)

    bit_acc = []
    image_paths = os.listdir(tamper_image_path)
    image_paths.sort()

    # iterate files and run inference on each image (single-image inference)
    for image_path in tqdm(image_paths):
        # load and resize image to expected input size
        image = Image.open(os.path.join(tamper_image_path, image_path)).resize((size, size))

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

        elif target_model == "wam":
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image).unsqueeze(0)  # shape [1, C, H, W]
            image_down = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
            outputs = wam.detect(image_down)["preds"]
            pred_mask = F.sigmoid(outputs[:, 0, :, :]).unsqueeze(0)
            pred_mask = F.interpolate(pred_mask, size=(size, size), mode="bilinear", align_corners=False)
        
        elif target_model == "omniguard":
            dwt = DWT()
            iwt = IWT()
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2 - 1),
            ])
            image = transform(image).unsqueeze(0)  # shape [1, C, H, W]
            image_down = F.interpolate(image, size=(256, 256), mode="bilinear", align_corners=False)
            output_steg = dwt(image_down)
            output_image, bits = net(output_steg.cuda(), rev=True)
            secret_rev = output_image.narrow(1, 0, 12)
            secret_rev = iwt(secret_rev)

        # save predicted mask image to disk
        save_image(pred_mask, os.path.join(save_path, f"pred_mask_{eval_setting}", image_path), normalize=False, scale_each=True)

        # load ground-truth message that was saved earlier during generation step
        save_msgs = torch.load(os.path.join(save_path, 'msgs', image_path.split('.')[0] + '.pt'))

        # compute bitwise accuracy between predicted messages and saved messages
        pred_msgs_bin = torch.round(torch.sigmoid(pred_msgs))
        msgs_bin = torch.round(torch.sigmoid(save_msgs.squeeze(1)))
        acc = (((pred_msgs_bin.eq(msgs_bin.data)).sum()) / num_bits).mean().item()
        bit_acc.append(acc)

    # write bit accuracy summary to record file (append)
    msg = f"Bit Acc:{np.mean(bit_acc):.5f} \n"
    msg += "-" * 100 + "\n"
    print(msg)
    with open(os.path.join(f"pred_mask_{eval_setting}", "record.txt"), "a+") as f:
        f.write(msg)

def save_and_print_config(config, save_path):
    """
    Save and print the configuration dictionary to a YAML file.
    """
    os.makedirs(save_path, exist_ok=True)
    
    config_path = os.path.join(save_path, "config.yaml")

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
    with open('config.yaml', 'r') as f:
        default_config = yaml.safe_load(f)
    # ------------------ Configuration ------------------
    run_config = {
        'src_image_path': "/mnt/nas5/suhyeon/datasets/valAGE-Set",
        'target_model': "omniguard", # ["omniguard", "wam", "stableguard"]
        'save_path': "/mnt/nas5/suhyeon/projects/eval_spliceless/omniguard/512_valAGE_sd",
        'edit_model_name': "sd-legacy/stable-diffusion-inpainting",
        'num_bits': 48,
        'size': 512,
    }
    # ---------------------------------------------------

    c = {}
    c.update(default_config['defaults'])
    c.update(run_config)
    c['weight_path'] = default_config['weight_paths'][c['target_model']]
    c['normalization'] = default_config['normalization'][c['target_model']]
    save_and_print_cfg = save_and_print_config(c, c['save_path'])

    set_seed(c['seed'])
    # 1) generate watermarked/ tampered images and save cover/tamper/gt/msg to disk
    generate_watermark_image(norm=c['normalization'],
                             weight_path=c['weight_path'],
                             target_model=c['target_model'],
                             src_image_path=c['src_image_path'],
                             save_path=c['save_path'],
                             edit_model_name=c['edit_model_name'],
                             num_bits=c['num_bits'],
                             size=c['size'])

    # 2) run detector over the saved spliced/spliceless images to generate predicted masks and message predictions
    eval_setting = ["spliced", "spliceless"]
    for setting in eval_setting:
        generate_tamper_mask(weight_path=c['weight_path'],
                            eval_setting=setting,
                            target_model=c['target_model'],
                            save_path=c['save_path'],
                            num_bits=c['num_bits'],
                            size=c['size'])
        # 3) Evaluate predicted masks against ground-truth masks saved in disk
        eva = Evaluation(f"{c['save_path']}/pred_mask_{setting}", f"{c['save_path']}/gt")
        eva.run(f"{c['save_path']}/pred_mask_{setting}")