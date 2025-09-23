import os
os.environ["CUDA_VISIBLE_DEVICES"]='4'
import warnings
warnings.filterwarnings("ignore")
from PIL import Image
from torchvision.transforms import transforms as T
from tqdm import tqdm
import numpy as np
from diffusers import AutoencoderKL
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torchvision.transforms import transforms
from piq import ssim, psnr, LPIPS


from evaluation import PixelF1, PixelAUC, PixelIOU, PixelAccuracy
from dataset import ImageDataset, image_collate_fn
import random
from models import MultiplexingWatermarkVAEDecoder, MoEGuidedForensicNet



def denormalize_tensor(tensor, value_range=None, scale_each=True):
    tensor = tensor.clone()  # avoid modifying tensor in-place
    if value_range is not None and not isinstance(value_range, tuple):
        raise TypeError("value_range has to be a tuple (min, max) if specified. min and max are numbers")

    def norm_ip(img, low, high):
        img.clamp_(min=low, max=high)
        img.sub_(low).div_(max(high - low, 1e-5))

    def norm_range(t, value_range):
        if value_range is not None:
            norm_ip(t, value_range[0], value_range[1])
        else:
            norm_ip(t, float(t.min()), float(t.max()))

    if scale_each is True:
        for t in tensor:  # loop over mini-batch dimension
            norm_range(t, value_range)
    else:
        norm_range(tensor, value_range)
    
    return tensor

class Evalution(object):
    def __init__(self, pred_path, gt_path) -> None:
        self.pred_path = pred_path
        self.gt_path = gt_path
        self.f1 = PixelF1()
        self.auc = PixelAUC()
        self.iou = PixelIOU()
        self.acc = PixelAccuracy()
        


    def run(self, save_path):
        total_f1 = []
        total_auc = []
        total_iou = []
        total_acc = []
        total_fpr = []

        pred_images_path = os.listdir(self.pred_path)

        for pred_image_path in tqdm(pred_images_path):
            try:
                pred_image = Image.open(os.path.join(self.pred_path, pred_image_path)).convert("L")
                gt_image = Image.open(os.path.join(self.gt_path, pred_image_path)).convert("L")
            except:
                continue

            pred_tensor = T.ToTensor()(pred_image).unsqueeze(0)
            gt_tensor = T.ToTensor()(gt_image).unsqueeze(0)
            gt_tensor = torch.nn.functional.interpolate(gt_tensor, (pred_tensor.size(2), pred_tensor.size(3)))
            
            f1 = self.f1.batch_update(predict=pred_tensor, mask=gt_tensor)
            auc = self.auc.batch_update(predict=pred_tensor, mask=gt_tensor)
            iou = self.iou.batch_update(predict=pred_tensor, mask=gt_tensor)
            acc = self.acc.batch_update(predict=pred_tensor, mask=gt_tensor)
            fpr = self.f1.Cal_FPR(predict=pred_tensor, mask=gt_tensor)


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

        msg = f"Tampering F1:{np.mean(total_f1):.5f}, AUC:{np.mean(total_auc):.5f}, IoU: {np.mean(total_iou):.5f}, Acc: {np.mean(total_acc):.5f}, FPR: {np.mean(total_fpr):.5f}\n"

        print(msg)
        
        with open(os.path.join(save_path, "record.txt"), "a+") as f:
            f.write(msg)

@torch.no_grad()
def generate_watermark_image(weight_path, src_image_path, save_path, num_bits=48, size=512):
    res = ['cover_images', 'tamper_images', 'gt', 'msgs']
    for n in res:
        os.makedirs(os.path.join(save_path, '%s' % n), exist_ok=True)

    original_vae = AutoencoderKL.from_pretrained("stabilityai/stable-diffusion-2-1-base", subfolder="vae")
    mpw_vae_decoder = MultiplexingWatermarkVAEDecoder(num_bits=num_bits)
    mpw_vae_decoder_weight = torch.load(os.path.join(weight_path, "mpw_vae_decoder.bin"), map_location="cpu")
    mpw_vae_decoder.load_state_dict(mpw_vae_decoder_weight)
    original_vae = original_vae.cuda()
    mpw_vae_decoder = mpw_vae_decoder.cuda()
    original_vae.eval()
    mpw_vae_decoder.eval()

    val_dataset = ImageDataset(data_root=src_image_path, mode="val", size=size)
    val_dataloader = DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=image_collate_fn,
        batch_size=8,
        num_workers=4,
        drop_last=False,
    )

    sd_real_psnr_list = []
    sd_real_ssim_list = []
    sd_real_lpips_list = []

    wm_real_psnr_list = []
    wm_real_ssim_list = []
    wm_real_lpips_list = []

    wm_sd_psnr_list = []
    wm_sd_ssim_list = []
    wm_sd_lpips_list = []

    for batch in tqdm(val_dataloader):
        images = batch["images"].cuda()
        generated_images = batch["generated_images"].cuda()
        masks = batch["masks"].cuda()
        image_names = batch["image_names"]

        latents = original_vae.encode(images).latent_dist.sample()
        decode_images = original_vae.decode(latents, return_dict=False)[0]
        latents = original_vae.post_quant_conv(latents)
        phi = torch.empty(latents.size(0), num_bits).uniform_(0,1).cuda()
        msgs = (torch.bernoulli(phi) + 1e-8)
        cover_images = mpw_vae_decoder(latents, msgs=msgs)

        tamper_images = masks * generated_images + (1-masks) * cover_images

        
        sd_real_psnr_list.append(psnr(denormalize_tensor(images), denormalize_tensor(decode_images), data_range=1).item())
        sd_real_ssim_list.append(ssim(denormalize_tensor(images), denormalize_tensor(decode_images), data_range=1).item())
        sd_real_lpips_list.append(LPIPS(reduction='none')(denormalize_tensor(images), denormalize_tensor(decode_images)).mean().item())

        wm_real_psnr_list.append(psnr(denormalize_tensor(images), denormalize_tensor(cover_images), data_range=1).item())
        wm_real_ssim_list.append(ssim(denormalize_tensor(images), denormalize_tensor(cover_images), data_range=1).item())
        wm_real_lpips_list.append(LPIPS(reduction='none')(denormalize_tensor(images), denormalize_tensor(cover_images)).mean().item())

        wm_sd_psnr_list.append(psnr(denormalize_tensor(decode_images), denormalize_tensor(cover_images), data_range=1).item())
        wm_sd_ssim_list.append(ssim(denormalize_tensor(decode_images), denormalize_tensor(cover_images), data_range=1).item())
        wm_sd_lpips_list.append(LPIPS(reduction='none')(denormalize_tensor(decode_images), denormalize_tensor(cover_images)).mean().item())

        for i in range(images.size(0)):
            save_file_name = image_names[i]
            cover_image = cover_images[i]
            tamper_image = tamper_images[i]
            mask = masks[i]
            msg = msgs[i]

            cover_image = cover_image.unsqueeze(0)
            tamper_image = tamper_image.unsqueeze(0)
            mask = mask.unsqueeze(0)
            msg = msg.unsqueeze(0)


            cover_image = (cover_image / 2 + 0.5).clamp(0, 1) 
            cover_image = cover_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
            cover_image = (cover_image * 255).astype(np.uint8)
            cover_image_pil = Image.fromarray(cover_image)

            tamper_image = (tamper_image / 2 + 0.5).clamp(0, 1)
            tamper_image = tamper_image.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0)
            tamper_image = (tamper_image * 255).astype(np.uint8)
            tamper_image_pil = Image.fromarray(tamper_image)

            cover_image_pil.save(os.path.join(save_path, 'cover_images', save_file_name.replace("jpg", "png")))
            tamper_image_pil.save(os.path.join(save_path, 'tamper_images', save_file_name.replace("jpg", "png")))

            save_image(mask, os.path.join(save_path, 'gt', save_file_name.replace("jpg", "png")), normalize=True, scale_each=True)
            torch.save(msg, os.path.join(save_path, 'msgs', save_file_name.split(".")[0]+'.pt'))

    # SD vs Real:  Image similarity between the real image and the original diffusion VAE reconstruction
    # WM vs Real:  Image similarity between the real image and the watermarked image
    # WM vs SD:  Image similarity between the StableGuard image and the original diffusion VAE reconstruction
    # The error mainly comes from the original diffusion VAE reconstruction
    msg =  f"SD vs Real | PSNR: {np.mean(sd_real_psnr_list):.5f}, SSIM: {np.mean(sd_real_ssim_list):.5f}, LPIPS: {np.mean(sd_real_lpips_list):.5f} \n" 
    msg += f"WM vs Real | PSNR: {np.mean(wm_real_psnr_list):.5f}, SSIM: {np.mean(wm_real_ssim_list):.5f}, LPIPS: {np.mean(wm_real_lpips_list):.5f} \n" 
    msg += f"WM vs SD   | PSNR: {np.mean(wm_sd_psnr_list):.5f}, SSIM: {np.mean(wm_sd_ssim_list):.5f}, LPIPS: {np.mean(wm_sd_lpips_list):.5f} \n" 
    msg += "-" * 100 + "\n"
    print(msg)
    with open(os.path.join(save_path, "record.txt"), "w") as f:
        f.write(msg)

@torch.no_grad()
def generate_tamper_mask(weight_path, tamper_image_path, save_path, num_bits=48, size=512):
    os.makedirs(os.path.join(save_path, "pred_mask"), exist_ok=True)
    moe_gfn = MoEGuidedForensicNet(num_bits=num_bits)
    moe_gfn_weight = torch.load(os.path.join(weight_path, "moe_gfn.bin"), map_location="cpu")
    moe_gfn.load_state_dict(moe_gfn_weight)
    moe_gfn = moe_gfn.cuda()
    moe_gfn.eval()
    bit_acc = []
    image_paths = os.listdir(tamper_image_path)
    image_paths.sort()

    i = 0
    for image_path in tqdm(image_paths):
        image = Image.open(os.path.join(tamper_image_path, image_path)).resize((size, size))
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1),
        ])
        image = transform(image).unsqueeze(0)
        pred_msgs, pred_mask = moe_gfn(image.cuda())

        pred_mask = torch.sigmoid(pred_mask)
        save_image(pred_mask, os.path.join(save_path, 'pred_mask', image_path), normalize=False, scale_each=True)
        save_msgs = torch.load(os.path.join(save_path, 'msgs', image_path.split('.')[0]+'.pt'))

        pred_msgs_bin = torch.round(torch.sigmoid(pred_msgs))
        msgs_bin = torch.round(torch.sigmoid(save_msgs.squeeze(1)))
        acc = (((pred_msgs_bin.eq(msgs_bin.data)).sum()) / num_bits).mean().item()
        bit_acc.append(acc)

    msg = f"Bit Acc:{np.mean(bit_acc):.5f} \n"
    msg += "-" * 100 + "\n"
    print(msg)
    with open(os.path.join(save_path, "record.txt"), "a+") as f:
        f.write(msg)





if __name__ == "__main__":
    weight_path = "weights/clean"
    src_image_path = "datasets/demo/t2i/"
    save_path = "results/t2i"
    num_bits = 48
    size = 512

    generate_watermark_image(weight_path=weight_path,
                             src_image_path=src_image_path,
                             save_path=save_path,
                             num_bits=num_bits, size=size)
    
    generate_tamper_mask(weight_path=weight_path,
                         tamper_image_path=f"{save_path}/tamper_images",
                         save_path=save_path,
                         num_bits=num_bits,
                         size=size)

    eva = Evalution(f"{save_path}/pred_mask", f"{save_path}/gt")
    eva.run(save_path)