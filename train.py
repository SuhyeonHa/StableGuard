import os
import random
import argparse
from pathlib import Path
import itertools
import time
import random
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, StableDiffusionInpaintPipeline
from diffusers.training_utils import cast_training_params
from tqdm import tqdm
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler

from torch.utils.tensorboard import SummaryWriter
import logging
from lpips import LPIPS

from dataset import CocoDataset, collate_fn
from losses import WatsonDistanceVgg, weighted_binary_cross_entropy, dice_loss
from models import MultiplexingWatermarkVAEDecoder, MoEGuidedForensicNet
from utils_img import round_pixel
from models.mpw_vae import _get_feature_maps

import numpy as np
from evaluation import PixelF1, PixelAUC, PixelIOU, PixelAccuracy



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--mask_pool_path",
        type=str,
        default="",
        required=True,
        help="Mask pool path",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="run",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--num_bits",
        type=int,
        default=48,
        help=(
            "Number of watermark bit."
        ),
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant_with_warmup",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=5000, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--cosine_cycle_epoch",
        type=int,
        default=5,
        help=(
            "cosine_with_restarts option for cycle"
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--noise_strength", 
                    type=float, 
                    nargs='+', 
                    default=[0.0, 0.8], 
                    help="VAE decoder noise strength (e.g., --noise_strength 0.1 0.5)")
    parser.add_argument("--cache_dir", type=str, default=None, help="Path to a directory to store the pretrained models downloaded from huggingface")
    parser.add_argument("--seed", type=int, default=42, help="A seed for reproducible training.")
    parser.add_argument("--num_inference_steps_train", type=int, default=20, help="Number of inference steps during training.")
    parser.add_argument("--guidance_scale", type=float, default=7.5, help="Guidance scale for classifier-free guidance.")
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
        kwargs_handlers=[ddp_kwargs]
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(os.path.join(args.output_dir, 'images/train'), exist_ok=True)
            os.makedirs(os.path.join(args.output_dir, 'images/test'), exist_ok=True)
            
    writer = SummaryWriter(args.output_dir)
    logger = get_logger(__name__)
    logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
            filename=os.path.join(args.output_dir, 'log.log'))

    # Load scheduler, tokenizer and models.
    original_vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    mpw_vae_decoder = MultiplexingWatermarkVAEDecoder(num_bits=args.num_bits)
    moe_gfn = MoEGuidedForensicNet()
    lpips = LPIPS(net="vgg") # WatsonDistanceVgg() both Perceptual loss is ok, WatsonDistanceVgg can get better image quality

    for name, param in original_vae.decoder.named_parameters():
        if name in mpw_vae_decoder.state_dict():
            mpw_vae_decoder.state_dict()[name].copy_(param.detach().clone())
        else:
            print(name)

    # freeze parameters of models to save more memory
    original_vae.requires_grad_(False)
    mpw_vae_decoder.requires_grad_(True) # training all decoder params
    # mpw_vae_decoder.requires_grad_(False)

    # for param in mpw_vae_decoder.msg_adapters.parameters():
    #     param.requires_grad = True

    original_vae.encoder.forward = _get_feature_maps.__get__(
        original_vae.encoder, 
        original_vae.encoder.__class__
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16": # may result in ``Nan`` error 
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16": # bf16 is recommended
        weight_dtype = torch.bfloat16

    inpaint_pipe = StableDiffusionInpaintPipeline.from_pretrained( 
        "sd-legacy/stable-diffusion-inpainting",
        torch_dtype=torch.float16,
        safety_checker=None,
        cache_dir=args.cache_dir,
    ).to(accelerator.device)

    original_vae = original_vae.to(accelerator.device, dtype=weight_dtype)
    lpips = lpips.to(accelerator.device)
    mpw_vae_decoder = mpw_vae_decoder.to(accelerator.device, dtype=weight_dtype)

    cast_training_params([mpw_vae_decoder])

    # optimizer
    params_to_opt = itertools.chain(mpw_vae_decoder.msg_adapters.parameters(),
                                    moe_gfn.parameters())
    
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)

    # dataloader
    train_dataset = CocoDataset(data_root=args.data_root_path, mask_path=args.mask_pool_path, mode="train", size=args.resolution)   
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    val_dataset = CocoDataset(data_root=args.data_root_path, mask_path=args.mask_pool_path, mode="val", size=args.resolution)
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        drop_last=True,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.num_train_epochs * len(train_dataloader),
        num_cycles = int(args.num_train_epochs // args.cosine_cycle_epoch),
    )

    # Prepare everything with our `accelerator`.
    # mpw_vae_decoder, moe_gfn, optimizer, lr_scheduler, train_dataloader, val_dataloader = \
    #     accelerator.prepare(mpw_vae_decoder, moe_gfn, optimizer, lr_scheduler, train_dataloader, val_dataloader)
    mpw_vae_decoder, moe_gfn, optimizer, lr_scheduler, train_dataloader, val_dataloader, inpaint_pipe = \
    accelerator.prepare(mpw_vae_decoder, moe_gfn, optimizer, lr_scheduler, train_dataloader, val_dataloader, inpaint_pipe)

    for epoch in range(0, args.num_train_epochs):
        train_one_epoch(args, epoch, accelerator, train_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, inpaint_pipe, optimizer, lr_scheduler, lpips, writer, logger)
        val(args, epoch, accelerator, val_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, inpaint_pipe, lpips, logger)
        save_path = os.path.join(args.output_dir, f"checkpoint-last")
        accelerator.save_state(save_path, safe_serialization=False)


def train_one_epoch(args, epoch, accelerator, train_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, inpaint_pipe, optimizer, lr_scheduler, lpips, writer, logger):
    mpw_vae_decoder.train()
    moe_gfn.train()
    original_vae.eval()
    inpaint_pipe.unet.eval()
    global global_step
    begin = time.perf_counter()
    set_seed(args.seed)

    noise_generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    for step, batch in enumerate(train_dataloader):
        lr = lr_scheduler.get_last_lr()[0]
        load_data_time = time.perf_counter() - begin
        with accelerator.autocast():
            images = batch["images"]
            random_masks = batch["random_masks"] # 1: tampered region
            # Convert images to latent space
            with torch.no_grad():
                # get random watermark
                phi = torch.empty(images.size(0), args.num_bits).uniform_(0,1)
                msgs = (torch.bernoulli(phi) + 1e-8).to(accelerator.device, dtype=weight_dtype)
                    
                msgs_ = []
                random_masks_ = []
                for msg, random_mask in zip(msgs, random_masks):
                    if random.random() < 0.1: # all zero msg
                        msgs_.append(torch.zeros_like(msg))
                    else:
                        msgs_.append(msg)

                    if random.random() < 0.1: # fully untamper mask
                        random_masks_.append(torch.zeros_like(random_mask))
                    else:
                        random_masks_.append(random_mask)
                msgs = torch.stack(msgs_, dim=0)
                random_masks = torch.stack(random_masks_, dim=0)

                latents = original_vae.encode(images).latent_dist.sample()
                decode_images = original_vae.decode(latents, return_dict=False)[0]

            # watermarked image
            cover_images = mpw_vae_decoder(images, secret=msgs, vae=original_vae)

                # visualize
                # timesteps_to_visualize = list(range(0, 1000, 100)) + [999]
                # all_noisy_images = []
                # for t_val in tqdm(timesteps_to_visualize):

            orig_dtype = cover_images.dtype
            pipe_dtype = inpaint_pipe.unet.dtype

            cover_latents = original_vae.encode(cover_images).latent_dist.sample().to(dtype=pipe_dtype)
            down_masks = F.interpolate(random_masks, size=(cover_latents.shape[2], cover_latents.shape[3]), mode="nearest")
            down_masks = (down_masks > 0.5).to(dtype=pipe_dtype)

            spliced_latents = down_masks * latents + (1 - down_masks) * cover_latents
            prompt_embeds = inpaint_pipe._encode_prompt([""], accelerator.device, 1, True, None).to(dtype=pipe_dtype)
            prompt_embeds = torch.cat([prompt_embeds[0].expand(images.size(0), -1, -1),
                                                prompt_embeds[1].expand(images.size(0), -1, -1)], dim=0)
            inpaint_pipe.scheduler.set_timesteps(args.num_inference_steps_train, device=accelerator.device)

            timesteps = torch.randint(low=200, high=600, size=(images.size(0),), device=accelerator.device)
            # timesteps = torch.tensor([t_val] * images.size(0), device=accelerator.device)
            epsilon = torch.randn(cover_latents.shape, generator=noise_generator, device=accelerator.device, dtype=pipe_dtype)
            noisy_latents = inpaint_pipe.scheduler.add_noise(cover_latents, epsilon, timesteps)

            latent_model_input = torch.cat([noisy_latents] * 2)
            latent_model_input = inpaint_pipe.scheduler.scale_model_input(latent_model_input, timesteps)

            unet_mask = torch.cat([down_masks] * 2)
            unet_context = torch.cat([spliced_latents] * 2)
            latent_model_input = torch.cat([latent_model_input, unet_mask, unet_context], dim=1).to(dtype=pipe_dtype)
            
            realistic_noise_pred = inpaint_pipe.unet(latent_model_input, torch.cat([timesteps] * 2), encoder_hidden_states=prompt_embeds, return_dict=False)[0]
            
            noise_pred_uncond, noise_pred_text = realistic_noise_pred.chunk(2)
            realistic_noise_pred = noise_pred_uncond + args.guidance_scale * (noise_pred_text - noise_pred_uncond)

            # DDIM Inversion
            alpha_prod_t = inpaint_pipe.scheduler.alphas_cumprod[timesteps].to(device=noisy_latents.device, dtype=pipe_dtype)
            alpha_prod_t = alpha_prod_t.view(-1, 1, 1, 1)

            sqrt_alpha_prod_t = alpha_prod_t ** 0.5
            sqrt_one_minus_alpha_prod_t = (1 - alpha_prod_t) ** 0.5

            attacked_latent = (noisy_latents - sqrt_one_minus_alpha_prod_t * realistic_noise_pred) / sqrt_alpha_prod_t
            noisy_images = original_vae.decode(attacked_latent.to(dtype=orig_dtype), return_dict=False)[0]
            # all_noisy_images.append(noisy_images.to(dtype=orig_dtype))

            # if accelerator.is_main_process:
            #     images_to_save = []
            #     images_to_save.append(cover_images[2].detach().clone())
            #     images_to_save.append(random_masks[2].detach().clone().repeat(3, 1, 1))
            #     for img in all_noisy_images:
            #         images_to_save.append(img[2].detach().clone())
            #     result_images = torch.stack(images_to_save)
                
            #     save_image(
            #         result_images, 
            #         os.path.join(f'epoch_{epoch}_step_attack_viz.jpg'), 
            #         normalize=True, 
            #         scale_each=True, 
            #         nrow=len(images_to_save)
            #     )

            # random splicing
            rand_num = random.random()
            
            # tamper_images = random_masks * decode_images.detach().clone() + (1 - random_masks) * cover_images
            # tamper_noisy_images = random_masks * decode_images.detach().clone() + (1 - random_masks) * noisy_images

            if rand_num <= 0.5:
                tamper_images = random_masks * decode_images.detach().clone() + (1 - random_masks) * cover_images
                # tamper_images = (1 - random_masks) * decode_images.detach().clone() + random_masks * cover_images # invert
            elif rand_num > 0.5:
                tamper_images = random_masks * images.detach().clone() + (1 - random_masks) * cover_images
                # tamper_images = (1 - random_masks) * images.detach().clone() + random_masks * cover_images # invert

            tamper_noisy_images = noisy_images

            # if rand_num <= 0.5:
            #     tamper_noisy_images = random_masks * decode_images.detach().clone() + (1 - random_masks) * noisy_images
            #     # tamper_noisy_images = (1 - random_masks) * decode_images.detach().clone() + random_masks * noisy_images # invert
            # elif rand_num > 0.5:
            #     tamper_noisy_images = random_masks * images.detach().clone() + (1 - random_masks) * noisy_images
            #     # tamper_noisy_images = (1 - random_masks) * images.detach().clone() + random_masks * noisy_images # invert

            # add_quantization
            tamper_images = round_pixel(tamper_images)
            pred_mask = moe_gfn(tamper_images.to(dtype=weight_dtype))

            tamper_noisy_images = round_pixel(tamper_noisy_images)
            pred_noisy_mask = moe_gfn(tamper_noisy_images.to(dtype=weight_dtype))

            # Loss
            # similarity loss
            lpips_loss = lpips(cover_images, images.float().detach().clone()).mean() # 0.1
            mae_loss = F.l1_loss(cover_images, images.float().detach().clone()) # 0.1

            # watermark loss
            # gt_msgs = msgs[:, :, None, None].float().detach().clone().expand_as(pred_msgs)
            # msg_loss = F.binary_cross_entropy_with_logits(pred_msgs, gt_msgs)
            # noisy_msg_loss = F.binary_cross_entropy_with_logits(pred_noisy_msgs, gt_msgs)

            # tamper loss
            mask_loss = 0.2 * weighted_binary_cross_entropy(pred_mask, F.interpolate(random_masks, (pred_mask.size(2), pred_mask.size(3))).detach().clone()) + \
                        0.8 * dice_loss(pred_mask, F.interpolate(random_masks, (pred_mask.size(2), pred_mask.size(3))).detach().clone())

            noisy_mask_loss = 0.2 * weighted_binary_cross_entropy(pred_noisy_mask, F.interpolate(random_masks, (pred_noisy_mask.size(2), pred_noisy_mask.size(3))).detach().clone()) + \
                        0.8 * dice_loss(pred_noisy_mask, F.interpolate(random_masks, (pred_noisy_mask.size(2), pred_noisy_mask.size(3))).detach().clone())

            # total loss
            # loss = mae_loss + lpips_loss + msg_loss + mask_loss
            # if global_step < 1000:
            #     loss = mae_loss + lpips_loss + mask_loss
            # else:
            #     loss = mae_loss + lpips_loss + mask_loss + noisy_mask_loss
            loss = mae_loss + lpips_loss + mask_loss + noisy_mask_loss #+ msg_loss + noisy_msg_loss

            # for bit acc
            # pred_msgs_bin = torch.round(torch.sigmoid(pred_msgs))
            # pred_noisy_msgs_bin = torch.round(torch.sigmoid(pred_noisy_msgs))
            # msgs_bin = torch.round(torch.sigmoid(msgs.squeeze(1)))

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
            # avg_msg_loss = accelerator.gather(msg_loss.repeat(args.train_batch_size)).mean().item()
            # avg_noisy_msg_loss = accelerator.gather(noisy_msg_loss.repeat(args.train_batch_size)).mean().item()
            avg_mask_loss = accelerator.gather(mask_loss.repeat(args.train_batch_size)).mean().item()
            avg_noisy_mask_loss = accelerator.gather(noisy_mask_loss.repeat(args.train_batch_size)).mean().item()
            avg_mae_loss = accelerator.gather(mae_loss.repeat(args.train_batch_size)).mean().item()
            avg_lpips_loss = accelerator.gather(lpips_loss.repeat(args.train_batch_size)).mean().item()
            # avg_bit_correct = accelerator.gather(((pred_msgs_bin.eq(msgs_bin.data[:, :, None, None])).float().mean()) / (args.train_batch_size * args.num_bits)).mean().item()
            # avg_noisy_bit_correct = accelerator.gather(((pred_noisy_msgs_bin.eq(msgs_bin.data[:, :, None, None])).float().mean()) / (args.train_batch_size * args.num_bits)).mean().item()
            
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(moe_gfn.parameters()) + list(mpw_vae_decoder.parameters()), 5.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            step_time = time.perf_counter() - begin

            if accelerator.is_main_process:
                writer.add_scalar("LR", lr, global_step)
                # writer.add_scalar("Loss/bit_correct", avg_bit_correct, global_step)
                # writer.add_scalar("Loss/noisy_bit_correct", avg_noisy_bit_correct, global_step)
                writer.add_scalar("Loss/total_loss", avg_loss, global_step)
                # writer.add_scalar("Loss/msg_loss", avg_msg_loss, global_step)
                # writer.add_scalar("Loss/noisy_msg_loss", avg_noisy_msg_loss, global_step)
                writer.add_scalar("Loss/mask_loss", avg_mask_loss, global_step)
                writer.add_scalar("Loss/noisy_mask_loss", avg_noisy_mask_loss, global_step)
                writer.add_scalar("Loss/lpips_loss", avg_lpips_loss, global_step)
                writer.add_scalar("Loss/mse_loss", avg_mae_loss, global_step)
                if step % 10 == 0: # log
                    msg = (
                        f"Epoch: {epoch:02d}/{args.num_train_epochs:02d} | "
                        f"Step: {step:05d}/{len(train_dataloader):05d} | "
                        f"Global Step: {global_step:07d} | "
                        f"{'Data Time:'}{load_data_time:6.3f} | "
                        f"{'Step Time:'}{step_time:6.3f} | "
                        f"{'LR:'}{lr:6.6f} | "
                        f"{'Step Loss:'}{avg_loss:6.3f} | "
                        # f"{'Msg Loss:'}{avg_msg_loss:6.3f} | "
                        # f"{'Noisy Msg Loss:'}{avg_noisy_msg_loss:6.3f} | "
                        f"{'Mask Loss:'}{avg_mask_loss:6.3f} | "
                        f"{'Noisy Mask Loss:'}{avg_noisy_mask_loss:6.3f} | "
                        f"{'LPIPS Loss:'}{avg_lpips_loss:6.3f} | "
                        f"{'MAE Loss:'}{avg_mae_loss:6.3f} | "
                        # f"{'Bit Correct:'}{avg_bit_correct:6.3f} | "
                        # f"{'Noisy Bit Correct:'}{avg_noisy_bit_correct:6.3f}"
                    )
                    print(msg)
                    logger.info(msg)
                if step % 100 == 0: # visualization
                    result_images = torch.cat([images[:args.train_batch_size], 
                                               cover_images[:args.train_batch_size], 
                                               ((cover_images - images) *10)[:args.train_batch_size],
                                               random_masks.repeat(1, 3, 1, 1)[:args.train_batch_size], 
                                               tamper_images[:args.train_batch_size],
                                               tamper_noisy_images[:args.train_batch_size],
                                               F.sigmoid(F.interpolate(pred_mask, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size],
                                               F.sigmoid(F.interpolate(pred_noisy_mask, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size]],
                                               dim=0).detach().clone()
                    save_image(result_images, os.path.join(args.output_dir, 'images/train', '%s_%s.jpg' % (epoch, step)), normalize=True, scale_each=True, nrow=args.train_batch_size)

        global_step += 1
        
        if global_step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint_{global_step}")
            accelerator.save_state(save_path, safe_serialization=False)
        
        begin = time.perf_counter()

@torch.no_grad()
def val(args, epoch, accelerator, val_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, inpaint_pipeline, lpips, logger):
    mpw_vae_decoder.eval()
    moe_gfn.eval()
    original_vae.eval()
    # inpaint_pipeline.set_progress_bar_config(disable=True)

    f1_metric = PixelF1()
    auc_metric = PixelAUC()
    iou_metric = PixelIOU()
    acc_metric = PixelAccuracy()

    total_f1_spliceless, total_auc_spliceless, total_iou_spliceless, total_acc_spliceless, total_fpr_spliceless = 0.0, 0.0, 0.0, 0.0, 0.0
    total_f1_spliced, total_auc_spliced, total_iou_spliced, total_acc_spliced, total_fpr_spliced = 0.0, 0.0, 0.0, 0.0, 0.0
    avg_psnr, avg_lpips, avg_bit_correct_spliceless, avg_bit_correct_spliced = 0.0, 0.0, 0.0, 0.0

    for step, batch in enumerate(tqdm(val_dataloader)):
        with accelerator.autocast():
            images = batch["images"]
            random_masks = batch["random_masks"]
            
            with torch.no_grad():
                phi = torch.empty(images.size(0), args.num_bits).uniform_(0,1)
                msgs = (torch.bernoulli(phi) + 1e-8).to(accelerator.device, dtype=weight_dtype)

                msgs_ = []
                random_masks_ = []
                for msg, random_mask in zip(msgs, random_masks):
                    msgs_.append(msg)
                    random_masks_.append(random_mask)
                msgs = torch.stack(msgs_, dim=0)
                random_masks = torch.stack(random_masks_, dim=0)

                null_prompt = [""]*images.size(0)

                latents = original_vae.encode(images).latent_dist.sample()
                decode_images = original_vae.decode(latents, return_dict=False)[0]
            
            cover_images = mpw_vae_decoder(images, secret=msgs, vae=original_vae)
            cover_images = cover_images.clamp(-1.0, 1.0)

            inpainted_images = inpaint_pipeline(
                prompt=null_prompt, 
                image=cover_images.detach().clone(),
                mask_image=random_masks,
                output_type="pt",
                num_inference_steps=20
            ).images
            
            inpainted_images = F.interpolate(inpainted_images, size=(args.resolution, args.resolution))
            spliceless = (inpainted_images * 2 - 1).to(dtype=weight_dtype) # [-1, 1]
            gt_mask = F.interpolate(random_masks, (inpainted_images.size(2), inpainted_images.size(3))).detach().clone()
            
            spliced = random_masks * spliceless.detach().clone() + (1 - random_masks) * cover_images # [-1, 1]

            mask_spliceless = moe_gfn(spliceless.to(dtype=weight_dtype))
            mask_spliced = moe_gfn(spliced.to(dtype=weight_dtype))

            # msgs_spliceless_bin = torch.round(torch.sigmoid(msgs_spliceless)) # [B, num_bits, H, W]
            # msgs_spliced_bin = torch.round(torch.sigmoid(msgs_spliced)) # [B, num_bits, H, W]
            # msgs_bin = torch.round(torch.sigmoid(msgs.squeeze(1))) # [B, num_bits]

            # Spliceless (Inpaint)
            f1_sl = f1_metric.batch_update(predict=mask_spliceless, mask=gt_mask)
            auc_sl = auc_metric.batch_update(predict=mask_spliceless, mask=gt_mask)
            iou_sl = iou_metric.batch_update(predict=mask_spliceless, mask=gt_mask)
            acc_sl = acc_metric.batch_update(predict=mask_spliceless, mask=gt_mask)
            fpr_sl = f1_metric.Cal_FPR(predict=mask_spliceless, mask=gt_mask)

            # Spliced (Pixel)
            f1_sp = f1_metric.batch_update(predict=mask_spliced, mask=gt_mask)
            auc_sp = auc_metric.batch_update(predict=mask_spliced, mask=gt_mask)
            iou_sp = iou_metric.batch_update(predict=mask_spliced, mask=gt_mask)
            acc_sp = acc_metric.batch_update(predict=mask_spliced, mask=gt_mask)
            fpr_sp = f1_metric.Cal_FPR(predict=mask_spliced, mask=gt_mask)

            lpips_loss = lpips(cover_images, images.float().detach().clone()).mean()
            psnr = 10 * torch.log10(1 / F.mse_loss(cover_images, images.float().detach().clone()))

            # avg_bit_correct_spliceless += accelerator.gather((msgs_spliceless_bin.eq(msgs_bin.data[:, :, None, None]).float().mean()) / (args.train_batch_size * args.num_bits)).mean().item()
            # avg_bit_correct_spliced += accelerator.gather((msgs_spliced_bin.eq(msgs_bin.data[:, :, None, None]).float().mean()) / (args.train_batch_size * args.num_bits)).mean().item()
            
            f1_sl, auc_sl, iou_sl, acc_sl, fpr_sl = f1_sl.to(accelerator.device), auc_sl.to(accelerator.device), iou_sl.to(accelerator.device), acc_sl.to(accelerator.device), fpr_sl.to(accelerator.device)
            f1_sp, auc_sp, iou_sp, acc_sp, fpr_sp = f1_sp.to(accelerator.device), auc_sp.to(accelerator.device), iou_sp.to(accelerator.device), acc_sp.to(accelerator.device), fpr_sp.to(accelerator.device)
            lpips_loss, psnr = lpips_loss.to(accelerator.device), psnr.to(accelerator.device)

            # Spliceless
            total_f1_spliceless += accelerator.gather(f1_sl.repeat(args.train_batch_size)).mean().item()
            total_auc_spliceless += accelerator.gather(auc_sl.repeat(args.train_batch_size)).mean().item()
            total_iou_spliceless += accelerator.gather(iou_sl.repeat(args.train_batch_size)).mean().item()
            total_acc_spliceless += accelerator.gather(acc_sl.repeat(args.train_batch_size)).mean().item()
            total_fpr_spliceless += accelerator.gather(fpr_sl.repeat(args.train_batch_size)).mean().item()
            
            # Spliced
            total_f1_spliced += accelerator.gather(f1_sp.repeat(args.train_batch_size)).mean().item()
            total_auc_spliced += accelerator.gather(auc_sp.repeat(args.train_batch_size)).mean().item()
            total_iou_spliced += accelerator.gather(iou_sp.repeat(args.train_batch_size)).mean().item()
            total_acc_spliced += accelerator.gather(acc_sp.repeat(args.train_batch_size)).mean().item()
            total_fpr_spliced += accelerator.gather(fpr_sp.repeat(args.train_batch_size)).mean().item()
            
            # Perceptual
            avg_lpips += accelerator.gather(lpips_loss.repeat(args.train_batch_size)).mean().item()
            avg_psnr += accelerator.gather(psnr.repeat(args.train_batch_size)).mean().item()

        break # execute only one batch

    if accelerator.is_main_process:
        msg = "Val Epoch {:05d} \n".format(epoch)
        msg += "Spliceless - F1: {:.3f}, AUC: {:.3f}, IOU: {:.3f}, Acc: {:.3f}, FPR: {:.3f} \n".format(
            total_f1_spliceless / (step + 1),
            total_auc_spliceless / (step + 1),
            total_iou_spliceless / (step + 1),
            total_acc_spliceless / (step + 1),
            total_fpr_spliceless / (step + 1)
        )
        msg += "Spliced   - F1: {:.3f}, AUC: {:.3f}, IOU: {:.3f}, Acc: {:.3f}, FPR: {:.3f} \n".format(
            total_f1_spliced / (step + 1),
            total_auc_spliced / (step + 1),
            total_iou_spliced / (step + 1),
            total_acc_spliced / (step + 1),
            total_fpr_spliced / (step + 1)
        )
        msg += "Perceptual - PSNR: {:.3f}, LPIPS: {:.3f} \n".format(
            avg_psnr,
            avg_lpips / (step + 1)
        )
        # msg += "Bit Correct - Spliceless: {:.3f}, Spliced: {:.3f} \n".format(
        #     avg_bit_correct_spliceless / (step + 1),
        #     avg_bit_correct_spliced / (step + 1)
        # )
        print(msg)
        logger.info(msg)
        
        result_images = torch.cat([
            images[:args.train_batch_size], 
            cover_images[:args.train_batch_size], 
            ((cover_images - images) *10)[:args.train_batch_size],
            spliced[:args.train_batch_size],
            spliceless[:args.train_batch_size],
            random_masks.repeat(1, 3, 1, 1)[:args.train_batch_size], 
            F.sigmoid(F.interpolate(mask_spliced, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size],
            F.sigmoid(F.interpolate(mask_spliceless, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size]
            ], dim=0).detach().clone()
        
        save_image(result_images, os.path.join(args.output_dir, 'images/test', '%s.jpg' % epoch), normalize=True, scale_each=True, nrow=args.train_batch_size)

# @torch.no_grad()
# def val(args, epoch, accelerator, val_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, lpips, logger):
#     mpw_vae_decoder.eval()
#     moe_gfn.eval()
#     original_vae.eval()
#     avg_msg_loss = 0
#     avg_mask_loss = 0
#     avg_lpips_loss = 0
#     avg_bit_correct = 0

#     for step, batch in enumerate(tqdm(val_dataloader)):
#         with accelerator.autocast():
#             images = batch["images"]
#             random_masks = batch["random_masks"]
#             # Convert images to latent space
#             with torch.no_grad():
#                 latents = original_vae.encode(images).latent_dist.sample()
#                 decode_images = original_vae.decode(latents, return_dict=False)[0]
#                 latents = original_vae.post_quant_conv(latents) # to process for another model (not vae)
#                 # phi = torch.empty(latents.size(0), args.num_bits).uniform_(0,1)
#                 # msgs = (torch.bernoulli(phi) + 1e-8).to(accelerator.device, dtype=weight_dtype)

#             cover_images = mpw_vae_decoder(latents)

#             with torch.no_grad():
#                 cover_latents = original_vae.encode(cover_images).latent_dist.sample()
#                 # rand_strength = random.uniform(args.noise_strength[0], args.noise_strength[1])
#                 # noise = torch.randn_like(cover_latents) * rand_strength
#                 noisy_images = original_vae.decode(cover_latents, return_dict=False)[0]

#             # rand_num = random.random()
#             decode_cover = random_masks * decode_images.detach().clone() + (1 - random_masks) * cover_images
#             # decode_cover = (1 - random_masks) * decode_images.detach().clone() + random_masks * cover_images # invert

#             orig_cover = random_masks * images.detach().clone() + (1 - random_masks) * cover_images
#             # orig_cover = (1 - random_masks) * images.detach().clone() + random_masks * cover_images # invert

#             decode_noisy = random_masks * decode_images.detach().clone() + (1 - random_masks) * noisy_images
#             # decode_noisy = (1 - random_masks) * decode_images.detach().clone() + random_masks * noisy_images # invert

#             orig_noisy = random_masks * images.detach().clone() + (1 - random_masks) * noisy_images
#             # orig_noisy = (1 - random_masks) * images.detach().clone() + random_masks * noisy_images # invert

#             pred_decode_cover = moe_gfn(decode_cover.to(dtype=weight_dtype))
#             pred_orig_cover = moe_gfn(orig_cover.to(dtype=weight_dtype))
#             pred_decode_noisy = moe_gfn(decode_noisy.to(dtype=weight_dtype))
#             pred_orig_noisy = moe_gfn(orig_noisy.to(dtype=weight_dtype))

#             # msg_loss = F.binary_cross_entropy_with_logits(msg_decode_cover, msgs.float().detach().clone()) + \
#             #             F.binary_cross_entropy_with_logits(msg_orig_cover, msgs.float().detach().clone()) + \
#             #             F.binary_cross_entropy_with_logits(msg_decode_noisy, msgs.float().detach().clone()) + \
#             #             F.binary_cross_entropy_with_logits(msg_orig_noisy, msgs.float().detach().clone())
#             mask_loss = 0.8 * weighted_binary_cross_entropy(pred_decode_cover, F.interpolate(random_masks, (pred_decode_cover.size(2), pred_decode_cover.size(3))).detach().clone()) + \
#                         0.2 * dice_loss(pred_decode_cover, F.interpolate(random_masks, (pred_decode_cover.size(2), pred_decode_cover.size(3))).detach().clone()) + \
#                         0.8 * weighted_binary_cross_entropy(pred_orig_cover, F.interpolate(random_masks, (pred_orig_cover.size(2), pred_orig_cover.size(3))).detach().clone()) + \
#                         0.2 * dice_loss(pred_orig_cover, F.interpolate(random_masks, (pred_orig_cover.size(2), pred_orig_cover.size(3))).detach().clone())
#             noisy_mask_loss = 0.8 * weighted_binary_cross_entropy(pred_decode_noisy, F.interpolate(random_masks, (pred_decode_noisy.size(2), pred_decode_noisy.size(3))).detach().clone()) + \
#                         0.2 * dice_loss(pred_decode_noisy, F.interpolate(random_masks, (pred_decode_noisy.size(2), pred_decode_noisy.size(3))).detach().clone()) + \
#                         0.8 * weighted_binary_cross_entropy(pred_orig_noisy, F.interpolate(random_masks, (pred_orig_noisy.size(2), pred_orig_noisy.size(3))).detach().clone()) + \
#                         0.2 * dice_loss(pred_orig_noisy, F.interpolate(random_masks, (pred_orig_noisy.size(2), pred_orig_noisy.size(3))).detach().clone())
#             lpips_loss = lpips(cover_images, images.float().detach().clone()).mean() 

#             # pred_msgs_decode_cover_bin = torch.round(torch.sigmoid(msg_decode_cover))
#             # pred_msgs_orig_cover_bin = torch.round(torch.sigmoid(msg_orig_cover))
#             # pred_msgs_decode_noisy_bin = torch.round(torch.sigmoid(msg_decode_noisy))
#             # pred_msgs_orig_noisy_bin = torch.round(torch.sigmoid(msg_orig_noisy))
#             # msgs_bin = torch.round(torch.sigmoid(msgs.squeeze(1)))
#             # Gather the losses across all processes for logging (if we use distributed training).
#             # avg_msg_loss += accelerator.gather(msg_loss.repeat(args.train_batch_size)).mean().item()
#             avg_mask_loss += accelerator.gather(mask_loss.repeat(args.train_batch_size)).mean().item()
#             avg_noisy_mask_loss = accelerator.gather(noisy_mask_loss.repeat(args.train_batch_size)).mean().item()
#             avg_lpips_loss += accelerator.gather(lpips_loss.repeat(args.train_batch_size)).mean().item()
#             # avg_bit_correct += accelerator.gather((pred_msgs_decode_cover_bin.eq(msgs_bin.data).sum()) / (args.train_batch_size * args.num_bits)+ \
#             #                                       (pred_msgs_decode_cover_bin.eq(pred_msgs_orig_cover_bin.data).sum()) / (args.train_batch_size * args.num_bits)+ \
#             #                                       (pred_msgs_decode_cover_bin.eq(pred_msgs_decode_noisy_bin.data).sum()) / (args.train_batch_size * args.num_bits)+ \
#             #                                       (pred_msgs_decode_cover_bin.eq(pred_msgs_orig_noisy_bin.data).sum()) / (args.train_batch_size * args.num_bits)).mean().item()

#     avg_msg_loss = avg_msg_loss / (step + 1)
#     avg_mask_loss = avg_mask_loss / (step + 1)
#     avg_noisy_mask_loss = avg_noisy_mask_loss / (step + 1)
#     avg_lpips_loss = avg_lpips_loss / (step + 1)
#     avg_bit_correct = avg_bit_correct / (step + 1)

#     if accelerator.is_main_process:
#         msg = "Eval: " \
#               "Epoch {:05d}, msg_loss: {:.3f}, mask_loss: {:.3f}, noisy_mask_loss: {:.3f}, lpips_loss: {:.3f} bit correct: {:.3f} \n" \
#               "-------------------------------------------------------------------------------------------------------------------------".format(
#                 epoch, avg_msg_loss, avg_mask_loss, avg_noisy_mask_loss, avg_lpips_loss, avg_bit_correct)
#         print(msg)
#         logger.info(msg)
#         result_images = torch.cat([images[:args.train_batch_size], 
#                                    cover_images[:args.train_batch_size], 
#                                    ((cover_images - images) *10)[:args.train_batch_size],
#                                    decode_cover[:args.train_batch_size],
#                                    orig_cover[:args.train_batch_size],
#                                    decode_noisy[:args.train_batch_size],
#                                    orig_noisy[:args.train_batch_size],
#                                    random_masks.repeat(1, 3, 1, 1)[:args.train_batch_size], 
#                                    F.sigmoid(F.interpolate(pred_decode_cover, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size],
#                                    F.sigmoid(F.interpolate(pred_orig_cover, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size],
#                                    F.sigmoid(F.interpolate(pred_decode_noisy, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size],
#                                    F.sigmoid(F.interpolate(pred_orig_noisy, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size]
#                                    ], dim=0).detach().clone()
#         save_image(result_images, os.path.join(args.output_dir, 'images/test', '%s.jpg' % epoch), normalize=True, scale_each=True, nrow=args.train_batch_size)   





if __name__ == "__main__":
    global_step = 0
    main()    