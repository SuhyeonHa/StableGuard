import os
import random
import argparse
from pathlib import Path
import itertools
import time
import random
import torch
import torch.nn.functional as F
from torchvision import transforms
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL
from diffusers.training_utils import cast_training_params
from tqdm import tqdm
from torch import nn
import types
from torchvision.utils import save_image
from diffusers.optimization import get_scheduler
import omegaconf

from torch.utils.tensorboard import SummaryWriter
import logging
from lpips import LPIPS


from dataset import CocoDataset, collate_fn
from losses import WatsonDistanceVgg, weighted_binary_cross_entropy, dice_loss, hinge_d_loss
from aug.augmentation.augmenter import Augmenter
from models import MultiplexingWatermarkVAEDecoder, MoEGuidedForensicNet, NLayerDiscriminator
from utils_img import round_pixel



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
        default="sd-ip_adapter",
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
    parser.add_argument("--data_augmentation", type=str, default="none", help="Type of data augmentation to use at marking time. (Default: combined)")
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
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args

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
    moe_gfn = MoEGuidedForensicNet(num_bits=args.num_bits)
    discriminator = NLayerDiscriminator()
    lpips = LPIPS(net="vgg") # WatsonDistanceVgg() both Perceptual loss is ok, WatsonDistanceVgg can get better image quality
    for name, param in original_vae.decoder.named_parameters():
        if name in mpw_vae_decoder.state_dict():
            mpw_vae_decoder.state_dict()[name].copy_(param.detach().clone())
        else:
            print(name)

    # freeze parameters of models to save more memory
    original_vae.requires_grad_(False)
    mpw_vae_decoder.requires_grad_(False)

    for param in mpw_vae_decoder.msg_adapters.parameters():
        param.requires_grad = True

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    original_vae = original_vae.to(accelerator.device, dtype=weight_dtype)
    lpips = lpips.to(accelerator.device)
    discriminator = discriminator.to(accelerator.device)
    mpw_vae_decoder = mpw_vae_decoder.to(accelerator.device, dtype=weight_dtype)

    cast_training_params([mpw_vae_decoder])

    # optimizer
    params_to_opt = itertools.chain(mpw_vae_decoder.msg_adapters.parameters(),
                                    moe_gfn.parameters())
    
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

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
    mpw_vae_decoder, moe_gfn, discriminator, optimizer, optimizer_d, lr_scheduler, train_dataloader, val_dataloader = \
        accelerator.prepare(mpw_vae_decoder, moe_gfn, discriminator, optimizer, optimizer_d, lr_scheduler, train_dataloader, val_dataloader)

    for epoch in range(0, args.num_train_epochs):
        train_one_epoch(args, epoch, accelerator, train_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, discriminator, original_vae, optimizer, optimizer_d, lr_scheduler, lpips, writer, logger)
        val(args, epoch, accelerator, val_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, lpips, logger)
        save_path = os.path.join(args.output_dir, f"checkpoint-last")
        accelerator.save_state(save_path, safe_serialization=False)


def adopt_weight(weight, global_step, threshold=0, value=0.):
    if global_step < threshold:
        weight = value
    return weight


def train_one_epoch(args, epoch, accelerator, train_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, discriminator, original_vae, optimizer, optimizer_d, lr_scheduler, lpips, writer, logger):
    augmenter_cfg = omegaconf.OmegaConf.load("aug/augs.yaml")
    augmenter = Augmenter(**augmenter_cfg).to(accelerator.device)
    mpw_vae_decoder.train()
    moe_gfn.train()
    original_vae.train()
    global global_step
    begin = time.perf_counter()
    for step, batch in enumerate(train_dataloader):
        lr = lr_scheduler.get_last_lr()[0]
        load_data_time = time.perf_counter() - begin
        with accelerator.autocast():
            images = batch["images"]
            random_masks = batch["random_masks"]
            # Convert images to latent space
            with torch.no_grad():
                latents = original_vae.encode(images).latent_dist.sample()
                decode_images = original_vae.decode(latents, return_dict=False)[0]
                latents = original_vae.post_quant_conv(latents)

                # get random watermark
                phi = torch.empty(latents.size(0), args.num_bits).uniform_(0,1)
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

            # watermarked image
            cover_images = mpw_vae_decoder(latents, msgs=msgs)
  
            # train discriminator
            logit_real = discriminator(decode_images.contiguous().detach())
            logit_fake = discriminator(cover_images.contiguous().detach())
            disc_factor = adopt_weight(1.0, global_step, threshold=0)
            disc_loss = disc_factor * hinge_d_loss(logit_real, logit_fake)
            avg_disc_loss = accelerator.gather(disc_loss.repeat(args.train_batch_size)).mean().item()

            optimizer_d.zero_grad()
            accelerator.backward(disc_loss)
            optimizer_d.step()

            # face score
            logits_fake = discriminator(cover_images.contiguous())
            gen_loss = - logits_fake.mean()

            # random splicing
            rand_num = random.random()
            if rand_num <= 0.5:
                tamper_images = random_masks * decode_images.detach().clone() + (1 - random_masks) * cover_images
            elif rand_num > 0.5:
                tamper_images = random_masks * images.detach().clone() + (1 - random_masks) * cover_images   

            augmented_image, augmented_masks, selected_aug = augmenter.post_augment(tamper_images, random_masks)

            # add_quantization
            augmented_image = round_pixel(augmented_image)
            pred_msgs, pred_mask = moe_gfn(augmented_image.to(dtype=weight_dtype))

            # Loss
            # similarity loss
            lpips_loss = lpips(cover_images, decode_images.float().detach().clone()).mean() 
            mae_loss = F.l1_loss(cover_images, decode_images.float().detach().clone())

            # watermark loss
            msg_loss = F.binary_cross_entropy_with_logits(pred_msgs, msgs.float().detach().clone())

            # tamper loss
            mask_loss = 0.2 * weighted_binary_cross_entropy(pred_mask, F.interpolate(random_masks, (pred_mask.size(2), pred_mask.size(3))).detach().clone()) + \
                        0.8 * dice_loss(pred_mask, F.interpolate(random_masks, (pred_mask.size(2), pred_mask.size(3))).detach().clone())

            # total loss
            loss = mae_loss + lpips_loss + msg_loss + mask_loss + gen_loss

            # for bit acc
            pred_msgs_bin = torch.round(torch.sigmoid(pred_msgs))
            msgs_bin = torch.round(torch.sigmoid(msgs.squeeze(1)))

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
            avg_msg_loss = accelerator.gather(msg_loss.repeat(args.train_batch_size)).mean().item()
            avg_mask_loss = accelerator.gather(mask_loss.repeat(args.train_batch_size)).mean().item()
            avg_mae_loss = accelerator.gather(mae_loss.repeat(args.train_batch_size)).mean().item()
            avg_lpips_loss = accelerator.gather(lpips_loss.repeat(args.train_batch_size)).mean().item()
            avg_gen_loss = accelerator.gather(gen_loss.repeat(args.train_batch_size)).mean().item()
            avg_bit_correct = accelerator.gather(((pred_msgs_bin.eq(msgs_bin.data)).sum()) / (args.train_batch_size * args.num_bits)).mean().item()

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(list(moe_gfn.parameters()) + list(mpw_vae_decoder.parameters()), 6.0)
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()

            step_time = time.perf_counter() - begin
            if accelerator.is_main_process:
                writer.add_scalar("LR", lr, global_step)
                writer.add_scalar("Loss/bit_correct", avg_bit_correct, global_step)
                writer.add_scalar("Loss/total_loss", avg_loss, global_step)
                writer.add_scalar("Loss/msg_loss", avg_msg_loss, global_step)
                writer.add_scalar("Loss/mask_loss", avg_mask_loss, global_step)
                writer.add_scalar("Loss/lpips_loss", avg_lpips_loss, global_step)
                writer.add_scalar("Loss/mse_loss", avg_mae_loss, global_step)
                writer.add_scalar("Loss/gen_loss", avg_gen_loss, global_step)
                writer.add_scalar("Loss/disc_loss", avg_disc_loss, global_step)               
                if step % 10 == 0:
                    msg = (
                        f"Epoch: {epoch:02d}/{args.num_train_epochs:02d} | "
                        f"Step: {step:05d}/{len(train_dataloader):05d} | "
                        f"Global Step: {global_step:07d} | "
                        f"{'Data Time:'}{load_data_time:6.3f} | "
                        f"{'Step Time:'}{step_time:6.3f} | "
                        f"{'LR:'}{lr:6.6f} | "
                        f"{'Step Loss:'}{avg_loss:6.3f} | "
                        f"{'Msg Loss:'}{avg_msg_loss:6.3f} | "
                        f"{'Mask Loss:'}{avg_mask_loss:6.3f} | "
                        f"{'LPIPS Loss:'}{avg_lpips_loss:6.3f} | "
                        f"{'MAE Loss:'}{avg_mae_loss:6.3f} | "
                        f"{'Gen Loss:'}{avg_gen_loss:6.3f} | "
                        f"{'Disc Loss:'}{avg_disc_loss:6.3f} | "
                        f"{'Bit Correct:'}{avg_bit_correct:6.3f}"
                    )
                    print(msg)
                    logger.info(msg)
                if step % 100 == 0:
                    result_images = torch.cat([decode_images[:args.train_batch_size], 
                                               cover_images[:args.train_batch_size], 
                                               augmented_image[:args.train_batch_size],
                                               ((cover_images - decode_images) *10)[:args.train_batch_size],
                                               random_masks.repeat(1, 3, 1, 1)[:args.train_batch_size], 
                                               augmented_masks.repeat(1, 3, 1, 1)[:args.train_batch_size],
                                               F.sigmoid(F.interpolate(pred_mask, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size]],
                                               dim=0).detach().clone()
                    save_image(result_images, os.path.join(args.output_dir, 'images/train', '%s_%s.jpg' % (epoch, step)), normalize=True, scale_each=True, nrow=args.train_batch_size)

        global_step += 1
        
        if global_step % args.save_steps == 0:
            save_path = os.path.join(args.output_dir, f"checkpoint")
            accelerator.save_state(save_path, safe_serialization=False)
        
        begin = time.perf_counter()
                

@torch.no_grad()
def val(args, epoch, accelerator, val_dataloader, weight_dtype, mpw_vae_decoder, moe_gfn, original_vae, lpips, logger):
    mpw_vae_decoder.eval()
    moe_gfn.eval()
    original_vae.eval()
    avg_msg_loss = 0
    avg_mask_loss = 0
    avg_lpips_loss = 0
    avg_bit_correct = 0
    for step, batch in enumerate(tqdm(val_dataloader)):
        with accelerator.autocast():
            images = batch["images"]
            random_masks = batch["random_masks"]
            # Convert images to latent space
            with torch.no_grad():
                latents = original_vae.encode(images).latent_dist.sample()
                decode_images = original_vae.decode(latents, return_dict=False)[0]
                latents = original_vae.post_quant_conv(latents)
                phi = torch.empty(latents.size(0), args.num_bits).uniform_(0,1)
                msgs = (torch.bernoulli(phi) + 1e-8).to(accelerator.device, dtype=weight_dtype)

            cover_images = mpw_vae_decoder(latents, msgs=msgs)
            rand_num = random.random()
            if rand_num <= 0.5:
                tamper_images = random_masks * decode_images.detach().clone() + (1 - random_masks) * cover_images
            elif rand_num > 0.5:
                tamper_images = random_masks * images.detach().clone() + (1 - random_masks) * cover_images   

            pred_msgs, pred_mask = moe_gfn(tamper_images.to(dtype=weight_dtype))
            msg_loss = F.binary_cross_entropy_with_logits(pred_msgs, msgs.float().detach().clone())
            mask_loss = 0.2 * weighted_binary_cross_entropy(pred_mask, F.interpolate(random_masks, (pred_mask.size(2), pred_mask.size(3))).detach().clone()) + \
                        0.8 * dice_loss(pred_mask, F.interpolate(random_masks, (pred_mask.size(2), pred_mask.size(3))).detach().clone())
            lpips_loss = lpips(cover_images, decode_images.float().detach().clone()).mean() 

            pred_msgs_bin = torch.round(torch.sigmoid(pred_msgs))
            msgs_bin = torch.round(torch.sigmoid(msgs.squeeze(1)))
            # Gather the losses across all processes for logging (if we use distributed training).
            avg_msg_loss += accelerator.gather(msg_loss.repeat(args.train_batch_size)).mean().item()
            avg_mask_loss += accelerator.gather(mask_loss.repeat(args.train_batch_size)).mean().item()
            avg_lpips_loss += accelerator.gather(lpips_loss.repeat(args.train_batch_size)).mean().item()
            avg_bit_correct += accelerator.gather((pred_msgs_bin.eq(msgs_bin.data).sum()) / (args.train_batch_size * args.num_bits)).mean().item()

    avg_msg_loss = avg_msg_loss / (step + 1)
    avg_mask_loss = avg_mask_loss / (step + 1)
    avg_lpips_loss = avg_lpips_loss / (step + 1)
    avg_bit_correct = avg_bit_correct / (step + 1)

    if accelerator.is_main_process:
        msg = "Eval: " \
              "Epoch {:05d}, msg_loss: {:.3f}, mask_loss: {:.3f}, lpips_loss: {:.3f} bit correct: {:.3f} \n" \
              "-------------------------------------------------------------------------------------------------------------------------".format(
                epoch, avg_msg_loss, avg_mask_loss, avg_lpips_loss, avg_bit_correct)
        print(msg)
        logger.info(msg)
        result_images = torch.cat([decode_images[:args.train_batch_size], 
                                    cover_images[:args.train_batch_size], 
                                    ((cover_images - decode_images) *10)[:args.train_batch_size],
                                    random_masks.repeat(1, 3, 1, 1)[:args.train_batch_size], 
                                    F.sigmoid(F.interpolate(pred_mask, (args.resolution, args.resolution))).repeat(1, 3, 1, 1)[:args.train_batch_size]],
                                    dim=0).detach().clone()
        save_image(result_images, os.path.join(args.output_dir, 'images/test', '%s.jpg' % epoch), normalize=True, scale_each=True, nrow=args.train_batch_size)   





if __name__ == "__main__":
    global_step = 0
    main()    