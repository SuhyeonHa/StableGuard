import argparse
import gc
import json
import os
import statistics
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image
from torchvision import transforms

from locmark.locmark import LocMark
from locmark.main import Params
from omniguard.iml_vit_model import iml_vit_model
from omniguard.model_invert import Model, init_model
from omniguard.modules.Unet_common import DWT, IWT
from stableguard.models import MoEGuidedForensicNet, MultiplexingWatermarkVAEDecoder
from watermark_anything.wam_utils import load_model_from_checkpoint


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DENORM_IMAGENET = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark per-image protection and verification time.")
    parser.add_argument("--dataset", default="/mnt/nas5/suhyeon/datasets/valAGE-Set")
    parser.add_argument("--num-images", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", default="/mnt/nas5/suhyeon/projects/apt_rebuttal/timing_results.json")
    parser.add_argument("--apt-cover-dir", default="/mnt/nas5/suhyeon/projects/apt_rebuttal/ours/cover_images")
    parser.add_argument("--cache-dir", default="/mnt/nas5/suhyeon/caches/")
    parser.add_argument("--models", default="wam,omniguard,stableguard,apt,apt_refiner")
    parser.add_argument("--wam-weight-path", default="/mnt/nas5/suhyeon/checkpoints/wam/")
    parser.add_argument("--omniguard-weight-path", default="/mnt/nas5/suhyeon/checkpoints/omniguard/")
    parser.add_argument("--stableguard-weight-path", default="/mnt/nas5/suhyeon/checkpoints/stableguard/weights/clean")
    parser.add_argument("--wam-strength", type=float, default=3.0)
    parser.add_argument("--omniguard-strength", type=float, default=2.0)
    parser.add_argument("--anchor-type", default="submitted")
    return parser.parse_args()


def cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def image_paths(folder: str, limit: int) -> List[str]:
    valid_exts = (".jpg", ".jpeg", ".png")
    paths = [
        os.path.join(folder, name)
        for name in sorted(os.listdir(folder))
        if name.lower().endswith(valid_exts)
    ]
    return paths[:limit]


def load_tensors(paths: Iterable[str], size: int, norm: str, device: torch.device) -> List[torch.Tensor]:
    to_tensor = transforms.ToTensor()
    tensors = []
    for path in paths:
        image = Image.open(path).convert("RGB").resize((size, size))
        tensor = to_tensor(image).unsqueeze(0)
        if norm == "imagenet":
            tensor = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(tensor.squeeze(0)).unsqueeze(0)
        elif norm == "rescale":
            tensor = tensor * 2.0 - 1.0
        elif norm == "unit":
            pass
        else:
            raise ValueError(f"Unknown norm='{norm}'")
        tensors.append(tensor.to(device))
    return tensors


def load_secret(size: int, device: torch.device) -> torch.Tensor:
    image = Image.open("./omniguard/bluesky_white2.png").convert("RGB").resize((size, size))
    arr = np.array(image) / 255.0
    tensor = torch.from_numpy(np.ascontiguousarray(arr)).float().permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device)


def cuda_elapsed_ms(fn: Callable[[], object]) -> Tuple[float, object]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    out = fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def summarize(times_ms: List[float]) -> Dict[str, float]:
    return {
        "n": len(times_ms),
        "mean_ms": float(statistics.mean(times_ms)),
        "std_ms": float(statistics.pstdev(times_ms)) if len(times_ms) > 1 else 0.0,
        "median_ms": float(statistics.median(times_ms)),
        "min_ms": float(min(times_ms)),
        "max_ms": float(max(times_ms)),
    }


def measure(
    items: List[object],
    fn: Callable[[object], object],
    warmup: int,
    collect_outputs: bool = False,
) -> Tuple[Dict[str, float], Optional[List[object]]]:
    warmup_items = items[: min(warmup, len(items))]
    with torch.no_grad():
        for item in warmup_items:
            fn(item)
        torch.cuda.synchronize()

        times = []
        outputs = [] if collect_outputs else None
        for item in items:
            elapsed, out = cuda_elapsed_ms(lambda item=item: fn(item))
            times.append(elapsed)
            if collect_outputs:
                outputs.append(out.detach())
    return summarize(times), outputs


def to_ms_text(stats: Optional[Dict[str, float]]) -> str:
    if stats is None:
        return "-"
    return f"{stats['mean_ms']:.2f} ms"


def imagenet_to_unit(x: torch.Tensor) -> torch.Tensor:
    return DENORM_IMAGENET(x.squeeze(0).cpu()).unsqueeze(0).to(x.device).clamp(0, 1)


def pad_and_normalize_1024(x_unit: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x_unit.device)
    std = IMAGENET_STD.to(x_unit.device)
    _, _, h, w = x_unit.shape
    pad_h = max(0, 1024 - h)
    pad_w = max(0, 1024 - w)
    padded = F.pad(x_unit, (0, pad_w, 0, pad_h), value=0.0)
    padded = padded[:, :, :1024, :1024]
    return (padded - mean) / std


def benchmark_wam(args, paths, device):
    wam = load_model_from_checkpoint(args.wam_weight_path, num_bits=32, scaling_w=args.wam_strength).to(device).eval()
    images = load_tensors(paths, size=256, norm="imagenet", device=device)
    msgs = [wam.get_random_msg(1).to(device) for _ in images]
    pairs = list(zip(images, msgs))

    embed_stats, covers = measure(
        pairs,
        lambda item: wam.embed(item[0], item[1])["imgs_w"],
        args.warmup,
        collect_outputs=True,
    )
    verify_stats, _ = measure(
        covers,
        lambda img: wam.detect(img)["preds"],
        args.warmup,
        collect_outputs=False,
    )
    del wam, images, msgs, covers
    cleanup()
    return embed_stats, verify_stats


def benchmark_stableguard(args, paths, device):
    vae = AutoencoderKL.from_pretrained(
        "stabilityai/stable-diffusion-2-1-base",
        subfolder="vae",
        cache_dir=args.cache_dir,
    ).to(device).eval()
    decoder = MultiplexingWatermarkVAEDecoder(num_bits=48).to(device).eval()
    decoder_weight = torch.load(os.path.join(args.stableguard_weight_path, "mpw_vae_decoder.bin"), map_location="cpu")
    decoder.load_state_dict(decoder_weight)
    detector = MoEGuidedForensicNet(num_bits=48).to(device).eval()
    detector_weight = torch.load(os.path.join(args.stableguard_weight_path, "moe_gfn.bin"), map_location="cpu")
    detector.load_state_dict(detector_weight)

    images = load_tensors(paths, size=256, norm="rescale", device=device)
    msgs = [(torch.bernoulli(torch.empty(1, 48, device=device).uniform_(0, 1)) + 1e-8) for _ in images]
    pairs = list(zip(images, msgs))

    def embed(item):
        image, msg = item
        latents = vae.encode(image).latent_dist.sample()
        latents = vae.post_quant_conv(latents)
        return decoder(latents, msgs=msg)

    embed_stats, covers = measure(pairs, embed, args.warmup, collect_outputs=True)
    verify_stats, _ = measure(covers, lambda img: detector(img)[1], args.warmup, collect_outputs=False)
    del vae, decoder, detector, images, msgs, covers
    cleanup()
    return embed_stats, verify_stats


def benchmark_omniguard(args, paths, device):
    net = Model(checkpoint=args.omniguard_weight_path).to(device).eval()
    init_model(net)
    state_dicts = torch.load(
        os.path.join(args.omniguard_weight_path, "model_checkpoint_01500.pt"),
        map_location="cpu",
        weights_only=False,
    )
    network_state_dict = {k.removeprefix("module."): v for k, v in state_dicts["net"].items()}
    net.load_state_dict(network_state_dict)

    extractor = iml_vit_model()
    extractor.load_state_dict(
        torch.load(os.path.join(args.omniguard_weight_path, "checkpoint-175.pth"), weights_only=False)["model"],
        strict=True,
    )
    extractor = extractor.to(device).eval()

    dwt = DWT()
    iwt = IWT()
    images = load_tensors(paths, size=512, norm="rescale", device=device)
    secret = load_secret(512, device)
    msgs = [torch.randint(2, (1, 64), dtype=torch.float32, device=device) for _ in images]
    pairs = list(zip(images, msgs))

    def embed(item):
        image, msg = item
        cover_input = dwt((image + 1.0) / 2.0)
        secret_input = dwt(secret)
        cover, _, _, _ = net(cover_input, secret_input, msg, wm_strength=args.omniguard_strength)
        return cover * 2.0 - 1.0

    def verify(image):
        image_unit = (image + 1.0) / 2.0
        output_steg = dwt(image_unit)
        output_image, _ = net(output_steg, rev=True)
        secret_rev = iwt(output_image.narrow(1, 0, 12))
        artifact = pad_and_normalize_1024(secret_rev)
        fuse = pad_and_normalize_1024(image_unit)
        pred = extractor(artifact, fuse)
        return pred[:, :, :512, :512]

    embed_stats, covers = measure(pairs, embed, args.warmup, collect_outputs=True)
    verify_stats, _ = measure(covers, verify, args.warmup, collect_outputs=False)
    del net, extractor, dwt, iwt, images, secret, msgs, covers
    cleanup()
    return embed_stats, verify_stats


def benchmark_apt(args, device, use_refiner: bool):
    paths = image_paths(args.apt_cover_dir, args.num_images)
    if not paths:
        raise FileNotFoundError(f"No APT cover images found in {args.apt_cover_dir}")
    locmark_args = Params()
    locmark_args.device = device
    locmark_args.anchor_type = args.anchor_type
    locmark = LocMark(args=locmark_args)
    images = load_tensors(paths, size=256, norm="unit", device=device)
    verify_stats, _ = measure(
        images,
        lambda img: locmark.decode_watermark(img, use_refiner=use_refiner)[1],
        args.warmup,
        collect_outputs=False,
    )
    del locmark, images
    cleanup()
    return None, verify_stats


def write_outputs(args, results):
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    md_path = os.path.splitext(args.output)[0] + ".md"
    rows = [
        ("WAM", "many GPU-days", to_ms_text(results.get("wam", {}).get("embedding")), to_ms_text(results.get("wam", {}).get("verification"))),
        ("OmniGuard", "many GPU-days", to_ms_text(results.get("omniguard", {}).get("embedding")), to_ms_text(results.get("omniguard", {}).get("verification"))),
        ("StableGuard", "many GPU-days", to_ms_text(results.get("stableguard", {}).get("embedding")), to_ms_text(results.get("stableguard", {}).get("verification"))),
        ("APT", "none", "14.34s (50 steps), 42.52s (150 steps)", to_ms_text(results.get("apt", {}).get("verification"))),
        ("APT*", "~hours (decoder)", "14.34-42.52s", to_ms_text(results.get("apt_refiner", {}).get("verification"))),
    ]
    with open(md_path, "w") as f:
        f.write("| Model | Training (one-time) | Embedding/Protection (per-image) | Verification (per-image) |\n")
        f.write("|---|---:|---:|---:|\n")
        for row in rows:
            f.write(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n")

    print(f"Saved JSON: {args.output}")
    print(f"Saved table: {md_path}")
    print(open(md_path).read())


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for GPU timing.")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    paths = image_paths(args.dataset, args.num_images)
    if len(paths) < args.num_images:
        print(f"Warning: requested {args.num_images} images, found {len(paths)}.")

    requested = {name.strip() for name in args.models.split(",") if name.strip()}
    results = {
        "metadata": {
            "dataset": args.dataset,
            "num_images": len(paths),
            "warmup": args.warmup,
            "gpu": args.gpu,
            "timing": "CUDA events; model loading and disk I/O excluded",
        }
    }

    runners = {
        "wam": lambda: benchmark_wam(args, paths, device),
        "omniguard": lambda: benchmark_omniguard(args, paths, device),
        "stableguard": lambda: benchmark_stableguard(args, paths, device),
        "apt": lambda: benchmark_apt(args, device, use_refiner=False),
        "apt_refiner": lambda: benchmark_apt(args, device, use_refiner=True),
    }

    for name, runner in runners.items():
        if name not in requested:
            continue
        print(f"===== Benchmarking {name} =====")
        try:
            embedding, verification = runner()
            results[name] = {
                "embedding": embedding,
                "verification": verification,
            }
        except Exception as exc:
            cleanup()
            results[name] = {"error": repr(exc)}
            print(f"[{name}] ERROR: {exc}")

    write_outputs(args, results)


if __name__ == "__main__":
    main()
