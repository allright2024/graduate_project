#!/usr/bin/env python3
"""
leffa_photoguard_pgd.py

Run PGD on the reference/cloth image to perform a PhotoGuard-style attack.
The objective is to minimize the L2 distance between the Generative UNet's noise 
prediction for the adversarial cloth and a target cloth (defaulting to a gray image),
at a chosen diffusion timestep.

Key properties
--------------
- Perturbation is applied ONLY where cloth_mask == 1.
- Target is the Generative UNet output.
- No internal attention maps are recorded/optimized.

Example
-------
python leffa_photoguard_pgd.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-inpainting \
  --pretrained_model ./ckpts/virtual_tryon.pth \
  --src_image /workspace/vton/train/image \
  --ref_image /workspace/vton/train/cloth \
  --model_mask_image /workspace/vton/train/agnostic-mask \
  --cloth_mask_image /workspace/vton/train/cloth-mask \
  --densepose_image /workspace/vton/train/image-densepose \
  --target_step_index 0 \
  --dtype float16 \
  --pgd_steps 30 \
  --epsilon 8.0 \
  --alpha 1.0 \
  --output_dir ./leffa_photoguard_out
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from huggingface_hub import login, snapshot_download

# Hugging Face 토큰 적용 및 모델 다운로드
HF_TOKEN = "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"
login(token=HF_TOKEN)

if not os.path.exists("./ckpts/virtual_tryon.pth"):
    print("[INFO] Checkpoints not found. Downloading from HuggingFace Hub...")
    snapshot_download(repo_id="franciszzj/Leffa", local_dir="./ckpts", token=HF_TOKEN)
else:
    print("[INFO] Checkpoints found locally.")

from leffa.model import LeffaModel
from leffa.transform import LeffaTransform

def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device

def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a [1,3,H,W] tensor in [-1,1] to PIL RGB."""
    x = image_tensor.detach().float().cpu().clamp(-1.0, 1.0)
    x = (x[0] / 2.0 + 0.5).clamp(0.0, 1.0)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(x)

def delta_to_pil(delta_tensor: torch.Tensor, scale: float = 8.0) -> Image.Image:
    """Visualize perturbation delta in [-1,1] as an RGB image centered at 127."""
    x = delta_tensor.detach().float().cpu()[0]
    x = (x / (scale * 2.0 / 255.0)).clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)

@torch.no_grad()
def build_model(args: argparse.Namespace, device: torch.device) -> LeffaModel:
    model = LeffaModel(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_model=args.pretrained_model,
        dtype=args.dtype,
    )
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model

@torch.no_grad()
def prepare_batch_and_perturb_mask(
    args: argparse.Namespace,
    paths: Dict[str, Path],
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    transform = LeffaTransform(dataset=args.dataset)

    # 1. Prepare main batch
    batch = {
        "src_image": [Image.open(paths["src_image"]).convert("RGB")],
        "ref_image": [Image.open(paths["ref_image"]).convert("RGB")],
        "mask": [Image.open(paths["model_mask_image"])],
        "densepose": [Image.open(paths["densepose_image"]).convert("RGB")],
    }
    batch = transform(batch)

    # 2. Prepare cloth mask for perturbation
    cloth_mask_pil = Image.open(paths["cloth_mask_image"])
    cloth_mask = transform.mask_processor.preprocess(cloth_mask_pil, transform.height, transform.width)[0]
    cloth_mask = transform.prepare_mask(cloth_mask)  # [1,1,H,W]
    cloth_mask = cloth_mask.expand(-1, 3, -1, -1)

    # 3. Prepare target cloth image
    if paths.get("target_cloth_image"):
        target_pil = Image.open(paths["target_cloth_image"]).convert("RGB")
    else:
        target_arr = np.full((transform.height, transform.width, 3), 128, dtype=np.uint8)
        target_pil = Image.fromarray(target_arr, mode="RGB")
        
    target_tensor = transform.vae_processor.preprocess(target_pil, transform.height, transform.width)[0]
    target_tensor = transform.prepare_image(target_tensor)

    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
            
    cloth_mask = cloth_mask.to(device=device, dtype=moved["ref_image"].dtype)
    target_tensor = target_tensor.to(device=device, dtype=moved["ref_image"].dtype)
    return moved, cloth_mask, target_tensor

@torch.no_grad()
def resolve_target_timestep(args: argparse.Namespace, model: LeffaModel) -> int:
    device = module_device(model.vae)
    model.noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = model.noise_scheduler.timesteps

    if args.target_timestep is not None:
        return int(args.target_timestep)
    idx = int(args.target_step_index)
    if idx < 0 or idx >= len(timesteps):
        raise ValueError(f"target_step_index={idx} is out of range for {len(timesteps)} timesteps.")
    return int(timesteps[idx].item())

def get_noise_prediction(
    model: LeffaModel,
    batch: Dict[str, torch.Tensor],
    target_timestep: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    ref_image_latent: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    device = module_device(model.vae)
    vae_dtype = next(model.vae.parameters()).dtype

    src_image = batch["src_image"].to(device=device, dtype=vae_dtype)
    mask = batch["mask"].to(device=device, dtype=vae_dtype)
    densepose = batch["densepose"].to(device=device, dtype=vae_dtype)

    masked_image = src_image * (mask < 0.5)

    masked_image_latent = model.vae.encode(masked_image).latent_dist.sample()
    masked_image_latent = masked_image_latent * model.vae.config.scaling_factor

    if ref_image_latent is None:
        ref_image = batch["ref_image"].to(device=device, dtype=vae_dtype)
        ref_image_latent = model.vae.encode(ref_image).latent_dist.sample()
        ref_image_latent = ref_image_latent * model.vae.config.scaling_factor

    mask_latent = F.interpolate(mask, size=masked_image_latent.shape[-2:], mode="nearest")
    densepose_latent = F.interpolate(densepose, size=masked_image_latent.shape[-2:], mode="nearest")

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        masked_image_latent.shape,
        generator=generator,
        device=device,
        dtype=masked_image_latent.dtype,
    )

    model.noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = model.noise_scheduler.timesteps
    noise = noise * model.noise_scheduler.init_noise_sigma
    latent = noise

    masked_image_latent = torch.cat([masked_image_latent] * 2)
    ref_image_latent = torch.cat([torch.zeros_like(ref_image_latent), ref_image_latent])
    mask_latent = torch.cat([mask_latent] * 2)
    densepose_latent = torch.cat([densepose_latent] * 2)

    resolved_target_idx = None
    for idx, t in enumerate(timesteps):
        if int(t.item()) == int(target_timestep):
            resolved_target_idx = idx
            break
    if resolved_target_idx is None:
        raise ValueError(f"Target timestep {target_timestep} not found in scheduler timesteps.")

    # We run the forward pass up to the target timestep
    for idx, t in enumerate(timesteps):
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = model.noise_scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = torch.cat(
            [latent_model_input, mask_latent, masked_image_latent, densepose_latent],
            dim=1,
        )

        _, reference_features = model.unet_encoder(
            ref_image_latent,
            t,
            encoder_hidden_states=None,
            return_dict=False,
        )
        reference_features = list(reference_features)

        noise_pred = model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            reference_features=reference_features,
            return_dict=False,
        )[0]

        if int(t.item()) == int(target_timestep):
            return noise_pred

        # We only need to step if target_timestep is not the first step
        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = model.noise_scheduler.step(noise_pred, t, latent, return_dict=False)[0]

    return noise_pred

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def collect_paths(args) -> List[Dict[str, Path]]:
    ref_path = Path(args.ref_image)
    if ref_path.is_file():
        return [{
            "ref_image": Path(args.ref_image),
            "src_image": Path(args.src_image),
            "model_mask_image": Path(args.model_mask_image),
            "cloth_mask_image": Path(args.cloth_mask_image),
            "densepose_image": Path(args.densepose_image),
            "target_cloth_image": Path(args.target_cloth_image) if args.target_cloth_image else None
        }]
    
    # It's a directory
    pairs = []
    ref_files = [x for x in sorted(ref_path.iterdir()) if is_image_file(x)]
    
    # Limit to 100 images
    ref_files = ref_files[:30]
    
    for ref_file in ref_files:
        stem = ref_file.stem
        src_cand = Path(args.src_image) / ref_file.name
        cloth_mask_cand = Path(args.cloth_mask_image) / ref_file.name
        densepose_cand = Path(args.densepose_image) / ref_file.name
        
        # model mask is tricky. It could be stem.png, stem_mask.png, etc.
        model_mask_dir = Path(args.model_mask_image)
        model_mask_cand1 = model_mask_dir / f"{stem}_mask.png"
        model_mask_cand2 = model_mask_dir / f"{stem}.png"
        model_mask_cand3 = model_mask_dir / f"{stem}.jpg"
        
        model_mask_cand = None
        for cand in [model_mask_cand1, model_mask_cand2, model_mask_cand3]:
            if cand.exists():
                model_mask_cand = cand
                break
                
        if model_mask_cand is None:
            print(f"[WARN] Skipping {ref_file.name}: model_mask not found.")
            continue
            
        if not src_cand.exists() or not cloth_mask_cand.exists() or not densepose_cand.exists():
            print(f"[WARN] Skipping {ref_file.name}: One or more required files not found.")
            continue
            
        target_cand = None
        if args.target_cloth_image:
            t_cand = Path(args.target_cloth_image) / ref_file.name
            if t_cand.exists():
                target_cand = t_cand
            
        pairs.append({
            "ref_image": ref_file,
            "src_image": src_cand,
            "model_mask_image": model_mask_cand,
            "cloth_mask_image": cloth_mask_cand,
            "densepose_image": densepose_cand,
            "target_cloth_image": target_cand
        })
    return pairs

def process_one_pair(args, model, paths, target_timestep, out_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch, cloth_mask, target_tensor = prepare_batch_and_perturb_mask(args, paths, device)
    
    clean_ref = batch["ref_image"].detach().clone()

    # Pre-compute target_noise
    target_batch = dict(batch)
    target_batch["ref_image"] = target_tensor
    with torch.no_grad():
        target_noise = get_noise_prediction(
            model=model,
            batch=target_batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
        ).detach()

    # Encode clean_ref once to latent space
    with torch.no_grad():
        vae_dtype = next(model.vae.parameters()).dtype
        clean_ref_device = clean_ref.to(device=device, dtype=vae_dtype)
        clean_latent = model.vae.encode(clean_ref_device).latent_dist.sample()
        clean_latent = clean_latent * model.vae.config.scaling_factor
        clean_latent = clean_latent.detach()

    adv_latent = clean_latent.clone().detach()

    # Resize cloth_mask to latent resolution (usually 1/8)
    # cloth_mask has shape [1, 3, H, W]. We take just 1 channel to broadcast correctly with 4-channel latents.
    latent_mask = F.interpolate(cloth_mask[:, :1, :, :], size=clean_latent.shape[-2:], mode="nearest")

    # Apply epsilon and alpha directly to latent space
    eps = float(args.epsilon) * (2.0 / 255.0) * model.vae.config.scaling_factor
    alpha = float(args.alpha) * (2.0 / 255.0) * model.vae.config.scaling_factor

    if args.random_start:
        delta = torch.empty_like(adv_latent).uniform_(-eps, eps)
        delta = delta * latent_mask
        adv_latent = adv_latent + delta

    history: List[Dict[str, Any]] = []

    # Evaluate the clean image once.
    with torch.enable_grad():
        clean_noise = get_noise_prediction(
            model=model,
            batch=batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            ref_image_latent=clean_latent,
        )
        clean_loss = (clean_noise - target_noise).norm(p=2).item()
    history.append({"iteration": -1, "kind": "clean", "loss": clean_loss})
    print(f"[{paths['ref_image'].name}] [clean] loss={clean_loss:.6f}")

    for step in range(args.pgd_steps):
        adv_latent = adv_latent.detach().clone().requires_grad_(True)
        
        model.zero_grad(set_to_none=True)
        
        adv_noise = get_noise_prediction(
            model=model,
            batch=batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            ref_image_latent=adv_latent,
        )
        
        loss = (adv_noise - target_noise).norm(p=2)
        loss.backward()

        grad = adv_latent.grad
        if grad is None:
            raise RuntimeError("No gradient was produced for adv_latent.")

        with torch.no_grad():
            signed = grad.sign()
            # Gradient descent to MINIMIZE the distance to target
            adv_latent = adv_latent - alpha * signed * latent_mask
            delta = torch.clamp(adv_latent - clean_latent, min=-eps, max=eps)
            delta = delta * latent_mask
            adv_latent = clean_latent + delta
            adv_latent = adv_latent.to(dtype=clean_latent.dtype)

        loss_val = float(loss.detach().cpu().item())
        history.append({"iteration": step, "kind": "pgd", "loss": loss_val})
        if (step+1) % 10 == 0 or step == 0:
            print(f"[{paths['ref_image'].name}] [iter {step:03d}] loss={loss_val:.6f}")

    # Final evaluation on the optimized latent.
    with torch.enable_grad():
        final_noise = get_noise_prediction(
            model=model,
            batch=batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            ref_image_latent=adv_latent.detach(),
        )
        final_loss = (final_noise - target_noise).norm(p=2).item()
    history.append({"iteration": args.pgd_steps, "kind": "final", "loss": final_loss})

    out_dir.mkdir(parents=True, exist_ok=True)

    # Decode latent back to pixel space
    with torch.no_grad():
        adv_ref_decoded = model.vae.decode(adv_latent.detach() / model.vae.config.scaling_factor).sample
        adv_ref_decoded = torch.clamp(adv_ref_decoded, -1.0, 1.0)
        
        # Blend the decoded image with the original clean_ref using the original pixel-space mask
        adv_ref = clean_ref * (1.0 - cloth_mask) + adv_ref_decoded * cloth_mask

    adv_pil = tensor_to_pil(adv_ref.detach())
    delta_pil = delta_to_pil((adv_ref.detach() - clean_ref).detach(), scale=max(args.epsilon, 1.0))
    clean_pil = tensor_to_pil(clean_ref)

    clean_pil.save(out_dir / "cloth_clean.png")
    adv_pil.save(out_dir / "cloth_adv.png")
    delta_pil.save(out_dir / "cloth_delta_vis.png")

    result = {
        "config": {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "pretrained_model": args.pretrained_model,
            "src_image": str(paths["src_image"]),
            "ref_image": str(paths["ref_image"]),
            "model_mask_image": str(paths["model_mask_image"]),
            "cloth_mask_image": str(paths["cloth_mask_image"]),
            "densepose_image": str(paths["densepose_image"]),
            "target_cloth_image": str(paths["target_cloth_image"]) if paths.get("target_cloth_image") else None,
            "dataset": args.dataset,
            "target_timestep": int(target_timestep),
            "target_step_index": args.target_step_index,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "dtype": args.dtype,
            "pgd_steps": args.pgd_steps,
            "epsilon_pixels": args.epsilon,
            "alpha_pixels": args.alpha,
            "random_start": args.random_start,
        },
        "history": history,
        "final_delta_linf_normalized": float((adv_ref.detach() - clean_ref).abs().max().cpu().item()),
    }
    (out_dir / "optimization_log.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def pgd_optimize(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.dtype == "float16":
        print("[WARN] CUDA is unavailable, forcing dtype from float16 to float32.")
        args.dtype = "float32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    target_timestep = resolve_target_timestep(args, model)

    paths_list = collect_paths(args)
    if not paths_list:
        print("[ERROR] No valid image pairs found.")
        return

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    
    is_batch = len(paths_list) > 1 or Path(args.ref_image).is_dir()
    summaries = []

    for idx, paths in enumerate(paths_list, start=1):
        print(f"\n[INFO] Processing {idx}/{len(paths_list)}: {paths['ref_image'].name}")
        this_out = output_root / paths["ref_image"].stem if is_batch else output_root
        
        try:
            summary = process_one_pair(args, model, paths, target_timestep, this_out)
            summaries.append({
                "file_name": paths["ref_image"].name,
                "output_dir": str(this_out),
                "final_loss": summary["history"][-1]["loss"],
                "final_delta_linf_normalized": summary["final_delta_linf_normalized"]
            })
        except Exception as e:
            print(f"[ERROR] Failed on {paths['ref_image'].name}: {e}")

    if is_batch:
        batch_summary = {
            "num_pairs": len(summaries),
            "target_timestep": int(target_timestep),
            "items": summaries,
        }
        (output_root / "batch_summary.json").write_text(
            json.dumps(batch_summary, indent=2, ensure_ascii=False)
        )
        print(f"\n[OK] Saved batch summary to: {output_root / 'batch_summary.json'}")
    else:
        print(f"\n[OK] Saved optimized cloth to: {output_root / 'cloth_adv.png'}")
        print(f"[OK] Saved logs to: {output_root / 'optimization_log.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--src_image", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--model_mask_image", type=str, required=True)
    parser.add_argument("--cloth_mask_image", type=str, required=True)
    parser.add_argument("--densepose_image", type=str, required=True)
    parser.add_argument(
        "--target_cloth_image",
        type=str,
        default=None,
        help="Optional target cloth image to match. If none, uses gray image.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="virtual_tryon",
        choices=["virtual_tryon", "pose_transfer"],
    )
    parser.add_argument("--target_timestep", type=int, default=None)
    parser.add_argument("--target_step_index", type=int, default=0)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--pgd_steps", type=int, default=10)
    parser.add_argument(
        "--epsilon",
        type=float,
        default=8.0,
        help="Linf epsilon in pixel space (0..255). Internal scale is converted to [-1,1].",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="PGD step size in pixel space (0..255). Internal scale is converted to [-1,1].",
    )
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--output_dir", type=str, default="leffa_photoguard_out")
    return parser.parse_args()


if __name__ == "__main__":
    pgd_optimize(parse_args())
