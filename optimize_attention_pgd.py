#!/usr/bin/env python3
"""
optimize_attention_pgd.py

Run PGD on the cloth/reference image so that the sum of three Leffa attention-map
L2 norms becomes smaller at a chosen diffusion timestep.

Optimized objective (summed over selected layers):
    1) reference_self        : reference UNet attn1 self-attention
    2) denoising_self        : generative UNet attn1 current->current attention
    3) denoising_ref_cross   : generative UNet attn1 current->reference attention

The PGD update is applied ONLY inside the cloth mask region (mask==1).
Leffa model weights stay frozen and the model runs in eval mode.

Important
---------
- This script is intended primarily for target_step_index=0, which is much more memory-friendly.
- For virtual try-on, `--model_mask_image` should ideally be the person/inpainting mask used by Leffa.
- `--cloth_mask_image` is the perturbation support mask for the reference/cloth image.
- The "cross" term here is NOT Leffa attn2. It is the current->reference block inside
  Leffa's modified attn1 in the generative UNet.

Example
-------
python optimize_attention_pgd.py \
  --pretrained_model_name_or_path ./ckpts/stable-diffusion-inpainting \
  --pretrained_model ./ckpts/virtual_tryon.pth \
  --src_image /workspace/train/image/00000_00.jpg \
  --ref_image /workspace/train/cloth/00000_00.jpg \
  --model_mask_image /workspace/train/agnostic-mask/00000_00.png \
  --cloth_mask_image /workspace/train/cloth-mask/00000_00.jpg \
  --densepose_image /workspace/train/image-densepose/00000_00.jpg \
  --target_step_index 0 \
  --layer_scope mid \
  --dtype float16 \
  --pgd_steps 10 \
  --epsilon 8.0 \
  --alpha 1.0 \
  --output_dir ./pgd_attention_out
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from leffa.model import LeffaModel, SkipAttnProcessor
from leffa.transform import LeffaTransform


MID_TOKEN = "mid_block"
CATEGORIES = ["reference_self", "denoising_self", "denoising_ref_cross"]


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


class LossContext:
    def __init__(
        self,
        target_timestep: int,
        layer_scope: str = "mid",
        layer_substr: Optional[str] = None,
    ) -> None:
        self.target_timestep = int(target_timestep)
        self.layer_scope = layer_scope
        self.layer_substr = layer_substr
        self.current_timestep: Optional[int] = None
        self.reset()

    def reset(self) -> None:
        self.loss_terms: Dict[str, List[torch.Tensor]] = {k: [] for k in CATEGORIES}
        self.layer_records: Dict[str, List[Dict[str, Any]]] = {k: [] for k in CATEGORIES}

    def should_record(self, layer_name: str) -> bool:
        if self.current_timestep != self.target_timestep:
            return False
        if self.layer_scope == "mid" and MID_TOKEN not in layer_name:
            return False
        if self.layer_substr and self.layer_substr not in layer_name:
            return False
        return True

    def add_loss(self, category: str, layer_name: str, tensor: torch.Tensor) -> None:
        norm = torch.linalg.vector_norm(tensor.float())
        self.loss_terms[category].append(norm)
        self.layer_records[category].append(
            {
                "layer_name": layer_name,
                "l2": float(norm.detach().cpu().item()),
                "shape": list(tensor.shape),
            }
        )

    def reduced(self, device: torch.device) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        zero = torch.zeros((), device=device, dtype=torch.float32)
        for key in CATEGORIES:
            out[key] = torch.stack(self.loss_terms[key]).sum() if self.loss_terms[key] else zero
        out["total"] = out["reference_self"] + out["denoising_self"] + out["denoising_ref_cross"]
        return out

    def detached_summary(self) -> Dict[str, Any]:
        cat_sums = {}
        for key in CATEGORIES:
            cat_sums[key] = float(sum(row["l2"] for row in self.layer_records[key]))
        cat_sums["total"] = float(sum(cat_sums[k] for k in CATEGORIES))
        return {
            "category_sums": cat_sums,
            "layers": self.layer_records,
        }


class ReferenceLossAttnProcessor(torch.nn.Module):
    def __init__(self, layer_name: str, ctx: LossContext) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.ctx = ctx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        scale = 1.0 / math.sqrt(head_dim)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self.ctx.should_record(self.layer_name):
            self.ctx.add_loss("reference_self", self.layer_name, attn_probs)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


class GenerativeLossAttnProcessor(torch.nn.Module):
    def __init__(self, layer_name: str, ctx: LossContext) -> None:
        super().__init__()
        self.layer_name = layer_name
        self.ctx = ctx

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        scale = 1.0 / math.sqrt(head_dim)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        attn_probs = torch.softmax(attn_scores, dim=-1)

        if self.ctx.should_record(self.layer_name):
            total_q = attn_probs.shape[-2]
            total_k = attn_probs.shape[-1]
            if total_q % 2 != 0 or total_k % 2 != 0:
                raise ValueError(
                    f"Expected even token counts in generative attn1 for {self.layer_name}, got q={total_q}, k={total_k}."
                )
            q_cur = total_q // 2
            k_cur = total_k // 2
            denoising_self = attn_probs[:, :, :q_cur, :k_cur]
            denoising_ref_cross = attn_probs[:, :, :q_cur, k_cur:]
            self.ctx.add_loss("denoising_self", self.layer_name, denoising_self)
            self.ctx.add_loss("denoising_ref_cross", self.layer_name, denoising_ref_cross)

        hidden_states = torch.matmul(attn_probs, value)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor
        return hidden_states


@torch.no_grad()
def install_loss_processors(unet: torch.nn.Module, ctx: LossContext, branch: str) -> None:
    processors: Dict[str, torch.nn.Module] = {}
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            if branch == "reference":
                processors[name] = ReferenceLossAttnProcessor(name, ctx)
            elif branch == "generative":
                processors[name] = GenerativeLossAttnProcessor(name, ctx)
            else:
                raise ValueError(f"Unsupported branch: {branch}")
        elif name.endswith("attn2.processor"):
            processors[name] = SkipAttnProcessor()
        else:
            processors[name] = proc if proc is not None else SkipAttnProcessor()
    unet.set_attn_processor(processors)


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
    device: torch.device,
) -> tuple[Dict[str, torch.Tensor], torch.Tensor]:
    transform = LeffaTransform(dataset=args.dataset)

    batch = {
        "src_image": [Image.open(args.src_image).convert("RGB")],
        "ref_image": [Image.open(args.ref_image).convert("RGB")],
        "mask": [Image.open(args.model_mask_image)],
        "densepose": [Image.open(args.densepose_image).convert("RGB")],
    }
    batch = transform(batch)

    cloth_mask_pil = Image.open(args.cloth_mask_image)
    cloth_mask = transform.mask_processor.preprocess(cloth_mask_pil, transform.height, transform.width)[0]
    cloth_mask = transform.prepare_mask(cloth_mask)  # [1,1,H,W]
    cloth_mask = cloth_mask.expand(-1, 3, -1, -1)

    moved = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    cloth_mask = cloth_mask.to(device=device, dtype=moved["ref_image"].dtype)
    return moved, cloth_mask


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


def run_and_compute_loss(
    model: LeffaModel,
    batch: Dict[str, torch.Tensor],
    target_timestep: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    ref_acceleration: bool,
    ctx: LossContext,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    ctx.reset()

    device = module_device(model.vae)
    vae_dtype = next(model.vae.parameters()).dtype

    src_image = batch["src_image"].to(device=device, dtype=vae_dtype)
    ref_image = batch["ref_image"].to(device=device, dtype=vae_dtype)
    mask = batch["mask"].to(device=device, dtype=vae_dtype)
    densepose = batch["densepose"].to(device=device, dtype=vae_dtype)

    masked_image = src_image * (mask < 0.5)

    masked_image_latent = model.vae.encode(masked_image).latent_dist.sample()
    ref_image_latent = model.vae.encode(ref_image).latent_dist.sample()
    masked_image_latent = masked_image_latent * model.vae.config.scaling_factor
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

    reference_features = None
    cached_ref_timestep = None
    if ref_acceleration:
        cached_ref_timestep = int(timesteps[num_inference_steps // 2].item())
        ctx.current_timestep = cached_ref_timestep
        _, reference_features = model.unet_encoder(
            ref_image_latent,
            timesteps[num_inference_steps // 2],
            encoder_hidden_states=None,
            return_dict=False,
        )
        reference_features = list(reference_features)

    for _, t in enumerate(timesteps):
        current_t = int(t.item())
        latent_model_input = torch.cat([latent] * 2)
        latent_model_input = model.noise_scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = torch.cat(
            [latent_model_input, mask_latent, masked_image_latent, densepose_latent],
            dim=1,
        )

        if not ref_acceleration:
            ctx.current_timestep = current_t
            _, reference_features = model.unet_encoder(
                ref_image_latent,
                t,
                encoder_hidden_states=None,
                return_dict=False,
            )
            reference_features = list(reference_features)

        ctx.current_timestep = current_t
        noise_pred = model.unet(
            latent_model_input,
            t,
            encoder_hidden_states=None,
            cross_attention_kwargs=None,
            added_cond_kwargs=None,
            reference_features=reference_features,
            return_dict=False,
        )[0]

        if current_t == target_timestep:
            break

        noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
        latent = model.noise_scheduler.step(noise_pred, t, latent, return_dict=False)[0]

    reduced = ctx.reduced(device=device)
    summary = ctx.detached_summary()
    meta = {
        "target_timestep": int(target_timestep),
        "target_step_index": int(resolved_target_idx),
        "cached_reference_timestep": cached_ref_timestep,
        "summary": summary,
        "reduced": {k: float(v.detach().cpu().item()) for k, v in reduced.items()},
    }
    return reduced["total"], meta


def pgd_optimize(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.dtype == "float16":
        print("[WARN] CUDA is unavailable, forcing dtype from float16 to float32.")
        args.dtype = "float32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    batch, cloth_mask = prepare_batch_and_perturb_mask(args, device)
    target_timestep = resolve_target_timestep(args, model)

    ctx = LossContext(
        target_timestep=target_timestep,
        layer_scope=args.layer_scope,
        layer_substr=args.layer_substr,
    )
    install_loss_processors(model.unet_encoder, ctx, branch="reference")
    install_loss_processors(model.unet, ctx, branch="generative")

    clean_ref = batch["ref_image"].detach().clone()
    adv_ref = clean_ref.clone().detach()

    eps = float(args.epsilon) * (2.0 / 255.0)
    alpha = float(args.alpha) * (2.0 / 255.0)

    if args.random_start:
        delta = torch.empty_like(adv_ref).uniform_(-eps, eps)
        delta = delta * cloth_mask
        adv_ref = (clean_ref + delta).clamp(-1.0, 1.0)

    history: List[Dict[str, Any]] = []

    # Evaluate the clean image once.
    clean_batch = dict(batch)
    clean_batch["ref_image"] = clean_ref
    with torch.enable_grad():
        clean_loss, clean_meta = run_and_compute_loss(
            model=model,
            batch=clean_batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            ref_acceleration=args.ref_acceleration,
            ctx=ctx,
        )
    history.append({"iteration": -1, "kind": "clean", **clean_meta})

    for step in range(args.pgd_steps):
        adv_ref = adv_ref.detach().clone().requires_grad_(True)
        iter_batch = dict(batch)
        iter_batch["ref_image"] = adv_ref

        model.zero_grad(set_to_none=True)
        loss, meta = run_and_compute_loss(
            model=model,
            batch=iter_batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            ref_acceleration=args.ref_acceleration,
            ctx=ctx,
        )
        loss.backward()

        grad = adv_ref.grad
        if grad is None:
            raise RuntimeError("No gradient was produced for adv_ref.")

        with torch.no_grad():
            signed = grad.sign()
            adv_ref = adv_ref - alpha * signed * cloth_mask
            delta = torch.clamp(adv_ref - clean_ref, min=-eps, max=eps)
            delta = delta * cloth_mask
            adv_ref = torch.clamp(clean_ref + delta, min=-1.0, max=1.0)

        history.append({"iteration": step, "kind": "pgd", **meta})
        cat = meta["reduced"]
        print(
            f"[iter {step:03d}] total={cat['total']:.6f} | "
            f"ref_self={cat['reference_self']:.6f} | "
            f"den_self={cat['denoising_self']:.6f} | "
            f"den_ref_cross={cat['denoising_ref_cross']:.6f}"
        )

    # Final evaluation on the optimized image.
    final_batch = dict(batch)
    final_batch["ref_image"] = adv_ref.detach()
    with torch.enable_grad():
        final_loss, final_meta = run_and_compute_loss(
            model=model,
            batch=final_batch,
            target_timestep=target_timestep,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            seed=args.seed,
            ref_acceleration=args.ref_acceleration,
            ctx=ctx,
        )
    history.append({"iteration": args.pgd_steps, "kind": "final", **final_meta})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
            "src_image": args.src_image,
            "ref_image": args.ref_image,
            "model_mask_image": args.model_mask_image,
            "cloth_mask_image": args.cloth_mask_image,
            "densepose_image": args.densepose_image,
            "dataset": args.dataset,
            "target_timestep": int(target_timestep),
            "target_step_index": args.target_step_index,
            "num_inference_steps": args.num_inference_steps,
            "guidance_scale": args.guidance_scale,
            "seed": args.seed,
            "layer_scope": args.layer_scope,
            "layer_substr": args.layer_substr,
            "ref_acceleration": args.ref_acceleration,
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

    print(f"[OK] Saved optimized cloth to: {out_dir / 'cloth_adv.png'}")
    print(f"[OK] Saved logs to: {out_dir / 'optimization_log.json'}")


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
        "--dataset",
        type=str,
        default="virtual_tryon",
        choices=["virtual_tryon", "pose_transfer"],
    )
    parser.add_argument("--target_timestep", type=int, default=None)
    parser.add_argument("--target_step_index", type=int, default=0)
    parser.add_argument(
        "--layer_scope",
        type=str,
        default="mid",
        choices=["mid", "all"],
        help="Default is mid, i.e. only layers whose name contains 'mid_block'.",
    )
    parser.add_argument("--layer_substr", type=str, default=None)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_acceleration", action="store_true")
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
    parser.add_argument("--output_dir", type=str, default="pgd_attention_out")
    return parser.parse_args()


if __name__ == "__main__":
    pgd_optimize(parse_args())
