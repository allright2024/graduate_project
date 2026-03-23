#!/usr/bin/env python3
"""
check_attention.py

Compute L2 norms of selected Leffa attention maps at a chosen diffusion timestep T.
This version focuses on MID layers only by default and reports three categories:

1) reference_self:   reference UNet self-attention
2) denoising_self:   generative UNet self-attention over current tokens only
3) denoising_ref_cross: generative UNet attention from current tokens to reference tokens

Notes
-----
- Inputs must already be prepared: src_image, ref_image, mask_image, densepose_image.
- No gradients are tracked. The model runs in eval mode.
- For the generative UNet, Leffa concatenates [current_tokens, reference_tokens]
  before attn1. This script splits the resulting attention probability tensor into:
    * current -> current   (denoising_self)
    * current -> reference (denoising_ref_cross)
  assuming the concatenated sequence is evenly split, which matches Leffa's design
  for paired blocks.
- Cross-attention attn2 remains disabled via SkipAttnProcessor, matching Leffa.

Example
-------
python check_attention.py \
    --pretrained_model_name_or_path ./ckpts/stable-diffusion-inpainting \
    --pretrained_model ./ckpts/virtual_tryon.pth \
    --src_image person.png \
    --ref_image garment.png \
    --mask_image mask.png \
    --densepose_image densepose.png \
    --target_step_index 0 \
    --output_json attention_norms.json
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from PIL import Image

from leffa.model import LeffaModel, SkipAttnProcessor
from leffa.transform import LeffaTransform


MID_TOKEN = "mid_block"


@dataclass
class AttentionRecord:
    category: str
    layer_name: str
    timestep: int
    shape: List[int]
    total_l2: float
    per_batch_l2: List[float]
    per_head_l2: List[List[float]]
    mean: float
    max_value: float
    min_value: float


def _module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


class RecordingContext:
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
        self.records: Dict[str, List[AttentionRecord]] = defaultdict(list)

    def should_record(self, layer_name: str) -> bool:
        if self.current_timestep != self.target_timestep:
            return False
        if self.layer_scope == "mid" and MID_TOKEN not in layer_name:
            return False
        if self.layer_substr and self.layer_substr not in layer_name:
            return False
        return True

    def _make_record(
        self,
        category: str,
        layer_name: str,
        tensor: torch.Tensor,
    ) -> AttentionRecord:
        x = tensor.detach().float()
        total_l2 = torch.linalg.vector_norm(x).item()
        per_batch_l2 = torch.linalg.vector_norm(x.reshape(x.shape[0], -1), dim=1).tolist()
        per_head_l2 = torch.linalg.vector_norm(
            x.reshape(x.shape[0], x.shape[1], -1), dim=2
        ).tolist()
        return AttentionRecord(
            category=category,
            layer_name=layer_name,
            timestep=int(self.target_timestep),
            shape=list(x.shape),
            total_l2=float(total_l2),
            per_batch_l2=[float(v) for v in per_batch_l2],
            per_head_l2=[[float(v) for v in row] for row in per_head_l2],
            mean=float(x.mean().item()),
            max_value=float(x.max().item()),
            min_value=float(x.min().item()),
        )

    def add(self, category: str, layer_name: str, tensor: torch.Tensor) -> None:
        self.records[category].append(self._make_record(category, layer_name, tensor))

    def summarize(self) -> Dict[str, List[AttentionRecord]]:
        for category in self.records:
            self.records[category].sort(key=lambda r: r.layer_name)
        return dict(self.records)


class ReferenceRecordingAttnProcessor(torch.nn.Module):
    """Record reference UNet attn1 self-attention."""

    def __init__(self, layer_name: str, ctx: RecordingContext) -> None:
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
            self.ctx.add("reference_self", self.layer_name, attn_probs)

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


class GenerativeRecordingAttnProcessor(torch.nn.Module):
    """
    Record generative UNet attn1 in two parts:
      - denoising_self: current -> current
      - denoising_ref_cross: current -> reference
    """

    def __init__(self, layer_name: str, ctx: RecordingContext) -> None:
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
                    f"Expected even token counts in generative attn1 for {self.layer_name}, "
                    f"but got q={total_q}, k={total_k}."
                )
            q_cur = total_q // 2
            k_cur = total_k // 2
            denoising_self = attn_probs[:, :, :q_cur, :k_cur]
            denoising_ref_cross = attn_probs[:, :, :q_cur, k_cur:]
            self.ctx.add("denoising_self", self.layer_name, denoising_self)
            self.ctx.add("denoising_ref_cross", self.layer_name, denoising_ref_cross)

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
def install_recording_processors(
    unet: torch.nn.Module,
    ctx: RecordingContext,
    branch: str,
) -> None:
    new_processors: Dict[str, torch.nn.Module] = {}
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            if branch == "reference":
                new_processors[name] = ReferenceRecordingAttnProcessor(name, ctx)
            elif branch == "generative":
                new_processors[name] = GenerativeRecordingAttnProcessor(name, ctx)
            else:
                raise ValueError(f"Unsupported branch: {branch}")
        elif name.endswith("attn2.processor"):
            new_processors[name] = SkipAttnProcessor()
        else:
            new_processors[name] = proc if proc is not None else SkipAttnProcessor()
    unet.set_attn_processor(new_processors)


@torch.no_grad()
def prepare_inputs(args: argparse.Namespace) -> Dict[str, torch.Tensor]:
    batch = {
        "src_image": [Image.open(args.src_image).convert("RGB")],
        "ref_image": [Image.open(args.ref_image).convert("RGB")],
        "mask": [Image.open(args.mask_image)],
        "densepose": [Image.open(args.densepose_image).convert("RGB")],
    }
    transform = LeffaTransform(dataset=args.dataset)
    return transform(batch)


@torch.no_grad()
def move_batch_to_device(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


@torch.no_grad()
def run_until_target_timestep(
    model: LeffaModel,
    batch: Dict[str, torch.Tensor],
    target_timestep: int,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    ref_acceleration: bool,
    ctx: RecordingContext,
) -> Dict[str, Any]:
    device = _module_device(model.vae)
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

    return {
        "target_step_index": resolved_target_idx,
        "target_timestep": target_timestep,
        "cached_reference_timestep": cached_ref_timestep,
    }


@torch.no_grad()
def build_model(args: argparse.Namespace, device: torch.device) -> LeffaModel:
    model = LeffaModel(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        pretrained_model=args.pretrained_model,
        dtype=args.dtype,
    )
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def resolve_target_timestep(args: argparse.Namespace, model: LeffaModel) -> int:
    device = _module_device(model.vae)
    model.noise_scheduler.set_timesteps(args.num_inference_steps, device=device)
    timesteps = model.noise_scheduler.timesteps

    if args.target_timestep is not None:
        return int(args.target_timestep)
    if args.target_step_index is None:
        raise ValueError("Either --target_timestep or --target_step_index must be provided.")

    idx = int(args.target_step_index)
    if idx < 0 or idx >= len(timesteps):
        raise ValueError(f"target_step_index={idx} is out of range for {len(timesteps)} timesteps.")
    return int(timesteps[idx].item())


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.dtype == "float16":
        print("[WARN] CUDA is unavailable, so dtype is forced from float16 to float32.")
        args.dtype = "float32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    batch = prepare_inputs(args)
    batch = move_batch_to_device(batch, device)

    target_timestep = resolve_target_timestep(args, model)
    ctx = RecordingContext(
        target_timestep=target_timestep,
        layer_scope=args.layer_scope,
        layer_substr=args.layer_substr,
    )

    install_recording_processors(model.unet_encoder, ctx, branch="reference")
    install_recording_processors(model.unet, ctx, branch="generative")

    run_meta = run_until_target_timestep(
        model=model,
        batch=batch,
        target_timestep=target_timestep,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        ref_acceleration=args.ref_acceleration,
        ctx=ctx,
    )

    summary = ctx.summarize()
    output = {
        "meta": {
            "target_timestep": int(target_timestep),
            "target_step_index": int(run_meta["target_step_index"]),
            "num_inference_steps": int(args.num_inference_steps),
            "guidance_scale": float(args.guidance_scale),
            "seed": int(args.seed),
            "layer_scope": args.layer_scope,
            "layer_substr": args.layer_substr,
            "ref_acceleration": bool(args.ref_acceleration),
            "cached_reference_timestep": run_meta["cached_reference_timestep"],
            "dataset": args.dataset,
            "categories": [
                "reference_self",
                "denoising_self",
                "denoising_ref_cross",
            ],
        },
        "results": {
            category: [record.__dict__ for record in records]
            for category, records in summary.items()
        },
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))

    print(f"[OK] Saved attention norms to: {output_path}")
    for category in ["reference_self", "denoising_self", "denoising_ref_cross"]:
        records = output["results"].get(category, [])
        print(f"\n[{category}]")
        if not records:
            print("  (no records captured)")
            continue
        for row in records:
            print(f"  {row['layer_name']}: total_l2={row['total_l2']:.6f}, shape={row['shape']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--src_image", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--mask_image", type=str, required=True)
    parser.add_argument("--densepose_image", type=str, required=True)
    parser.add_argument(
        "--dataset",
        type=str,
        default="virtual_tryon",
        choices=["virtual_tryon", "pose_transfer"],
    )
    parser.add_argument("--target_timestep", type=int, default=None)
    parser.add_argument("--target_step_index", type=int, default=None)
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
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--output_json", type=str, default="attention_norms.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
