#!/usr/bin/env python3
"""
optimize_reference_attention_pgd.py

PGD over the reference/cloth image only, using ONLY Leffa's reference UNet.
The objective is to minimize the sum of L2 norms of the reference UNet's
internal self-attention maps at a chosen diffusion timestep.

Optimized objective (summed over selected layers):
    total_loss = sum_l || A_ref_self^(l) ||_2

where A_ref_self^(l) is the attn1 self-attention probability tensor of the
reference UNet at the selected timestep.

Also supports visualization of selected self-attention maps:
- mean attention matrix over batch and heads
- mean key activation map (averaged over queries)
- center-query key activation map

Key properties
--------------
- Denoising UNet is NEVER used.
- src_image / model_mask_image / densepose_image are NOT needed.
- PGD updates are applied ONLY where cloth_mask == 1.
- Model weights stay frozen; gradients flow only to the reference image.
- Default layer scope is mid_block for lower memory use.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

from leffa.model import LeffaModel, SkipAttnProcessor
from leffa.transform import LeffaTransform


MID_TOKEN = "mid_block"


def module_device(module: torch.nn.Module) -> torch.device:
    return next(module.parameters()).device


def tensor_to_pil(image_tensor: torch.Tensor) -> Image.Image:
    """Convert a [1,3,H,W] tensor in [-1,1] to PIL RGB."""
    x = image_tensor.detach().float().cpu().clamp(-1.0, 1.0)
    x = (x[0] / 2.0 + 0.5).clamp(0.0, 1.0)
    x = (x.permute(1, 2, 0).numpy() * 255.0).round().astype("uint8")
    return Image.fromarray(x)


def delta_to_pil(delta_tensor: torch.Tensor, scale: float = 8.0) -> Image.Image:
    """Visualize perturbation delta in [-1,1] as RGB centered at 127."""
    x = delta_tensor.detach().float().cpu()[0]
    denom = max(scale, 1e-6) * (2.0 / 255.0)
    x = (x / denom).clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
    x = x.permute(1, 2, 0).numpy()
    return Image.fromarray(x)


def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def normalize_to_uint8(x: torch.Tensor) -> Image.Image:
    x = x.detach().float().cpu()
    x_min = float(x.min().item())
    x_max = float(x.max().item())
    if x_max - x_min < 1e-12:
        arr = torch.zeros_like(x, dtype=torch.uint8)
    else:
        arr = ((x - x_min) / (x_max - x_min) * 255.0).round().clamp(0, 255).to(torch.uint8)
    return Image.fromarray(arr.numpy(), mode="L")


def infer_hw_from_seq_len(seq_len: int, aspect_ratio: float) -> Optional[Tuple[int, int]]:
    best = None
    best_err = float("inf")
    root = int(math.sqrt(seq_len))
    for h in range(1, root + 1):
        if seq_len % h != 0:
            continue
        w = seq_len // h
        for hh, ww in ((h, w), (w, h)):
            err = abs((hh / ww) - aspect_ratio)
            if err < best_err:
                best_err = err
                best = (hh, ww)
    return best


class LossContext:
    def __init__(
        self,
        target_timestep: int,
        layer_scope: str = "mid",
        layer_substr: Optional[str] = None,
        vis_max_layers: int = 8,
        vis_max_side: int = 512,
        image_aspect_ratio: float = 1024 / 768,
    ) -> None:
        self.target_timestep = int(target_timestep)
        self.layer_scope = layer_scope
        self.layer_substr = layer_substr
        self.current_timestep: Optional[int] = None
        self.vis_max_layers = int(vis_max_layers)
        self.vis_max_side = int(vis_max_side)
        self.image_aspect_ratio = float(image_aspect_ratio)
        self.capture_visuals = False
        self.reset()

    def reset(self) -> None:
        self.loss_terms: List[torch.Tensor] = []
        self.layer_records: List[Dict[str, Any]] = []
        self.visual_records: Dict[str, Dict[str, Any]] = {}

    def should_record(self, layer_name: str) -> bool:
        if self.current_timestep != self.target_timestep:
            return False
        if self.layer_scope == "mid" and MID_TOKEN not in layer_name:
            return False
        if self.layer_substr and self.layer_substr not in layer_name:
            return False
        return True

    def maybe_capture_visual(self, layer_name: str, attn_probs: torch.Tensor) -> None:
        if not self.capture_visuals:
            return
        if layer_name in self.visual_records:
            return
        if len(self.visual_records) >= self.vis_max_layers:
            return

        # attn_probs: [B, H, Q, K]
        mean_matrix = attn_probs.mean(dim=(0, 1))  # [Q, K]
        q_len, k_len = mean_matrix.shape

        matrix_for_vis = mean_matrix
        max_side = max(q_len, k_len)
        if max_side > self.vis_max_side:
            scale = self.vis_max_side / max_side
            new_h = max(1, int(round(q_len * scale)))
            new_w = max(1, int(round(k_len * scale)))
            matrix_for_vis = F.interpolate(
                mean_matrix[None, None].float(),
                size=(new_h, new_w),
                mode="bilinear",
                align_corners=False,
            )[0, 0]

        mean_key = mean_matrix.mean(dim=0)
        center_query_idx = int(q_len // 2)
        center_query = mean_matrix[center_query_idx]

        inferred_hw = infer_hw_from_seq_len(k_len, self.image_aspect_ratio)
        mean_key_map = None
        center_query_map = None
        if inferred_hw is not None and inferred_hw[0] * inferred_hw[1] == k_len:
            h, w = inferred_hw
            mean_key_map = mean_key.view(h, w)
            center_query_map = center_query.view(h, w)

        self.visual_records[layer_name] = {
            "matrix": matrix_for_vis.detach().float().cpu(),
            "mean_key_map": None if mean_key_map is None else mean_key_map.detach().float().cpu(),
            "center_query_map": None if center_query_map is None else center_query_map.detach().float().cpu(),
            "shape": [int(q_len), int(k_len)],
            "center_query_index": center_query_idx,
            "spatial_hw": None if inferred_hw is None else [int(inferred_hw[0]), int(inferred_hw[1])],
        }

    def add_loss(self, layer_name: str, tensor: torch.Tensor) -> None:
        norm = torch.linalg.vector_norm(tensor.float())
        self.loss_terms.append(norm)
        self.layer_records.append(
            {
                "layer_name": layer_name,
                "l2": float(norm.detach().cpu().item()),
                "shape": list(tensor.shape),
            }
        )
        self.maybe_capture_visual(layer_name, tensor)

    def reduced(self, device: torch.device) -> torch.Tensor:
        if not self.loss_terms:
            return torch.zeros((), device=device, dtype=torch.float32)
        return torch.stack(self.loss_terms).sum()

    def detached_summary(self) -> Dict[str, Any]:
        return {
            "total": float(sum(row["l2"] for row in self.layer_records)),
            "layers": self.layer_records,
        }


class ReferenceLossAttnProcessor(torch.nn.Module):
    """Records only reference-UNet attn1 self-attention probabilities."""

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
            self.ctx.add_loss(self.layer_name, attn_probs)

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
def install_reference_loss_processors(unet: torch.nn.Module, ctx: LossContext) -> None:
    processors: Dict[str, torch.nn.Module] = {}
    for name, proc in unet.attn_processors.items():
        if name.endswith("attn1.processor"):
            processors[name] = ReferenceLossAttnProcessor(name, ctx)
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
def prepare_ref_and_mask(
    ref_image_path: str,
    cloth_mask_path: str,
    device: torch.device,
    image_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    transform = LeffaTransform()

    ref_pil = Image.open(ref_image_path).convert("RGB")
    ref = transform.vae_processor.preprocess(ref_pil, transform.height, transform.width)[0]
    ref = transform.prepare_image(ref).to(device=device, dtype=image_dtype)

    cloth_mask_pil = Image.open(cloth_mask_path)
    cloth_mask = transform.mask_processor.preprocess(cloth_mask_pil, transform.height, transform.width)[0]
    cloth_mask = transform.prepare_mask(cloth_mask).expand(-1, 3, -1, -1)
    cloth_mask = cloth_mask.to(device=device, dtype=image_dtype)

    return ref, cloth_mask


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


def run_reference_loss(
    model: LeffaModel,
    ref_image: torch.Tensor,
    target_timestep: int,
    ctx: LossContext,
    capture_visuals: bool = False,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    ctx.reset()
    ctx.current_timestep = int(target_timestep)
    ctx.capture_visuals = bool(capture_visuals)

    device = module_device(model.vae)
    vae_dtype = next(model.vae.parameters()).dtype

    ref_image = ref_image.to(device=device, dtype=vae_dtype)
    ref_latent = model.vae.encode(ref_image).latent_dist.sample()
    ref_latent = ref_latent * model.vae.config.scaling_factor

    t = torch.tensor([target_timestep], device=device, dtype=torch.long)
    # Reference branch uses CFG-style duplication in Leffa pipeline; reproduce that.
    ref_latent = torch.cat([torch.zeros_like(ref_latent), ref_latent], dim=0)

    _ = model.unet_encoder(
        ref_latent,
        t,
        encoder_hidden_states=None,
        return_dict=False,
    )

    total = ctx.reduced(device=device)
    meta = {
        "target_timestep": int(target_timestep),
        "summary": ctx.detached_summary(),
        "reduced": {"reference_self": float(total.detach().cpu().item())},
        "visual_records": ctx.visual_records,
    }
    return total, meta


@torch.no_grad()
def save_attention_visuals(visual_records: Dict[str, Dict[str, Any]], phase_dir: Path) -> List[Dict[str, Any]]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, Any]] = []

    for layer_name, record in visual_records.items():
        safe = sanitize_name(layer_name)
        layer_dir = phase_dir / safe
        layer_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = layer_dir / "matrix_mean.png"
        normalize_to_uint8(record["matrix"]).save(matrix_path)

        row = {
            "layer_name": layer_name,
            "shape": record["shape"],
            "center_query_index": record["center_query_index"],
            "spatial_hw": record["spatial_hw"],
            "files": {
                "matrix_mean": str(matrix_path.name),
            },
        }

        if record["mean_key_map"] is not None:
            mean_key_path = layer_dir / "mean_key_map.png"
            normalize_to_uint8(record["mean_key_map"]).save(mean_key_path)
            row["files"]["mean_key_map"] = str(mean_key_path.name)

        if record["center_query_map"] is not None:
            center_path = layer_dir / "center_query_map.png"
            normalize_to_uint8(record["center_query_map"]).save(center_path)
            row["files"]["center_query_map"] = str(center_path.name)

        manifest.append(row)

    return manifest


def pgd_optimize(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.dtype == "float16":
        print("[WARN] CUDA is unavailable, forcing dtype from float16 to float32.")
        args.dtype = "float32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)

    image_dtype = next(model.vae.parameters()).dtype
    clean_ref, cloth_mask = prepare_ref_and_mask(
        ref_image_path=args.ref_image,
        cloth_mask_path=args.cloth_mask_image,
        device=device,
        image_dtype=image_dtype,
    )
    target_timestep = resolve_target_timestep(args, model)

    ctx = LossContext(
        target_timestep=target_timestep,
        layer_scope=args.layer_scope,
        layer_substr=args.layer_substr,
        vis_max_layers=args.vis_max_layers,
        vis_max_side=args.vis_max_side,
        image_aspect_ratio=args.image_height / args.image_width,
    )
    install_reference_loss_processors(model.unet_encoder, ctx)

    adv_ref = clean_ref.clone().detach()

    eps = float(args.epsilon) * (2.0 / 255.0)
    alpha = float(args.alpha) * (2.0 / 255.0)

    if args.random_start:
        delta = torch.empty_like(adv_ref).uniform_(-eps, eps)
        delta = delta * cloth_mask
        adv_ref = (clean_ref + delta).clamp(-1.0, 1.0)

    history: List[Dict[str, Any]] = []

    with torch.no_grad():
        clean_loss, clean_meta = run_reference_loss(
            model=model,
            ref_image=clean_ref,
            target_timestep=target_timestep,
            ctx=ctx,
            capture_visuals=args.save_attention_vis,
        )
    history.append({"iteration": -1, "kind": "clean", **{k: v for k, v in clean_meta.items() if k != "visual_records"}})

    for step in range(args.pgd_steps):
        adv_ref = adv_ref.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        loss, meta = run_reference_loss(
            model=model,
            ref_image=adv_ref,
            target_timestep=target_timestep,
            ctx=ctx,
            capture_visuals=False,
        )
        loss.backward()

        grad = adv_ref.grad
        if grad is None:
            raise RuntimeError("No gradient was produced for adv_ref.")

        with torch.no_grad():
            adv_ref = adv_ref - alpha * grad.sign() * cloth_mask
            delta = torch.clamp(adv_ref - clean_ref, min=-eps, max=eps)
            delta = delta * cloth_mask
            adv_ref = torch.clamp(clean_ref + delta, min=-1.0, max=1.0)

        history.append({"iteration": step, "kind": "pgd", **{k: v for k, v in meta.items() if k != "visual_records"}})
        print(f"[iter {step:03d}] reference_self={meta['reduced']['reference_self']:.6f}")

    with torch.no_grad():
        final_loss, final_meta = run_reference_loss(
            model=model,
            ref_image=adv_ref.detach(),
            target_timestep=target_timestep,
            ctx=ctx,
            capture_visuals=args.save_attention_vis,
        )
    history.append({"iteration": args.pgd_steps, "kind": "final", **{k: v for k, v in final_meta.items() if k != "visual_records"}})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    clean_pil = tensor_to_pil(clean_ref)
    adv_pil = tensor_to_pil(adv_ref.detach())
    delta_pil = delta_to_pil((adv_ref.detach() - clean_ref).detach(), scale=max(args.epsilon, 1.0))

    clean_pil.save(out_dir / "cloth_clean.png")
    adv_pil.save(out_dir / "cloth_adv.png")
    delta_pil.save(out_dir / "cloth_delta_vis.png")

    vis_manifest = {}
    if args.save_attention_vis:
        vis_dir = out_dir / "attention_vis"
        vis_manifest["clean"] = save_attention_visuals(clean_meta["visual_records"], vis_dir / "clean")
        vis_manifest["final"] = save_attention_visuals(final_meta["visual_records"], vis_dir / "final")

    result = {
        "config": {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "pretrained_model": args.pretrained_model,
            "ref_image": args.ref_image,
            "cloth_mask_image": args.cloth_mask_image,
            "target_timestep": int(target_timestep),
            "target_step_index": args.target_step_index,
            "num_inference_steps": args.num_inference_steps,
            "layer_scope": args.layer_scope,
            "layer_substr": args.layer_substr,
            "dtype": args.dtype,
            "pgd_steps": args.pgd_steps,
            "epsilon_pixels": args.epsilon,
            "alpha_pixels": args.alpha,
            "random_start": args.random_start,
            "save_attention_vis": args.save_attention_vis,
            "vis_max_layers": args.vis_max_layers,
            "vis_max_side": args.vis_max_side,
            "image_height": args.image_height,
            "image_width": args.image_width,
        },
        "history": history,
        "visualization_manifest": vis_manifest,
        "final_delta_linf_normalized": float((adv_ref.detach() - clean_ref).abs().max().cpu().item()),
    }
    (out_dir / "optimization_log.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    print(f"[OK] Saved optimized cloth to: {out_dir / 'cloth_adv.png'}")
    if args.save_attention_vis:
        print(f"[OK] Saved attention maps to: {out_dir / 'attention_vis'}")
    print(f"[OK] Saved logs to: {out_dir / 'optimization_log.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument("--ref_image", type=str, required=True)
    parser.add_argument("--cloth_mask_image", type=str, required=True)
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
    parser.add_argument("--save_attention_vis", action="store_true")
    parser.add_argument("--vis_max_layers", type=int, default=8)
    parser.add_argument("--vis_max_side", type=int, default=512)
    parser.add_argument("--image_height", type=int, default=1024)
    parser.add_argument("--image_width", type=int, default=768)
    parser.add_argument("--output_dir", type=str, default="pgd_ref_attention_out")
    return parser.parse_args()


if __name__ == "__main__":
    pgd_optimize(parse_args())
