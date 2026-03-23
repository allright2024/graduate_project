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

Key properties
--------------
- Denoising UNet is NEVER used.
- src_image / model_mask_image / densepose_image are NOT needed.
- PGD updates are applied ONLY where cloth_mask == 1.
- Model weights stay frozen; gradients flow only to the reference image.
- Default layer scope is mid_block for lower memory use.

Example
-------
python optimize_reference_attention_pgd.py \
  --pretrained_model_name_or_path ./ckpts/stable-diffusion-inpainting \
  --pretrained_model ./ckpts/virtual_tryon.pth \
  --ref_image /workspace/train/cloth/00000_00.jpg \
  --cloth_mask_image /workspace/train/cloth-mask/00000_00.jpg \
  --target_step_index 0 \
  --layer_scope mid \
  --dtype float16 \
  --pgd_steps 10 \
  --epsilon 8.0 \
  --alpha 1.0 \
  --output_dir ./pgd_ref_attention_out
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

import torch
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
        self.loss_terms: List[torch.Tensor] = []
        self.layer_records: List[Dict[str, Any]] = []

    def should_record(self, layer_name: str) -> bool:
        if self.current_timestep != self.target_timestep:
            return False
        if self.layer_scope == "mid" and MID_TOKEN not in layer_name:
            return False
        if self.layer_substr and self.layer_substr not in layer_name:
            return False
        return True

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
) -> tuple[torch.Tensor, Dict[str, Any]]:
    ctx.reset()
    ctx.current_timestep = int(target_timestep)

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
    }
    return total, meta


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

def sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def collect_matched_pairs(ref_path: Path, mask_path: Path) -> List[Tuple[Path, Path]]:
    if ref_path.is_file() and mask_path.is_file():
        return [(ref_path, mask_path)]
    if ref_path.is_dir() and mask_path.is_dir():
        ref_files = {x.name: x for x in sorted(ref_path.iterdir()) if is_image_file(x)}
        mask_files = {x.name: x for x in sorted(mask_path.iterdir()) if is_image_file(x)}
        common = sorted(set(ref_files.keys()) & set(mask_files.keys()))
        if not common:
            raise ValueError(
                f"No matching image filenames found between {ref_path} and {mask_path}."
            )
        missing_masks = sorted(set(ref_files.keys()) - set(mask_files.keys()))
        missing_refs = sorted(set(mask_files.keys()) - set(ref_files.keys()))
        if missing_masks:
            print(f"[WARN] {len(missing_masks)} ref images have no matching mask and will be skipped.")
        if missing_refs:
            print(f"[WARN] {len(missing_refs)} mask images have no matching ref and will be skipped.")
        return [(ref_files[name], mask_files[name]) for name in common]
    raise ValueError(
        "ref_image and cloth_mask_image must either both be files or both be directories."
    )


def optimize_one_pair(
    args: argparse.Namespace,
    model: LeffaModel,
    target_timestep: int,
    ref_image_path: Path,
    cloth_mask_path: Path,
    out_dir: Path,
) -> Dict[str, Any]:
    device = module_device(model.vae)
    image_dtype = next(model.vae.parameters()).dtype

    clean_ref, cloth_mask = prepare_ref_and_mask(
        ref_image_path=str(ref_image_path),
        cloth_mask_path=str(cloth_mask_path),
        device=device,
        image_dtype=image_dtype,
    )

    ctx = LossContext(
        target_timestep=target_timestep,
        layer_scope=args.layer_scope,
        layer_substr=args.layer_substr,
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
        _, clean_meta = run_reference_loss(
            model=model,
            ref_image=clean_ref,
            target_timestep=target_timestep,
            ctx=ctx,
        )
    history.append({"iteration": -1, "kind": "clean", **clean_meta})

    for step in range(args.pgd_steps):
        adv_ref = adv_ref.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)

        loss, meta = run_reference_loss(
            model=model,
            ref_image=adv_ref,
            target_timestep=target_timestep,
            ctx=ctx,
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

        history.append({"iteration": step, "kind": "pgd", **meta})
        print(f"[{ref_image_path.name}] [iter {step:03d}] reference_self={meta['reduced']['reference_self']:.6f}")

    with torch.no_grad():
        _, final_meta = run_reference_loss(
            model=model,
            ref_image=adv_ref.detach(),
            target_timestep=target_timestep,
            ctx=ctx,
        )
    history.append({"iteration": args.pgd_steps, "kind": "final", **final_meta})

    out_dir.mkdir(parents=True, exist_ok=True)
    clean_pil = tensor_to_pil(clean_ref)
    adv_pil = tensor_to_pil(adv_ref.detach())
    delta_pil = delta_to_pil((adv_ref.detach() - clean_ref).detach(), scale=max(args.epsilon, 1.0))

    clean_pil.save(out_dir / "cloth_clean.png")
    adv_pil.save(out_dir / "cloth_adv.png")
    delta_pil.save(out_dir / "cloth_delta_vis.png")

    result = {
        "config": {
            "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
            "pretrained_model": args.pretrained_model,
            "ref_image": str(ref_image_path),
            "cloth_mask_image": str(cloth_mask_path),
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
        },
        "history": history,
        "final_delta_linf_normalized": float((adv_ref.detach() - clean_ref).abs().max().cpu().item()),
    }
    (out_dir / "optimization_log.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))

    return {
        "file_name": ref_image_path.name,
        "output_dir": str(out_dir),
        "final_reference_self": final_meta["reduced"]["reference_self"],
        "final_delta_linf_normalized": result["final_delta_linf_normalized"],
    }


def pgd_optimize(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available() and args.dtype == "float16":
        print("[WARN] CUDA is unavailable, forcing dtype from float16 to float32.")
        args.dtype = "float32"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)
    target_timestep = resolve_target_timestep(args, model)

    ref_path = Path(args.ref_image)
    mask_path = Path(args.cloth_mask_image)
    pairs = collect_matched_pairs(ref_path, mask_path)

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    summaries: List[Dict[str, Any]] = []
    multi = len(pairs) > 1 or ref_path.is_dir()

    for idx, (this_ref, this_mask) in enumerate(pairs, start=1):
        print(f"[INFO] Processing {idx}/{len(pairs)}: {this_ref.name}")
        this_out = output_root / sanitize_name(this_ref.stem) if multi else output_root
        summary = optimize_one_pair(
            args=args,
            model=model,
            target_timestep=target_timestep,
            ref_image_path=this_ref,
            cloth_mask_path=this_mask,
            out_dir=this_out,
        )
        summaries.append(summary)

    if multi:
        batch_summary = {
            "num_pairs": len(summaries),
            "target_timestep": int(target_timestep),
            "items": summaries,
        }
        (output_root / "batch_summary.json").write_text(
            json.dumps(batch_summary, indent=2, ensure_ascii=False)
        )
        print(f"[OK] Saved batch summary to: {output_root / 'batch_summary.json'}")
    elif summaries:
        print(f"[OK] Saved optimized cloth to: {output_root / 'cloth_adv.png'}")
        print(f"[OK] Saved logs to: {output_root / 'optimization_log.json'}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    parser.add_argument("--pretrained_model", type=str, required=True)
    parser.add_argument(
        "--ref_image",
        type=str,
        required=True,
        help="Path to a single cloth image or a directory of cloth images.",
    )
    parser.add_argument(
        "--cloth_mask_image",
        type=str,
        required=True,
        help="Path to a single cloth mask or a directory of cloth masks with matching filenames.",
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
    parser.add_argument("--output_dir", type=str, default="pgd_ref_attention_out")
    return parser.parse_args()


if __name__ == "__main__":
    pgd_optimize(parse_args())
