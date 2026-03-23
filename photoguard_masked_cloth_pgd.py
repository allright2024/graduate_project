import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
import torch
from diffusers import AutoencoderKL


VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PhotoGuard-style masked PGD for cloth images. "
            "Only pixels where cloth_mask==1 are perturbed. "
            "Supports single-file mode and directory batch mode."
        )
    )
    parser.add_argument(
        "--cloth_image",
        type=str,
        required=True,
        help="Path to a clean cloth image or a directory of cloth images.",
    )
    parser.add_argument(
        "--cloth_mask_image",
        type=str,
        required=True,
        help=(
            "Path to a binary mask image or a directory of mask images. "
            "Only mask==1 regions are perturbed."
        ),
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="runwayml/stable-diffusion-inpainting",
        help="Diffusers model id or local directory containing a VAE subfolder.",
    )
    parser.add_argument(
        "--target_image",
        type=str,
        default=None,
        help=(
            "Optional target image, file or directory. "
            "If omitted, a uniform mid-gray image is used. "
            "If a directory is provided in batch mode, file names are matched."
        ),
    )
    parser.add_argument("--height", type=int, default=None, help="Optional resize height.")
    parser.add_argument("--width", type=int, default=None, help="Optional resize width.")
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Model/image dtype used during optimization.",
    )
    parser.add_argument("--pgd_steps", type=int, default=200, help="Number of PGD iterations.")
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.12,
        help="L-infinity perturbation budget in normalized [-1, 1] image space.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.01,
        help="PGD step size in normalized [-1, 1] image space.",
    )
    parser.add_argument(
        "--random_start",
        action="store_true",
        help="Initialize delta from uniform[-epsilon, epsilon] inside the mask.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./photoguard_cloth_out",
        help="Directory where outputs are written.",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def round_down_multiple(x: int, base: int = 32) -> int:
    y = x - (x % base)
    return max(y, base)



def resolve_size(image: Image.Image, width: int | None, height: int | None) -> Tuple[int, int]:
    if width is not None and height is not None:
        return round_down_multiple(width), round_down_multiple(height)
    w, h = image.size
    return round_down_multiple(w), round_down_multiple(h)



def load_rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img.resize(size, resample=Image.LANCZOS)



def load_mask(path: str, size: Tuple[int, int]) -> Image.Image:
    mask = Image.open(path).convert("L")
    return mask.resize(size, resample=Image.NEAREST)



def pil_to_normalized_tensor(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    return tensor * 2.0 - 1.0



def pil_mask_to_tensor(mask: Image.Image) -> torch.Tensor:
    arr = np.array(mask).astype(np.float32) / 255.0
    arr = (arr >= 0.5).astype(np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor



def tensor_to_pil_image(t: torch.Tensor) -> Image.Image:
    t = t.detach().cpu().clamp(-1.0, 1.0)
    t = (t[0] / 2.0 + 0.5).clamp(0.0, 1.0)
    arr = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)



def delta_to_vis(delta: torch.Tensor, amplify: float = 8.0) -> Image.Image:
    d = delta.detach().cpu()[0]
    d = d / max(amplify, 1e-8)
    d = d.clamp(-1.0, 1.0)
    d = (d + 1.0) / 2.0
    arr = (d.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)



def build_target_image(size: Tuple[int, int], target_path: str | None) -> Image.Image:
    if target_path is not None:
        return load_rgb(target_path, size)
    gray = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    return Image.fromarray(gray, mode="RGB")



def sanitize_stem(path: Path) -> str:
    return path.stem.replace("/", "_").replace("\\", "_")



def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS



def collect_images(directory: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(directory.iterdir()) if is_image_file(p)}



def resolve_target_for_name(target_arg: str | None, name: str) -> str | None:
    if target_arg is None:
        return None
    p = Path(target_arg)
    if p.is_dir():
        candidate = p / name
        return str(candidate) if candidate.exists() else None
    return str(p)



def build_pairs(cloth_arg: str, cloth_mask_arg: str) -> List[Tuple[Path, Path]]:
    cloth_path = Path(cloth_arg)
    mask_path = Path(cloth_mask_arg)

    cloth_is_dir = cloth_path.is_dir()
    mask_is_dir = mask_path.is_dir()

    if cloth_is_dir != mask_is_dir:
        raise ValueError(
            "--cloth_image and --cloth_mask_image must both be files or both be directories."
        )

    if not cloth_is_dir:
        if not cloth_path.exists():
            raise FileNotFoundError(f"Cloth image not found: {cloth_path}")
        if not mask_path.exists():
            raise FileNotFoundError(f"Cloth mask image not found: {mask_path}")
        return [(cloth_path, mask_path)]

    cloth_files = collect_images(cloth_path)
    mask_files = collect_images(mask_path)

    common_names = sorted(set(cloth_files.keys()) & set(mask_files.keys()))
    if not common_names:
        raise ValueError(
            f"No matching image file names found between {cloth_path} and {mask_path}."
        )

    cloth_only = sorted(set(cloth_files.keys()) - set(mask_files.keys()))
    mask_only = sorted(set(mask_files.keys()) - set(cloth_files.keys()))
    if cloth_only:
        print(f"[warn] {len(cloth_only)} cloth files had no matching mask and were skipped.")
    if mask_only:
        print(f"[warn] {len(mask_only)} mask files had no matching cloth and were skipped.")

    return [(cloth_files[name], mask_files[name]) for name in common_names]


class MaskedPhotoGuardAttack:
    def __init__(self, pretrained_model_name_or_path: str, dtype: torch.dtype) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained(
            pretrained_model_name_or_path,
            subfolder="vae",
        ).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)
        if self.device == "cuda":
            self.vae.to(dtype=dtype)

    def encode_mean(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae.encode(x).latent_dist.mean

    def attack(
        self,
        cloth: torch.Tensor,
        cloth_mask: torch.Tensor,
        target: torch.Tensor,
        epsilon: float,
        alpha: float,
        pgd_steps: int,
        random_start: bool,
    ) -> Tuple[torch.Tensor, List[dict]]:
        cloth = cloth.to(self.device)
        cloth_mask = cloth_mask.to(self.device)
        target = target.to(self.device)

        if self.device == "cuda":
            cloth = cloth.to(self.dtype)
            cloth_mask = cloth_mask.to(self.dtype)
            target = target.to(self.dtype)

        target_latent = self.encode_mean(target).detach()

        if random_start:
            delta = torch.empty_like(cloth).uniform_(-epsilon, epsilon)
            delta = delta * cloth_mask
            delta = torch.max(torch.min(delta, 1.0 - cloth), -1.0 - cloth)
        else:
            delta = torch.zeros_like(cloth)

        delta = delta.detach()
        history: List[dict] = []

        for step in range(pgd_steps):
            delta.requires_grad_(True)
            adv = torch.clamp(cloth + delta, -1.0, 1.0)
            latent = self.encode_mean(adv)
            loss = (latent - target_latent).norm(p=2)

            grad = torch.autograd.grad(loss, [delta], retain_graph=False, create_graph=False)[0]

            step_size = alpha - (alpha - alpha / 100.0) / max(pgd_steps, 1) * step
            with torch.no_grad():
                delta = delta - step_size * grad.sign()
                delta = delta.clamp(-epsilon, epsilon)
                delta = delta * cloth_mask
                delta = torch.max(torch.min(delta, 1.0 - cloth), -1.0 - cloth)

                adv_now = torch.clamp(cloth + delta, -1.0, 1.0)
                current_loss = (self.encode_mean(adv_now) - target_latent).norm(p=2).item()
                linf = delta.abs().max().item()
                masked_linf = (delta.abs() * cloth_mask).max().item()
                history.append(
                    {
                        "step": step,
                        "loss": current_loss,
                        "step_size": float(step_size),
                        "linf": linf,
                        "masked_linf": masked_linf,
                    }
                )

        adv = torch.clamp(cloth + delta, -1.0, 1.0).detach()
        return adv, history



def process_one(
    attacker: MaskedPhotoGuardAttack,
    cloth_path: Path,
    mask_path: Path,
    target_path: str | None,
    args: argparse.Namespace,
    sample_out_dir: Path,
) -> dict:
    orig = Image.open(cloth_path).convert("RGB")
    size = resolve_size(orig, args.width, args.height)

    cloth_img = load_rgb(str(cloth_path), size)
    cloth_mask_img = load_mask(str(mask_path), size)
    target_img = build_target_image(size, target_path)

    cloth = pil_to_normalized_tensor(cloth_img)
    cloth_mask = pil_mask_to_tensor(cloth_mask_img)
    target = pil_to_normalized_tensor(target_img)

    adv, history = attacker.attack(
        cloth=cloth,
        cloth_mask=cloth_mask,
        target=target,
        epsilon=args.epsilon,
        alpha=args.alpha,
        pgd_steps=args.pgd_steps,
        random_start=args.random_start,
    )

    delta = adv.cpu() - cloth.cpu()
    adv_pil = tensor_to_pil_image(adv.cpu())
    delta_vis = delta_to_vis(delta)

    sample_out_dir.mkdir(parents=True, exist_ok=True)
    cloth_img.save(sample_out_dir / "cloth_clean.png")
    cloth_mask_img.save(sample_out_dir / "cloth_mask.png")
    target_img.save(sample_out_dir / "target_image.png")
    adv_pil.save(sample_out_dir / "cloth_adv.png")
    delta_vis.save(sample_out_dir / "cloth_delta_vis.png")

    metadata = {
        "cloth_image": str(cloth_path),
        "cloth_mask_image": str(mask_path),
        "target_image": target_path,
        "pretrained_model_name_or_path": args.pretrained_model_name_or_path,
        "size": {"width": size[0], "height": size[1]},
        "dtype": args.dtype,
        "pgd_steps": args.pgd_steps,
        "epsilon": args.epsilon,
        "alpha": args.alpha,
        "random_start": args.random_start,
        "seed": args.seed,
        "history": history,
        "final_loss": history[-1]["loss"] if history else None,
    }

    with open(sample_out_dir / "optimization_log.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"[done] {cloth_path.name} -> {sample_out_dir}")
    return {
        "name": cloth_path.name,
        "sample_dir": str(sample_out_dir),
        "final_loss": metadata["final_loss"],
        "steps": len(history),
    }



def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = build_pairs(args.cloth_image, args.cloth_mask_image)
    batch_mode = Path(args.cloth_image).is_dir() and Path(args.cloth_mask_image).is_dir()

    attacker = MaskedPhotoGuardAttack(
        pretrained_model_name_or_path=args.pretrained_model_name_or_path,
        dtype=DTYPE_MAP[args.dtype],
    )

    results = []
    for cloth_path, mask_path in pairs:
        target_path = resolve_target_for_name(args.target_image, cloth_path.name)
        sample_dir = out_dir / sanitize_stem(cloth_path)
        if not batch_mode:
            sample_dir = out_dir
        try:
            result = process_one(
                attacker=attacker,
                cloth_path=cloth_path,
                mask_path=mask_path,
                target_path=target_path,
                args=args,
                sample_out_dir=sample_dir,
            )
            results.append(result)
        except Exception as e:
            print(f"[error] Failed on {cloth_path.name}: {e}")
            results.append(
                {
                    "name": cloth_path.name,
                    "sample_dir": str(sample_dir),
                    "error": str(e),
                }
            )

    if batch_mode:
        summary = {
            "num_pairs": len(pairs),
            "num_success": sum(1 for r in results if "error" not in r),
            "num_failed": sum(1 for r in results if "error" in r),
            "results": results,
        }
        with open(out_dir / "batch_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved batch outputs to: {out_dir}")
        print(f"Successful: {summary['num_success']} / {summary['num_pairs']}")
    else:
        success = next((r for r in results if "error" not in r), None)
        if success is not None:
            print(f"Saved outputs to: {out_dir}")
            print(f"Final loss: {success['final_loss']}")


if __name__ == "__main__":
    main()
