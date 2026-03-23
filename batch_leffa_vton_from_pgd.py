import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from leffa.transform import LeffaTransform
from leffa.model import LeffaModel
from leffa.inference import LeffaInference
from leffa_utils.densepose_predictor import DensePosePredictor
from leffa_utils.utils import resize_and_center


COMMON_EXTS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Batch virtual try-on with Leffa using cloth_clean.png / cloth_adv.png produced "
            "under pgd_ref_attention_out/<sample_id>/ folders."
        )
    )
    parser.add_argument("--pgd_root", type=str, default="/workspace/Leffa/pgd_ref_attention_out")
    parser.add_argument("--image_dir", type=str, default="/workspace/test/image")
    parser.add_argument("--agnostic_mask_dir", type=str, default="/workspace/test/agnostic-mask")
    parser.add_argument("--cloth_mask_dir", type=str, default="/workspace/test/cloth-mask")
    parser.add_argument("--output_clean_dir", type=str, default="/workspace/outputs/clean")
    parser.add_argument("--output_adv_dir", type=str, default="/workspace/outputs/adv")
    parser.add_argument("--pretrained_model_name_or_path", type=str, default="./ckpts/stable-diffusion-inpainting")
    parser.add_argument("--pretrained_model", type=str, default="./ckpts/virtual_tryon.pth")
    parser.add_argument("--densepose_config_path", type=str, default="./ckpts/densepose/densepose_rcnn_R_50_FPN_s1x.yaml")
    parser.add_argument("--densepose_weights_path", type=str, default="./ckpts/densepose/model_final_162be9.pkl")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--scale", type=float, default=2.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ref_acceleration", action="store_true")
    parser.add_argument("--repaint", action="store_true")
    parser.add_argument("--vt_model_type", type=str, default="viton_hd", choices=["viton_hd", "dress_code"])
    parser.add_argument("--folder_prefix", type=str, default="00")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--save_summary", action="store_true")
    return parser.parse_args()


class BatchLeffaVTON:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.transform = LeffaTransform(height=args.height, width=args.width)
        self.densepose_predictor = DensePosePredictor(
            config_path=args.densepose_config_path,
            weights_path=args.densepose_weights_path,
        )
        vt_model = LeffaModel(
            pretrained_model_name_or_path=args.pretrained_model_name_or_path,
            pretrained_model=args.pretrained_model,
            dtype=args.dtype,
        )
        self.inference = LeffaInference(model=vt_model)

    def run_one(self, src_image_path: Path, agnostic_mask_path: Path, cloth_image_path: Path, cloth_mask_path: Path, output_path: Path) -> Dict:
        src_image = Image.open(src_image_path).convert("RGB")
        src_image = resize_and_center(src_image, self.args.width, self.args.height)

        agnostic_mask = load_and_pad_mask(agnostic_mask_path, self.args.width, self.args.height)
        cloth_image = prepare_cloth_image(cloth_image_path, cloth_mask_path, self.args.width, self.args.height)
        densepose = make_densepose(self.densepose_predictor, src_image, self.args.vt_model_type)

        data = {
            "src_image": [src_image],
            "ref_image": [cloth_image],
            "mask": [agnostic_mask],
            "densepose": [densepose],
        }
        data = self.transform(data)
        output = self.inference(
            data,
            ref_acceleration=self.args.ref_acceleration,
            num_inference_steps=self.args.steps,
            guidance_scale=self.args.scale,
            seed=self.args.seed,
            repaint=self.args.repaint,
        )
        gen_image = output["generated_image"][0]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        gen_image.save(output_path)
        return {
            "src_image": str(src_image_path),
            "agnostic_mask": str(agnostic_mask_path),
            "cloth_image": str(cloth_image_path),
            "cloth_mask": str(cloth_mask_path),
            "output": str(output_path),
        }



def make_densepose(predictor: DensePosePredictor, src_image: Image.Image, vt_model_type: str) -> Image.Image:
    src_array = np.array(src_image)
    if vt_model_type == "viton_hd":
        seg_array = predictor.predict_seg(src_array)[:, :, ::-1]
        return Image.fromarray(seg_array)
    iuv_array = predictor.predict_iuv(src_array)
    seg_array = iuv_array[:, :, 0:1]
    seg_array = np.concatenate([seg_array] * 3, axis=-1)
    return Image.fromarray(seg_array)



def prepare_cloth_image(cloth_image_path: Path, cloth_mask_path: Path, width: int, height: int) -> Image.Image:
    cloth = Image.open(cloth_image_path).convert("RGB")
    cloth_mask = Image.open(cloth_mask_path).convert("L")
    if cloth_mask.size != cloth.size:
        cloth_mask = cloth_mask.resize(cloth.size, Image.NEAREST)
    cloth_mask_np = np.array(cloth_mask, dtype=np.uint8)
    cloth_mask_bin = Image.fromarray(np.where(cloth_mask_np >= 128, 255, 0).astype(np.uint8), mode="L")
    white_bg = Image.new("RGB", cloth.size, (255, 255, 255))
    cloth_fg = Image.composite(cloth, white_bg, cloth_mask_bin)
    return resize_and_center(cloth_fg, width, height)



def load_and_pad_mask(mask_path: Path, width: int, height: int) -> Image.Image:
    mask = Image.open(mask_path).convert("L")
    original_width, original_height = mask.size
    scale = min(width / original_width, height / original_height)
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    resized = mask.resize((new_width, new_height), Image.NEAREST)
    canvas = Image.new("L", (width, height), 0)
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    canvas.paste(resized, (left, top))
    canvas_np = np.array(canvas, dtype=np.uint8)
    canvas_np = np.where(canvas_np >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(canvas_np, mode="L")



def find_matching_file(image_dir: Path, stem: str) -> Optional[Path]:
    for ext in COMMON_EXTS:
        candidate = image_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    matches = sorted([p for p in image_dir.glob(f"{stem}.*") if p.suffix.lower() in COMMON_EXTS])
    return matches[0] if matches else None



def build_tasks(args: argparse.Namespace) -> Tuple[List[Dict], List[Dict]]:
    pgd_root = Path(args.pgd_root)
    image_dir = Path(args.image_dir)
    agnostic_mask_dir = Path(args.agnostic_mask_dir)
    cloth_mask_dir = Path(args.cloth_mask_dir)
    output_clean_dir = Path(args.output_clean_dir)
    output_adv_dir = Path(args.output_adv_dir)

    tasks: List[Dict] = []
    skipped: List[Dict] = []

    for folder in sorted(pgd_root.iterdir()):
        if not folder.is_dir() or not folder.name.startswith(args.folder_prefix):
            continue

        stem = folder.name
        src_image = find_matching_file(image_dir, stem)
        agnostic_mask = agnostic_mask_dir / f"{stem}_mask.png"
        cloth_mask = find_matching_file(cloth_mask_dir, stem)
        clean_cloth = folder / "cloth_clean.png"
        adv_cloth = folder / "cloth_adv.png"

        missing = []
        if src_image is None:
            missing.append("src_image")
        if not agnostic_mask.exists():
            missing.append("agnostic_mask")
        if cloth_mask is None:
            missing.append("cloth_mask")
        if not clean_cloth.exists():
            missing.append("cloth_clean.png")
        if not adv_cloth.exists():
            missing.append("cloth_adv.png")

        if missing:
            skipped.append({"sample": stem, "reason": "missing_inputs", "missing": missing})
            continue

        output_name = src_image.name
        tasks.append(
            {
                "sample": stem,
                "mode": "clean",
                "src_image": src_image,
                "agnostic_mask": agnostic_mask,
                "cloth_image": clean_cloth,
                "cloth_mask": cloth_mask,
                "output": output_clean_dir / output_name,
            }
        )
        tasks.append(
            {
                "sample": stem,
                "mode": "adv",
                "src_image": src_image,
                "agnostic_mask": agnostic_mask,
                "cloth_image": adv_cloth,
                "cloth_mask": cloth_mask,
                "output": output_adv_dir / output_name,
            }
        )

    return tasks, skipped



def main(args: argparse.Namespace) -> None:
    tasks, skipped = build_tasks(args)
    runner = BatchLeffaVTON(args)

    processed: List[Dict] = []
    failed: List[Dict] = []

    for task in tasks:
        output_path: Path = task["output"]
        if output_path.exists() and not args.overwrite:
            processed.append({
                "sample": task["sample"],
                "mode": task["mode"],
                "status": "skipped_exists",
                "output": str(output_path),
            })
            continue

        try:
            meta = runner.run_one(
                src_image_path=task["src_image"],
                agnostic_mask_path=task["agnostic_mask"],
                cloth_image_path=task["cloth_image"],
                cloth_mask_path=task["cloth_mask"],
                output_path=output_path,
            )
            meta.update({"sample": task["sample"], "mode": task["mode"], "status": "ok"})
            processed.append(meta)
            print(f"[OK] {task['sample']} [{task['mode']}] -> {output_path}")
        except Exception as exc:
            failed.append({
                "sample": task["sample"],
                "mode": task["mode"],
                "status": "failed",
                "error": repr(exc),
            })
            print(f"[FAIL] {task['sample']} [{task['mode']}] -> {exc}")

    if args.save_summary:
        summary = {
            "processed": processed,
            "failed": failed,
            "skipped": skipped,
            "num_tasks": len(tasks),
            "num_processed_entries": len(processed),
            "num_failed": len(failed),
            "num_skipped": len(skipped),
        }
        summary_path = Path(args.output_clean_dir).parent / "batch_leffa_vton_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main(parse_args())
