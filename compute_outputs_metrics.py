import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError as e:
    raise ImportError("scikit-image가 필요합니다. `pip install scikit-image`") from e


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="outputs/clean 과 outputs/adv 의 매칭 이미지들에 대해 SSIM, PSNR 평균 계산"
    )
    parser.add_argument("--clean_dir", type=str, required=True, help="예: /workspace/outputs/clean")
    parser.add_argument("--adv_dir", type=str, required=True, help="예: /workspace/outputs/adv")
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="결과 JSON 저장 경로. 기본값: <adv_dir>/metrics_summary.json",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VALID_EXTS


def collect_images_recursive(root: Path):
    files = [p for p in root.rglob("*") if is_image_file(p)]
    rel_map = {}
    for p in files:
        rel_key = str(p.relative_to(root))
        rel_map[rel_key] = p
    return rel_map


def load_rgb_image(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.array(img, dtype=np.uint8)


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(255.0 / math.sqrt(mse))


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        return float(ssim_fn(img1, img2, data_range=255, channel_axis=-1))
    except TypeError:
        return float(ssim_fn(img1, img2, data_range=255, multichannel=True))


def safe_mean(values):
    if not values:
        return None
    finite_vals = [v for v in values if np.isfinite(v)]
    if not finite_vals:
        return None
    return float(np.mean(finite_vals))


def main():
    args = parse_args()

    clean_dir = Path(args.clean_dir)
    adv_dir = Path(args.adv_dir)

    if not clean_dir.is_dir():
        raise FileNotFoundError(f"clean_dir가 디렉터리가 아닙니다: {clean_dir}")
    if not adv_dir.is_dir():
        raise FileNotFoundError(f"adv_dir가 디렉터리가 아닙니다: {adv_dir}")

    output_json = Path(args.output_json) if args.output_json else adv_dir / "metrics_summary.json"

    clean_map = collect_images_recursive(clean_dir)
    adv_map = collect_images_recursive(adv_dir)

    common_keys = sorted(set(clean_map.keys()) & set(adv_map.keys()))
    clean_only = sorted(set(clean_map.keys()) - set(adv_map.keys()))
    adv_only = sorted(set(adv_map.keys()) - set(clean_map.keys()))

    results = []
    skipped = []

    for key in common_keys:
        clean_path = clean_map[key]
        adv_path = adv_map[key]

        try:
            clean_img = load_rgb_image(clean_path)
            adv_img = load_rgb_image(adv_path)

            if clean_img.shape != adv_img.shape:
                skipped.append(
                    {
                        "image": key,
                        "reason": "shape_mismatch",
                        "clean_shape": list(clean_img.shape),
                        "adv_shape": list(adv_img.shape),
                    }
                )
                continue

            ssim_val = compute_ssim(clean_img, adv_img)
            psnr_val = compute_psnr(clean_img, adv_img)

            results.append(
                {
                    "image": key,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                    "clean_path": str(clean_path),
                    "adv_path": str(adv_path),
                }
            )
        except Exception as e:
            skipped.append(
                {
                    "image": key,
                    "reason": "exception",
                    "message": str(e),
                }
            )

    summary = {
        "clean_dir": str(clean_dir),
        "adv_dir": str(adv_dir),
        "num_clean_images": len(clean_map),
        "num_adv_images": len(adv_map),
        "num_matched_images": len(common_keys),
        "num_valid_pairs": len(results),
        "num_skipped_pairs": len(skipped),
        "mean_ssim": safe_mean([r["ssim"] for r in results]),
        "mean_psnr": safe_mean([r["psnr"] for r in results]),
        "clean_only": clean_only,
        "adv_only": adv_only,
        "per_image": results,
        "skipped": skipped,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Clean images     : {len(clean_map)}")
    print(f"Adv images       : {len(adv_map)}")
    print(f"Matched images   : {len(common_keys)}")
    print(f"Valid pairs      : {len(results)}")
    print(f"Skipped pairs    : {len(skipped)}")
    print(f"Mean SSIM        : {summary['mean_ssim']}")
    print(f"Mean PSNR        : {summary['mean_psnr']}")
    print(f"Saved JSON       : {output_json}")


if __name__ == "__main__":
    main()