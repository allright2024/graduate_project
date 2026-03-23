import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

try:
    from skimage.metrics import structural_similarity as ssim_fn
except ImportError as e:
    raise ImportError(
        "scikit-image가 필요합니다. 설치: pip install scikit-image"
    ) from e


VALID_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="pgd_ref_attention_out 내 00* 폴더의 cloth_clean.png / cloth_adv.png에 대해 SSIM, PSNR 평균 계산"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="예: /workspace/Leffa/pgd_ref_attention_out",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="결과 JSON 저장 경로. 기본값: <root>/adv_clean_metrics.json",
    )
    return parser.parse_args()


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
    # skimage 버전 차이를 감안해 channel_axis / multichannel 둘 다 대응
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

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise FileNotFoundError(f"유효한 디렉터리가 아닙니다: {root}")

    output_json = (
        Path(args.output_json)
        if args.output_json is not None
        else root / "adv_clean_metrics.json"
    )

    sample_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("00")]
    )

    results = []
    skipped = []

    for sample_dir in sample_dirs:
        clean_path = sample_dir / "cloth_clean.png"
        adv_path = sample_dir / "cloth_adv.png"

        missing = []
        if not clean_path.exists():
            missing.append(str(clean_path.name))
        if not adv_path.exists():
            missing.append(str(adv_path.name))

        if missing:
            skipped.append(
                {
                    "sample": sample_dir.name,
                    "reason": "missing_files",
                    "missing": missing,
                }
            )
            continue

        try:
            clean_img = load_rgb_image(clean_path)
            adv_img = load_rgb_image(adv_path)

            if clean_img.shape != adv_img.shape:
                skipped.append(
                    {
                        "sample": sample_dir.name,
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
                    "sample": sample_dir.name,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                    "clean_path": str(clean_path),
                    "adv_path": str(adv_path),
                }
            )
        except Exception as e:
            skipped.append(
                {
                    "sample": sample_dir.name,
                    "reason": "exception",
                    "message": str(e),
                }
            )

    ssim_values = [r["ssim"] for r in results]
    psnr_values = [r["psnr"] for r in results]

    summary = {
        "root": str(root),
        "num_sample_dirs": len(sample_dirs),
        "num_valid_samples": len(results),
        "num_skipped_samples": len(skipped),
        "mean_ssim": safe_mean(ssim_values),
        "mean_psnr": safe_mean(psnr_values),
        "per_sample": results,
        "skipped": skipped,
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Scanned sample dirs : {len(sample_dirs)}")
    print(f"Valid samples       : {len(results)}")
    print(f"Skipped samples     : {len(skipped)}")
    print(f"Mean SSIM           : {summary['mean_ssim']}")
    print(f"Mean PSNR           : {summary['mean_psnr']}")
    print(f"Saved JSON          : {output_json}")


if __name__ == "__main__":
    main()