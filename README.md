# Adversarial Perturbation for Clothing Data Protection and Virtual Try-on Disruption


## Adversarial & PGD Attack Scripts

This repository includes several scripts for generating adversarial examples (PGD attacks) and evaluating adversarial robustness on Leffa:

### 1. PGD Optimization Scripts
- **`leffa_photoguard_pgd.py`**: Runs a PhotoGuard-style PGD attack on the reference/cloth image. The objective is to minimize the L2 distance between the Generative UNet's noise prediction for the adversarial cloth and a target cloth.
- **`photoguard_masked_cloth_pgd.py`**: PhotoGuard-style masked PGD for cloth images. It perturbs pixels where the mask equals 1 using VAE latent target loss.
- **`optimize_ref_attention_pgd.py`**: PGD attack over the reference/cloth image using only Leffa's Reference UNet. It aims to minimize the sum of L2 norms of the reference UNet's internal self-attention maps.
- **`optimize_reference_attention_pgd.py`**: Extended version of `optimize_ref_attention_pgd.py` that includes options to visualize selected self-attention maps (e.g., mean attention matrix, mean key activation map).

### 2. Batch Inference Script
- **`batch_leffa_vton_from_pgd.py`**: Performs batch virtual try-on inference with Leffa using the clean and adversarial cloth images produced by the PGD scripts.

### 3. Metric Evaluation Scripts
- **`clean_adv_SSIM_PSNR.py`**: Computes the average SSIM and PSNR between the original (clean) cloth and the optimized (adversarial) cloth images.
- **`compute_outputs_metrics.py`**: Computes the average SSIM and PSNR between the virtual try-on results generated using the clean cloth versus the adversarial cloth.



