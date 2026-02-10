import os
import warnings
warnings.filterwarnings("ignore")

# 禁止 torchmetrics / tokenizer 的隐式输出
os.environ["TORCHMETRICS_DEBUG"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import random
import numpy as np
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import (
    MultiScaleStructuralSimilarityIndexMeasure,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
)


# --------------------------------------------------
# FID / IS / MS-SSIM (unaligned)
# --------------------------------------------------
def compute_metrics(val_set, model, device=None, batch_count=5, max_samples=200):
    if device is None:
        device = model.device

    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    transform_to_uint8 = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: (x * 255).byte()),
    ])

    model.eval()
    total_samples = len(val_set)

    # ⭐ 限制 validation 样本数（避免超慢 + stdout flush）
    num_samples = min(max_samples, total_samples)
    selected_indices = set(random.sample(range(total_samples), num_samples))

    fake_images = []
    fake_images_msssim = []

    # ⭐ 关键：禁止 autograd + 同步
    with torch.no_grad():
        for idx, data in enumerate(val_set):
            if idx not in selected_indices:
                continue

            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            real_B = transform_to_uint8(visuals["real_B"].cpu()).to(device)
            fake_B = transform_to_uint8(visuals["fake_B"].cpu()).to(device)

            # ⭐ 禁止 expand（会导致隐式同步）
            real_B = real_B.repeat(1, 3, 1, 1)
            fake_B = fake_B.repeat(1, 3, 1, 1)

            fid.update(real_B, real=True)
            fid.update(fake_B, real=False)

            fake_images.append(fake_B)
            fake_images_msssim.append(fake_B.float() / 255.0)

    fake_images = torch.cat(fake_images, dim=0)
    fake_images_msssim = torch.cat(fake_images_msssim, dim=0)

    # Inception / FID
    is_mean, _ = inception(fake_images)
    fid_score = fid.compute()

    # MS-SSIM（避免 print / tqdm）
    ms_ssim_scores = []
    batch_size = max(1, len(fake_images_msssim) // batch_count)

    for i in range(0, len(fake_images_msssim) - 1, batch_size):
        batch = fake_images_msssim[i:i + batch_size]
        for j in range(len(batch) - 1):
            score = ms_ssim_metric(
                batch[j].unsqueeze(0),
                batch[j + 1].unsqueeze(0)
            )
            ms_ssim_scores.append(score.item())

    avg_ms_ssim = float(np.mean(ms_ssim_scores)) if len(ms_ssim_scores) > 0 else 0.0

    return fid_score.item(), is_mean.item(), avg_ms_ssim


# --------------------------------------------------
# PSNR / SSIM (aligned)
# --------------------------------------------------
def compute_ssim_and_psnr(val_set, model, device=None, max_samples=200):
    if device is None:
        device = model.device

    model.eval()
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(device)

    ssim_scores, psnr_scores = [], []
    total_samples = len(val_set)
    num_samples = min(max_samples, total_samples)
    selected_indices = set(random.sample(range(total_samples), num_samples))

    # ⭐ 关键：no_grad
    with torch.no_grad():
        for idx, data in enumerate(val_set):
            if idx not in selected_indices:
                continue

            model.set_input(data)
            model.test()
            visuals = model.get_current_visuals()

            real_B = visuals["real_B"].to(device).float()
            fake_B = visuals["fake_B"].to(device).float()

            assert real_B.shape == fake_B.shape, \
                f"Shape mismatch: {real_B.shape} vs {fake_B.shape}"

            for real_img, fake_img in zip(real_B, fake_B):
                ssim_val = ssim_metric(
                    preds=fake_img.unsqueeze(0),
                    target=real_img.unsqueeze(0)
                ).item()
                psnr_val = psnr_metric(
                    preds=fake_img.unsqueeze(0),
                    target=real_img.unsqueeze(0)
                ).item()

                ssim_scores.append(ssim_val)
                psnr_scores.append(psnr_val)

    avg_ssim = float(np.mean(ssim_scores))
    avg_psnr = float(np.mean(psnr_scores))

    return avg_ssim, avg_psnr


# --------------------------------------------------
# Entry
# --------------------------------------------------
def validation(val_set, model, opt, device=None):
    if device is None:
        device = model.device

    if "unaligned" in opt.dataset_mode:
        fid, is_mean, ms_ssim = compute_metrics(val_set, model, device=device)
        print(f"[Validation] FID: {fid:.4f}")
        print(f"[Validation] Inception Score: {is_mean:.4f}")
        print(f"[Validation] MS-SSIM: {ms_ssim:.4f}")
        return fid, is_mean, ms_ssim

    else:
        ssim, psnr = compute_ssim_and_psnr(val_set, model, device=device)
        print(f"[Validation] SSIM: {ssim:.4f}")
        print(f"[Validation] PSNR: {psnr:.4f}")
        return ssim, psnr
