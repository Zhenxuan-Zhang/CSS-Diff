import os
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio

def compute_ssim_and_psnr(test_dir, real_prefix='real_B', fake_prefix='fake_B'):
    """
    Compute SSIM and PSNR (mean ± std) between real and fake grayscale images in a directory.

    Args:
        test_dir (str): Path to the directory containing real and fake images.
        real_prefix (str): Prefix for real images (default: 'real_B').
        fake_prefix (str): Prefix for fake images (default: 'fake_B').

    Returns:
        (str, str): LaTeX-formatted SSIM and PSNR strings
    """
    transform = transforms.ToTensor()
    # transform = transforms.Compose([
    # transforms.ToTensor(),                      # Converts to [0, 1]
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Then to [-1, 1]
    # ])
    real_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if real_prefix in f])
    fake_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if fake_prefix in f])

    assert len(real_paths) == len(fake_paths), "Mismatch in number of real and fake images."

    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0)

    ssim_scores, psnr_scores = [], []

    for real_path, fake_path in zip(real_paths, fake_paths):
        real_img = transform(Image.open(real_path).convert("L")).unsqueeze(0)
        fake_img = transform(Image.open(fake_path).convert("L")).unsqueeze(0)

        assert real_img.shape == fake_img.shape, f"Image size mismatch: {real_path}, {fake_path}"

        ssim_val = ssim_metric(preds=fake_img, target=real_img).item()
        psnr_val = psnr_metric(preds=fake_img, target=real_img).item()

        ssim_scores.append(ssim_val)
        psnr_scores.append(psnr_val)

    avg_ssim = np.mean(ssim_scores)
    std_ssim = np.std(ssim_scores)
    avg_psnr = np.mean(psnr_scores)
    std_psnr = np.std(psnr_scores)

    # LaTeX-friendly output
    ssim_latex = f"{avg_ssim:.4f}$\\pm${std_ssim:.4f}"
    psnr_latex = f"{avg_psnr:.2f}$\\pm${std_psnr:.2f}"

    print("LaTeX-ready results:")
    print(f"SSIM: {ssim_latex}")
    print(f"PSNR: {psnr_latex}")

    return ssim_latex, psnr_latex

def main():
    print("Choose a metric to compute:")
    print("1. SSIM and PSNR")

    choice = input("Enter the number of your choice: ")
    test_dir = input("Enter the test directory path: ")

    if choice == "1":
        compute_ssim_and_psnr(test_dir)
    else:
        print("Invalid choice. Please choose 1.")

if __name__ == "__main__":
    main()
