import os
import torch
from torchvision import transforms
from PIL import Image
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure
import numpy as np

def compute_fid_and_is_from_test_dir(test_dir, real_prefix='real_B', fake_prefix='fake_B', device='cuda:0'):
    """
    Compute FID and Inception Score for test dataset, ensuring all images are resized to 224x224 if necessary.

    Args:
        test_dir (str): Directory containing test images.
        real_prefix (str): Prefix for the real images (default: 'real_B').
        fake_prefix (str): Prefix for the fake/synthetic images (default: 'fake_B').
        device (str): Computation device ('cuda:X' or 'cpu').

    Returns:
        fid_score (float): FID score between the real and fake images.
        is_mean (float): Mean Inception Score for the fake images.
    """
    # Initialize FID and IS on the specified device
    fid = FrechetInceptionDistance(feature=2048).to(device)
    inception = InceptionScore().to(device)

    # Transform to uint8 tensor with conditional resizing
    transform_to_uint8 = transforms.Compose([
        transforms.Lambda(lambda img: img if img.size == (224, 224) else img.resize((224, 224))),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x * 255).byte()),
    ])

    # Get image paths
    real_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if real_prefix in f])
    fake_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if fake_prefix in f])

    # Process real images for FID
    for img_path in real_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform_to_uint8(img).unsqueeze(0).to(device)
        fid.update(img_tensor, real=True)

    # Process fake images for FID and IS
    fake_images = []
    for img_path in fake_paths:
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform_to_uint8(img).unsqueeze(0).to(device)
        fid.update(img_tensor, real=False)
        fake_images.append(img_tensor)

    # Compute IS
    fake_images = torch.cat(fake_images, dim=0)
    is_mean, is_std = inception(fake_images)

    # Compute FID
    fid_score = fid.compute()

    print(f"FID: {fid_score.item():.4f}")
    print(f"Inception Score: {is_mean.item():.4f} ± {is_std.item():.4f}")

    return fid_score.item(), is_mean.item()
def compute_ms_ssim_among_synthetic_with_torchmetrics(test_dir, fake_prefix='fake_B', batch_count=40, device='cuda:0'):
    """
    Compute MS-SSIM among synthetic images with the same normalization strategy as compute_metrics.

    Args:
        test_dir (str): Directory containing test images.
        fake_prefix (str): Prefix for fake images (default: 'fake_B').
        batch_count (int): Number of batches to divide images into.
        device (str): Computation device ('cuda:X' or 'cpu').

    Returns:
        avg_ms_ssim (float): Average MS-SSIM score.
        std_ms_ssim (float): Standard deviation of MS-SSIM scores.
    """
    # Initialize MS-SSIM metric on the specified device
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # Match normalization

    # Image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to [0,1] range
        transforms.Lambda(lambda x: (x * 255).byte())  # Convert to uint8 (0-255)
    ])

    # Load and preprocess images
    fake_b_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if fake_prefix in f])
    assert len(fake_b_paths) >= batch_count, f"Need at least {batch_count} images."

    fake_images = []
    for path in fake_b_paths:
        img = Image.open(path).convert("RGB")
        if img.size[0] < 160 or img.size[1] < 160:
            img = img.resize((224, 224))
        img = transform(img).to(device)  # Convert and move tensor to device
        #img = img.expand(3, -1, -1)  # Ensure 3-channel consistency
        fake_images.append(img)

    # Split into batches
    batch_size = len(fake_images) // batch_count
    batches = [torch.stack(fake_images[i*batch_size:(i+1)*batch_size]) for i in range(batch_count)]

    # Compute MS-SSIM
    ms_ssim_scores = []
    for batch in batches:
        for i in range(len(batch)-1):
            score = ms_ssim_metric(
                preds=batch[i].unsqueeze(0),
                target=batch[i+1].unsqueeze(0)
            )
            ms_ssim_scores.append(score.item())

    avg_ms_ssim = np.mean(ms_ssim_scores)
    std_ms_ssim = np.std(ms_ssim_scores)
    print(f"MS-SSIM: {avg_ms_ssim:.4f} ± {std_ms_ssim:.4f}")
    
    return avg_ms_ssim, std_ms_ssim



def compute_ms_ssim_among_synthetic_with_torchmetrics(test_dir, fake_prefix='fake_B', batch_count=5, device='cuda:0'):
    """
    Compute MS-SSIM among synthetic images with the same normalization strategy as compute_metrics.

    Args:
        test_dir (str): Directory containing test images.
        fake_prefix (str): Prefix for fake images (default: 'fake_B').
        batch_count (int): Number of batches to divide images into.
        device (str): Computation device ('cuda:X' or 'cpu').

    Returns:
        avg_ms_ssim (float): Average MS-SSIM score.
        std_ms_ssim (float): Standard deviation of MS-SSIM scores.
    """
    # Initialize MS-SSIM metric on the specified device
    ms_ssim_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # Ensures [0,1] range

    # Image transformation
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts image to range [0,1]
    ])

    # Load and preprocess images
    fake_b_paths = sorted([os.path.join(test_dir, f) for f in os.listdir(test_dir) if fake_prefix in f])
    assert len(fake_b_paths) >= batch_count, f"Need at least {batch_count} images."

    fake_images = []
    for path in fake_b_paths:
        img = Image.open(path).convert("RGB")
        if img.size[0] < 160 or img.size[1] < 160:
            img = img.resize((224, 224))  # Resize if too small
        img = transform(img).to(device)  # Convert to tensor and move to device
        fake_images.append(img)

    # Split into batches
    batch_size = len(fake_images) // batch_count
    batches = [torch.stack(fake_images[i*batch_size:(i+1)*batch_size]) for i in range(batch_count)]

    # Compute MS-SSIM
    ms_ssim_scores = []
    for batch in batches:
        for i in range(len(batch)-1):
            score = ms_ssim_metric(
                preds=batch[i].unsqueeze(0),  # Add batch dim
                target=batch[i+1].unsqueeze(0)  # Add batch dim
            )
            ms_ssim_scores.append(score.item())

    avg_ms_ssim = np.mean(ms_ssim_scores)
    std_ms_ssim = np.std(ms_ssim_scores)
    print(f"MS-SSIM: {avg_ms_ssim:.4f} ± {std_ms_ssim:.4f}")
    
    return avg_ms_ssim, std_ms_ssim

def main():
    print("Choose a metric to compute:")
    print("1. FID and IS")
    print("2. MS-SSIM")
    choice = input("Enter your choice (1/2): ").strip()
    
    test_dir = input("Enter test directory path: ").strip()
    
    # Device selection
    default_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = input(f"Enter device [{default_device}]: ").strip() or default_device

    if choice == "1":
        real_prefix = input("Real image prefix [real_B]: ").strip() or 'real_B'
        fake_prefix = input("Fake image prefix [fake_B]: ").strip() or 'fake_B'
        compute_fid_and_is_from_test_dir(test_dir, real_prefix, fake_prefix, device=device)
    elif choice == "2":
        fake_prefix = input("Fake image prefix [fake_B]: ").strip() or 'fake_B'
        batch_count = input("Batch count [5]: ").strip()
        batch_count = int(batch_count) if batch_count else 5
        compute_ms_ssim_among_synthetic_with_torchmetrics(
            test_dir, fake_prefix, batch_count, device=device
        )
    else:
        print("Invalid choice")

if __name__ == "__main__":
    main()
