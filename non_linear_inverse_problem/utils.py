"""
Utility functions for denoiser training, evaluation, and visualization.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import json


# ============================================================================
# DENOISER ARCHITECTURE
# ============================================================================

class BF_CNN(nn.Module):
    """
    Blind Denoiser architecture. Same as in the paper. 
    
    Residual learning network that learns to estimate noise rather than the
    clean image, enabling it to act as a blind denoiser for multiple noise levels.
    """
    
    def __init__(self, depth: int = 20, n_channels: int = 64, image_channels: int = 1):
        """
        Args:
            depth: Number of convolutional layers
            n_channels: Number of feature channels
            image_channels: Number of input image channels (1 for grayscale)
        """
        super(BF_CNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(image_channels, n_channels, kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Intermediate layers: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(n_channels, n_channels, kernel_size, padding=padding, bias=False))
            
            # Batch normalization without bias
            bn = nn.BatchNorm2d(n_channels, affine=True)
            bn.register_parameter('bias', None)
            layers.append(bn)

            layers.append(nn.ReLU(inplace=True))
            
        # Last layer: Conv (no activation)
        layers.append(nn.Conv2d(n_channels, image_channels, kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing residual estimate.
        
        Args:
            x: Noisy input [B, C, H, W]
        
        Returns:
            Clean image estimate (x - noise)
        """
        noise = self.dncnn(x)
        return x - noise


# ============================================================================
# DENOISER TRAINING
# ============================================================================

def train_blind_denoiser(
    model: nn.Module,
    config,
    train_loader: DataLoader,
    val_loader: DataLoader = None,
) -> Dict[str, List[float]]:
    """
    Train the blind denoiser on Gaussian noise corruption.
    
    The denoiser learns to estimate E[x | y] for Gaussian noise of any level σ.
    Training uses varying noise levels sampled uniformly at each epoch.
    
    Args:
        model: BF_CNN denoiser instance
        config: Configuration object with training parameters
        train_loader: Training data loader
        val_loader: Validation data loader (optional)
    
    Returns:
        Dictionary with training history
    """
    device = config.device
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
        'learning_rate': []
    }

    best_val_loss = float('inf')
    patience_counter = 0
    
    print("Starting denoiser training...")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(config.num_epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        n_batches = 0
        
        for x_clean, _ in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}"):
            x_clean = x_clean.to(device)
            
            # Sample noise level for each image in batch
            batch_size = x_clean.shape[0]
            sigma = torch.rand(batch_size, 1, 1, 1, device=device) * config.sigma_max
            
            # Add Gaussian noise
            noise = torch.randn_like(x_clean) * sigma
            y_noisy = x_clean + noise
            
            # Forward pass
            x_hat = model(y_noisy)
            
            # Compute loss
            loss = criterion(x_hat, x_clean)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        history['train_loss'].append(train_loss)
        
        # Validation Phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            n_val_batches = 0
            
            with torch.no_grad():
                for x_clean, _ in tqdm.tqdm(val_loader, desc=f"Validation Epoch {epoch+1}/{config.num_epochs}"):
                    x_clean = x_clean.to(device)
                    
                    # Sample noise levels
                    batch_size = x_clean.shape[0]
                    sigma = torch.rand(batch_size, 1, 1, 1, device=device) * config.sigma_max
                    
                    noise = torch.randn_like(x_clean) * sigma
                    y_noisy = x_clean + noise
                    x_hat = model(y_noisy)
                    val_loss += criterion(x_hat, x_clean).item()
                    n_val_batches += 1
            
            val_loss /= n_val_batches
            history['val_loss'].append(val_loss)
            history['epoch'].append(epoch)
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping with best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'best_denoiser.pt')
            else:
                patience_counter += 1
                if patience_counter >= 12:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            print(
                f"Epoch {epoch+1:3d}/{config.num_epochs} - "
                f"Train Loss: {train_loss:.8f} - Val Loss: {val_loss:.8f} - "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
        else:
            print(f"Epoch {epoch+1:3d}/{config.num_epochs} - Train Loss: {train_loss:.8f}")
    
    print("Denoiser training complete!")

    # Load best model
    try:
        model.load_state_dict(torch.load('best_denoiser.pt'))
    except FileNotFoundError:
        print("Warning: 'best_denoiser.pt' not found, using last epoch's weights.")

    return history


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def evaluate_reconstruction(
    x_reconstructed: torch.Tensor,
    x_ground_truth: torch.Tensor,
    A,  # Can be torch.Tensor (matrix) or task object
    b_observed: torch.Tensor,
) -> Dict[str, float]:
    """
    Evaluate reconstruction quality.
    
    Computes image-space MSE, measurement error, and PSNR.
    
    Args:
        x_reconstructed: Reconstructed image [N]
        x_ground_truth: Ground truth image [N]
        A: Measurement matrix [M, N] (tensor) OR PhaseRetrievalTask object
        b_observed: Observed magnitudes [M]
    
    Returns:
        Dictionary with metrics: {mse_image, measurement_error, psnr}
    """
    # MSE in image space
    mse_img = (torch.norm(x_reconstructed - x_ground_truth) ** 2).item() / x_ground_truth.numel()
    
    # Measurement error
    if isinstance(A, torch.Tensor):
        b_recon = torch.abs(A @ x_reconstructed)
    else:
        # A is a PhaseRetrievalTask object with get_magnitudes method
        b_recon = A.get_magnitudes(x_reconstructed)
    
    measurement_error = (torch.norm(b_recon - b_observed) ** 2).item() / b_observed.numel()
    
    # PSNR
    max_val = x_ground_truth.max().item()
    psnr = 10 * np.log10(max_val ** 2 / mse_img) if mse_img > 0 else float('inf')
    
    return {
        'mse_image': mse_img,
        'measurement_error': measurement_error,
        'psnr': psnr,
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_reconstruction_results(
    x_orig: torch.Tensor,
    x_recon: torch.Tensor,
    measurement: torch.Tensor,
    trajectory: List[torch.Tensor],
    metrics: Dict,
    save_path: Path,
    image_size: int = 28,
):
    """
    Visualize reconstruction results and convergence trajectory.
    
    Creates a figure showing:
    - Ground truth image
    - Observed measurements
    - Reconstructed image
    - Absolute error
    - Intermediate reconstructions
    
    Args:
        x_orig: Original image [N]
        x_recon: Reconstructed image [N]
        measurement: Observed magnitudes [M]
        trajectory: List of intermediate reconstructions
        metrics: Evaluation metrics
        save_path: Path to save figure
        image_size: Image height/width
    """
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Original image
    axes[0, 0].imshow(x_orig.view(image_size, image_size), cmap='gray')
    axes[0, 0].set_title('Ground Truth')
    axes[0, 0].axis('off')

    # ===== ROBUST MEASUREMENT VISUALIZATION =====
    # Try to reshape measurements intelligently
    m = measurement.numel()
    
    # Strategy: Try to make one dimension equal to image_size if possible
    if m % image_size == 0:
        # M is a multiple of image_size: reshape as (M/image_size, image_size)
        b_img = measurement.view(m // image_size, image_size).cpu()
    else:
        # Find closest factors to create a reasonable rectangular shape
        best_dim1 = int(np.sqrt(m))
        for i in range(int(np.sqrt(m)), 0, -1):
            if m % i == 0:
                best_dim1 = i
                break
        b_img = measurement.view(best_dim1, m // best_dim1).cpu()
    
    axes[0, 1].imshow(b_img, cmap='viridis')
    axes[0, 1].set_title(f'Observed Magnitudes [{b_img.shape[0]}×{b_img.shape[1]}]')
    axes[0, 1].axis('off')
    
    # Reconstructed image
    axes[0, 2].imshow(x_recon.view(image_size, image_size), cmap='gray')
    axes[0, 2].set_title('Reconstructed')
    axes[0, 2].axis('off')
    
    # Absolute error
    diff = (x_recon - x_orig).abs()
    axes[1, 2].imshow(diff.view(image_size, image_size), cmap='hot')
    axes[1, 2].set_title('Absolute Error')
    axes[1, 2].axis('off')
    
    # Intermediate reconstructions
    n_traj = min(2, len(trajectory))
    for i in range(n_traj):
        idx = (i + 1) * len(trajectory) // (n_traj + 1)
        ax = axes[1, i]
        im = trajectory[idx].view(image_size, image_size)
        ax.imshow(im, cmap='gray')
        ax.set_title(f'Iter {idx * 10}')
        ax.axis('off')    
    
    plt.suptitle(
        f"Phase Retrieval | MSE={metrics['mse_image']:.6f} | "
        f"Meas.Err={metrics['measurement_error']:.6f} | "
        f"PSNR={metrics['psnr']:.2f}dB",
        fontsize=12
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved reconstruction figure to {save_path}")
    plt.show()


def visualize_convergence(
    sigma_trajectory: List[float],
    save_path: Path,
    sigma_L: float = 0.01,
):
    """
    Visualize convergence behavior.
    
    Args:
        sigma_trajectory: Estimated noise levels at each iteration
        mse_trajectory: MSE to ground truth at each iteration
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sigma_trajectory, linewidth=2, label='σ_t (estimated noise)')
    ax.axhline(y=sigma_L, color='r', linestyle='--', label='σ_L (convergence threshold)')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Noise Level')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_title('Convergence: Noise Level Over Iterations')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved convergence plot to {save_path}")
    plt.show()


def visualize_denoiser_performance(
    model: nn.Module,
    dataset,
    config,
    save_path: Path,
    test_indices: List[int] = None,
    noise_levels: List[float] = None,
):
    """
    Visualize blind denoiser performance on images with various noise levels.
    
    Args:
        model: Trained denoiser
        dataset: Test dataset
        config: Configuration object
        test_indices: Image indices to visualize (default: [0, 50, 100])
        noise_levels: Noise levels to test (default: [0.1, 0.2, 0.3])
    """
    if test_indices is None:
        test_indices = [0, 50, 100]
    if noise_levels is None:
        noise_levels = [0.1, 0.2, 0.3]
    
    print("\n--- Denoiser Test ---\n")
    
    model.eval()
    
    for test_idx in test_indices:
        x_clean, label = dataset[test_idx]
        x_clean = x_clean.to(config.device)
        
        print(f"Image {test_idx} - Label: {label}")
        
        fig, axes = plt.subplots(
            len(noise_levels) + 1, 3,
            figsize=(12, 4 * (len(noise_levels) + 1))
        )
        
        # Display clean image
        axes[0, 0].imshow(x_clean.squeeze().cpu(), cmap='gray')
        axes[0, 0].set_title('Clean Image', fontsize=11, fontweight='bold')
        axes[0, 0].axis('off')
        axes[0, 1].axis('off')
        axes[0, 2].axis('off')
        
        # Test with different noise levels
        with torch.no_grad():
            for i, sigma in enumerate(noise_levels, 1):
                # Add noise
                noise = torch.randn_like(x_clean) * sigma
                x_noisy = x_clean + noise
                x_noisy_input = x_noisy.unsqueeze(0)
                
                # Denoise
                x_denoised = model(x_noisy_input)
                
                # Compute metrics
                mse_noisy = torch.mean((x_noisy - x_clean) ** 2).item()
                mse_denoised = torch.mean((x_denoised - x_clean) ** 2).item()
                psnr_noisy = 10 * np.log10(1.0 / mse_noisy) if mse_noisy > 0 else 0
                psnr_denoised = 10 * np.log10(1.0 / mse_denoised) if mse_denoised > 0 else 0
                
                # Display noisy image
                axes[i, 0].imshow(x_noisy.squeeze().cpu(), cmap='gray')
                axes[i, 0].set_title(
                    f'σ={sigma:.2f}\nMSE={mse_noisy:.4f}, PSNR={psnr_noisy:.2f}dB',
                    fontsize=10
                )
                axes[i, 0].axis('off')
                
                # Display denoised image
                axes[i, 1].imshow(x_denoised.squeeze().cpu(), cmap='gray')
                axes[i, 1].set_title(
                    f'Denoised\nMSE={mse_denoised:.4f}, PSNR={psnr_denoised:.2f}dB',
                    fontsize=10,
                    color='green' if mse_denoised < mse_noisy else 'red'
                )
                axes[i, 1].axis('off')
                
                # Display error
                error = torch.abs(x_denoised - x_clean)
                axes[i, 2].imshow(error.squeeze().cpu(), cmap='hot')
                axes[i, 2].set_title(f'Error (max={error.max():.4f})', fontsize=10)
                axes[i, 2].axis('off')
        
        plt.suptitle(f'Denoiser Test - Image {test_idx}', fontsize=14, fontweight='bold')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.tight_layout()
        plt.show()


def plot_training_history(
    history: Dict[str, List[float]],
    save_path: Path,
):
    """
    Plot denoiser training history.
    
    Args:
        history: Dictionary from train_blind_denoiser
        save_path: Path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(history['train_loss'], label='Training Loss', linewidth=2)
    if history['val_loss']:
        ax.plot(history['epoch'], history['val_loss'], 
                label='Validation Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('MSE Loss', fontsize=11)
    ax.set_yscale('log')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Denoiser Training History', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved training history to {save_path / 'training_history.png'}")


def save_results_summary(
    results: Dict,
    save_path: Path,
    filename: str = 'summary.json',
):
    """
    Save results summary to JSON file.
    
    Args:
        results: Results dictionary
        save_path: Directory to save file
        filename: Name of output file
    """
    with open(save_path / filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved results to {save_path / filename}")
