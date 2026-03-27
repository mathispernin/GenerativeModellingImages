"""
Linear Inverse Problems Using Denoiser Implicit Prior.

Implementation of general linear inverse problems (e.g., inpainting)
using the denoiser prior approach from:
"Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
(Kadkhodaie & Simoncelli, 2020)
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, Dict, Optional
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# LINEAR OPERATORS
# ============================================================================

class LinearOperator:
    """Base class for linear operators (measurement matrix)."""
    
    def M(self, x: torch.Tensor) -> torch.Tensor:
        """Forward operator: apply measurement."""
        raise NotImplementedError
    
    def M_T(self, x: torch.Tensor) -> torch.Tensor:
        """Adjoint operator: adjoint of measurement."""
        raise NotImplementedError


class InpaintingOperator(LinearOperator):
    """
    Inpainting operator: zeros out central region of image.
    
    Creates a mask where central pixels are missing (value 0).
    M: applies mask (observation)
    M_T: adjoint (restoration)
    """
    
    def __init__(self, image_shape: Tuple[int, ...], mask_frac: float = 0.25, device: str = "cpu"):
        """
        Initialize inpainting operator.
        
        Args:
            image_shape: Shape of image (C, H, W)
            mask_frac: Fraction of central pixels to mask
            device: "cpu" or "cuda"
        """
        self.image_shape = image_shape
        self.device = device
        n_ch, im_d1, im_d2 = image_shape
        
        # Create mask (1 = observed, 0 = missing)
        self.mask = torch.ones(image_shape, device=device)
        
        # Mask central region
        h_mask = int(im_d1 * np.sqrt(mask_frac) / 2)
        w_mask = int(im_d2 * np.sqrt(mask_frac) / 2)
        c1, c2 = im_d1 // 2, im_d2 // 2
        
        self.mask[:, c1-h_mask:c1+h_mask, c2-w_mask:c2+w_mask] = 0
        
        masked_pixels = (1 - self.mask.mean()).item() * 100
        print(f"Inpainting mask created: {masked_pixels:.1f}% of pixels masked")
    
    def M(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: observe non-masked pixels."""
        return x * self.mask
    
    def M_T(self, x: torch.Tensor) -> torch.Tensor:
        """Adjoint: restore by filling with observations."""
        return x * self.mask


class IdentityOperator(LinearOperator):
    """Identity operator (no measurement loss)."""
    
    def __init__(self, image_shape: Tuple[int, ...], device: str = "cpu"):
        self.image_shape = image_shape
        self.device = device
    
    def M(self, x: torch.Tensor) -> torch.Tensor:
        """Forward: identity."""
        return x
    
    def M_T(self, x: torch.Tensor) -> torch.Tensor:
        """Adjoint: identity."""
        return x


# ============================================================================
# LINEAR INVERSE PROBLEM SOLVER
# ============================================================================

def solve_inverse_problem_with_denoiser(
    denoiser: nn.Module,
    operator: LinearOperator,
    x_observed: torch.Tensor,
    config,
    x_ground_truth: Optional[torch.Tensor] = None,
    sigma_0: float = 1.0,
    sigma_L: float = 0.01,
    h0: float = 0.01,
    beta: float = 0.01,
    verbose: bool = True,
) -> Tuple[torch.Tensor, Dict]:
    """
    Solve linear inverse problems using implicit denoiser prior.
    
    Algorithm from: "Solving Linear Inverse Problems Using the Prior
    Implicit in a Denoiser" (Kadkhodaie & Simoncelli, 2020)
    
    Key formula:
        d_t = (I - M M_T) f(y_{t-1}) + M(x^obs - M_T y_{t-1})
    
    where f(y) = D(y) - y (denoiser residual)
    
    Args:
        denoiser: Trained blind denoiser D(y)
        operator: LinearOperator with M() and M_T() methods
        x_observed: Observed measurements M_T(x*)
        config: Configuration object
        x_ground_truth: Ground truth (optional)
        sigma_0: Initial noise level
        sigma_L: Convergence threshold
        h0: Initial step size
        beta: Langevin noise parameter
        verbose: Print progress
    
    Returns:
        Tuple of:
            - Reconstructed image
            - Dictionary with metrics
    """
    device = config.device
    denoiser = denoiser.to(device).eval()
    
    n_ch, im_d1, im_d2 = x_observed.shape
    N = n_ch * im_d1 * im_d2
    
    x_observed = x_observed.to(device)
    
    results = {
        'iterations': 0,
        'sigma_trajectory': [],
        'mse_trajectory': [],
        'intermediates': [],
        'time_per_iter': [],
    }
    
    # ========== INITIALIZATION ==========
    # Create mask to identify missing/observed regions
    ones = torch.ones_like(x_observed)
    mask = operator.M_T(operator.M(ones))
    missing_mask = ones - mask
    
    # Initialize mean over missing region and observations over observed region
    y_mean = 0.5 * missing_mask + x_observed
    
    # Add Gaussian noise
    z_init = torch.randn_like(x_observed) * sigma_0
    y = y_mean + z_init
    
    # Reshape for denoiser
    y_for_denoiser = y.unsqueeze(0) if y.dim() == 3 else y
    
    if x_ground_truth is not None:
        x_ground_truth = x_ground_truth.to(device)
        mse = torch.norm(y - x_ground_truth).item() ** 2 / N
        results['mse_trajectory'].append(mse)
        if verbose:
            print(f"Initial MSE: {mse:.6f}")
    
    # ========== MAIN LOOP ==========
    t = 1
    time_start = time.time()
    sigma_t = sigma_0
    
    while sigma_t >= sigma_L and t <= config.max_iterations:
        
        # Step 1: Denoise current estimate
        with torch.no_grad():
            y_denoised = denoiser(y_for_denoiser)
        
        if y_denoised.dim() == 4:
            y_denoised = y_denoised.squeeze(0)
        
        # Step 2: Compute descent direction
        # d_t = (I - M M_T) f(y_{t-1}) + M(x^obs - M_T y_{t-1})
        
        # Residual of denoiser: f(y) = D(y) - y
        f_y = y_denoised - y
        
        # First term: Project residual onto unobserved space
        P_f_y = operator.M_T(operator.M(f_y))
        term1 = f_y - P_f_y
        
        # Second term: Constraint violation
        P_y = operator.M_T(operator.M(y))
        term2 = x_observed - P_y
        
        d = term1 + term2
        
        # Step 3: Estimate noise level
        sigma_t = torch.norm(d).item() / np.sqrt(N)
        results['sigma_trajectory'].append(sigma_t)
        
        if sigma_t < sigma_L:
            if verbose:
                print(f"Converged at iteration {t}: σ_t = {sigma_t:.6f} < σ_L = {sigma_L}")
            break
        
        # Step 4: Adaptive step size
        h_t = h0 * t / (1 + h0 * (t - 1))
        
        # Step 5: Gradient ascent step
        y = y + h_t * d
        
        # Step 6: Langevin noise
        gamma_squared = ((1 - beta * h_t) ** 2 - (1 - h_t) ** 2) * (sigma_t ** 2)
        gamma_t = np.sqrt(max(gamma_squared, 0.0))
        
        noise = torch.randn_like(y) * gamma_t
        y = y + noise
        
        # Reshape for next denoiser call
        y_for_denoiser = y.unsqueeze(0) if y.dim() == 3 else y
        
        # Track progress
        if x_ground_truth is not None:
            mse = torch.norm(y - x_ground_truth).item() ** 2 / N
            results['mse_trajectory'].append(mse)
        
        if t % config.save_intermediates_freq == 0:
            results['intermediates'].append(y.cpu().clone())
            elapsed = time.time() - time_start
            results['time_per_iter'].append(elapsed / t)
            
            log_str = f"Iter {t:3d} | σ={sigma_t:.6f} | h={h_t:.6f} | γ={gamma_t:.6f}"
            if x_ground_truth is not None:
                log_str += f" | MSE={mse:.6f}"
            if verbose:
                print(log_str)
        
        t += 1
    
    results['iterations'] = t - 1
    results['total_time'] = time.time() - time_start
    
    if verbose:
        print(f"Finished in {results['iterations']} iterations ({results['total_time']:.2f}s)")
    
    return y.detach().cpu(), results


# ============================================================================
# INPAINTING EVALUATION
# ============================================================================

def evaluate_inpainting(
    denoiser: nn.Module,
    val_dataset,
    config,
    num_test_images: int = 10,
) -> Dict:
    """
    Evaluate inpainting performance on test dataset.
    
    Args:
        denoiser: Trained denoiser
        val_dataset: Validation dataset
        config: Configuration object
        num_test_images: Number of images to test
    
    Returns:
        Dictionary with results
    """
    print("\nEvaluating Inpainting Performance")
    print("=" * 60)
    
    # Create inpainting operator
    inpaint_op = InpaintingOperator(
        image_shape=(1, config.image_size, config.image_size),
        mask_frac=0.15,
        device=config.device
    )
    
    # Configuration for inpainting
    inpaint_config = type('Config', (), {
        'device': config.device,
        'image_size': config.image_size,
        'max_iterations': 500,
        'save_intermediates_freq': 10,
    })()
    
    results_list = []
    
    for idx in range(num_test_images):
        x_true, _ = val_dataset[idx]
        if x_true.shape[0] == 1:
            x_true = x_true.squeeze(0).unsqueeze(0)
        
        # Generate observations
        x_observed = inpaint_op.M_T(x_true.to(config.device))
        
        # Solve
        x_recon, pr_results = solve_inverse_problem_with_denoiser(
            denoiser,
            inpaint_op,
            x_observed,
            inpaint_config,
            x_ground_truth=x_true.squeeze(0),
            sigma_0=1.0,
            sigma_L=0.01,
            h0=0.01,
            beta=0.01,
            verbose=False,
        )
        
        # Evaluate
        mse = torch.norm(x_recon - x_true.cpu().squeeze(0)).item() ** 2 / (config.image_size ** 2)
        max_item = torch.max(torch.abs(x_true.cpu())).item()
        psnr = 10 * np.log10(max_item ** 2 / mse) if mse > 0 else float('inf')
        
        results_list.append({
            'idx': idx,
            'x_true': x_true.squeeze(0),
            'x_observed': x_observed.cpu().squeeze(0),
            'x_recon': x_recon.squeeze(0),
            'mse': mse,
            'psnr': psnr,
            'iterations': pr_results['iterations'],
        })
    
    return {
        'operator': inpaint_op,
        'results': results_list,
        'mean_mse': np.mean([r['mse'] for r in results_list]),
        'mean_psnr': np.mean([r['psnr'] for r in results_list]),
        'mean_iterations': np.mean([r['iterations'] for r in results_list]),
    }


def visualize_inpainting_results(
    results_dict: Dict,
    save_path: Path,
):
    """
    Visualize inpainting reconstruction results.
    
    Args:
        results_dict: Results from evaluate_inpainting
        save_path: Directory to save figures
    """
    results = results_dict['results']
    
    fig, axes = plt.subplots(len(results), 4, figsize=(12, 3 * len(results)))
    
    if len(results) == 1:
        axes = axes.reshape(1, -1)
    
    for i, res in enumerate(results):
        # Original image
        axes[i, 0].imshow(res['x_true'].cpu().numpy(), cmap='gray')
        axes[i, 0].set_title(f"Original #{res['idx']}")
        axes[i, 0].axis('off')
        
        # Masked image
        axes[i, 1].imshow(res['x_observed'].cpu().numpy(), cmap='gray')
        axes[i, 1].set_title("Inpainted (masked)")
        axes[i, 1].axis('off')
        
        # Reconstruction
        axes[i, 2].imshow(res['x_recon'].cpu().numpy(), cmap='gray')
        axes[i, 2].set_title(f"Reconstructed\nPSNR={res['psnr']:.2f}dB")
        axes[i, 2].axis('off')
        
        # Error
        error = torch.abs(res['x_recon'] - res['x_true'].cpu())
        axes[i, 3].imshow(error.numpy(), cmap='hot')
        axes[i, 3].set_title(f"Error\n(max={error.max():.4f})")
        axes[i, 3].axis('off')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved inpainting results to {save_path / 'inpainting_results.png'}")
