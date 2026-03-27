"""
Phase Retrieval with Fourier Measurements.

Implementation of the phase retrieval problem using Fourier magnitude
measurements and the implicit denoiser prior approach.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from typing import Tuple, Dict, Optional
import tqdm


# ============================================================================
# FOURIER MEASUREMENT TASK
# ============================================================================

class FourierPhaseRetrievalTask:
    """
    Encapsulates Fourier phase retrieval forward and inverse operators.
    
    Complex-domain formulation with zero-padding:
        - Observations: b = |F(P(x*))| (magnitude of zero-padded FFT)
        - Constraint set: S_PR = {x : |F(P(x))| matches b}
        - Projector: P_S_PR(y) = Crop(Adj(IFFT(b ⊙ Phase(FFT(Pad(y))))))
    """
    
    def __init__(
        self,
        image_size: int,
        oversample_ratio: int = 2,
        device: str = "cpu"
    ):
        """
        Initialize Fourier measurement task.
        
        Args:
            image_size: Image height/width (assuming square, e.g., 28)
            oversample_ratio: Factor to pad the image (e.g., 2 → 56x56)
            device: "cpu" or "cuda"
        """
        self.image_size = image_size
        self.N = image_size * image_size
        self.pad_size = int(image_size * oversample_ratio)
        self.M_size = self.pad_size * self.pad_size
        self.device = device
        
        # Calculate padding to center the image
        pad_total = self.pad_size - image_size
        self.pad_l = pad_total // 2
        self.pad_r = pad_total - self.pad_l
        self.padding = (self.pad_l, self.pad_r, self.pad_l, self.pad_r)
        
        print(f"Fourier Task: Image {image_size}x{image_size}, Padded to {self.pad_size}x{self.pad_size}")
        print(f"Measurements M: {self.M_size}, Image N: {self.N}")

    def _format_input(self, x: torch.Tensor):
        """Convert input to 4D format [B, C, H, W] for FFT operations."""
        is_flat = x.dim() == 1
        if is_flat:
            x_fmt = x.view(1, 1, self.image_size, self.image_size)
        elif x.dim() == 2 and x.size(1) == self.N:
            x_fmt = x.view(-1, 1, self.image_size, self.image_size)
        else:
            x_fmt = x
        return x_fmt, is_flat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward operator: F(P(x)) (zero-padded FFT).
        
        Args:
            x: Image [N] or batch [B, N]
        
        Returns:
            Fourier coefficients (complex)
        """
        x_fmt, is_flat = self._format_input(x)
        
        # Zero-pad and compute FFT
        x_padded = F.pad(x_fmt, self.padding)
        X_freq = torch.fft.fft2(x_padded)
        
        if is_flat:
            return X_freq.view(-1)
        elif x.dim() == 2 and x.size(1) == self.N:
            return X_freq.view(x.size(0), -1)
        return X_freq

    def get_magnitudes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get observation magnitudes b = |F(P(x))|.
        
        Args:
            x: Image [N] or batch [B, N]
        
        Returns:
            Magnitudes [M] or [B, M]
        """
        return torch.abs(self.forward(x))

    def get_phases(self, X_complex: torch.Tensor) -> torch.Tensor:
        """Extract phase information X / |X|."""
        mags = torch.abs(X_complex)
        mags = torch.clamp(mags, min=1e-8)
        return X_complex / mags

    def project_to_constraint_set(
        self,
        y: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """
        Project y onto the constraint set in Fourier domain.
        
        Formula: P_S_PR(z) = Crop(Real(IFFT(b ⊙ Phase(FFT(Pad(z))))))
        
        Args:
            y: Current estimate [N] or [B, N]
            b: Target magnitudes [M] or [B, M]
        
        Returns:
            Projected image [N] or [B, N]
        """
        y_fmt, is_flat = self._format_input(y)
        
        # Forward to Fourier domain
        y_padded = F.pad(y_fmt, self.padding)
        Y_freq = torch.fft.fft2(y_padded)
        
        # Extract and impose magnitude
        phases = self.get_phases(Y_freq)
        
        # Reshape magnitude to match Fourier domain
        if b.dim() == 1:
            b_fmt = b.view(1, 1, self.pad_size, self.pad_size)
        elif b.dim() == 4:
            b_fmt = b
        else:
            b_fmt = b.view_as(phases.real)
        
        constrained_freq = b_fmt * phases
        
        # Inverse Fourier transform
        y_proj_padded = torch.fft.ifft2(constrained_freq)
        y_proj_real = torch.real(y_proj_padded)
        
        # Crop padding
        y_proj = y_proj_real[
            ...,
            self.pad_l : self.pad_l + self.image_size,
            self.pad_l : self.pad_l + self.image_size
        ]
        
        if is_flat:
            return y_proj.reshape(-1)
        elif y.dim() == 2 and y.size(1) == self.N:
            return y_proj.reshape(y.size(0), -1)
        return y_proj


# ============================================================================
# FOURIER PHASE RETRIEVAL ALGORITHM
# ============================================================================

def phase_retrieval_with_fourier_measurements(
    denoiser: nn.Module,
    task: FourierPhaseRetrievalTask,
    measurements: torch.Tensor,
    config,
    test_ground_truth: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Solve phase retrieval using denoiser implicit prior (Fourier case).
    
    Algorithm:
        1. Initialize with weighted phase reconstruction
        2. For each iteration:
            a. Compute denoiser residual: d = D(y_{t-1}) - y_{t-1}
            b. Estimate noise variance: σ_t = ||d|| / √N
            c. Adaptive step: y' = y_{t-1} + h_t * d
            d. Project onto constraint: x_proj = P_S_PR(y') with relaxation
            e. Add Langevin noise: y_t = x_proj + γ_t * z_t
    
    Args:
        denoiser: Trained blind denoiser
        task: FourierPhaseRetrievalTask instance
        measurements: Target magnitudes [M]
        config: Configuration object
        test_ground_truth: Ground truth for evaluation (optional)
    
    Returns:
        Tuple of:
            - Final reconstructed image [N]
            - Dictionary with metrics and intermediates
    """
    device = config.device
    denoiser = denoiser.to(device).eval()
    
    N = task.N
    
    if measurements.dim() > 1:
        measurements = measurements.squeeze()
    measurements = measurements.to(device)
    
    results = {
        'iterations': 0,
        'sigma_trajectory': [],
        'time_per_iter': [],
        'mse_trajectory': [],
        'intermediates': [],
    }
    
    # ========== INITIALIZATION ==========
    with torch.no_grad():
        # Initialize with random phases in Fourier domain
        random_phases = torch.exp(
            1j * torch.rand(1, 1, task.pad_size, task.pad_size, device=device) * 2 * np.pi
        )
        initial_freq = measurements.view(1, 1, task.pad_size, task.pad_size) * random_phases
        
        # Transform back to image space
        initial_padded = torch.fft.ifft2(initial_freq)
        initial_real = torch.real(initial_padded)
        
        # Crop to image size
        z_raw = initial_real[
            :, :,
            task.pad_l:task.pad_l + task.image_size,
            task.pad_l:task.pad_l + task.image_size
        ].reshape(-1)
    
    # Add some noise for exploration
    z_raw = z_raw + torch.randn(N, device=device) * config.sigma_0 * 0.5
    
    # Project to constraint set
    y = task.project_to_constraint_set(z_raw, measurements)
    y_im = y.view(1, 1, config.image_size, config.image_size)
    
    if test_ground_truth is not None:
        test_ground_truth = test_ground_truth.to(device)
    
    # ========== MAIN LOOP ==========
    t = 1
    time_start = time.time()
    sigma_t = config.sigma_0
    
    while sigma_t >= config.sigma_L and t <= config.max_iterations:
        
        # Step 1: Compute denoiser residual
        with torch.no_grad():
            y_hat = denoiser(y_im).view(-1)
        
        y_hat_proj = task.project_to_constraint_set(y_hat, measurements)
        d_raw = y_hat_proj - y  # Residual in image space
        
        # Step 2: Estimate noise variance
        sigma_t = torch.norm(d_raw).item() / (N ** 0.5)
        results['sigma_trajectory'].append(sigma_t)
        
        if sigma_t < config.sigma_L:
            break
        
        # Step 3: Adaptive step size
        h_t = config.h0 * t / (1 + config.h0 * (t - 1))
        
        # Step 4: Gradient ascent
        y_prime = y + h_t * d_raw
        
        # Step 5: Project onto constraint
        y_proj = task.project_to_constraint_set(y_prime, measurements)
        
        # Step 6: Langevin noise
        gamma_squared = ((1 - config.beta * h_t) ** 2 - (1 - h_t) ** 2) * (sigma_t ** 2)
        gamma_t = np.sqrt(max(gamma_squared, 0.0))
        
        # Step 7: Add noise
        z_t = torch.randn(N, device=device)
        y = y_proj + gamma_t * z_t
        y_im = y.view(1, 1, config.image_size, config.image_size)
        
        # Track intermediates
        if t % config.save_intermediates_freq == 0:
            results['intermediates'].append(y.detach().cpu().clone())
            elapsed = time.time() - time_start
            results['time_per_iter'].append(elapsed / t)
        
        t += 1
    
    results['iterations'] = t - 1
    results['total_time'] = time.time() - time_start
    
    # Final denoising pass
    with torch.no_grad():
        #y_final = y_im.view(-1)
        y_final = denoiser(y_im).view(-1)
    
    return y_final.cpu(), results


# ============================================================================
# FOURIER AMBIGUITY RESOLUTION
# ============================================================================

def align_fourier_reconstruction(
    x_recon: torch.Tensor,
    x_test: torch.Tensor,
    image_size: int
) -> Tuple[torch.Tensor, float]:
    """
    Resolve Fourier phase retrieval ambiguities (shifts, flips, sign).
    
    Fourier measurements are invariant under:
    - Circular shifts
    - Spatial reflection (180° rotation)
    - Global sign flip
    
    This function searches for the best alignment by testing all combinations.
    
    Args:
        x_recon: Reconstructed image [N]
        x_test: Ground truth image [N]
        image_size: Image height/width
    
    Returns:
        Tuple of:
            - Best aligned reconstruction
            - Minimum MSE achieved by alignment
    """
    x_r = x_recon.view(image_size, image_size)
    x_t = x_test.view(image_size, image_size)
    
    best_mse = float('inf')
    best_aligned = x_recon.clone()
    
    # Test spatial inversion (flip)
    for flip in [False, True]:
        curr_x = torch.flip(x_r, dims=[0, 1]) if flip else x_r.clone()
        
        # Test sign
        for sign in [1, -1]:
            curr_x_signed = curr_x * sign
            
            # Test all shifts
            for dy in range(image_size):
                for dx in range(image_size):
                    shifted = torch.roll(curr_x_signed, shifts=(dy, dx), dims=(0, 1))
                    mse = torch.mean((shifted - x_t) ** 2).item()
                    
                    if mse < best_mse:
                        best_mse = mse
                        best_aligned = shifted.view(-1)
    
    return best_aligned, best_mse


# ============================================================================
# HYPERPARAMETER SWEEP
# ============================================================================

def run_fourier_phase_retrieval_evaluation(
    denoiser: nn.Module,
    val_dataset,
    config,
    param_ranges: Dict,
):
    """
    Run Fourier phase retrieval evaluation over multiple configurations.
    
    Args:
        denoiser: Trained denoiser
        val_dataset: Validation dataset
        config: Configuration object
        param_ranges: Dictionary with parameter lists
    
    Returns:
        Dictionary with results for each configuration
    """
    from itertools import product as iterproduct
    
    # Generate all combinations
    param_names = list(param_ranges.keys())
    param_lists = list(param_ranges.values())
    configurations = list(iterproduct(*param_lists))
    
    print(f"\nFourier Phase Retrieval Evaluation")
    print(f"Number of configurations: {len(configurations)}")
    
    results_by_config = {}
    
    # Create task once
    task = FourierPhaseRetrievalTask(
        image_size=config.image_size,
        oversample_ratio=config.oversample_ratio,
        device=config.device
    )
    
    for config_idx, params in enumerate(configurations):
        param_dict = dict(zip(param_names, params))
        config_name = f"Config_{config_idx:02d}" + "".join(
            f"_{k}={v:.4f}".replace('0.', '.') for k, v in param_dict.items()
        )
        
        # Update config
        for key, value in param_dict.items():
            setattr(config, key, value)
        
        # Prepare metrics
        metrics = {
            'mse': [],
            'measurement_error': [],
            'psnr': [],
            'iterations': [],
            'time': [],
        }
        
        # Evaluate on test set
        test_indices = list(range(0, len(val_dataset), 10))
        
        for idx in tqdm.tqdm(test_indices, desc=config_name[-40:], leave=False):
            x_test, _ = val_dataset[idx]
            x_test = x_test.view(-1).to(config.device)
            
            # Generate Fourier measurements
            b_test = task.get_magnitudes(x_test)
            
            # Run algorithm
            x_recon, pr_results = phase_retrieval_with_fourier_measurements(
                denoiser, task, b_test, config, test_ground_truth=None
            )
            
            # Resolve ambiguities
            x_aligned, _ = align_fourier_reconstruction(
                x_recon.to(config.device),
                x_test,
                config.image_size
            )
            
            # Evaluate
            from utils import evaluate_reconstruction
            eval_result = evaluate_reconstruction(
                x_aligned.to(config.device),
                x_test,
                task,
                b_test
            )
            
            # Store metrics
            metrics['mse'].append(eval_result['mse_image'])
            metrics['measurement_error'].append(eval_result['measurement_error'])
            metrics['psnr'].append(eval_result['psnr'])
            metrics['iterations'].append(pr_results['iterations'])
            metrics['time'].append(pr_results.get('total_time', 0))
        
        # Compute statistics
        results_by_config[config_name] = {
            'params': param_dict,
            'metrics': metrics,
            'mean_mse': np.mean(metrics['mse']),
            'std_mse': np.std(metrics['mse']),
            'mean_psnr': np.mean(metrics['psnr']),
            'std_psnr': np.std(metrics['psnr']),
            'mean_iterations': np.mean(metrics['iterations']),
            'total_time': np.sum(metrics['time']),
        }
        
        print(f"[{config_idx+1}/{len(configurations)}] PSNR: {results_by_config[config_name]['mean_psnr']:.4f}")
    
    return results_by_config
