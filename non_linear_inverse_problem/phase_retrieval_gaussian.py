"""
Phase Retrieval with Random Gaussian Measurements.

Implementation of the phase retrieval problem using random Gaussian
measurement matrices and the implicit denoiser prior approach.
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Tuple, Dict, Optional
import tqdm


# ============================================================================
# GAUSSIAN MEASUREMENT TASK
# ============================================================================

class GaussianPhaseRetrievalTask:
    """
    Encapsulates Gaussian phase retrieval forward and inverse operators.
    
    Real-domain formulation:
        - Observations: b = |A x*| (magnitude of linear measurements)
        - Constraint set: S_PR = {x : |Ax| matches magnitudes b}
        - Projector: P_S_PR(y) = A^+ (b ⊙ sign(A y))
    """
    
    def __init__(
        self,
        measurement_dim: int,
        image_dim: int,
        seed: int = 42,
        device: str = "cpu"
    ):
        """
        Initialize Gaussian measurement task.
        
        Args:
            measurement_dim: M (number of measurements)
            image_dim: N (image dimension, e.g., 784 for 28x28)
            seed: Random seed for reproducibility
            device: "cpu" or "cuda"
        """
        self.M_shape = (measurement_dim, image_dim)
        self.N = image_dim
        self.M_size = measurement_dim
        self.device = device
        
        # Generate random measurement matrix A ~ N(0, 1/N)
        rng = np.random.RandomState(seed)
        A = rng.randn(measurement_dim, image_dim) / np.sqrt(image_dim)
        self.A = torch.from_numpy(A).float().to(device)
        
        # Compute pseudo-inverse A^+ once
        self.A_pinv = torch.linalg.pinv(self.A)
        
        print(f"Measurement matrix A: {self.A.shape}")
        print(f"Pseudo-inverse A^+: {self.A_pinv.shape}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward operator: A @ x (linear measurement).
        
        Args:
            x: Image [N] or batch [B, N]
        
        Returns:
            Measurements [M] or [B, M]
        """
        if x.dim() == 1:
            return self.A @ x
        else:  # Batch
            return x @ self.A.T
    
    def get_magnitudes(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get observation magnitudes b = |A x|.
        
        Args:
            x: Image [N] or batch [B, N]
        
        Returns:
            Magnitudes [M] or [B, M]
        """
        return torch.abs(self.forward(x))
    
    def get_signs(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get signs of measurements sign(A x).
        
        Args:
            x: Image [N] or batch [B, N]
        
        Returns:
            Signs {-1, +1} [M] or [B, M]
        """
        return torch.sign(self.forward(x))
    
    def project_to_constraint_set(
        self,
        y: torch.Tensor,
        b: torch.Tensor
    ) -> torch.Tensor:
        """
        Project y onto the constraint set S_PR (hard projection).
        
        Implementation: P_S_PR(z) = A^+ (b ⊙ sign(A z))
        
        Args:
            y: Current estimate [N] or [B, N]
            b: Target magnitudes [M] or [B, M]
        
        Returns:
            Projected image [N] or [B, N]
        """
        # Get signs of current estimate's measurements
        signs = self.get_signs(y)
        
        # Element-wise product of magnitudes and signs
        constrained_meas = b * signs
        
        # Project back to image space via pseudo-inverse
        if y.dim() == 1:
            return self.A_pinv @ constrained_meas
        else:  # Batch
            return constrained_meas @ self.A_pinv.T


# ============================================================================
# PHASE RETRIEVAL ALGORITHM
# ============================================================================

def phase_retrieval_with_gaussian_measurements(
    denoiser: nn.Module,
    task: GaussianPhaseRetrievalTask,
    measurements: torch.Tensor,
    config,
    test_ground_truth: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict]:
    """
    Solve phase retrieval using denoiser implicit prior (Gaussian case).
    
    Algorithm:
        1. Initialize y_0 from constraint set with noise
        2. For each iteration:
            a. Compute denoiser residual: d = D(y_{t-1}) - y_{t-1}
            b. Estimate noise variance: σ_t = ||d|| / √N
            c. Adaptive step: y' = y_{t-1} + h_t * d
            d. Project onto constraint: x_proj = P_S_PR(y') with relaxation
            e. Add Langevin noise: y_t = x_proj + γ_t * z_t
    
    Args:
        denoiser: Trained blind denoiser D(y)
        task: GaussianPhaseRetrievalTask instance
        measurements: Target magnitudes b = |A x*| [M]
        config: Configuration object
        test_ground_truth: Ground truth for evaluation (optional) [N]
    
    Returns:
        Tuple of:
            - Final reconstructed image [N]
            - Dictionary with metrics and intermediates
    """
    device = config.device
    denoiser = denoiser.to(device).eval()
    
    N = task.N
    
    # Prepare measurements
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
    # y_0 = P_S_PR(z_0) where z_0 ~ N(0, σ_0^2 I)
    z_raw = torch.randn(N, device=device) * config.sigma_0
    y = task.project_to_constraint_set(z_raw, measurements)
    
    # Reshape to [1, 1, H, W] for denoiser
    y_im = y.view(1, 1, config.image_size, config.image_size)
    
    if test_ground_truth is not None:
        test_ground_truth = test_ground_truth.to(device)
        mse = torch.norm(y - test_ground_truth).item() ** 2 / N
        results['mse_trajectory'].append(mse)
    
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
        sigma_t = torch.norm(d_raw).item() / np.sqrt(N)
        results['sigma_trajectory'].append(sigma_t)
        
        if sigma_t < config.sigma_L:
            break
        
        # Step 3: Adaptive step size
        h_t = config.h0 * t / (1 + config.h0 * (t - 1))
        
        # Step 4: Gradient ascent on manifold
        y_prime = y + h_t * d_raw
        
        # Step 5: Project onto constraint
        y_proj = task.project_to_constraint_set(y_prime, measurements)
        
        # Step 6: Compute Langevin noise amplitude
        gamma_squared = (
            (1 - config.beta * h_t) ** 2 - (1 - h_t) ** 2
        ) * (sigma_t ** 2)
        gamma_t = np.sqrt(max(gamma_squared, 0.0))
        
        # Step 7: Inject noise
        z_t = torch.randn(N, device=device)
        y = y_proj + gamma_t * z_t
        
        # Reshape for next denoiser call
        y_im = y.view(1, 1, config.image_size, config.image_size)
        
        # Track metrics
        if test_ground_truth is not None:
            mse = torch.norm(y - test_ground_truth).item() ** 2 / N
            results['mse_trajectory'].append(mse)
        
        # Save intermediates
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
# HYPERPARAMETER COMPARISON
# ============================================================================

def run_gaussian_phase_retrieval_evaluation(
    denoiser: nn.Module,
    val_dataset,
    config,
    param_ranges: Dict,
):
    """
    Run phase retrieval evaluation over multiple hyperparameter configurations.
    
    Args:
        denoiser: Trained denoiser
        val_dataset: Validation dataset
        config: Configuration object
        param_ranges: Dictionary with parameter lists
            Example: {
                'h0': [0.01, 0.02],
                'beta': [0.01, 0.05],
                ...
            }
    
    Returns:
        Dictionary with results for each configuration
    """
    from itertools import product as iterproduct
    
    # Generate all combinations of parameters
    param_names = list(param_ranges.keys())
    param_lists = list(param_ranges.values())
    configurations = list(iterproduct(*param_lists))
    
    print(f"\nGaussian Phase Retrieval Evaluation")
    print(f"Number of configurations: {len(configurations)}")
    
    results_by_config = {}
    
    for config_idx, params in enumerate(configurations):
        # Create config tuple
        param_dict = dict(zip(param_names, params))
        config_name = f"Config_{config_idx:02d}" + "".join(
            f"_{k}={v:.4f}".replace('0.', '.') for k, v in param_dict.items()
        )
        
        # Update config
        for key, value in param_dict.items():
            setattr(config, key, value)
        
        # Create task
        task = GaussianPhaseRetrievalTask(
            measurement_dim=config.measurement_dim,
            image_dim=config.n_pixels,
            seed=config.seed_measurement,
            device=config.device
        )
        
        # Prepare metrics dictionary
        metrics = {
            'mse': [],
            'measurement_error': [],
            'psnr': [],
            'iterations': [],
            'time': [],
        }
        
        # Evaluate on test set (sampling)
        test_indices = list(range(0, len(val_dataset), 10))
        
        for idx in tqdm.tqdm(test_indices, desc=config_name[-40:], leave=False):
            x_test, _ = val_dataset[idx]
            x_test = x_test.view(-1)
            
            # Generate measurements
            b_test = task.get_magnitudes(x_test.to(config.device))
            
            # Run algorithm
            x_recon, pr_results = phase_retrieval_with_gaussian_measurements(
                denoiser, task, b_test, config, test_ground_truth=None
            )
            
            # Check sign correction
            from utils import evaluate_reconstruction
            
            eval_original = evaluate_reconstruction(
                x_recon.to(config.device), x_test.to(config.device), task.A, b_test
            )
            
            eval_flipped = evaluate_reconstruction(
                (-x_recon).to(config.device), x_test.to(config.device), task.A, b_test
            )
            
            eval_best = eval_flipped if eval_flipped['mse_image'] < eval_original['mse_image'] else eval_original
            
            # Store metrics
            metrics['mse'].append(eval_best['mse_image'])
            metrics['measurement_error'].append(eval_best['measurement_error'])
            metrics['psnr'].append(eval_best['psnr'])
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
