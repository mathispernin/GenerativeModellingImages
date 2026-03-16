"""
This file contains code for low-dimensional visualisation experiments.
"""

##########################################################################
# Imports and setup
##########################################################################

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from scipy.stats import multivariate_normal, norm

##########################################################################
# Utility functions
##########################################################################

def prior_density(x, weights, means, covs):
    """Compute p(x) for a Gaussian mixture."""
    p = 0
    for w, mu, cov in zip(weights, means, covs):
        p += w * multivariate_normal.pdf(x, mean=mu, cov=cov)
    return p

def noisy_density(y, sigma, weights, means, covariances):
    p = 0
    for w, mu, cov in zip(weights, means, covariances):
        p += w * multivariate_normal.pdf(y, mean=mu, cov=cov + sigma**2 * np.eye(2))
    return p

def mmse_denoiser(y, sigma, weights, means, covs):
    """Compute E[x|y] for Gaussian mixture prior."""
    sigma2I = sigma**2 * np.eye(2)
    posterior_weights = []
    posterior_means = []
    
    for w, mu, cov in zip(weights, means, covs):
        cov_y = cov + sigma2I
        pw = w * multivariate_normal.pdf(y, mean=mu, cov=cov_y)
        posterior_weights.append(pw)
        # E[x|y, component k] = μ_k + Σ_k (Σ_k + σ²I)^{-1} (y - μ_k)
        inv_cov_y = np.linalg.inv(cov_y)
        cond_mean = mu + cov @ inv_cov_y @ (y - mu)
        posterior_means.append(cond_mean)
    
    total_w = sum(posterior_weights)
    posterior_weights = [pw / (total_w + 1e-300) for pw in posterior_weights]
    
    xhat = np.zeros(2)
    for pw, cm in zip(posterior_weights, posterior_means):
        xhat += pw * cm
    return xhat


def grad_log_noisy_density(y, sigma, weights, means, covs):
    """Compute ∇_y log p(y) analytically."""
    sigma2I = sigma**2 * np.eye(2)
    p_y = 0
    grad_p_y = np.zeros(2)
    for w, mu, cov in zip(weights, means, covs):
        cov_y = cov + sigma2I
        inv_cov_y = np.linalg.inv(cov_y)
        pdf_val = w * multivariate_normal.pdf(y, mean=mu, cov=cov_y)
        p_y += pdf_val
        grad_p_y += pdf_val * (-inv_cov_y @ (y - mu))
    return grad_p_y / (p_y + 1e-300)


##########################################################################
# Experiment 1: Visualizing Miyasawa's identity in 2D
##########################################################################

"""
Miyasawa's identity : x̂(y) = y + σ² ∇_y log p(y)

We use a 2D Gaussian mixture as the prior p(x), for which:
- p(y) = p(x) * g_σ  is known analytically (still a Gaussian mixture)
- The MMSE denoiser x̂(y) = E[x|y] can be computed in closed form
- ∇_y log p(y) can be computed analytically

We verify that the denoiser residual f(y) = x̂(y) - y equals σ² ∇_y log p(y).
We also train a small MLP denoiser and show it learns this relationship.
"""

# ============================================================
# 1. Define the 2D Gaussian Mixture Prior p(x)
# ============================================================

def log_prior(x, means, covs, weights):
    """Compute log p(x) for a Gaussian mixture."""
    p = 0
    for w, mu, cov in zip(weights, means, covs):
        p += w * multivariate_normal.pdf(x, mean=mu, cov=cov)
    return np.log(p + 1e-300)

# ============================================================
# 2. Compute p(y) = p(x) * g_σ analytically
# ============================================================
# If p(x) = Σ w_k N(μ_k, Σ_k), then p(y) = Σ w_k N(μ_k, Σ_k + σ²I)

# ============================================================
# 3. Compute MMSE denoiser x̂(y) = E[x|y] analytically
# ============================================================
# For Gaussian mixture: E[x|y] = Σ w_k(y) * (Σ_k(Σ_k+σ²I)^{-1}(y-μ_k) + μ_k)
# where w_k(y) ∝ w_k * N(y; μ_k, Σ_k+σ²I)

# ============================================================
# 4. Train a small MLP denoiser and verify it also learns this
# ============================================================

class MLPDenoiser(nn.Module):
    """Small MLP that takes noisy y and outputs x̂(y)."""
    def __init__(self, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 2),
        )
    def forward(self, y):
        return self.net(y)

def sample_from_mixture(n, means, covs, weights):
    """Sample from Gaussian mixture."""
    components = np.random.choice(len(weights), size=n, p=weights)
    samples = np.zeros((n, 2))
    for i, k in enumerate(components):
        samples[i] = np.random.multivariate_normal(means[k], covs[k])
    return samples

###########################################################
# Experiment 2: Coarse-to-fine stochastic ascent to find modes of p(x)
###########################################################

"""
Experiment 2: Coarse-to-fine stochastic ascent in 2D
===================================================================
Implements Algorithm 1 from the paper in 2D.
Compares with annealed Langevin dynamics (Song & Ermon, NeurIPS 2019).
"""

# ============================================================
# 1. Define prior p(x) — mixture of Gaussians
# ============================================================

def sample_prior(n, weights, means, covariances):
    comps = np.random.choice(len(weights), size=n, p=weights)
    return np.array([np.random.multivariate_normal(means[k], covariances[k]) for k in comps])

# ============================================================
# 2. Algorithm 1: Coarse-to-fine stochastic ascent
# ============================================================

def algorithm1_sample(sigma_0=3.0, sigma_L=0.001, h0=0.05, beta=0.5, max_iter=1000, weights=None, means=None, covariances=None):
    """
    Algorithm 1 from the paper, adapted for 2D.
    Returns trajectory and sigma history.
    """
    N = 2  # dimension
    y = np.random.randn(2) * sigma_0 + 0.5  # initialize ~ N(0.5, σ_0² I)
    
    trajectory = [y.copy()]
    sigma_history = [sigma_0]
    h_history = []
    
    t = 1
    sigma_t = sigma_0
    
    while sigma_t > sigma_L and t < max_iter:
        h_t = h0 * t / (1 + h0 * (t - 1))
        
        # Denoiser residual: f(y) = x̂(y) - y  (proportional to ∇ log p(y))
        xhat = mmse_denoiser(y, sigma_t, weights, means, covariances)
        d_t = xhat - y  # this is f(y)
        
        # Estimate sigma from residual magnitude
        sigma_t = np.linalg.norm(d_t) / np.sqrt(N)
        
        # Compute noise injection amplitude (Eq. 8)
        gamma2 = ((1 - beta * h_t)**2 - (1 - h_t)**2) * sigma_t**2
        gamma_t = np.sqrt(max(gamma2, 0))
        
        # Update (Eq. 5)
        z_t = np.random.randn(2)
        y = y + h_t * d_t + gamma_t * z_t
        
        trajectory.append(y.copy())
        sigma_history.append(sigma_t)
        h_history.append(h_t)
        t += 1
    
    return np.array(trajectory), np.array(sigma_history), np.array(h_history)

# ============================================================
# 3. Annealed Langevin Dynamics (Song & Ermon) for comparison
# ============================================================

def score_function(y, sigma, weights, means, covariances):
    """∇_y log p(y) for Gaussian mixture."""
    sigma2I = sigma**2 * np.eye(2)
    p_y = 0
    grad_p_y = np.zeros(2)
    for w, mu, cov in zip(weights, means, covariances):
        cov_y = cov + sigma2I
        inv_cov_y = np.linalg.inv(cov_y)
        pdf_val = w * multivariate_normal.pdf(y, mean=mu, cov=cov_y)
        p_y += pdf_val
        grad_p_y += pdf_val * (-inv_cov_y @ (y - mu))
    return grad_p_y / (p_y + 1e-300)

def annealed_langevin(sigmas, epsilon=0.01, T_per_level=100, weights=None, means=None, covariances=None):
    """
    Annealed Langevin dynamics (Song & Ermon, 2019).
    Uses a discrete set of noise levels with T steps per level.
    """
    y = np.random.randn(2) * sigmas[0]
    trajectory = [y.copy()]
    sigma_history = [sigmas[0]]
    
    for sigma in sigmas:
        alpha = epsilon * (sigma / sigmas[-1])**2  # step size
        for _ in range(T_per_level):
            score = score_function(y, sigma, weights, means, covariances)
            z = np.random.randn(2)
            y = y + alpha * score + np.sqrt(2 * alpha) * z
            trajectory.append(y.copy())
            sigma_history.append(sigma)
    
    return np.array(trajectory), np.array(sigma_history)

##############################################################
# Experiment 3: Constrained coarse-to-fine ascent (Algorithm 2)
##############################################################

"""
Experiment 3: Linear Inverse Problem — Constrained Sampling (Algorithm 2) in 2D
================================================================================
We observe only one coordinate of a 2D point: x_c = M^T x = x_1
The algorithm must reconstruct x_2 using the prior.

σ² ∇_y log p(y|x_c) = (I - MM^T) f(y) + M(x_c - M^T y)

This decomposes the gradient into:
  - Prior gradient projected to the unconstrained subspace
  - Data fidelity term in the measurement subspace
"""

# ============================================================
# 1. Algorithm 2: Constrained coarse-to-fine ascent
# ============================================================

def algorithm2_sample(x_c, sigma_0=3.0, sigma_L=0.01, h0=0.05, beta=0.3, max_iter=500, weights=None, means=None, covariances=None, M=np.array([[1], [0]])):
    """
    Algorithm 2 from the paper in 2D.
    x_c: scalar measurement (x_1 coordinate of the true image)
    """
    N = 2
    M_T = M.T
    I_minus_MMT = np.eye(N) - M @ M_T

    # Initialize: measured component = x_c + noise, unobserved = 0.5 + noise
    e = np.ones(2)
    y_init = I_minus_MMT @ e * 0.5 + M.flatten() * x_c + sigma_0 * np.random.randn(2)
    y = y_init.copy()
    
    trajectory = [y.copy()]
    sigma_history = [sigma_0]
    grad_prior_hist = []
    grad_constraint_hist = []
    
    t = 1
    sigma_t = sigma_0
    
    while sigma_t > sigma_L and t < max_iter:
        h_t = h0 * t / (1 + h0 * (t - 1))
        
        xhat = mmse_denoiser(y, sigma_t, weights, means, covariances)
        f_y = xhat - y
        
        # Gradient decomposition
        grad_prior = I_minus_MMT @ f_y          # prior gradient in unconstrained subspace
        grad_constraint = M.flatten() * (x_c - M_T @ y)  # data fidelity in measurement subspace
        d_t = grad_prior + grad_constraint
        
        sigma_t = np.linalg.norm(d_t) / np.sqrt(N)
        gamma2 = ((1 - beta * h_t)**2 - (1 - h_t)**2) * sigma_t**2
        gamma_t = np.sqrt(max(gamma2, 0))
        
        z_t = np.random.randn(2)
        y = y + h_t * d_t + gamma_t * z_t
        
        trajectory.append(y.copy())
        sigma_history.append(sigma_t)
        grad_prior_hist.append(grad_prior.copy())
        grad_constraint_hist.append(grad_constraint.copy())
        t += 1
    
    return (np.array(trajectory), np.array(sigma_history), 
            np.array(grad_prior_hist), np.array(grad_constraint_hist))

################################################################
# Experiment 4: Visualization of the Gaussian Scale-Space and Forward Diffusion Process
################################################################

"""
Experiment 4: Gaussian Scale-Space & Forward Diffusion Process
==============================================================
The family of observation densities p_σ(y) forms a Gaussian scale-space representation of the prior

This is alike the forward process of DDPM/Score-based diffusion models:
    q(x_t | x_0) = N(x_t; √ᾱ_t x_0, (1-ᾱ_t)I)

We visualize the evolution of p_σ(y) as σ increases (= diffusion forward)
"""

# ============================================================
# 1. 1D Scale-Space Visualization
# ============================================================

def p_1d(x, weights_1d, mus_1d, sigmas_1d):
    return sum(w * norm.pdf(x, mu, s) for w, mu, s in zip(weights_1d, mus_1d, sigmas_1d))

def p_sigma_1d(y, sigma, weights_1d, mus_1d, sigmas_1d):
    """p_σ(y) = p(x) * g_σ — Gaussian-blurred version of p(x)."""
    return sum(w * norm.pdf(y, mu, np.sqrt(s**2 + sigma**2)) 
               for w, mu, s in zip(weights_1d, mus_1d, sigmas_1d))

# DDPM forward process marginal: q(x_t) for a schedule
def ddpm_marginal_1d(y, alpha_bar, weights_1d, mus_1d, sigmas_1d):
    """q(x_t | x_0) = N(√ᾱ x_0, (1-ᾱ)I) → marginalize over p(x_0)."""
    equiv_sigma = np.sqrt((1 - alpha_bar) / alpha_bar)  # effective noise level
    return sum(w * norm.pdf(y, np.sqrt(alpha_bar) * mu, 
               np.sqrt(alpha_bar * s**2 + (1 - alpha_bar)))
               for w, mu, s in zip(weights_1d, mus_1d, sigmas_1d))

# ============================================================
# 2. 2D Circle Scale-Space
# ============================================================

def sample_circle(n, radius=2.0, noise_std=0.05):
    """Sample from a noisy circle manifold."""
    theta = np.random.uniform(0, 2*np.pi, n)
    x = radius * np.cos(theta) + noise_std * np.random.randn(n)
    y = radius * np.sin(theta) + noise_std * np.random.randn(n)
    return np.stack([x, y], axis=1)

"""
Experiment 5A: Visualizing the backward evolution of the noisy scale-space p_sigma(y)
====================================================================================

We compare two reverse processes in 2D:

(1) Coarse-to-fine ascent (Algorithm 1):
    y_{t+1} = y_t + h_t * f(y_t) + gamma_t z_t
    where f(y) = xhat(y) - y, and xhat is the MMSE denoiser.

(2) Score-based reverse SDE, Euler discretization:
    dy = [-sigma^2 * score(y, sigma)] d(log sigma)  +  sigma * sqrt(2) dW
A practical discretization:
    y_{k+1} = y_k + a_k * score(y_k, sigma_k) + b_k * eps

We use an analytic Gaussian mixture prior, so:
- p_sigma(y) is analytic (Gaussian mixture with inflated covariance)
- score(y, sigma) = ∇_y log p_sigma(y) is analytic
- MMSE denoiser xhat(y) is analytic
and Miyasawa identity holds: xhat(y) - y = sigma^2 * score(y, sigma)
"""

# ---------------------------
# Prior: 2D Gaussian mixture
# ---------------------------

# ---------------------------
# Process 1: Paper-like reverse
# ---------------------------

def reverse_paper_like(n_particles=100, sigma_0=3.0, sigma_L=0.05, h0=0.05, beta=0.4, weights=None, means=None, covs=None, seed=0):
    """
    We run Algorithm-1-like updates.
    """
    rng = np.random.default_rng(seed)
    
    y = rng.normal(loc=0.5, scale=sigma_0, size=(n_particles, 2))
    sigmas_t = np.full(n_particles, sigma_0)
    
    target_snapshots = np.geomspace(sigma_0, sigma_L, 5) # 5 photos
    snapshots = []
    current_snap_idx = 0

    t = 1
    while np.median(sigmas_t) > sigma_L and t < 500:
        
        if current_snap_idx < len(target_snapshots) and np.median(sigmas_t) <= target_snapshots[current_snap_idx]:
            snapshots.append((np.median(sigmas_t), y.copy()))
            current_snap_idx += 1

        h_t = h0 * t / (1 + h0 * (t - 1))
        
        # 1. Denoiser residual f(y) = x̂(y) - y for each particle
        xhat = np.array([mmse_denoiser(y[i], sigmas_t[i], weights, means, covs) for i in range(n_particles)])
        d_t = xhat - y # C'est le résidu f(y)
        
        # 2. Update sigma estimate
        sigmas_t = np.linalg.norm(d_t, axis=1) / np.sqrt(2)
        
        # 3. Compute noise injection amplitude for each particle
        gamma2 = ((1 - beta * h_t)**2 - (1 - h_t)**2) * (sigmas_t**2)
        gamma_t = np.sqrt(np.maximum(gamma2, 0.0))
        
        # 4. Noise injection
        z_t = rng.normal(size=y.shape)
        y = y + h_t * d_t + (gamma_t[:, None] * z_t)
        
        t += 1

    snapshots.append((np.median(sigmas_t), y.copy()))
    
    return snapshots

# ---------------------------
# Process 2: Score-based reverse
# ---------------------------
def reverse_score_sde(n_particles=2000, sigma_0=3.0, sigma_L=0.05, weights=None, means=None, covs=None, n_levels=12, n_inner=50, seed=0):
    """
    We run a simple Euler discretization of the reverse SDE:
    dy = [-sigma^2 * score(y, sigma)] d(log sigma) + sigma * sqrt(2) dW
    Discretized as:
    y_{k+1} = y_k + a_k * score(y_k, sigma_k) + b_k * eps
    where a_k = -sigma_k^2 * (log sigma_{k+1} - log sigma_k), b_k = sigma_k * sqrt(2 * |log sigma_{k+1} - log sigma_k|)
    """
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=0.0, scale=sigma_0, size=(n_particles, 2))

    sigmas = np.geomspace(sigma_0, sigma_L, n_levels)
    snapshots = []

    for k in tqdm.tqdm(range(len(sigmas)-1), desc="Score-based reverse SDE"):
        s = sigmas[k]
        s_next = sigmas[k+1]
        # step in log-sigma
        dlog = np.log(s_next) - np.log(s)  # negative
        # choose coefficients (heuristic but standard-ish)
        # drift magnitude proportional to s^2 * score, scaled by |dlog|
        a = - (s**2) * dlog
        # diffusion magnitude proportional to s * sqrt(|dlog|)
        b = s * np.sqrt(2 * abs(dlog))

        for _ in range(n_inner):
            sc = np.array([grad_log_noisy_density(yi, s, weights, means, covs) for yi in y])
            y = y + a * sc + b * rng.normal(size=y.shape)

        snapshots.append((s_next, y.copy()))
    # include initial too
    snapshots = [(sigmas[0], rng.normal(loc=0.0, scale=sigma_0, size=(n_particles, 2)))] + snapshots
    return snapshots

# ---------------------------
# Visualization helper
# ---------------------------

def plot_snapshots(snapshots, title, filename, weights=None, means=None, covs=None, grid_lim=5.0):
    n = len(snapshots)
    cols = 5
    rows = 1
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = np.array(axes).reshape(-1)

    # density grid for contours
    fine = np.linspace(-grid_lim, grid_lim, 200)
    X, Y = np.meshgrid(fine, fine)
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)

    count = 0
    for i, (sigma, y) in tqdm.tqdm(enumerate(snapshots), desc="Plotting snapshots", total=len(snapshots)):
        if i in [0, 2, 4, 6, 11]:
            ax = axes[count]
            count += 1
            Z = np.array([noisy_density(p, sigma, weights, means, covs) for p in pts]).reshape(200, 200)
            ax.contourf(X, Y, Z, levels=25, cmap='Blues', alpha=0.65)
            ax.scatter(y[:,0], y[:,1], s=1, c='black', alpha=0.7)
            ax.set_title(rf"$\sigma \approx {sigma:.3f}$", fontsize=12)
            ax.set_xlim(-grid_lim, grid_lim); ax.set_ylim(-grid_lim, grid_lim)
            ax.set_aspect('equal')
            ax.set_xticks([]); ax.set_yticks([])

    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()
