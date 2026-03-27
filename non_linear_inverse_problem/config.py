"""
Configuration module for Phase Retrieval experiments.

This module contains all configurable parameters for training and evaluation.
"""

import torch
from pathlib import Path


class Config:
    """Configuration for the Phase Retrieval experiment."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        # Dataset
        self.image_size = 28
        self.n_pixels = 28 * 28  # N = 784
        
        # Measurement Matrix
        self.oversample_ratio = 2.0  # M/N ratio for Gaussian measurements
        self.measurement_dim = int(self.n_pixels * self.oversample_ratio)  # M = 1568
        self.pad_size = int(self.image_size * self.oversample_ratio)  # Padding for Fourier (28 * 2 = 56)
        self.seed_measurement = 42
        
        # BF-CNN Denoiser Architecture
        self.depth = 15  # Number of convolutional layers
        self.n_channels = 64  # Base feature channels
        
        # Denoiser Training
        self.batch_size_train = 128
        self.batch_size_test = 128
        self.num_epochs = 70
        self.learning_rate = 1e-3
        self.sigma_max = 0.4  # Maximum noise level for training
        self.seed_denoiser = 123
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Phase Retrieval Algorithm
        self.sigma_0 = 3.0         # Initial noise level
        self.sigma_L = 0.01        # Final noise level (convergence criterion)
        self.h0 = 0.01             # Initial step size parameter
        self.beta = 0.01           # Noise injection parameter [0, 1]
        self.max_iterations = 500  # Maximum iterations
        
        # Output
        self.save_dir = Path("./outputs")
        self.save_intermediates_freq = 10

    @classmethod
    def get_fourier_config(cls):
        """Return a config optimized for Fourier phase retrieval."""
        config = cls()
        config.sigma_0 = 1.0
        config.h0 = 0.01
        config.beta = 0.01
        config.max_iterations = 500
        return config
    
    @classmethod
    def get_gaussian_config(cls):
        """Return a config optimized for Gaussian phase retrieval."""
        config = cls()
        config.sigma_0 = 3.0
        config.h0 = 0.01
        config.beta = 0.01
        config.max_iterations = 500
        return config


def create_output_directory(config: Config) -> Path:
    """Create output directory structure."""
    config.save_dir.mkdir(parents=True, exist_ok=True)
    return config.save_dir
