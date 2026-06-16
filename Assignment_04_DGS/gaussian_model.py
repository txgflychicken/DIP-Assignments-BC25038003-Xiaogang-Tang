import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple
from dataclasses import dataclass


def knn_points(p1: torch.Tensor, p2: torch.Tensor, K: int):
    """Simple K-nearest neighbors in pure PyTorch (replaces pytorch3d).

    Args:
        p1: (B, N, 3) query points
        p2: (B, M, 3) reference points
        K:  number of neighbors

    Returns:
        dists: (B, N, K) squared distances to K nearest neighbors
        idx:   (B, N, K) indices of K nearest neighbors
        nn:    (B, N, K, 3) coordinates of K nearest neighbors
    """
    B, N, _ = p1.shape
    M = p2.shape[1]
    K = min(K, M)
    # Pairwise squared distances: (B, N, M)
    dists = torch.cdist(p1, p2, p=2).pow(2)
    # Get top-K smallest distances
    topk = torch.topk(dists, k=K, dim=-1, largest=False)
    # Build nn coords via advanced indexing
    nn = torch.zeros(B, N, K, 3, device=p1.device)
    for b in range(B):
        nn[b] = p2[b][topk.indices[b]]
    return topk.values, topk.indices, nn


@dataclass
class GaussianParameters:
    positions: torch.Tensor   # (N, 3) World space positions
    colors: torch.Tensor      # (N, 3) RGB colors in [0,1]
    opacities: torch.Tensor   # (N, 1) Opacity values in [0,1]
    covariance: torch.Tensor  # (N, 3, 3) Covariance matrices
    rotations: torch.Tensor   # (N, 4) Quaternions
    scales: torch.Tensor      # (N, 3) Log-space scales

class GaussianModel(nn.Module):
    def __init__(self, points3D_xyz: torch.Tensor, points3D_rgb: torch.Tensor):
        """
        Initialize 3D Gaussian Splatting model
        
        Args:
            points3D_xyz: (N, 3) tensor of point positions
            points3D_rgb: (N, 3) tensor of RGB colors in [0, 255]
        """
        super().__init__()
        self.n_points = len(points3D_xyz)
        
        # Initialize learnable parameters
        self._init_positions(points3D_xyz)
        self._init_rotations()
        self._init_scales(points3D_xyz)
        self._init_colors(points3D_rgb)
        self._init_opacities()

    def _init_positions(self, points3D_xyz: torch.Tensor) -> None:
        """Initialize 3D positions from input points"""
        self.positions = nn.Parameter(
            torch.as_tensor(points3D_xyz, dtype=torch.float32)
        )

    def _init_rotations(self) -> None:
        """Initialize rotations as identity quaternions [w,x,y,z]"""
        initial_rotations = torch.zeros((self.n_points, 4))
        initial_rotations[:, 0] = 1.0  # w=1, x=y=z=0 for identity
        self.rotations = nn.Parameter(initial_rotations)

    def _init_scales(self, points3D_xyz: torch.Tensor) -> None:
        """Initialize scales based on local point density"""
        # Compute mean distance to K nearest neighbors
        K = min(50, self.n_points - 1)
        points = points3D_xyz.unsqueeze(0)  # Add batch dimension
        dists, _, _ = knn_points(points, points, K=K)
        
        # Use log space for unconstrained optimization
        mean_dists = torch.mean(torch.sqrt(dists[0]), dim=1, keepdim=True) * 2.
        mean_dists = mean_dists.clamp(0.2*torch.median(mean_dists), 3.0*torch.median(mean_dists))  # Prevent infinite scales
        print('init_scales', torch.min(mean_dists), torch.max(mean_dists))
        
        log_scales = torch.log(mean_dists)
        self.scales = nn.Parameter(log_scales.repeat(1, 3))

    def _init_colors(self, points3D_rgb: torch.Tensor) -> None:
        """Initialize colors in logit space for sigmoid activation"""
        # Convert to [0,1] and apply logit for unconstrained optimization
        colors = torch.as_tensor(points3D_rgb, dtype=torch.float32) / 255.0
        colors = colors.clamp(0.001, 0.999)  # Prevent infinite logits
        self.colors = nn.Parameter(torch.logit(colors))

    def _init_opacities(self) -> None:
        """Initialize opacities in logit space for sigmoid activation"""
        # Initialize to high opacity (sigmoid(8.0) ≈ 0.9997)
        self.opacities = nn.Parameter(
            8.0 * torch.ones((self.n_points, 1), dtype=torch.float32)
        )

    def _compute_rotation_matrices(self) -> torch.Tensor:
        """Convert quaternions to 3x3 rotation matrices"""
        # Normalize quaternions to unit length
        q = F.normalize(self.rotations, dim=-1)
        w, x, y, z = q.unbind(-1)
        
        # Build rotation matrix elements
        R00 = 1 - 2*y*y - 2*z*z
        R01 = 2*x*y - 2*w*z
        R02 = 2*x*z + 2*w*y
        R10 = 2*x*y + 2*w*z
        R11 = 1 - 2*x*x - 2*z*z
        R12 = 2*y*z - 2*w*x
        R20 = 2*x*z - 2*w*y
        R21 = 2*y*z + 2*w*x
        R22 = 1 - 2*x*x - 2*y*y
        
        return torch.stack([
            R00, R01, R02,
            R10, R11, R12,
            R20, R21, R22
        ], dim=-1).reshape(-1, 3, 3)

    def compute_covariance(self) -> torch.Tensor:
        """Compute covariance matrices for all gaussians"""
        # Get rotation matrices
        R = self._compute_rotation_matrices()
        
        # Convert scales from log space and create diagonal matrices
        scales = torch.exp(self.scales)
        S = torch.diag_embed(scales)
        
        # Compute covariance
        # Σ = R S S^T R^T  (paper Eq. 6)
        # S is diagonal, so S = S^T and S S^T = S @ S (squared diagonal elements)
        # Covs3d = R @ S @ S @ R^T
        Covs3d = R @ S @ S @ R.transpose(1, 2)  # (N, 3, 3)

        return Covs3d

    def get_gaussian_params(self) -> GaussianParameters:
        """Get all gaussian parameters in world space"""
        return GaussianParameters(
            positions=self.positions,
            colors=torch.sigmoid(self.colors),
            opacities=torch.sigmoid(self.opacities),
            covariance=self.compute_covariance(),
            rotations=F.normalize(self.rotations, dim=-1),
            scales=torch.exp(self.scales)
        )

    def forward(self) -> Dict[str, torch.Tensor]:
        """Forward pass returns dictionary of parameters"""
        params = self.get_gaussian_params()
        return {
            'positions': params.positions,
            'covariance': params.covariance,
            'colors': params.colors,
            'opacities': params.opacities
        }