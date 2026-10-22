import numpy as np
from typing import Dict, Any

def run_svd(theta: np.ndarray) -> Dict[str, Any]:
    """
    Given a Document-Topic matrix Θ ∈ R^(N×K), calculates Θ = UΣVᵀ.
    Returns U, singular_values, Vt, and energy metrics.
    """
    if not isinstance(theta, np.ndarray):
        raise ValueError("Matrix must be a numpy array")
    if theta.ndim != 2:
        raise ValueError("Matrix must be 2D")
    if theta.size == 0 or theta.shape[0] == 0 or theta.shape[1] == 0:
        raise ValueError("Matrix cannot be empty")
    if np.isnan(theta).any() or np.isinf(theta).any():
        raise ValueError("Matrix contains invalid values (NaN/Inf)")
        
    try:
        U, S, Vt = np.linalg.svd(theta, full_matrices=False)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"SVD computation failed: {str(e)}")
        
    S = np.where(S < 1e-12, 0.0, S)
    squared_S = S ** 2
    total_energy = np.sum(squared_S)
    
    if total_energy > 0:
        latent_energy_ratio = squared_S / total_energy
    else:
        latent_energy_ratio = np.zeros_like(S)
        
    cumulative_energy = np.cumsum(latent_energy_ratio)
    rank = int(np.linalg.matrix_rank(theta, tol=1e-10))
    
    # Reconstruction error
    theta_reconstructed = U @ np.diag(S) @ Vt
    reconstruction_error = float(np.linalg.norm(theta - theta_reconstructed, ord='fro'))
    
    return {
        "U": U,
        "singular_values": S,
        "Vt": Vt,
        "squared_singular_values": squared_S,
        "latent_energy_ratio": latent_energy_ratio,
        "cumulative_energy": cumulative_energy,
        "rank": rank,
        "reconstruction_error": reconstruction_error
    }
