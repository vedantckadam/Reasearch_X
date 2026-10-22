import numpy as np
from typing import Dict, Any, Tuple

def run_eigen_analysis(theta: np.ndarray) -> Dict[str, Any]:
    """
    Given Document-Topic matrix Θ ∈ R^(N×K), calculates C = ΘᵀΘ,
    then computes symmetric eigendecomposition Cv = λv.
    Returns eigenvalues in descending order and eigenvectors.
    """
    if not isinstance(theta, np.ndarray):
        raise ValueError("Matrix must be a numpy array")
    if theta.ndim != 2:
        raise ValueError("Matrix must be 2D")
    if theta.size == 0 or theta.shape[0] == 0 or theta.shape[1] == 0:
        raise ValueError("Matrix cannot be empty")
    if np.isnan(theta).any() or np.isinf(theta).any():
        raise ValueError("Matrix contains invalid values (NaN/Inf)")

    # Calculate covariance-like matrix C = Theta^T @ Theta
    C = theta.T @ theta
    
    # Eigendecomposition of a symmetric matrix
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(C)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Eigen decomposition failed: {str(e)}")
        
    # eigh returns eigenvalues in ascending order; we want descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # Handle tiny negative floating-point values from numerical noise
    eigenvalues = np.where(eigenvalues < 1e-12, 0.0, eigenvalues)
    
    total_eigen = np.sum(eigenvalues)
    if total_eigen > 0:
        normalized_eigen = eigenvalues / total_eigen
    else:
        normalized_eigen = np.zeros_like(eigenvalues)
        
    cumulative_eigen = np.cumsum(normalized_eigen)
    
    return {
        "eigenvalues": eigenvalues,
        "eigenvectors": eigenvectors,
        "normalized_eigen_contribution": normalized_eigen,
        "cumulative_eigen_contribution": cumulative_eigen
    }
