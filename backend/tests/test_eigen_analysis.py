import pytest
import numpy as np
from app.services.eigen_analysis import run_eigen_analysis

def test_run_eigen_valid():
    theta = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.5, 0.5]
    ])
    res = run_eigen_analysis(theta)
    
    assert "eigenvalues" in res
    assert "eigenvectors" in res
    
    lam = res["eigenvalues"]
    v = res["eigenvectors"]
    
    # Check dimensions
    assert len(lam) == 2
    assert v.shape == (2, 2)
    
    # Check descending order
    assert lam[0] >= lam[1]
    
    # Verify Cv = \lambda v
    C = theta.T @ theta
    
    for i in range(len(lam)):
        Cv = C @ v[:, i]
        lv = lam[i] * v[:, i]
        np.testing.assert_allclose(Cv, lv, atol=1e-5)

def test_run_eigen_invalid():
    with pytest.raises(ValueError, match="must be a numpy array"):
        run_eigen_analysis([[1, 2]])
