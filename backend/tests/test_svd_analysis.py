import pytest
import numpy as np
from app.services.svd_analysis import run_svd

def test_run_svd_valid():
    theta = np.array([
        [0.8, 0.2],
        [0.1, 0.9],
        [0.5, 0.5]
    ])
    res = run_svd(theta)
    
    assert "U" in res
    assert "singular_values" in res
    assert "Vt" in res
    assert "squared_singular_values" in res
    
    U = res["U"]
    S = res["singular_values"]
    Vt = res["Vt"]
    
    # Check reconstruction
    theta_recon = U @ np.diag(S) @ Vt
    np.testing.assert_allclose(theta, theta_recon, atol=1e-5)
    
    assert res["reconstruction_error"] < 1e-5
    assert len(res["singular_values"]) == 2

def test_run_svd_invalid():
    with pytest.raises(ValueError, match="must be a numpy array"):
        run_svd([[1, 2], [3, 4]])
        
    with pytest.raises(ValueError, match="must be 2D"):
        run_svd(np.array([1, 2, 3]))
        
    with pytest.raises(ValueError, match="cannot be empty"):
        run_svd(np.array([[]]))
        
    with pytest.raises(ValueError, match="invalid values"):
        run_svd(np.array([[1.0, np.nan]]))
