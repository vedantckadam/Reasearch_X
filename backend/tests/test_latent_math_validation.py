import pytest
import numpy as np
from app.services.latent_analysis import run_latent_math_analysis

def test_latent_math_validation():
    theta = np.array([
        [0.8, 0.1, 0.1],
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1],
        [0.2, 0.7, 0.1]
    ])
    
    res = run_latent_math_analysis(theta)
    
    assert res["total_dimensions"] == 3
    assert res["rank"] <= 3
    assert "energy_80_dimensions" in res
    assert "dimensions" in res
    
    dims = res["dimensions"]
    assert len(dims) == 3
    
    for i, dim in enumerate(dims):
        # Verify mathematically that lambda_i == sigma_i^2
        assert np.isclose(dim["eigenvalue"], dim["squared_singular_value"], atol=1e-5)
        
        # Verify dominant topics is sorted by absolute loading descending
        topics = dim["dominant_topics"]
        for j in range(len(topics) - 1):
            assert topics[j]["absolute_loading"] >= topics[j+1]["absolute_loading"]
            
    assert res["reconstruction_error"] < 1e-5

def test_latent_math_validation_failure():
    # Test validation failure by intentionally passing mismatching arrays
    from app.services import latent_analysis
    import app.services.svd_analysis as svd
    
    original_svd = svd.run_svd
    
    def mock_bad_svd(theta):
        res = original_svd(theta)
        res["squared_singular_values"] = res["squared_singular_values"] + 10.0 # Force mismatch
        return res
        
    latent_analysis.run_svd = mock_bad_svd
    try:
        theta = np.array([[0.8, 0.2], [0.1, 0.9]])
        with pytest.raises(ValueError, match="Mathematical validation failed"):
            latent_analysis.run_latent_math_analysis(theta)
    finally:
        latent_analysis.run_svd = original_svd
