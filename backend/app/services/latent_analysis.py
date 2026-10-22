import numpy as np
from typing import Dict, Any
from app.services.svd_analysis import run_svd
from app.services.eigen_analysis import run_eigen_analysis

def load_document_topic_matrix(session, topic_analysis_id: str):
    """
    Reconstructs the Document x Topic matrix (Theta) from the database.
    Returns: numpy matrix (N x K), ordered paper_ids, ordered topic_ids, topic_labels
    """
    from app.models.paper_db import DocumentTopic, Topic
    
    docs = session.query(DocumentTopic).filter(DocumentTopic.topic_analysis_id == topic_analysis_id).all()
    if not docs:
        raise ValueError("No DocumentTopic records found.")
        
    topics = session.query(Topic).filter(Topic.topic_analysis_id == topic_analysis_id).order_by(Topic.topic_idx).all()
    topic_ids = [t.id for t in topics]
    topic_labels = {t.topic_idx: t.label for t in topics}
    
    paper_dict = {}
    for d in docs:
        if d.paper_id not in paper_dict:
            paper_dict[d.paper_id] = {}
        paper_dict[d.paper_id][d.topic_idx] = d.probability
        
    paper_ids = sorted(list(paper_dict.keys()))
    num_papers = len(paper_ids)
    num_topics = len(topics)
    
    theta = np.zeros((num_papers, num_topics))
    for i, p_id in enumerate(paper_ids):
        for j in range(num_topics):
            theta[i, j] = paper_dict[p_id].get(j, 0.0)
            
    # Normalize rows to sum to 1 to be safe, though LDA already does this
    row_sums = theta.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    theta = theta / row_sums
    
    return theta, paper_ids, topic_ids, topic_labels

def run_latent_math_analysis(theta: np.ndarray, low_energy_threshold: float = 0.05) -> Dict[str, Any]:
    """
    Orchestrates the actual mathematical latent dimension analysis.
    Validates relationship between singular values squared and eigenvalues.
    """
    svd_res = run_svd(theta)
    eigen_res = run_eigen_analysis(theta)
    
    # Validate relationship: lambda_i approx sigma_i^2
    # Ensure they have same shape before validation
    max_k = min(len(svd_res["squared_singular_values"]), len(eigen_res["eigenvalues"]))
    
    sigma_sq = svd_res["squared_singular_values"][:max_k]
    lam = eigen_res["eigenvalues"][:max_k]
    
    # Check within numerical tolerance
    if not np.allclose(sigma_sq, lam, atol=1e-5):
        raise ValueError("Mathematical validation failed: squared singular values do not match eigenvalues.")
        
    num_topics = theta.shape[1]
    dimensions = []
    
    Vt = svd_res["Vt"]
    
    for i in range(len(svd_res["singular_values"])):
        # Calculate topic loadings from Vt
        loadings = Vt[i, :]
        abs_loadings = np.abs(loadings)
        
        # Rank dominant topics
        sorted_indices = np.argsort(abs_loadings)[::-1]
        
        dominant_topics = []
        for rank, topic_idx in enumerate(sorted_indices):
            dominant_topics.append({
                "topic_index": int(topic_idx),
                "loading": float(loadings[topic_idx]),
                "absolute_loading": float(abs_loadings[topic_idx]),
                "rank": rank + 1
            })
            
        dim_data = {
            "dimension_index": i,
            "singular_value": float(svd_res["singular_values"][i]),
            "squared_singular_value": float(svd_res["squared_singular_values"][i]),
            "latent_energy_ratio": float(svd_res["latent_energy_ratio"][i]),
            "cumulative_energy": float(svd_res["cumulative_energy"][i]),
            "eigenvalue": float(eigen_res["eigenvalues"][i]) if i < len(eigen_res["eigenvalues"]) else 0.0,
            "normalized_eigen_contribution": float(eigen_res["normalized_eigen_contribution"][i]) if i < len(eigen_res["normalized_eigen_contribution"]) else 0.0,
            "cumulative_eigen_contribution": float(eigen_res["cumulative_eigen_contribution"][i]) if i < len(eigen_res["cumulative_eigen_contribution"]) else 0.0,
            "is_low_energy": float(svd_res["latent_energy_ratio"][i]) < low_energy_threshold,
            "dominant_topics": dominant_topics
        }
        dimensions.append(dim_data)
        
    # Calculate energy bounds
    cumulative_svd = svd_res["cumulative_energy"]
    energy_80 = list(np.where(cumulative_svd >= 0.80)[0])
    energy_90 = list(np.where(cumulative_svd >= 0.90)[0])
    
    return {
        "total_dimensions": len(dimensions),
        "rank": svd_res["rank"],
        "energy_80_dimensions": [int(x) for x in energy_80],
        "energy_90_dimensions": [int(x) for x in energy_90],
        "low_energy_threshold": low_energy_threshold,
        "reconstruction_error": svd_res["reconstruction_error"],
        "dimensions": dimensions
    }
