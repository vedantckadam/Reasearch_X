import numpy as np
from typing import Dict, List, Optional, Tuple, Any

def _validate_theta(theta: np.ndarray):
    if not isinstance(theta, np.ndarray):
        raise ValueError("theta must be a numpy array")
    if theta.ndim != 2:
        raise ValueError("theta must be 2-dimensional")
    if theta.shape[0] == 0:
        raise ValueError("theta must contain at least one document")
    if theta.shape[1] == 0:
        raise ValueError("theta must contain at least one topic")
    if not np.issubdtype(theta.dtype, np.number):
        raise ValueError("theta must contain numeric values")
    if np.isnan(theta).any():
        raise ValueError("Document-topic matrix contains NaN values.")
    if np.isinf(theta).any():
        raise ValueError("Document-topic matrix contains infinity values.")
    if (theta < 0).any():
        raise ValueError("theta must contain non-negative values")

def calculate_weighted_intersection(theta: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Calculates the weighted topic intersection matrix M = Theta^T @ Theta.
    M[i,j] represents weighted co-occurrence/association between topic i and topic j across the corpus.
    M[i,i] represents the total squared topic probability mass for topic i.
    Returns:
        intersection matrix (K x K)
        number of topics K
    """
    _validate_theta(theta)
    
    M = theta.T @ theta
    num_topics = theta.shape[1]
    
    return M, num_topics

def calculate_support_intersection(theta: np.ndarray, threshold: float = 0.20) -> np.ndarray:
    """
    Calculates the document-support topic intersection matrix.
    A count-based support measure of the number of documents where both topics exceed the threshold.
    """
    _validate_theta(theta)
    
    if not (0 <= threshold <= 1):
        raise ValueError("threshold must be between 0 and 1")
        
    presence = (theta >= threshold).astype(int)
    support_matrix = presence.T @ presence
    
    return support_matrix

def calculate_normalized_association(weighted_intersection: np.ndarray) -> np.ndarray:
    """
    Calculates the normalized topic association matrix.
    A[i,j] = M[i,j] / sqrt(M[i,i] * M[j,j])
    """
    if not isinstance(weighted_intersection, np.ndarray) or weighted_intersection.ndim != 2:
        raise ValueError("weighted_intersection must be a 2D numpy array")
        
    diag = np.diag(weighted_intersection)
    diag_matrix = np.outer(diag, diag)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        normalized = weighted_intersection / np.sqrt(diag_matrix)
        normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
    return normalized

def extract_topic_pairs(matrix: np.ndarray, topic_labels: Dict[int, str], top_n: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Extracts unique topic pairs from a symmetric matrix.
    Excludes the diagonal.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
        
    pairs = []
    k = matrix.shape[0]
    
    for i in range(k):
        for j in range(i + 1, k):
            pairs.append({
                "topic_a_id": i,
                "topic_a": topic_labels.get(i, f"Topic {i}"),
                "topic_b_id": j,
                "topic_b": topic_labels.get(j, f"Topic {j}"),
                "value": float(matrix[i, j])
            })
            
    # Sort descending by value
    pairs.sort(key=lambda x: x["value"], reverse=True)
    
    if top_n is not None:
        pairs = pairs[:top_n]
        
    return pairs
