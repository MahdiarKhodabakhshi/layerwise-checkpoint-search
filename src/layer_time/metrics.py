"""Representation metrics computation.

Step 4 of the workflow: Compute stationary metric features x_t,l from embeddings.

Core metrics (required):
- Prompt entropy
- Dataset entropy  
- Curvature
- Effective rank (dataset-level)

Optional metrics:
- Log-det covariance
- Anisotropy
- Spectral norm
- Mean pairwise cosine
- InfoNCE invariance score
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.linalg import svd
from scipy.stats import entropy as scipy_entropy


def compute_effective_rank(embeddings: np.ndarray, use_svd: bool = True) -> float:
    """
    Compute effective rank of the embedding matrix.
    
    Effective rank measures the dimensionality of the representation space.
    Uses singular value decomposition to compute the rank.
    
    Args:
        embeddings: Array of shape (n_examples, n_dimensions)
        use_svd: If True, use SVD-based computation (more accurate but slower)
    
    Returns:
        Effective rank (float)
    """
    if embeddings.shape[0] < 2:
        return 1.0
    
    # Center the embeddings
    centered = embeddings - embeddings.mean(axis=0, keepdims=True)
    
    if use_svd:
        # Compute SVD
        try:
            U, s, Vt = svd(centered, full_matrices=False)
            # Effective rank: exp of entropy of normalized singular values
            s_normalized = s / (s.sum() + 1e-10)
            s_normalized = s_normalized[s_normalized > 1e-10]  # Remove zeros
            if len(s_normalized) == 0:
                return 1.0
            ent = -np.sum(s_normalized * np.log(s_normalized + 1e-10))
            return float(np.exp(ent))
        except Exception:
            pass
    
    # Fallback: use covariance matrix trace
    cov = np.cov(centered.T)
    trace = np.trace(cov)
    if trace < 1e-10:
        return 1.0
    frobenius_norm = np.linalg.norm(cov, 'fro')
    return float((trace ** 2) / (frobenius_norm ** 2 + 1e-10))


def compute_prompt_entropy(embeddings: np.ndarray, n_bins: int = 50) -> float:
    """
    Compute prompt entropy (entropy of the distribution of embeddings).
    
    Measures the diversity/spread of representations.
    
    Args:
        embeddings: Array of shape (n_examples, n_dimensions)
        n_bins: Number of bins for histogram-based entropy estimation
    
    Returns:
        Prompt entropy (float)
    """
    if embeddings.shape[0] < 2:
        return 0.0
    
    # Use PCA to reduce dimensionality for entropy estimation
    try:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        U, s, Vt = svd(centered, full_matrices=False)
        # Use first principal component for entropy
        pc1 = U[:, 0] * s[0]
        
        # Compute histogram
        hist, _ = np.histogram(pc1, bins=n_bins)
        hist = hist + 1e-10  # Avoid zeros
        hist = hist / hist.sum()
        
        # Compute entropy
        ent = -np.sum(hist * np.log(hist))
        return float(ent)
    except Exception:
        # Fallback: use norm-based entropy
        norms = np.linalg.norm(embeddings, axis=1)
        hist, _ = np.histogram(norms, bins=n_bins)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        return float(-np.sum(hist * np.log(hist)))


def compute_dataset_entropy(embeddings: np.ndarray) -> float:
    """
    Compute dataset entropy (entropy of the embedding distribution).
    
    Similar to prompt entropy but computed differently - measures the
    entropy of the entire dataset's embedding distribution.
    
    Args:
        embeddings: Array of shape (n_examples, n_dimensions)
    
    Returns:
        Dataset entropy (float)
    """
    if embeddings.shape[0] < 2:
        return 0.0
    
    # Compute pairwise distances and use for entropy estimation
    try:
        # Sample a subset for efficiency if too large
        n_samples = min(1000, embeddings.shape[0])
        indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        sample = embeddings[indices]
        
        # Compute pairwise cosine distances
        normalized = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-10)
        pairwise_cosine = normalized @ normalized.T
        
        # Convert to distances (1 - cosine similarity)
        distances = 1 - pairwise_cosine
        distances = distances[np.triu_indices_from(distances, k=1)]
        
        # Estimate entropy from distance distribution
        hist, _ = np.histogram(distances, bins=50)
        hist = hist + 1e-10
        hist = hist / hist.sum()
        ent = -np.sum(hist * np.log(hist))
        return float(ent)
    except Exception:
        return 0.0


def compute_curvature(embeddings: np.ndarray) -> float:
    """
    Compute curvature metric.
    
    Measures the geometric curvature of the embedding manifold.
    
    Args:
        embeddings: Array of shape (n_examples, n_dimensions)
    
    Returns:
        Curvature (float)
    """
    if embeddings.shape[0] < 3:
        return 0.0
    
    try:
        # Compute local curvature using nearest neighbors
        from sklearn.neighbors import NearestNeighbors
        
        n_neighbors = min(10, embeddings.shape[0] - 1)
        if n_neighbors < 2:
            return 0.0
        
        nn = NearestNeighbors(n_neighbors=n_neighbors + 1)
        nn.fit(embeddings)
        distances, indices = nn.kneighbors(embeddings)
        
        # Compute average local curvature
        curvatures = []
        for i in range(min(100, embeddings.shape[0])):  # Sample for efficiency
            neighbors = embeddings[indices[i, 1:]]  # Exclude self
            center = embeddings[i]
            centered_neighbors = neighbors - center
            
            # Compute covariance of neighbors
            if centered_neighbors.shape[0] > 1:
                cov = np.cov(centered_neighbors.T)
                # Ensure symmetry (fix numerical errors)
                cov = (cov + cov.T) / 2
                # Curvature related to condition number of covariance
                eigvals = np.linalg.eigvals(cov)
                # Take real part and filter out negative/zero eigenvalues
                eigvals = np.real(eigvals)
                eigvals = eigvals[eigvals > 1e-10]
                if len(eigvals) > 1:
                    condition = np.max(eigvals) / (np.min(eigvals) + 1e-10)
                    curvatures.append(condition)
        
        if curvatures:
            return float(np.mean(curvatures))
    except ImportError:
        # Fallback: use variance-based approximation
        try:
            centered = embeddings - embeddings.mean(axis=0, keepdims=True)
            cov = np.cov(centered.T)
            # Ensure symmetry
            cov = (cov + cov.T) / 2
            eigvals = np.linalg.eigvals(cov)
            # Take real part and filter
            eigvals = np.real(eigvals)
            eigvals = eigvals[eigvals > 1e-10]
            if len(eigvals) > 1:
                return float(np.max(eigvals) / (np.min(eigvals) + 1e-10))
        except Exception:
            pass
    
    return 0.0


def compute_representation_metrics(embeddings: np.ndarray) -> Dict[str, float]:
    """
    Compute all representation metrics for a given embedding matrix.
    
    This is the core function that computes the feature vector x_t,l
    used by the bandit algorithm.
    
    Args:
        embeddings: Array of shape (n_examples, n_dimensions)
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics: Dict[str, float] = {}
    
    # Core metrics (required)
    metrics["prompt_entropy"] = compute_prompt_entropy(embeddings)
    metrics["dataset_entropy"] = compute_dataset_entropy(embeddings)
    metrics["curvature"] = compute_curvature(embeddings)
    metrics["effective_rank"] = compute_effective_rank(embeddings)
    
    # Optional metrics
    try:
        centered = embeddings - embeddings.mean(axis=0, keepdims=True)
        cov = np.cov(centered.T)
        
        # Log-determinant of covariance (use slogdet for numerical stability)
        try:
            # Ensure symmetry
            cov_sym = (cov + cov.T) / 2
            # Add regularization
            cov_reg = cov_sym + np.eye(cov_sym.shape[0]) * 1e-10
            sign, logdet = np.linalg.slogdet(cov_reg)
            if sign > 0:
                metrics["logdet_covariance"] = float(logdet)
            else:
                # If determinant is negative, use a safe fallback
                metrics["logdet_covariance"] = float(logdet)  # slogdet handles this
        except Exception:
            metrics["logdet_covariance"] = 0.0
        
        # Anisotropy (variance of singular values)
        try:
            s = np.linalg.svd(cov, compute_uv=False)
            metrics["anisotropy"] = float(np.std(s) / (np.mean(s) + 1e-10))
        except Exception:
            metrics["anisotropy"] = 0.0
        
        # Spectral norm
        try:
            metrics["spectral_norm"] = float(np.linalg.norm(cov, ord=2))
        except Exception:
            metrics["spectral_norm"] = 0.0
    except Exception:
        metrics["logdet_covariance"] = 0.0
        metrics["anisotropy"] = 0.0
        metrics["spectral_norm"] = 0.0
    
    # Mean pairwise cosine similarity
    try:
        normalized = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        # Sample for efficiency
        n_samples = min(500, embeddings.shape[0])
        indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
        sample = normalized[indices]
        pairwise_cosine = sample @ sample.T
        # Get upper triangle (excluding diagonal)
        triu_indices = np.triu_indices_from(pairwise_cosine, k=1)
        metrics["mean_pairwise_cosine"] = float(np.mean(pairwise_cosine[triu_indices]))
    except Exception:
        metrics["mean_pairwise_cosine"] = 0.0
    
    return metrics

