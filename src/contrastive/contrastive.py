# src/contrastive/contrastive.py
"""
Contrastive Learning Module

Implements contrastive learning components including:
- InfoNCE loss computation
- Section importance scoring using semantic similarity
- Proportional allocation of summary budget
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict
import math

try:
    from sentence_transformers import SentenceTransformer
    _has_sentence_transformers = True
except ImportError:
    _has_sentence_transformers = False

try:
    from sklearn.metrics.pairwise import cosine_similarity
    _has_sklearn = True
except ImportError:
    _has_sklearn = False


def info_nce_loss(z_i: torch.Tensor, 
                 z_j: torch.Tensor, 
                 temperature: float = 0.07) -> float:
    """
    Compute InfoNCE (Normalized Temperature-scaled Cross Entropy) loss.
    
    This loss function is used in contrastive learning to bring positive pairs
    closer and push negative pairs apart in the embedding space.
    
    Args:
        z_i: First batch of embeddings (positive pairs)
        z_j: Second batch of embeddings (positive pairs)
        temperature: Temperature parameter for scaling similarities
        
    Returns:
        InfoNCE loss value
    """
    # Concatenate positive pairs
    z = torch.cat([z_i, z_j], dim=0)
    
    # L2 normalization
    z = F.normalize(z, p=2, dim=1)
    
    # Compute similarity matrix
    sim_matrix = torch.matmul(z, z.T)
    
    N = z_i.shape[0]
    
    # Create mask to exclude self-similarities
    mask = (~torch.eye(2 * N, dtype=torch.bool)).to(z.device)
    
    # Compute exponential similarities (excluding self-similarities)
    exp_sim = torch.exp(sim_matrix / temperature) * mask
    
    # Denominator: sum of all negative pairs
    denominator = exp_sim.sum(dim=1)
    
    # Numerator: positive pairs similarity
    positive_sim = F.cosine_similarity(z_i, z_j)
    numerator = torch.exp(positive_sim / temperature)
    
    # Duplicate for both directions (i->j and j->i)
    numerator = torch.cat([numerator, numerator], dim=0)
    
    # Compute loss
    loss = -torch.log(numerator / (denominator + 1e-8))
    
    return float(loss.mean().item())


class ProjectionHead(nn.Module):
    """
    Projection head for contrastive learning.
    
    Maps embeddings to a lower-dimensional space for contrastive learning.
    """
    
    def __init__(self, input_dim: int = 384, projection_dim: int = 64):
        """
        Initialize projection head.
        
        Args:
            input_dim: Dimension of input embeddings
            projection_dim: Dimension of projected embeddings
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project embeddings."""
        return self.projection(x)


def compute_section_importance(texts: List[str], 
                               config: Dict) -> List[float]:
    """
    Compute importance scores for document sections using semantic similarity.
    
    Sections are scored based on their semantic similarity to a reference
    (typically the abstract or document average).
    
    Args:
        texts: List of section texts
        config: Configuration dictionary with model settings
        
    Returns:
        List of normalized importance scores (0-1)
    """
    if not texts:
        return []
    
    # Get model name from config
    model_name = config.get("feature_extraction", {}).get(
        "sentence_model", 
        "all-MiniLM-L6-v2"
    )
    
    # Try to load Sentence-BERT model
    if not _has_sentence_transformers:
        # Fallback: uniform importance
        return [1.0 / len(texts)] * len(texts)
    
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        # Fallback: uniform importance
        return [1.0 / len(texts)] * len(texts)
    
    # Generate embeddings for each section
    embeddings = []
    for text in texts:
        if not text.strip():
            # Zero embedding for empty text
            embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
        else:
            emb = model.encode(text, convert_to_numpy=True)
            embeddings.append(emb)
    
    embeddings = np.array(embeddings)
    
    if len(embeddings) == 0:
        return []
    
    # Use first section (usually abstract) as reference
    # Or use mean of all sections
    if len(embeddings) > 1:
        reference = embeddings[0]  # Abstract as reference
    else:
        reference = embeddings.mean(axis=0)
    
    # Compute cosine similarities to reference
    if not _has_sklearn:
        # Manual cosine similarity
        sims = []
        for emb in embeddings:
            sim = np.dot(emb, reference) / (np.linalg.norm(emb) * np.linalg.norm(reference) + 1e-8)
            sims.append(sim)
        sims = np.array(sims)
    else:
        sims = cosine_similarity(embeddings, reference.reshape(1, -1)).reshape(-1)
    
    # Normalize to [0, 1]
    sims = sims - sims.min()
    if sims.max() > 1e-8:
        sims = sims / sims.max()
    else:
        # All similarities are equal
        sims = np.ones_like(sims) / len(sims)
    
    return sims.tolist()


def exponential_allocation(scores: List[float], 
                          budget: int,
                          alpha: float = 2.0) -> List[int]:
    """
    Allocate summary budget proportionally using exponential weighting.
    
    Higher-scored sections receive exponentially more budget allocation.
    
    Args:
        scores: Importance scores for each section (0-1)
        budget: Total summary budget (number of sentences)
        alpha: Exponential weight parameter (higher = more aggressive allocation)
        
    Returns:
        List of budget allocations for each section
    """
    if not scores or budget <= 0:
        return [0] * len(scores)
    
    # Compute exponential weights
    weights = [math.exp(alpha * s) for s in scores]
    total_weight = sum(weights)
    
    if total_weight == 0:
        # Uniform allocation
        base = budget // len(scores)
        remainder = budget % len(scores)
        alloc = [base] * len(scores)
        for i in range(remainder):
            alloc[i] += 1
        return alloc
    
    # Proportional allocation
    allocations = [max(1, int(round(budget * (w / total_weight)))) for w in weights]
    
    # Adjust to match exact budget
    diff = sum(allocations) - budget
    iteration = 0
    max_iterations = 1000
    
    while diff != 0 and iteration < max_iterations:
        if diff > 0:
            # Reduce from largest allocation (if > 1)
            valid_idx = [i for i, a in enumerate(allocations) if a > 1]
            if not valid_idx:
                break
            idx = max(valid_idx, key=lambda i: allocations[i])
            allocations[idx] -= 1
            diff -= 1
        else:
            # Add to largest allocation
            idx = allocations.index(max(allocations))
            allocations[idx] += 1
            diff += 1
        iteration += 1
    
    return allocations
