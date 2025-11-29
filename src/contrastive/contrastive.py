# src/contrastive/contrastive.py
import torch
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def info_nce_loss(z_i: torch.Tensor, z_j: torch.Tensor, temperature: float = 0.07) -> float:
    z = torch.cat([z_i, z_j], dim=0)
    z = F.normalize(z, p=2, dim=1)
    sim = torch.matmul(z, z.T)
    N = z_i.shape[0]
    mask = (~torch.eye(2*N, dtype=torch.bool)).to(z.device)
    exp_sim = torch.exp(sim / temperature) * mask
    denom = exp_sim.sum(dim=1)
    pos = torch.exp((F.cosine_similarity(z_i, z_j) / temperature))
    pos = torch.cat([pos, pos], dim=0)
    loss = -torch.log(pos / denom)
    return float(loss.mean().item())

def compute_section_importance(texts, config):
    model_name = config["feature_extraction"]["sentence_model"]
    try:
        model = SentenceTransformer(model_name)
    except Exception:
        # fallback: uniform importance
        return [1.0 / max(1, len(texts)) for _ in texts]
    embeddings = []
    for t in texts:
        if not t.strip():
            embeddings.append(np.zeros(model.get_sentence_embedding_dimension()))
            continue
        emb = model.encode(t, convert_to_numpy=True)
        embeddings.append(emb)
    embeddings = np.array(embeddings)
    ref = embeddings[0] if len(embeddings) > 0 else embeddings.mean(axis=0)
    sims = cosine_similarity(embeddings, ref.reshape(1,-1)).reshape(-1)
    sims = sims - sims.min()
    if sims.max() > 0:
        sims = sims / sims.max()
    else:
        sims = np.ones_like(sims) / len(sims)
    return sims.tolist()

def exponential_allocation(scores, budget):
    import math
    alpha = 2.0
    weights = [math.exp(alpha * s) for s in scores]
    total = sum(weights)
    alloc = [max(1, int(round(budget * (w / total)))) for w in weights]
    # adjust to sum budget
    diff = sum(alloc) - budget
    idx = 0
    while diff != 0 and idx < 1000:
        if diff > 0:
            # reduce largest
            j = alloc.index(max(alloc))
            if alloc[j] > 1:
                alloc[j] -= 1
                diff -= 1
        else:
            # add to largest
            j = alloc.index(max(alloc))
            alloc[j] += 1
            diff += 1
        idx += 1
    return alloc
