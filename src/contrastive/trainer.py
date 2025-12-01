import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

RANDOM_SEED = 42
def set_seed(seed=RANDOM_SEED):
    import random, os
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=384, projection_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, projection_dim * 2),
            nn.ReLU(),
            nn.Linear(projection_dim * 2, projection_dim)
        )
    def forward(self, x):
        z = self.net(x)
        return F.normalize(z, p=2, dim=1)

class WeightedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    def forward(self, a, b, weights):
        # a,b: (batch, dim). weights: (batch,)
        sim = F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=2) / self.temperature
        # positive on diagonal
        pos = torch.diag(sim)
        # Simple loss: -pos*weights + logsumexp(sim, dim=1)
        loss = - pos * weights + torch.logsumexp(sim, dim=1)
        return loss.mean()

def to_tensor_batch(pairs):
    """
    Convert list of (emb1_numpy, emb2_numpy, weight) to (a,b,weights) tensors
    """
    import torch
    a = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    b = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    w = torch.tensor([p[2] for p in pairs], dtype=torch.float32)
    return a, b, w

def train_once(projection_head, pairs, epochs=5, lr=1e-4):
    set_seed()
    device = torch.device("cpu")
    projection_head.to(device)
    opt = torch.optim.Adam(projection_head.parameters(), lr=lr)
    loss_fn = WeightedContrastiveLoss()
    for epoch in range(epochs):
        projection_head.train()
        a,b,w = to_tensor_batch(pairs)
        a = a.to(device); b=b.to(device); w=w.to(device)
        opt.zero_grad()
        za = projection_head(a)
        zb = projection_head(b)
        loss = loss_fn(za, zb, w)
        loss.backward()
        opt.step()
    return projection_head
