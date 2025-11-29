# src/contrastive/demo_contrastive.py
import torch
from .contrastive import info_nce_loss

def demo():
    torch.manual_seed(0)
    batch = 4
    dim = 32
    z_i = torch.randn(batch, dim)
    z_j = torch.randn(batch, dim)
    loss = info_nce_loss(z_i, z_j, temperature=0.1)
    print("Demo InfoNCE loss:", loss)

if __name__ == "__main__":
    demo()
