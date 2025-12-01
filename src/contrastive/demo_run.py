from contrastive.keyphrase_enhanced import load_model, get_keyphrase_embeddings, create_keyphrase_positive_pairs, create_keyphrase_negative_pairs, combine_pairs
from contrastive.trainer import ProjectionHead, train_once
import numpy as np
import os
import json

# Synthetic example: tiny article with two sections
def synthetic_example():
    # two sections, sentences are small random vectors (demo only)
    np.random.seed(0)
    section_matrices = {
        "Introduction": [np.random.randn(384) for _ in range(6)],
        "Methods": [np.random.randn(384) for _ in range(7)],
        "Results": [np.random.randn(384) for _ in range(5)]
    }
    section_keyphrases = {
        "Introduction": [("background", 0.9), ("motivation", 0.8)],
        "Methods": [("dataset", 0.95), ("training", 0.8)],
        "Results": [("accuracy", 0.9), ("evaluation", 0.7)]
    }
    return section_matrices, section_keyphrases

def main():
    print("Running demo...")
    section_matrices, section_keyphrases = synthetic_example()
    model = load_model()  # will download if missing; small demo uses this default
    kp_embs = get_keyphrase_embeddings(section_keyphrases, model)
    pos, pos_sims, pos_info = create_keyphrase_positive_pairs(section_matrices, kp_embs)
    neg, neg_sims, neg_info = create_keyphrase_negative_pairs(section_matrices, kp_embs)
    combined_pos, combined_neg = combine_pairs(pos, [], neg, [])
    print(f"Positive pairs: {len(combined_pos)}, Negative pairs: {len(combined_neg)}")
    # quick train on a small subset if available
    if combined_pos:
        sub = combined_pos[:16]  # tiny batch for demo
        ph = ProjectionHead(input_dim=384, projection_dim=64)
        ph = train_once(ph, sub, epochs=3, lr=1e-4)
        print("Demo training completed.")
    else:
        print("No positive pairs created; demo ends.")

if __name__ == "__main__":
    main()
