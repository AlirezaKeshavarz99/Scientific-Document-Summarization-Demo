# src/contrastive/keyphrase_enhanced.py

import numpy as np
import random
from sentence_transformers import SentenceTransformer

MAX_PAIRS = 1000
KEYPHRASE_WEIGHT = 2.0
SIMILARITY_THRESHOLD = 0.45


def cosine_sim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def load_model(model_name="all-MiniLM-L6-v2"):
    """Load sentence transformer model."""
    return SentenceTransformer(model_name)


def encode_keyphrases(section_keyphrases, model):
    """
    section_keyphrases: dict(section -> [(phrase, score)])
    returns: dict(section -> embeddings)
    """
    out = {}

    for sec, phrases in section_keyphrases.items():
        texts = [p[0] for p in phrases]

        if not texts:
            continue

        out[sec] = model.encode(texts, convert_to_numpy=True)

    return out


def create_positive_pairs(section_sents, kp_embeddings, sample_limit=MAX_PAIRS):
    """
    Pair keyphrases with sentences from same section.
    """
    pairs = []
    sims = []

    for sec, sents in section_sents.items():
        kp = kp_embeddings.get(sec)

        if kp is None:
            continue

        for i in range(len(kp)):
            for j in range(len(sents)):

                if len(pairs) >= sample_limit:
                    return pairs, sims

                sim = cosine_sim(kp[i], sents[j])
                pairs.append((kp[i], sents[j], KEYPHRASE_WEIGHT))
                sims.append(sim)

    return pairs, sims


def create_negative_pairs(section_sents, kp_embeddings, sample_limit=MAX_PAIRS):
    """
    Keyphrases paired with sentences from OTHER sections.
    """
    pairs = []
    sims = []

    sections = list(section_sents.keys())

    for sec1, kp in kp_embeddings.items():
        for sec2 in sections:
            if sec1 == sec2:
                continue

            sents = section_sents.get(sec2, [])

            for i in range(len(kp)):
                for j in range(len(sents)):

                    if len(pairs) >= sample_limit:
                        return pairs, sims

                    sim = cosine_sim(kp[i], sents[j])

                    if sim > SIMILARITY_THRESHOLD:
                        continue

                    pairs.append((kp[i], sents[j], KEYPHRASE_WEIGHT))
                    sims.append(sim)

    return pairs, sims


def combine_pairs(pos_kp, neg_kp):
    """Simple merge utility."""
    return list(pos_kp), list(neg_kp)