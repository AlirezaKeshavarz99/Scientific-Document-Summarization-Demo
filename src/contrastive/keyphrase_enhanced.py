import os
import random
import numpy as np
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt

MAX_KEYPHRASE_PAIRS = 1000
KEYPHRASE_WEIGHT = 2.0
SIMILARITY_THRESHOLD = 0.45

def cos_sim(a, b):
    """Cosine similarity for 1-D numpy arrays."""
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def get_keyphrase_embeddings(section_keyphrases, model):
    """
    Encode keyphrases for each section. 
    `section_keyphrases` is dict: section_name -> [(kp, score), ...]
    Returns dict: section_name -> numpy array (n_keyphrases, dim)
    """
    out = {}
    for sec, kp_pairs in section_keyphrases.items():
        keyphrases = [p[0] for p in kp_pairs]
        if not keyphrases:
            continue
        emb = model.encode(keyphrases, convert_to_numpy=True)
        out[sec] = emb
    return out

def create_keyphrase_positive_pairs(section_matrices, keyphrase_embeddings, sample_limit=MAX_KEYPHRASE_PAIRS):
    """
    For each section, pair each keyphrase embedding with each sentence embedding
    from the same section (sampling if too many).
    section_matrices: dict section -> [sentence_embedding_numpy,...]
    keyphrase_embeddings: dict section -> numpy array (k, dim)
    Returns list of (emb_kp, emb_sent, weight), similarities list, info dict.
    """
    positives = []
    similarities = []
    info = {}
    for sec, sents in section_matrices.items():
        kp_embs = keyphrase_embeddings.get(sec)
        if kp_embs is None:
            info[sec] = {'kp':0,'sents':len(sents),'pairs':0}
            continue
        potential_pairs = len(kp_embs) * len(sents)
        sample_factor = 1.0 if potential_pairs <= sample_limit else (sample_limit / potential_pairs)
        cnt = 0
        sim_sum = 0.0
        for i in range(len(kp_embs)):
            for j in range(len(sents)):
                if sample_factor < 1.0 and random.random() > sample_factor:
                    continue
                sim = cos_sim(kp_embs[i], sents[j])
                positives.append((kp_embs[i], sents[j], KEYPHRASE_WEIGHT))
                similarities.append(sim)
                sim_sum += sim
                cnt += 1
                if cnt >= sample_limit:
                    break
            if cnt >= sample_limit:
                break
        info[sec] = {'kp':len(kp_embs),'sents':len(sents),'pairs':cnt,'avg_sim': (sim_sum/cnt if cnt>0 else 0)}
    return positives, similarities, info

def create_keyphrase_negative_pairs(section_matrices, keyphrase_embeddings, threshold=SIMILARITY_THRESHOLD, sample_limit=MAX_KEYPHRASE_PAIRS):
    """
    Pair keyphrases of one section with sentences from other sections and filter out
    any pair with similarity > threshold.
    """
    negatives = []
    similarities = []
    info = {}
    sections = list(section_matrices.keys())
    for sec1, kp_embs in keyphrase_embeddings.items():
        info[sec1] = {}
        for sec2 in sections:
            if sec1 == sec2:
                continue
            sents = section_matrices.get(sec2)
            if not sents:
                info[sec1][sec2] = {'pairs':0,'filtered':0,'avg_sim':0}
                continue
            potential_pairs = len(kp_embs) * len(sents)
            sample_factor = 1.0 if potential_pairs <= sample_limit else (sample_limit / potential_pairs)
            cnt = 0; filt = 0; sim_sum=0.0
            for i in range(len(kp_embs)):
                for j in range(len(sents)):
                    if sample_factor < 1.0 and random.random() > sample_factor:
                        continue
                    sim = cos_sim(kp_embs[i], sents[j])
                    if sim > threshold:
                        filt += 1
                        continue
                    negatives.append((kp_embs[i], sents[j], KEYPHRASE_WEIGHT))
                    similarities.append(sim)
                    sim_sum += sim
                    cnt += 1
                    if cnt >= sample_limit:
                        break
                if cnt >= sample_limit:
                    break
            info[sec1][sec2] = {'pairs':cnt,'filtered':filt,'avg_sim': (sim_sum/cnt if cnt>0 else 0)}
    return negatives, similarities, info

def combine_pairs(kp_pos, orig_pos, kp_neg, orig_neg):
    """
    Simple concatenation that keeps weight field (tuple size = 3).
    Accepts lists.
    """
    combined_pos = list(kp_pos) + list(orig_pos or [])
    combined_neg = list(kp_neg) + list(orig_neg or [])
    return combined_pos, combined_neg

# small helper to get a model (so demo_run.py can call it)
def load_model(name='all-MiniLM-L6-v2'):
    return SentenceTransformer(name)
