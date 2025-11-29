# src/evaluation/metrics.py
from rouge_score import rouge_scorer

def calculate_rouge(reference: str, hypothesis: str):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return {k: {"precision": v.precision, "recall": v.recall, "fmeasure": v.fmeasure} for k, v in scores.items()}

def evaluate_pair(reference: str, hypothesis: str):
    return calculate_rouge(reference, hypothesis)
