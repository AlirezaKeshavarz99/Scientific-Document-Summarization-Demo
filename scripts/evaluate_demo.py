# scripts/evaluate_demo.py
import argparse
from src.evaluation.metrics import evaluate_pair

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", required=True)
    parser.add_argument("--hypothesis", required=True)
    args = parser.parse_args()

    with open(args.reference, "r", encoding="utf-8") as f:
        ref = f.read()
    with open(args.hypothesis, "r", encoding="utf-8") as f:
        hyp = f.read()

    scores = evaluate_pair(ref, hyp)
    print("ROUGE results (fmeasure):")
    for k, v in scores.items():
        print(f"{k}: {v['fmeasure']:.4f}")

if __name__ == "__main__":
    main()
