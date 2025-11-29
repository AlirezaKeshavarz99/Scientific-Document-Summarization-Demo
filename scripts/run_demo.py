# scripts/run_demo.py
import argparse
import os
from src.pipeline import SummarizationPipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/outputs/demo_summary.txt")
    parser.add_argument("--config", default="configs/pipeline_config.yaml")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    pipeline = SummarizationPipeline(config_path=args.config)
    result = pipeline.summarize_document(args.input)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(result["final_summary"])
    print("Demo completed. Output written to", args.output)
    # Print brief summary
    print("\n=== Final Summary Preview ===\n")
    print(result["final_summary"][:1000])

if __name__ == "__main__":
    main()
