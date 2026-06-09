# scripts/evaluate_demo.py
"""
Evaluation script.

Evaluates a generated summary against a reference summary using ROUGE,
BERTScore, METEOR, and compression statistics.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.metrics import evaluate_pair, format_evaluation_results


logger = logging.getLogger(__name__)
ALLOWED_METRICS = {"rouge", "bertscore", "meteor", "compression"}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated summaries against reference summaries",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt
  python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt --metrics rouge,bertscore
  python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt --output results.txt
        """
    )

    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference summary file"
    )

    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Path to generated summary file"
    )

    parser.add_argument(
        "--metrics",
        default="rouge,bertscore,meteor,compression",
        help="Comma-separated list of metrics to compute"
    )

    parser.add_argument(
        "--output",
        help="Path to save the formatted evaluation results"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    return parser.parse_args()


def read_text_file(path: str) -> str:
    """Read a UTF-8 text file."""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    return file_path.read_text(encoding="utf-8").strip()


def parse_metric_list(metric_string: str):
    """Parse and validate metric names."""
    metrics = [m.strip().lower() for m in metric_string.split(",") if m.strip()]
    unknown = [m for m in metrics if m not in ALLOWED_METRICS]

    if unknown:
        raise ValueError(
            f"Unknown metric(s): {', '.join(unknown)}. "
            f"Allowed metrics are: {', '.join(sorted(ALLOWED_METRICS))}"
        )

    return metrics


def main():
    """Main execution function."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    try:
        reference = read_text_file(args.reference)
        hypothesis = read_text_file(args.hypothesis)

        metrics = parse_metric_list(args.metrics)

        logger.info("Computing metrics: %s", ", ".join(metrics))
        results = evaluate_pair(reference, hypothesis, metrics=metrics)

        formatted_results = format_evaluation_results(results)
        print("\n" + formatted_results)

        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(formatted_results, encoding="utf-8")
            logger.info("Results saved to: %s", args.output)

        print("\n" + "=" * 60)
        print("QUICK SUMMARY")
        print("=" * 60)

        if "rouge" in results:
            rouge_l = results["rouge"].get("rougeL", {})
            print(f"ROUGE-L F1:   {rouge_l.get('fmeasure', 0):.4f}")

        if "bertscore" in results:
            print(f"BERTScore F1: {results['bertscore'].get('f1', 0):.4f}")

        if "meteor" in results:
            print(f"METEOR:       {results['meteor']:.4f}")

        if "compression" in results:
            comp = results["compression"].get("word_compression", 0.0)
            print(f"Compression:  {comp:.1%}")

        print("=" * 60 + "\n")

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error during evaluation: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()