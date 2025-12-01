# scripts/evaluate_demo.py
"""
Evaluation Script

Evaluates generated summaries against reference summaries using multiple metrics:
- ROUGE (ROUGE-1, ROUGE-2, ROUGE-L)
- BERTScore
- METEOR
- Compression statistics
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import evaluate_pair, format_evaluation_results

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate generated summaries against reference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with all metrics
  python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt
  
  # Evaluate with specific metrics
  python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt --metrics rouge,bertscore
  
  # Save results to file
  python scripts/evaluate_demo.py --reference ref.txt --hypothesis gen.txt --output results.txt
        """
    )
    
    parser.add_argument(
        "--reference",
        required=True,
        help="Path to reference (gold) summary file"
    )
    
    parser.add_argument(
        "--hypothesis",
        required=True,
        help="Path to generated (system) summary file"
    )
    
    parser.add_argument(
        "--metrics",
        default="rouge,bertscore,meteor,compression",
        help="Comma-separated list of metrics to compute (default: all)"
    )
    
    parser.add_argument(
        "--output",
        help="Path to save evaluation results (optional)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Validate inputs
    if not os.path.exists(args.reference):
        logger.error(f"Reference file not found: {args.reference}")
        sys.exit(1)
    
    if not os.path.exists(args.hypothesis):
        logger.error(f"Hypothesis file not found: {args.hypothesis}")
        sys.exit(1)
    
    try:
        # Load texts
        logger.info(f"Loading reference from: {args.reference}")
        with open(args.reference, "r", encoding="utf-8") as f:
            reference = f.read().strip()
        
        logger.info(f"Loading hypothesis from: {args.hypothesis}")
        with open(args.hypothesis, "r", encoding="utf-8") as f:
            hypothesis = f.read().strip()
        
        # Parse metrics
        metrics = [m.strip() for m in args.metrics.split(",")]
        
        # Evaluate
        logger.info(f"Computing metrics: {', '.join(metrics)}")
        results = evaluate_pair(reference, hypothesis, metrics=metrics)
        
        # Format results
        formatted_results = format_evaluation_results(results)
        
        # Print results
        print("\n" + formatted_results)
        
        # Save results if output specified
        if args.output:
            os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(formatted_results)
            logger.info(f"\n✓ Results saved to: {args.output}")
        
        # Print quick summary
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        
        if "rouge" in results:
            rouge_l = results["rouge"].get("rougeL", {})
            print(f"ROUGE-L F1:    {rouge_l.get('fmeasure', 0):.4f}")
        
        if "bertscore" in results:
            print(f"BERTScore F1:  {results['bertscore'].get('f1', 0):.4f}")
        
        if "meteor" in results:
            print(f"METEOR:        {results['meteor']:.4f}")
        
        if "compression" in results:
            comp = results["compression"]["word_compression"]
            print(f"Compression:   {comp:.1%}")
        
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
