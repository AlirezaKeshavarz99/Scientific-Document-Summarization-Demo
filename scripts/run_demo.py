# scripts/run_demo.py
"""
Run Summarization Demo

Command-line interface for the Scientific Document Summarization Framework.
Generates summaries from scientific papers using the complete pipeline.
"""

import argparse
import os
import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import SummarizationPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scientific Document Summarization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python scripts/run_demo.py --input examples/sample_paper.txt --output summary.txt
  
  # Section-level summaries
  python scripts/run_demo.py --input paper.txt --output sections.txt --summary-type section
  
  # Custom compression ratio
  python scripts/run_demo.py --input paper.txt --output summary.txt --compression-ratio 0.15
  
  # Use CUDA if available
  python scripts/run_demo.py --input paper.txt --output summary.txt --device cuda
        """
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input document (text file)"
    )
    
    parser.add_argument(
        "--output",
        default="data/outputs/demo_summary.txt",
        help="Path to output summary file (default: data/outputs/demo_summary.txt)"
    )
    
    parser.add_argument(
        "--summary-type",
        choices=["section", "document"],
        default="document",
        help="Type of summary to generate (default: document)"
    )
    
    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.2,
        help="Target compression ratio 0-1 (default: 0.2 = 80%% reduction)"
    )
    
    parser.add_argument(
        "--config",
        default="configs/pipeline_config.yaml",
        help="Path to pipeline configuration file"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device to run on (default: cpu)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main execution function."""
    args = parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate inputs
    if not os.path.exists(args.input):
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)
    
    if not (0 < args.compression_ratio <= 1):
        logger.error("Compression ratio must be between 0 and 1")
        sys.exit(1)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    try:
        # Initialize pipeline
        logger.info("Initializing summarization pipeline...")
        pipeline = SummarizationPipeline(
            config_path=args.config,
            device=args.device
        )
        
        # Generate summary
        logger.info(f"Processing document: {args.input}")
        logger.info(f"Summary type: {args.summary_type}")
        logger.info(f"Compression ratio: {args.compression_ratio}")
        
        summary = pipeline.summarize(
            args.input,
            summary_type=args.summary_type,
            compression_ratio=args.compression_ratio
        )
        
        # Write output
        with open(args.output, "w", encoding="utf-8") as f:
            if args.summary_type == "section":
                # Format section summaries
                for section_name, section_summary in summary.items():
                    f.write(f"## {section_name.upper()}\n\n")
                    f.write(f"{section_summary}\n\n")
            else:
                # Document-level summary
                f.write(summary)
        
        logger.info(f"✓ Summary written to: {args.output}")
        
        # Print preview
        print("\n" + "="*70)
        print("SUMMARY PREVIEW")
        print("="*70 + "\n")
        
        if args.summary_type == "section":
            for section_name, section_summary in list(summary.items())[:3]:
                print(f"**{section_name.upper()}**")
                print(section_summary[:200] + "..." if len(section_summary) > 200 else section_summary)
                print()
        else:
            print(summary[:1000] + "..." if len(summary) > 1000 else summary)
        
        print("\n" + "="*70)
        print(f"Full summary saved to: {args.output}")
        print("="*70)
        
    except Exception as e:
        logger.error(f"Error during summarization: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
