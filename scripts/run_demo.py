# scripts/run_demo.py
"""
Run summarization demo.

Command-line interface for the Scientific Document Summarization Framework.
Reads a scientific text file and generates either a document-level or
section-level summary.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import SummarizationPipeline


logger = logging.getLogger(__name__)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Scientific Document Summarization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_demo.py --input examples/sample_paper.txt --output summary.txt
  python scripts/run_demo.py --input paper.txt --output sections.txt --summary-type section
  python scripts/run_demo.py --input paper.txt --output summary.txt --compression-ratio 0.15
  python scripts/run_demo.py --input paper.txt --output summary.txt --device cuda
        """
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to input document (.txt)"
    )

    parser.add_argument(
        "--output",
        default="data/outputs/demo_summary.txt",
        help="Path to output summary file"
    )

    parser.add_argument(
        "--summary-type",
        choices=["section", "document"],
        default="document",
        help="Type of summary to generate"
    )

    parser.add_argument(
        "--compression-ratio",
        type=float,
        default=0.2,
        help="Target compression ratio between 0 and 1"
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
        help="Device to run on"
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
        raise FileNotFoundError(f"Input file not found: {path}")

    return file_path.read_text(encoding="utf-8").strip()


def write_summary(output_path: str, summary, summary_type: str):
    """Write summary output to file."""
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as f:
        if summary_type == "section" and isinstance(summary, dict):
            for section_name, section_summary in summary.items():
                f.write(f"## {section_name.upper()}\n\n")
                f.write(f"{section_summary}\n\n")
        else:
            f.write(str(summary))


def print_preview(summary, summary_type: str, output_path: str):
    """Print a short preview to the console."""
    print("\n" + "=" * 70)
    print("SUMMARY PREVIEW")
    print("=" * 70 + "\n")

    if summary_type == "section" and isinstance(summary, dict):
        for section_name, section_summary in list(summary.items())[:3]:
            print(f"**{section_name.upper()}**")
            preview = section_summary[:200] + "..." if len(section_summary) > 200 else section_summary
            print(preview)
            print()
    else:
        text = str(summary)
        print(text[:1000] + "..." if len(text) > 1000 else text)

    print("\n" + "=" * 70)
    print(f"Full summary saved to: {output_path}")
    print("=" * 70)


def main():
    """Main execution function."""
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    if not (0 < args.compression_ratio <= 1):
        logger.error("Compression ratio must be between 0 and 1.")
        sys.exit(1)

    try:
        document_text = read_text_file(args.input)

        logger.info("Initializing summarization pipeline...")
        pipeline = SummarizationPipeline(
            config_path=args.config,
            device=args.device
        )

        logger.info("Processing document: %s", args.input)
        logger.info("Summary type: %s", args.summary_type)
        logger.info("Compression ratio: %s", args.compression_ratio)

        summary = pipeline.summarize(
            document_text,
            summary_type=args.summary_type,
            compression_ratio=args.compression_ratio
        )

        write_summary(args.output, summary, args.summary_type)
        logger.info("Summary written to: %s", args.output)

        print_preview(summary, args.summary_type, args.output)

    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.error("Error during summarization: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()