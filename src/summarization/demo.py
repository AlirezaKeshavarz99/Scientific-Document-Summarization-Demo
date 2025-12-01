# src/summarization/demo.py
"""
Interactive Demo for Summarization

Provides a simple interactive demo to test the summarization pipeline
with custom text input.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.pipeline import SummarizationPipeline


def demo_section_level():
    """Demo section-level summarization."""
    print("\n" + "="*70)
    print("SECTION-LEVEL SUMMARIZATION DEMO")
    print("="*70 + "\n")
    
    sample_document = """
    Abstract
    
    This research explores the application of deep learning to medical image analysis.
    We propose a novel convolutional neural network architecture for disease detection.
    
    Introduction
    
    Medical imaging plays a crucial role in modern healthcare. Accurate diagnosis
    requires careful analysis of complex images. Deep learning has shown promise
    in automating this process while maintaining high accuracy.
    
    Methods
    
    We developed a custom CNN with residual connections and attention mechanisms.
    The model was trained on 10,000 labeled medical images using data augmentation.
    
    Results
    
    Our approach achieved 94% accuracy on the test set, outperforming baseline
    methods by 7%. The model showed robust performance across different patient
    demographics.
    
    Conclusion
    
    Deep learning provides effective automation for medical image analysis.
    Future work will focus on explainability and clinical integration.
    """
    
    print("Sample Document:")
    print("-" * 70)
    print(sample_document[:300] + "...\n")
    
    pipeline = SummarizationPipeline()
    section_summaries = pipeline.summarize(
        sample_document,
        summary_type="section",
        compression_ratio=0.25
    )
    
    print("\nSection Summaries:")
    print("-" * 70)
    for section_name, summary in section_summaries.items():
        print(f"\n{section_name.upper()}:")
        print(summary)


def demo_document_level():
    """Demo document-level summarization."""
    print("\n" + "="*70)
    print("DOCUMENT-LEVEL SUMMARIZATION DEMO")
    print("="*70 + "\n")
    
    sample_document = """
    Abstract
    
    This study investigates climate change impacts on coastal ecosystems.
    
    Introduction
    
    Climate change poses significant threats to marine biodiversity.
    Coastal regions are particularly vulnerable to rising sea levels.
    
    Methods
    
    We conducted field surveys at 50 coastal sites and analyzed temperature
    and biodiversity data over 10 years.
    
    Results
    
    Results showed 15% decline in species diversity correlating with 2°C
    temperature increase. Coral reefs were most affected.
    
    Discussion
    
    Findings highlight urgent need for conservation efforts. Policy interventions
    should prioritize marine protected areas.
    
    Conclusion
    
    Climate change significantly impacts coastal ecosystems. Immediate action
    is required to preserve marine biodiversity.
    """
    
    print("Sample Document:")
    print("-" * 70)
    print(sample_document[:300] + "...\n")
    
    pipeline = SummarizationPipeline()
    document_summary = pipeline.summarize(
        sample_document,
        summary_type="document",
        compression_ratio=0.2
    )
    
    print("\nDocument Summary:")
    print("-" * 70)
    print(document_summary)


def interactive_demo():
    """Interactive demo with user input."""
    print("\n" + "="*70)
    print("INTERACTIVE SUMMARIZATION DEMO")
    print("="*70 + "\n")
    
    print("Enter or paste your scientific text below.")
    print("Type 'END' on a new line when finished:\n")
    
    lines = []
    while True:
        try:
            line = input()
            if line.strip().upper() == 'END':
                break
            lines.append(line)
        except EOFError:
            break
    
    text = '\n'.join(lines)
    
    if not text.strip():
        print("\nNo text provided. Exiting.")
        return
    
    print("\nProcessing...")
    
    pipeline = SummarizationPipeline()
    summary = pipeline.summarize(
        text,
        summary_type="document",
        compression_ratio=0.2
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70 + "\n")
    print(summary)
    print("\n" + "="*70)


def main():
    """Main demo function."""
    print("\n" + "="*70)
    print("SCIENTIFIC DOCUMENT SUMMARIZATION - INTERACTIVE DEMO")
    print("="*70)
    
    print("\nSelect a demo option:")
    print("1. Section-level summarization (pre-loaded example)")
    print("2. Document-level summarization (pre-loaded example)")
    print("3. Interactive mode (enter your own text)")
    print("4. Exit")
    
    while True:
        try:
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == '1':
                demo_section_level()
            elif choice == '2':
                demo_document_level()
            elif choice == '3':
                interactive_demo()
            elif choice == '4':
                print("\nExiting demo. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 1-4.")
            
            if choice in ['1', '2', '3']:
                cont = input("\nRun another demo? (y/n): ").strip().lower()
                if cont != 'y':
                    print("\nExiting demo. Goodbye!")
                    break
                    
        except KeyboardInterrupt:
            print("\n\nDemo interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            print("Please try again.")


if __name__ == '__main__':
    main()
