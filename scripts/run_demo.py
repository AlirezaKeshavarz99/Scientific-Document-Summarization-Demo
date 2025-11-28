#!/usr/bin/env python3
"""
Demo script for Scientific Document Summarization Framework

This script demonstrates the architecture and capabilities of the
multi-stage summarization framework described in the MSc thesis.
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.pipeline import SummarizationPipeline

def main():
    print("=" * 70)
    print("SCIENTIFIC DOCUMENT SUMMARIZATION FRAMEWORK - DEMO")
    print("MSc Thesis Implementation (Demo Version)")
    print("=" * 70)
    
    print("\n📚 Framework Overview:")
    print("• Two-phase architecture: Section-level + Document-level summarization")
    print("• Self-reliant: No external training data required")
    print("• LLM integration: Advanced prompt engineering with pre-trained models")
    print("• Contrastive learning: Semantic representation refinement")
    print("• Multi-stage pipeline: Preprocessing → Feature extraction → Summarization")
    
    print("\n🚀 Initializing Pipeline...")
    pipeline = SummarizationPipeline()
    
    print("\n🔧 Pipeline Components:")
    print("✓ Document loading and segmentation")
    print("✓ Keyphrase extraction and semantic analysis") 
    print("✓ Section importance classification")
    print("✓ LLM-powered summarization with prompt engineering")
    print("✓ Multi-stage summary generation and refinement")
    
    print("\n📊 Performance (from Thesis Evaluation):")
    print("• ROUGE-1: 0.50 (25% improvement over baseline)")
    print("• ROUGE-2: 0.25")
    print("• BERTScore: 0.88")
    print("• Human Evaluation: 4.3/5.0")
    
    print("\n💡 Research Innovations:")
    print("• Novel contrastive learning approach for semantic distinction")
    print("• Gini-based distinctiveness analysis for section importance")
    print("• Exponential allocation algorithm for summary length distribution")
    print("• Critical n-gram fusion process to preserve technical terminology")
    print("• Fine-tuning-free operation with 8-bit quantization")
    
    print("\n🔒 Note: This is a demonstration version.")
    print("The complete implementation with novel algorithms is part of")
    print("ongoing research being prepared for publication.")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully! 🎉")
    print("For the complete implementation and research details,")
    print("please refer to the MSc thesis document.")
    print("=" * 70)

if __name__ == "__main__":
    main()