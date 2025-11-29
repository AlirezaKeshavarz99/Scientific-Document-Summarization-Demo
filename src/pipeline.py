# src/pipeline.py
import yaml
from pathlib import Path
from src.preprocessing.segmenter import ScientificDocumentSegmenter
from src.feature_extraction.keyphrase_extractor import ScientificKeyphraseExtractor
from src.summarization.llm_integration import summarize_text
from src.contrastive.contrastive import compute_section_importance, exponential_allocation
from src.utils.helpers import load_config, ensure_directory

class SummarizationPipeline:
    def __init__(self, config_path="configs/pipeline_config.yaml"):
        self.config = load_config(config_path)
        self.device = self.config["pipeline"]["device"]
        self.segmenter = ScientificDocumentSegmenter(spacy_model=self.config["preprocessing"]["spacy_model"])
        self.keyphrase_extractor = ScientificKeyphraseExtractor()
        ensure_directory("data/outputs")

    def load_document(self, path: str) -> str:
        p = Path(path)
        return p.read_text(encoding='utf-8')

    def segment_document(self, text: str):
        return list(self.segmenter.extract_sections(text).items())

    def extract_keyphrases(self, text: str):
        kps = self.keyphrase_extractor.extract_keyphrases(text, top_n=self.config["feature_extraction"]["keyphrase"]["top_n"])
        return [kp for kp, score in kps]

    def generate_section_summary(self, section_text: str):
        cfg = self.config["summarization"]
        return summarize_text(section_text, model_name=cfg["model_name"], max_length=cfg["max_length"], device=self.device)

    def rank_and_allocate(self, sections):
        texts = [t for _, t in sections]
        scores = compute_section_importance(texts, self.config)
        # normalize scores -> compute allocations
        total_budget = max(1, sum(len(t.split('.')) for t in texts) // 10)
        allocations = exponential_allocation(scores, total_budget)
        return list(zip([s[0] for s in sections], texts, scores, allocations))

    def summarize_document(self, path: str):
        text = self.load_document(path)
        sections = self.segment_document(text)
        ranked = self.rank_and_allocate(sections)
        section_summaries = {}
        for title, text, score, alloc in ranked:
            # For demo: call summarizer on the whole section (LLM fallback handles length)
            summary = self.generate_section_summary(text)
            section_summaries[title] = summary
        final_summary = "\n\n".join([f"{t}:\n{s}" for t, s in section_summaries.items()])
        return {
            "section_summaries": section_summaries,
            "final_summary": final_summary,
            "importance_scores": {t: s for t, _, s, _ in ranked},
            "allocations": {t: a for t, _, _, a in ranked}
        }
