# tests/test_pipeline_basic.py
from src.pipeline import SummarizationPipeline

def test_demo_runs():
    p = SummarizationPipeline(config_path="configs/pipeline_config.yaml")
    res = p.summarize_document("examples/sample_paper.txt")
    assert "final_summary" in res or "final" in res
    assert isinstance(res["final_summary"], str)
    assert len(res["final_summary"]) > 0
