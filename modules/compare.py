"""
compare.py
----------
Multi-paper comparison: builds a structured side-by-side comparison table
(dataset, model, accuracy, etc.) across two or more papers, plus an optional
narrative comparison.
"""

from modules.llm_client import complete_json
from modules.summarizer import MAX_CHARS_FOR_FULL_TEXT

COMPARE_PROMPT = """You are comparing multiple research papers side by side.

For EACH paper below, extract these fields if mentioned (use "Not specified" if not found):
- dataset: the primary dataset(s) used
- model: the primary model/architecture name
- accuracy: the headline accuracy/F1/performance metric (include the number and metric name)
- key_method: one short phrase describing the core technique

Then write a 2-4 sentence narrative comparing the papers' approaches and results.

Respond with ONLY valid JSON in exactly this shape:
{
  "papers": [
    {"title": "...", "dataset": "...", "model": "...", "accuracy": "...", "key_method": "..."},
    ...
  ],
  "narrative_comparison": "..."
}

PAPERS:
{papers_block}
"""


def compare_papers(papers: list, model: str = None) -> dict:
    """papers: list of dicts with 'title' and 'text' keys.
    Returns {"papers": [...], "narrative_comparison": "..."}.
    """
    if len(papers) < 2:
        raise ValueError("Comparison requires at least 2 papers.")

    blocks = []
    for i, p in enumerate(papers, start=1):
        truncated = p["text"][:MAX_CHARS_FOR_FULL_TEXT]
        blocks.append(f"--- PAPER {i}: {p['title']} ---\n{truncated}\n")
    papers_block = "\n".join(blocks)

    prompt = COMPARE_PROMPT.replace("{papers_block}", papers_block)

    result = complete_json(prompt, temperature=0.2, model=model)
    if "_raw" in result:
        return {"papers": [], "narrative_comparison": result["_raw"].strip()}
    return result


def comparison_to_markdown_table(comparison: dict) -> str:
    """Render the structured comparison as a markdown table, e.g.:

    | Feature  | Paper A | Paper B |
    |----------|---------|---------|
    | Dataset  | SQuAD   | WebText |
    """
    papers = comparison.get("papers", [])
    if not papers:
        return "_No comparable data extracted._"

    headers = [p.get("title", f"Paper {i+1}")[:30] for i, p in enumerate(papers)]
    rows = ["Dataset", "Model", "Accuracy", "Key Method"]
    keys = ["dataset", "model", "accuracy", "key_method"]

    lines = []
    lines.append("| Feature | " + " | ".join(headers) + " |")
    lines.append("|---------|" + "|".join(["---"] * len(headers)) + "|")
    for row_label, key in zip(rows, keys):
        cells = [p.get(key, "Not specified") for p in papers]
        lines.append(f"| {row_label} | " + " | ".join(cells) + " |")
    return "\n".join(lines)
