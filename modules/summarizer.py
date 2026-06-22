"""
summarizer.py
-------------
LLM-powered structured extraction from a paper: summary, methodology,
results, conclusion, limitations.

Design decision: rather than relying on regex/section-header matching (which
breaks constantly on real papers -- inconsistent headers like "Method" vs
"Methodology" vs "Approach", multi-column PDFs, missing sections), we let the
LLM read the (retrieved, relevant) text and extract structured fields
directly. We feed it the full paper text if it's short enough, or the
most relevant chunks (via semantic search for each field) if not -- this
keeps token usage reasonable for long papers while staying robust to
section-naming differences.
"""

from modules.llm_client import complete_json

SUMMARY_SCHEMA_PROMPT = """You are a research assistant extracting structured information from an academic paper.

Read the paper text below and extract the following fields. Be concise and factual -- base everything strictly on the text given, and write "Not specified in the provided text" for any field you cannot find.

Respond with ONLY valid JSON, no markdown formatting, no preamble, in exactly this shape:
{
  "summary": "2-3 sentence plain-language summary of what the paper does",
  "methodology": "1-3 sentences describing the method/approach/architecture used",
  "results": "1-3 sentences describing the key quantitative or qualitative results",
  "conclusion": "1-2 sentences describing the paper's main conclusion or takeaway",
  "limitations": "1-3 sentences describing limitations the authors acknowledge"
}

PAPER TEXT:
\"\"\"
{paper_text}
\"\"\"
"""

MAX_CHARS_FOR_FULL_TEXT = 12000  # ~3-4k tokens, safely within budget for one call


def summarize_paper(paper_text: str, model: str = None) -> dict:
    """Extract {summary, methodology, results, conclusion, limitations} as a
    dict from paper text using the LLM. Truncates very long papers to the
    first MAX_CHARS_FOR_FULL_TEXT characters (introduction/methods/results
    are typically front-loaded in practice, and a more sophisticated
    long-paper strategy can route through the retriever instead -- see
    summarize_paper_via_retrieval below)."""
    text = paper_text[:MAX_CHARS_FOR_FULL_TEXT]
    prompt = SUMMARY_SCHEMA_PROMPT.replace("{paper_text}", text)

    result = complete_json(prompt, temperature=0.2, model=model)
    if "_raw" in result:  # parsing failed upstream
        return {
            "summary": result["_raw"].strip(),
            "methodology": "Could not parse structured output.",
            "results": "Could not parse structured output.",
            "conclusion": "Could not parse structured output.",
            "limitations": "Could not parse structured output.",
        }
    return result


def summarize_paper_via_retrieval(paper_id: str, paper_title: str, index, model: str = None) -> dict:
    """For long papers: instead of truncating, pull the most relevant chunks
    for each field via semantic search, then summarize from those. This
    keeps the approach robust to where in the paper each section actually
    lives, without needing exact section-header matching."""
    field_queries = {
        "methodology": "methodology, model architecture, approach, experimental setup",
        "results": "results, accuracy, performance, evaluation findings",
        "conclusion": "conclusion, summary of contributions, main takeaway",
        "limitations": "limitations, weaknesses, future work, threats to validity",
    }
    gathered = {"abstract/intro": ""}
    intro_chunks = index.search("abstract introduction overview", top_k=3, paper_id=paper_id)
    gathered["abstract/intro"] = "\n".join(r.chunk.text for r in intro_chunks)

    for field, q in field_queries.items():
        results = index.search(q, top_k=3, paper_id=paper_id)
        gathered[field] = "\n".join(r.chunk.text for r in results)

    combined_text = "\n\n".join(f"[{k.upper()}]\n{v}" for k, v in gathered.items())
    return summarize_paper(combined_text, model=model)
