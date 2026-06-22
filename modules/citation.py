"""
citation.py
-----------
Week 4 features:
- Citation extraction: pull the reference list from a paper's text.
- Literature review generation: synthesize a short lit-review paragraph
  across multiple loaded papers.
- Research gap finder: surface gaps/limitations across papers, looking for
  common threads rather than just repeating each paper's own stated
  limitations.

Note on "Research Gap Finder": this is explicitly the most subjective/
LLM-judgment-heavy feature in the whole project (see general discussion of
why this is the riskiest Week 4 feature). It's framed here as "limitations
synthesis across papers" -- grounded in what the papers themselves say plus
explicit cross-paper pattern-spotting -- rather than open-ended novel gap
invention, to keep its output defensible rather than generic-sounding.
"""

import re
from modules.llm_client import complete_json, complete_text
from modules.summarizer import MAX_CHARS_FOR_FULL_TEXT


# ---------- Citation extraction ----------

REFERENCE_HEADER_PATTERN = re.compile(
    r"\n\s*(References|Bibliography|Works Cited)\s*\n", re.IGNORECASE
)


def extract_references_heuristic(full_text: str) -> list:
    """Heuristic citation extraction: find the References section by header,
    then split into individual entries by looking for lines that start with
    a bracketed number [1] or a numbered/author-year pattern.

    This is a heuristic, not a perfect parser -- reference formatting varies
    a lot across venues. It's offered as a fast, free, no-LLM-call first
    pass; extract_references_llm below is the fallback for messier formats.
    """
    match = REFERENCE_HEADER_PATTERN.search(full_text)
    if not match:
        return []

    ref_section = full_text[match.end():]
    # Stop at a likely appendix/end marker if present
    end_match = re.search(r"\n\s*(Appendix|Acknowledgg?ements)\s*\n", ref_section, re.IGNORECASE)
    if end_match:
        ref_section = ref_section[:end_match.start()]

    # Try splitting on numbered bracket refs like [1], [2]
    bracket_entries = re.split(r"\n?\[\d+\]\s*", ref_section)
    bracket_entries = [e.strip() for e in bracket_entries if len(e.strip()) > 20]
    if len(bracket_entries) >= 2:
        return bracket_entries

    # Fall back: split on blank lines / newlines for "Author, Year." style lists
    line_entries = [l.strip() for l in ref_section.split("\n") if len(l.strip()) > 20]
    return line_entries


def extract_references_llm(full_text: str, model: str = None) -> list:
    """LLM fallback for citation extraction when the heuristic finds nothing
    or finds too little -- handles inconsistent reference formatting."""
    prompt = f"""Extract the list of references/citations from this paper's text. \
If there is no visible reference list in the text provided, return an empty list.

Respond with ONLY valid JSON: {{"references": ["ref 1 full text", "ref 2 full text", ...]}}

TEXT:
\"\"\"
{full_text[-MAX_CHARS_FOR_FULL_TEXT:]}
\"\"\"
"""
    # references are usually near the END of a paper, hence [-MAX_CHARS:]
    result = complete_json(prompt, temperature=0, model=model)
    return result.get("references", [])


def extract_citations(full_text: str, use_llm_fallback: bool = True) -> list:
    refs = extract_references_heuristic(full_text)
    if len(refs) < 2 and use_llm_fallback:
        refs = extract_references_llm(full_text)
    return refs


# ---------- Literature review generation ----------

LIT_REVIEW_PROMPT = """You are writing a short literature review paragraph synthesizing the papers below.

Write 1-2 paragraphs that:
- Identify the common research theme connecting these papers
- Note how the approaches relate, build on, or diverge from each other
- Stay strictly grounded in what's actually stated in the texts -- do not invent findings

Do not use markdown headers. Plain prose only.

PAPERS:
{papers_block}
"""


def generate_literature_review(papers: list, model: str = None) -> str:
    """papers: list of {"title": ..., "text": ...} dicts."""
    blocks = []
    for i, p in enumerate(papers, start=1):
        truncated = p["text"][:MAX_CHARS_FOR_FULL_TEXT]
        blocks.append(f"--- PAPER {i}: {p['title']} ---\n{truncated}\n")
    papers_block = "\n".join(blocks)
    prompt = LIT_REVIEW_PROMPT.replace("{papers_block}", papers_block)

    return complete_text(prompt, temperature=0.3, model=model)


# ---------- Research gap finder ----------

GAP_FINDER_PROMPT = """You are analyzing limitations across multiple research papers to identify research gaps.

Below are the stated limitations/future-work sections (or general text if limitations weren't explicitly labeled) from each paper, plus their general topic area.

Identify 3-5 concrete research gaps. A good gap is either:
(a) a limitation explicitly stated by one or more of the papers, or
(b) a pattern that emerges from comparing limitations ACROSS papers (e.g. "none of these papers evaluate X")

Do NOT invent gaps unrelated to what's in the text. Each gap should be one short sentence.

Respond with ONLY valid JSON: {"research_gaps": ["gap 1", "gap 2", ...]}

PAPER LIMITATIONS / CONTEXT:
{context_block}
"""


def find_research_gaps(papers: list, model: str = None) -> list:
    """papers: list of {"title": ..., "text": ..., "limitations": Optional[str]}.
    If 'limitations' is already extracted (e.g. from summarize_paper), pass it
    in to ground the gap analysis more tightly; otherwise falls back to raw text.
    """
    blocks = []
    for i, p in enumerate(papers, start=1):
        content = p.get("limitations") or p["text"][:MAX_CHARS_FOR_FULL_TEXT]
        blocks.append(f"--- PAPER {i}: {p['title']} ---\n{content}\n")
    context_block = "\n".join(blocks)
    prompt = GAP_FINDER_PROMPT.replace("{context_block}", context_block)

    result = complete_json(prompt, temperature=0.3, model=model)
    if "_raw" in result:
        return [result["_raw"].strip()]
    return result.get("research_gaps", [])
