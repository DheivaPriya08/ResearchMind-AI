"""
pdf_loader.py
-------------
Handles PDF upload/loading, text extraction, and chunking.

Uses pdfplumber for extraction (handles academic two-column layouts and
general formatting noticeably better than raw pypdf), with a pypdf fallback
for PDFs that pdfplumber chokes on.
"""

from dataclasses import dataclass, field
from pathlib import Path
import re
import hashlib


@dataclass
class Chunk:
    """A single chunk of text from a paper, with metadata for traceability."""
    chunk_id: str
    paper_id: str
    paper_title: str
    page_number: int
    text: str
    chunk_index: int  # position within the paper


@dataclass
class Paper:
    """A loaded paper: raw text per page + metadata."""
    paper_id: str
    title: str
    filepath: str
    num_pages: int
    pages: list = field(default_factory=list)  # list of raw page text
    full_text: str = ""


def _clean_text(text: str) -> str:
    """Light cleanup of extracted PDF text: fix hyphenation breaks, collapse
    whitespace, drop obvious header/footer noise like lone page numbers."""
    if not text:
        return ""
    # Rejoin words broken across a line by a hyphen, e.g. "trans-\nformer"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Collapse multiple newlines/spaces
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    # Drop lines that are just a number (likely a page number)
    text = re.sub(r"\n\d{1,4}\n", "\n", text)
    return text.strip()


def _extract_with_pdfplumber(filepath: str):
    import pdfplumber
    pages = []
    with pdfplumber.open(filepath) as pdf:
        for page in pdf.pages:
            raw = page.extract_text() or ""
            pages.append(_clean_text(raw))
    return pages


def _extract_with_pypdf(filepath: str):
    from pypdf import PdfReader
    reader = PdfReader(filepath)
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        pages.append(_clean_text(raw))
    return pages


def _guess_title(pages: list, fallback: str) -> str:
    """Heuristic: the title of an academic paper is usually one of the first
    non-empty, reasonably short lines on page 1 (not the abstract, which
    tends to be a long paragraph)."""
    if not pages:
        return fallback
    first_page = pages[0]
    lines = [l.strip() for l in first_page.split("\n") if l.strip()]
    for line in lines[:8]:
        # Skip lines that look like author lists, emails, or affiliations
        if "@" in line or re.search(r"\buniversity\b|\bdepartment\b", line, re.I):
            continue
        if 15 <= len(line) <= 200 and not line.isupper():
            return line
    return lines[0] if lines else fallback


def load_pdf(filepath: str, paper_id: str = None) -> Paper:
    """Load a single PDF, extracting text page by page.

    Tries pdfplumber first; falls back to pypdf if extraction yields
    essentially no text (e.g. some pdfplumber edge cases).
    """
    filepath = str(filepath)
    name = Path(filepath).stem

    try:
        pages = _extract_with_pdfplumber(filepath)
    except Exception:
        pages = []

    total_chars = sum(len(p) for p in pages)
    if total_chars < 50:  # extraction likely failed -> fallback
        try:
            pages = _extract_with_pypdf(filepath)
        except Exception:
            pages = pages or []

    full_text = "\n\n".join(pages)
    if paper_id is None:
        paper_id = hashlib.md5(filepath.encode()).hexdigest()[:10]

    title = _guess_title(pages, fallback=name)

    return Paper(
        paper_id=paper_id,
        title=title,
        filepath=filepath,
        num_pages=len(pages),
        pages=pages,
        full_text=full_text,
    )


def chunk_paper(paper: Paper, chunk_size: int = 900, overlap: int = 150) -> list:
    """Split a paper's text into overlapping chunks for embedding.

    Chunking is done by character count (not tokens) for simplicity, on a
    per-page basis so each chunk retains an accurate page_number — important
    for letting users trace a search result back to where it appeared.

    chunk_size=900 chars (~150-200 words) and overlap=150 chars are reasonable
    defaults for academic paper paragraphs: small enough for precise
    retrieval, large enough to keep a full sentence or two of context.
    """
    chunks = []
    idx = 0
    for page_num, page_text in enumerate(paper.pages, start=1):
        if not page_text or len(page_text) < 20:
            continue
        start = 0
        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            # try not to cut mid-sentence: extend to next period if close
            if end < len(page_text):
                next_period = page_text.find(". ", end)
                if 0 <= next_period - end < 200:
                    end = next_period + 1
            piece = page_text[start:end].strip()
            if len(piece) > 30:
                chunks.append(Chunk(
                    chunk_id=f"{paper.paper_id}_c{idx}",
                    paper_id=paper.paper_id,
                    paper_title=paper.title,
                    page_number=page_num,
                    text=piece,
                    chunk_index=idx,
                ))
                idx += 1
            if end >= len(page_text):
                break
            start = end - overlap if end - overlap > start else end
    return chunks
