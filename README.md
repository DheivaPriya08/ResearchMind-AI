# 🧠 ResearchMind AI

**Document Intelligence + Semantic Search + Literature Review + Research Gap Detection for academic papers.**

Upload one or more research paper PDFs, semantically search across them, get auto-extracted summaries (methodology / results / conclusion / limitations), compare papers side by side, generate a literature review, surface research gaps, and export everything as a PDF report.

## Demo flow

```
Upload PDF(s)
   ↓
Semantic search any topic → relevant paragraphs (with page numbers)
   ↓
Per-paper summary: Methodology / Results / Conclusion / Limitations
   ↓
Multi-paper comparison table (dataset / model / accuracy / method)
   ↓
Literature review synthesis + cross-paper research gap analysis
   ↓
Export everything as one PDF report
```

## Why this architecture

This project is deliberately split into two halves with very different reliability profiles:

- **Retrieval (works offline, fully deterministic):** PDF extraction → chunking → embeddings (`sentence-transformers`) → FAISS similarity search. No API key needed. This is the core, and it's the part that's tested most thoroughly here, because a RAG project is only as good as its retrieval.
- **Generation (needs an API key — OpenAI or Google Gemini, your choice):** summaries, comparisons, literature review, research gap finding. These all go through one model call with a structured JSON schema in the prompt, rather than relying on regex/section-header matching — academic papers are far too inconsistent about section naming ("Methods" vs "Methodology" vs "Approach") for a heuristic parser to be robust. Citation extraction is the exception: it tries a regex heuristic first (fast, free) and only falls back to an LLM call if that heuristic comes up empty.

  A provider dropdown in the sidebar lets you pick OpenAI or Gemini and paste the matching key — `modules/llm_client.py` is a thin abstraction so the rest of the app doesn't care which one is active. Useful in practice since OpenAI's API requires paid billing while Gemini has a usable free tier.

**A note on "Research Gap Finder":** this is the most subjective feature in the project — asking an LLM to do something close to genuine literature-review judgment doesn't have a ground truth to check against. It's implemented here as "limitations synthesis across papers" (grounded in what the papers themselves say, plus explicit cross-paper pattern-spotting) rather than open-ended gap invention, specifically to keep the output defensible rather than generic-sounding. Treat its output as a draft starting point, not a final answer.

## Project structure

```text
ResearchMind-AI/
├── app.py                  # Streamlit app (UI + orchestration)
├── requirements.txt
├── README.md
├── .env.example
│
├── modules/
│   ├── pdf_loader.py        # PDF extraction (pdfplumber + pypdf fallback) + chunking
│   ├── embeddings.py        # sentence-transformers wrapper
│   ├── retriever.py         # FAISS index, single & cross-paper search
│   ├── summarizer.py        # LLM-based structured summary extraction
│   ├── compare.py           # Multi-paper comparison table generation
│   ├── citation.py          # Citation extraction, lit review, gap finder
│   └── report.py            # PDF report export (reportlab)
│
└── sample_papers/           # Two synthetic sample "papers" for quick testing
```

## Deploying (Streamlit Community Cloud)

This makes the app live at a public URL, with your API key hidden server-side via Streamlit's secrets manager — visitors get a working demo with zero setup, and your key is never exposed in the public repo.

1. **Push this repo to GitHub** (public or private both work for Streamlit Community Cloud).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **"New app"**, pick this repo, branch `main`, and set the main file path to `app.py`.
4. Before (or after) deploying, open the app's **Settings → Secrets** panel and paste:
   ```toml
   GEMINI_API_KEY = "AQ.your-real-key-here"
   ```
   (Use `OPENAI_API_KEY = "sk-..."` instead if you'd rather embed an OpenAI key.)
5. Click **Deploy**. After the build finishes you'll get a public URL like `https://your-app-name.streamlit.app`.

**How the key-handling works:** `app.py` checks `st.secrets` first. If the deployer configured a key there (step 4), every visitor gets a working app immediately with no key prompt — the sidebar just shows "✅ Using built-in access." If no secret is configured (e.g. running locally without `.streamlit/secrets.toml`), it falls back to the manual paste-your-own-key field, auto-detecting OpenAI vs Gemini from the key's prefix.

**Abuse protection on the shared key:** each browser session is capped at 25 AI-feature calls (`MAX_LLM_CALLS_PER_SESSION` in `app.py`) — semantic search itself is unlimited and free since it doesn't call the LLM at all. Raise or lower this constant depending on how much of your free-tier quota you're comfortable exposing to public traffic.

**To test the embedded-key behavior locally before deploying:** copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml`, fill in a real key, and run `streamlit run app.py` — you should see the same "no key prompt needed" experience visitors will get.

## Setup (local development)

```bash
git clone <your-repo-url>
cd ResearchMind-AI
pip install -r requirements.txt
```

You'll need an API key for the LLM-powered features (summaries, comparison, lit review, gap finding) — either OpenAI or Google Gemini, pick one in the app's sidebar. Semantic search works without either.

- **OpenAI**: get a key at `platform.openai.com` (starts with `sk-...`). Requires billing set up on your account.
- **Gemini**: get a key at Google AI Studio (`aistudio.google.com`). Has a free tier.

```bash
cp .env.example .env
# edit .env and add your key, OR just paste it into the sidebar when the app is running
```

## Run

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`), paste your OpenAI key into the sidebar, and upload a PDF — or use the included samples in `sample_papers/`.

## Design decisions worth knowing for an interview

- **Chunking:** character-based (not token-based) chunks of ~900 chars with 150-char overlap, computed per-page so every chunk retains an accurate page number for traceability back to the source. Chunk boundaries are nudged to the nearest sentence end where possible to avoid mid-sentence cuts.
- **Embedding model:** `all-MiniLM-L6-v2` — small (~80MB), fast on CPU, solid quality/speed tradeoff for short-passage semantic search. Embeddings are L2-normalized so cosine similarity reduces to a dot product, letting the FAISS index be a plain `IndexFlatIP` rather than something more complex.
- **Single shared index across all papers**, not one index per paper — multi-paper search is the default case (each chunk carries its own `paper_id`), and single-paper search is just a filtered query on the same index. This mirrors how the tool actually gets used.
- **Long papers:** rather than truncating to the first N characters for summarization, `summarize_paper_via_retrieval` runs a semantic search for each target field (methodology, results, conclusion, limitations) and feeds the LLM only the most relevant chunks — robust to where in the paper a section actually lives.
- **Citation extraction is heuristic-first, LLM-fallback** — regex over the `References` section runs first since it's free and instant; the LLM is only invoked if that heuristic finds fewer than 2 entries.

## Known limitations

- PDF extraction quality depends on the source PDF; scanned (image-only) PDFs aren't OCR'd here and will extract little or no text.
- Citation heuristic extraction works well for numbered `[1]`-style reference lists; author-year styles with no clear separator are less reliable and rely more on the LLM fallback.
- Research Gap Finder output should be treated as a draft for a human researcher to refine, not a final answer (see note above).
- This is a local/demo app (Streamlit, in-memory FAISS index per session) — there's no persistent database, so reloading the page clears loaded papers.

## Possible extensions

- Persist the FAISS index + paper metadata to disk (or a vector DB) so papers don't need re-uploading every session.
- OCR fallback (`pytesseract` + `pdf2image`) for scanned PDFs.
- Swap in a more powerful embedding model for higher retrieval quality on longer, more technical passages.
- Add a small retrieval evaluation harness (e.g. a fixed set of questions with known correct paragraphs) to quantitatively track retrieval quality across changes.
