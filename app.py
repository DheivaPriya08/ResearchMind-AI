"""
app.py
------
ResearchMind AI -- Streamlit app.

Run with:  streamlit run app.py

Features (mapped to the original 4-week plan):
  Week 1: Upload PDF(s), semantic search over chunks
  Week 2: Per-paper summary / methodology / results / conclusion / limitations
  Week 3: Multi-paper upload, cross-paper search, comparison table
  Week 4: Literature review generator, citation extraction, research gap
          finder, export everything as a PDF report
"""

import os
import sys
import tempfile

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from modules.pdf_loader import load_pdf, chunk_paper
from modules.retriever import PaperIndex
from modules.summarizer import summarize_paper_via_retrieval, summarize_paper
from modules.compare import compare_papers, comparison_to_markdown_table
from modules.citation import extract_citations, generate_literature_review, find_research_gaps
from modules.report import build_report


st.set_page_config(page_title="ResearchMind AI", page_icon="🧠", layout="wide")


# ---------------- Session state ----------------
if "papers" not in st.session_state:
    st.session_state.papers = {}       # paper_id -> Paper object
if "index" not in st.session_state:
    st.session_state.index = PaperIndex()
if "summaries" not in st.session_state:
    st.session_state.summaries = {}    # paper_id -> summary dict
if "comparison" not in st.session_state:
    st.session_state.comparison = None
if "lit_review" not in st.session_state:
    st.session_state.lit_review = None
if "llm_calls_this_session" not in st.session_state:
    st.session_state.llm_calls_this_session = 0  # protects a shared/demo key from runaway usage
if "research_gaps" not in st.session_state:
    st.session_state.research_gaps = None
if "citations" not in st.session_state:
    st.session_state.citations = {}


def has_openai_key() -> bool:
    from modules.llm_client import has_api_key
    return has_api_key()


MAX_LLM_CALLS_PER_SESSION = 25  # generous for a real demo session, low enough to block abuse of a shared key


def llm_budget_ok() -> bool:
    """Guards against runaway usage of a shared/embedded demo key. Each
    person's session gets its own counter (Streamlit session_state is
    per-browser-session), so one visitor can't exhaust it for everyone --
    but it stops any single visitor (or a bot) from looping calls forever."""
    if st.session_state.llm_calls_this_session >= MAX_LLM_CALLS_PER_SESSION:
        st.error(
            f"This demo session has hit its limit of {MAX_LLM_CALLS_PER_SESSION} AI calls "
            "(protects the shared demo key from abuse). Refresh the page to start a new "
            "session, or use your own API key to remove this limit."
        )
        return False
    return True


# ---------------- Sidebar: API key + upload ----------------
with st.sidebar:
    st.title("🧠 ResearchMind AI")
    st.caption("Document Intelligence & Semantic Search for Research Papers")

    st.subheader("API Key")

    # Deployed version: if the app owner configured a key via Streamlit
    # secrets (.streamlit/secrets.toml locally, or the Secrets panel on
    # Streamlit Community Cloud), use it automatically -- visitors get a
    # zero-friction working demo without ever seeing or entering a key.
    # Local dev / no secret configured: fall back to the manual paste field.
    secret_key = None
    secret_provider = None
    try:
        if "GEMINI_API_KEY" in st.secrets:
            secret_key = st.secrets["GEMINI_API_KEY"]
            secret_provider = "gemini"
        elif "GOOGLE_API_KEY" in st.secrets:
            secret_key = st.secrets["GOOGLE_API_KEY"]
            secret_provider = "gemini"
        elif "OPENAI_API_KEY" in st.secrets:
            secret_key = st.secrets["OPENAI_API_KEY"]
            secret_provider = "openai"
    except Exception:
        pass  # no secrets.toml present at all -- totally normal for local dev

    if secret_key:
        os.environ["LLM_PROVIDER"] = secret_provider
        if secret_provider == "gemini":
            os.environ["GOOGLE_API_KEY"] = secret_key
        else:
            os.environ["OPENAI_API_KEY"] = secret_key
        st.success(f"✅ Using built-in **{'Gemini' if secret_provider == 'gemini' else 'OpenAI'}** access — no key needed.")
        key_input = secret_key  # so downstream "if key_input" checks still work
    else:
        st.caption("Paste either an OpenAI key (starts with `sk-`) or a Google AI Studio / Gemini key (starts with `AQ.` or `AIza`). The provider is auto-detected from the key itself.")

        key_input = st.text_input(
            "API key",
            type="password",
            value=os.environ.get("_RM_RAW_KEY", ""),
        )

        detected_provider = None
        if key_input:
            stripped = key_input.strip()
            if stripped.startswith("sk-"):
                detected_provider = "openai"
            elif stripped.startswith("AQ.") or stripped.startswith("AIza"):
                detected_provider = "gemini"

            os.environ["_RM_RAW_KEY"] = stripped  # remember raw input across reruns for the text_input above

            if detected_provider == "openai":
                os.environ["LLM_PROVIDER"] = "openai"
                os.environ["OPENAI_API_KEY"] = stripped
                os.environ.pop("GOOGLE_API_KEY", None)
                st.success(f"✅ Detected **OpenAI** key (ends in `...{stripped[-6:]}`)")
            elif detected_provider == "gemini":
                os.environ["LLM_PROVIDER"] = "gemini"
                os.environ["GOOGLE_API_KEY"] = stripped
                os.environ.pop("OPENAI_API_KEY", None)
                st.success(f"✅ Detected **Gemini** key (ends in `...{stripped[-6:]}`)")
            else:
                st.error(
                    "This doesn't look like a recognized OpenAI (`sk-...`) or Gemini "
                    "(`AQ....` / `AIza...`) key. Double-check you copied the whole "
                    "thing with no extra spaces."
                )
                os.environ.pop("OPENAI_API_KEY", None)
                os.environ.pop("GOOGLE_API_KEY", None)
        else:
            st.info("No API key entered yet.")

    st.divider()
    st.subheader("Upload Papers (PDF)")
    uploaded_files = st.file_uploader(
        "Upload one or more research papers", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files and st.button("Process Uploaded PDFs", type="primary"):
        progress = st.progress(0, text="Starting...")
        for i, f in enumerate(uploaded_files):
            progress.progress((i) / len(uploaded_files), text=f"Loading {f.name}...")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

            paper_id = f"p{len(st.session_state.papers)}_{f.name[:20]}"
            paper = load_pdf(tmp_path, paper_id=paper_id)
            st.session_state.papers[paper_id] = paper

            progress.progress((i + 0.5) / len(uploaded_files), text=f"Indexing {f.name}...")
            chunks = chunk_paper(paper)
            st.session_state.index.add_chunks(chunks)

        progress.progress(1.0, text="Done!")
        st.success(f"Processed {len(uploaded_files)} paper(s).")

    if st.session_state.papers:
        st.divider()
        st.subheader("Loaded Papers")
        for pid, paper in st.session_state.papers.items():
            st.write(f"📄 **{paper.title}** ({paper.num_pages} pages)")

        if st.button("Clear all papers"):
            st.session_state.papers = {}
            st.session_state.index = PaperIndex()
            st.session_state.summaries = {}
            st.session_state.comparison = None
            st.session_state.lit_review = None
            st.session_state.research_gaps = None
            st.session_state.citations = {}
            st.rerun()


# ---------------- Main area: tabs ----------------
if not st.session_state.papers:
    st.title("🧠 ResearchMind AI")
    st.markdown(
        """
        Upload one or more research paper PDFs in the sidebar to get started.

        **What you can do here:**
        - 🔍 **Semantic Search** — ask questions, get the most relevant passages back (works without an API key)
        - 📝 **Summaries** — auto-extracted methodology, results, conclusion, limitations
        - ⚖️ **Compare Papers** — side-by-side comparison table across multiple papers
        - 📚 **Literature Review** — auto-generated synthesis paragraph
        - 🔬 **Research Gaps** — cross-paper limitation analysis
        - 📤 **Export Report** — everything bundled into one PDF
        """
    )
    st.info("👈 Upload PDFs in the sidebar to begin. Try the included sample_papers/ if you don't have your own handy.")
    st.stop()

tab_search, tab_summary, tab_compare, tab_litreview, tab_export = st.tabs(
    ["🔍 Semantic Search", "📝 Summaries", "⚖️ Compare Papers", "📚 Lit Review & Gaps", "📤 Export Report"]
)


# ---------------- Tab 1: Semantic Search (Week 1 + Week 3) ----------------
with tab_search:
    st.header("Semantic Search")
    st.caption("Search across all loaded papers, or narrow to one.")

    paper_options = ["All papers"] + [p.title for p in st.session_state.papers.values()]
    selected = st.selectbox("Search scope", paper_options)

    query = st.text_input("Search query", placeholder="e.g. What accuracy did the model achieve?")
    top_k = st.slider("Number of results", 1, 10, 5)

    if query:
        paper_id_filter = None
        if selected != "All papers":
            for pid, p in st.session_state.papers.items():
                if p.title == selected:
                    paper_id_filter = pid
                    break

        results = st.session_state.index.search(query, top_k=top_k, paper_id=paper_id_filter)
        if not results:
            st.warning("No results found. Try a different query or check that papers are processed.")
        for r in results:
            with st.container(border=True):
                st.markdown(f"**{r.chunk.paper_title}** · page {r.chunk.page_number} · relevance: `{r.score:.3f}`")
                st.write(r.chunk.text)


# ---------------- Tab 2: Summaries (Week 2) ----------------
with tab_summary:
    st.header("Paper Summaries")
    if not has_openai_key():
        st.warning("Enter your API key in the sidebar (pick a provider first) to use this feature.")
    else:
        for pid, paper in st.session_state.papers.items():
            with st.expander(f"📄 {paper.title}", expanded=(len(st.session_state.papers) == 1)):
                if pid not in st.session_state.summaries:
                    if st.button(f"Generate summary", key=f"sum_{pid}") and llm_budget_ok():
                        with st.spinner("Analyzing paper..."):
                            try:
                                summary = summarize_paper_via_retrieval(
                                    pid, paper.title, st.session_state.index
                                )
                                st.session_state.summaries[pid] = summary
                                st.session_state.llm_calls_this_session += 1
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")
                else:
                    s = st.session_state.summaries[pid]
                    st.markdown(f"**Summary:** {s.get('summary')}")
                    st.markdown(f"**Methodology:** {s.get('methodology')}")
                    st.markdown(f"**Results:** {s.get('results')}")
                    st.markdown(f"**Conclusion:** {s.get('conclusion')}")
                    st.markdown(f"**Limitations:** {s.get('limitations')}")

                    st.divider()
                    st.markdown("**Citations**")
                    if pid not in st.session_state.citations:
                        if st.button("Extract citations", key=f"cite_{pid}"):
                            with st.spinner("Extracting references..."):
                                refs = extract_citations(paper.full_text)
                                st.session_state.citations[pid] = refs
                                st.rerun()
                    else:
                        refs = st.session_state.citations[pid]
                        st.caption(f"{len(refs)} reference(s) found")
                        for r in refs[:15]:
                            st.write(f"- {r}")


# ---------------- Tab 3: Compare Papers (Week 3) ----------------
with tab_compare:
    st.header("Compare Papers")
    if len(st.session_state.papers) < 2:
        st.info("Upload at least 2 papers to use comparison.")
    elif not has_openai_key():
        st.warning("Enter your API key in the sidebar (pick a provider first) to use this feature.")
    else:
        if st.button("Generate Comparison", type="primary") and llm_budget_ok():
            with st.spinner("Comparing papers..."):
                papers_payload = [
                    {"title": p.title, "text": p.full_text}
                    for p in st.session_state.papers.values()
                ]
                try:
                    st.session_state.comparison = compare_papers(papers_payload)
                    st.session_state.llm_calls_this_session += 1
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.comparison:
            st.markdown(comparison_to_markdown_table(st.session_state.comparison))
            st.divider()
            st.markdown("**Narrative Comparison**")
            st.write(st.session_state.comparison.get("narrative_comparison", ""))


# ---------------- Tab 4: Lit Review & Research Gaps (Week 4) ----------------
with tab_litreview:
    st.header("Literature Review Generator")
    if not has_openai_key():
        st.warning("Enter your API key in the sidebar (pick a provider first) to use this feature.")
    else:
        if st.button("Generate Literature Review", type="primary") and llm_budget_ok():
            with st.spinner("Synthesizing literature review..."):
                papers_payload = [
                    {"title": p.title, "text": p.full_text}
                    for p in st.session_state.papers.values()
                ]
                try:
                    st.session_state.lit_review = generate_literature_review(papers_payload)
                    st.session_state.llm_calls_this_session += 1
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.lit_review:
            st.write(st.session_state.lit_review)

        st.divider()
        st.header("Research Gap Finder")
        st.caption(
            "Synthesizes limitations across papers to surface gaps -- grounded in what the "
            "papers themselves state, plus cross-paper patterns. Treat this as a starting "
            "point for your own analysis, not a final answer."
        )
        if st.button("Find Research Gaps", type="primary") and llm_budget_ok():
            with st.spinner("Analyzing limitations across papers..."):
                papers_payload = []
                for pid, p in st.session_state.papers.items():
                    summary = st.session_state.summaries.get(pid, {})
                    papers_payload.append({
                        "title": p.title,
                        "text": p.full_text,
                        "limitations": summary.get("limitations"),
                    })
                try:
                    st.session_state.research_gaps = find_research_gaps(papers_payload)
                    st.session_state.llm_calls_this_session += 1
                except Exception as e:
                    st.error(f"Error: {e}")

        if st.session_state.research_gaps:
            for i, gap in enumerate(st.session_state.research_gaps, start=1):
                st.markdown(f"{i}. {gap}")


# ---------------- Tab 5: Export Report (Week 4) ----------------
with tab_export:
    st.header("Export Report")
    st.caption("Bundles whatever you've generated so far (summaries, comparison, lit review, gaps, citations) into a PDF.")

    if st.button("Build PDF Report", type="primary"):
        paper_summaries = []
        for pid, p in st.session_state.papers.items():
            if pid in st.session_state.summaries:
                paper_summaries.append({"title": p.title, "summary": st.session_state.summaries[pid]})

        if not paper_summaries:
            st.warning("Generate at least one paper summary first (see the Summaries tab).")
        else:
            citations_by_paper = {
                st.session_state.papers[pid].title: refs
                for pid, refs in st.session_state.citations.items()
            }
            output_path = os.path.join(tempfile.gettempdir(), "researchmind_report.pdf")
            build_report(
                output_path,
                paper_summaries=paper_summaries,
                comparison=st.session_state.comparison,
                literature_review=st.session_state.lit_review,
                research_gaps=st.session_state.research_gaps,
                citations_by_paper=citations_by_paper,
            )
            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Report PDF",
                    data=f.read(),
                    file_name="researchmind_report.pdf",
                    mime="application/pdf",
                )
            st.success("Report built successfully.")
