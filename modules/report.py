"""
report.py
---------
Exports a ResearchMind AI analysis session (summaries, comparison table,
literature review, research gaps) to a polished PDF report using reportlab.
"""

from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime


def _styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="GapItem", parent=styles["Normal"], leftIndent=14, bulletIndent=4, spaceAfter=6
    ))
    return styles


def build_report(
    output_path: str,
    paper_summaries: list,      # list of {"title": str, "summary": dict}
    comparison: dict = None,    # {"papers": [...], "narrative_comparison": str}
    literature_review: str = None,
    research_gaps: list = None,
    citations_by_paper: dict = None,  # {paper_title: [refs]}
):
    styles = _styles()
    doc = SimpleDocTemplate(output_path, pagesize=letter,
                             topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    story = []

    # --- Title page ---
    story.append(Paragraph("ResearchMind AI — Research Report", styles["Title"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(
        f"Generated {datetime.now().strftime('%B %d, %Y')} · {len(paper_summaries)} paper(s) analyzed",
        styles["Normal"]
    ))
    story.append(Spacer(1, 20))

    # --- Per-paper summaries ---
    story.append(Paragraph("Paper Summaries", styles["Heading1"]))
    for item in paper_summaries:
        title = item["title"]
        s = item["summary"]
        story.append(Paragraph(title, styles["Heading2"]))
        story.append(Paragraph(f"<b>Summary:</b> {s.get('summary', 'N/A')}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Methodology:</b> {s.get('methodology', 'N/A')}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Results:</b> {s.get('results', 'N/A')}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Conclusion:</b> {s.get('conclusion', 'N/A')}", styles["Normal"]))
        story.append(Spacer(1, 4))
        story.append(Paragraph(f"<b>Limitations:</b> {s.get('limitations', 'N/A')}", styles["Normal"]))
        story.append(Spacer(1, 14))

    # --- Comparison table ---
    if comparison and comparison.get("papers"):
        story.append(PageBreak())
        story.append(Paragraph("Paper Comparison", styles["Heading1"]))
        papers = comparison["papers"]
        headers = ["Feature"] + [p.get("title", "")[:24] for p in papers]
        rows = [
            ["Dataset"] + [p.get("dataset", "N/A") for p in papers],
            ["Model"] + [p.get("model", "N/A") for p in papers],
            ["Accuracy"] + [p.get("accuracy", "N/A") for p in papers],
            ["Key Method"] + [p.get("key_method", "N/A") for p in papers],
        ]
        table_data = [headers] + rows
        t = Table(table_data, hAlign="LEFT")
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
        ]))
        story.append(t)
        story.append(Spacer(1, 12))
        if comparison.get("narrative_comparison"):
            story.append(Paragraph("Narrative Comparison", styles["Heading2"]))
            story.append(Paragraph(comparison["narrative_comparison"], styles["Normal"]))

    # --- Literature review ---
    if literature_review:
        story.append(PageBreak())
        story.append(Paragraph("Literature Review", styles["Heading1"]))
        for para in literature_review.split("\n\n"):
            if para.strip():
                story.append(Paragraph(para.strip(), styles["Normal"]))
                story.append(Spacer(1, 8))

    # --- Research gaps ---
    if research_gaps:
        story.append(Spacer(1, 12))
        story.append(Paragraph("Research Gaps Identified", styles["Heading1"]))
        for i, gap in enumerate(research_gaps, start=1):
            story.append(Paragraph(f"{i}. {gap}", styles["GapItem"]))

    # --- Citations ---
    if citations_by_paper:
        story.append(PageBreak())
        story.append(Paragraph("Citations", styles["Heading1"]))
        for title, refs in citations_by_paper.items():
            story.append(Paragraph(title, styles["Heading2"]))
            if not refs:
                story.append(Paragraph("No references extracted.", styles["Normal"]))
            for ref in refs[:50]:  # cap to avoid runaway reports
                story.append(Paragraph(f"• {ref}", styles["GapItem"]))
            story.append(Spacer(1, 10))

    doc.build(story)
    return output_path
