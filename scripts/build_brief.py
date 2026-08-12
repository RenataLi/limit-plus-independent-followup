"""Build the two-page independent research brief as a polished PDF."""

from __future__ import annotations

import os
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Flowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from limitplus.release import load_release_metrics


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "limit_plus_independent_followup_brief.pdf"
# Keep PDF IDs and metadata deterministic while recording the public-release date.
os.environ["SOURCE_DATE_EPOCH"] = "1786492800"  # 2026-08-12 00:00:00 UTC
NAVY = colors.HexColor("#173B57")
BLUE = colors.HexColor("#2477A8")
TEAL = colors.HexColor("#159A9C")
PALE = colors.HexColor("#EAF3F7")
INK = colors.HexColor("#17232D")
MUTED = colors.HexColor("#5C6A73")
GRID = colors.HexColor("#D4E0E6")
GREEN = colors.HexColor("#087F5B")


class MetricBars(Flowable):
    def __init__(self, width: float, rows: list[tuple[str, float, float]]):
        super().__init__()
        self.width = width
        self.height = len(rows) * 8.5 * mm + 8 * mm
        self.rows = rows

    def draw(self):
        canvas = self.canv
        left = 55 * mm
        chart_width = self.width - left - 13 * mm
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(left, self.height - 4 * mm, "60")
        canvas.drawCentredString(left + chart_width / 2, self.height - 4 * mm, "80")
        canvas.drawRightString(left + chart_width, self.height - 4 * mm, "100%")
        for index, (label, recall, ndcg) in enumerate(self.rows):
            y = self.height - (index + 1) * 8.5 * mm - 2 * mm
            canvas.setFillColor(INK)
            canvas.setFont("Helvetica", 8.2)
            canvas.drawString(0, y + 2.1 * mm, label)
            for offset, value, color in ((3.2 * mm, recall, BLUE), (0, ndcg, TEAL)):
                x = left
                scaled = max(0.0, (value - 0.60) / 0.40) * chart_width
                canvas.setFillColor(PALE)
                canvas.roundRect(x, y + offset, chart_width, 2.2 * mm, 1 * mm, fill=1, stroke=0)
                canvas.setFillColor(color)
                canvas.roundRect(x, y + offset, scaled, 2.2 * mm, 1 * mm, fill=1, stroke=0)
            canvas.setFont("Helvetica", 7.4)
            canvas.setFillColor(BLUE)
            canvas.drawRightString(left + chart_width + 11 * mm, y + 5.0 * mm, f"R {recall:.3f}")
            canvas.setFillColor(TEAL)
            canvas.drawRightString(left + chart_width + 11 * mm, y + 1.8 * mm, f"N {ndcg:.3f}")


def footer(canvas, doc):
    canvas.saveState()
    width, _ = A4
    canvas.setStrokeColor(GRID)
    canvas.line(18 * mm, 12 * mm, width - 18 * mm, 12 * mm)
    canvas.setFont("Helvetica", 7.2)
    canvas.setFillColor(MUTED)
    canvas.drawString(18 * mm, 7.8 * mm, "Independent LIMIT+ follow-up | 12 Aug 2026")
    canvas.drawRightString(width - 18 * mm, 7.8 * mm, f"{doc.page} / 2")
    canvas.restoreState()


def paragraph(text: str, style):
    return Paragraph(text, style)


def callout_box(text: str, style, width: float):
    box = Table([[Paragraph(text, style)]], colWidths=[width])
    box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 8),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return box


def main() -> int:
    metrics = load_release_metrics(ROOT)
    non_negation_interval = (
        "crosses zero"
        if metrics.non_negation_ndcg20_ci_low <= 0 <= metrics.non_negation_ndcg20_ci_high
        else "does not cross zero"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    body = ParagraphStyle(
        "Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.5,
        leading=11.2,
        textColor=INK,
        spaceAfter=2.2 * mm,
    )
    small = ParagraphStyle(
        "Small",
        parent=body,
        fontSize=7.3,
        leading=9.1,
        textColor=MUTED,
    )
    title = ParagraphStyle(
        "Title",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=26,
        textColor=NAVY,
        spaceAfter=2 * mm,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        parent=body,
        fontSize=10.5,
        leading=13,
        textColor=MUTED,
        spaceAfter=5 * mm,
    )
    heading = ParagraphStyle(
        "Heading",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=11.2,
        leading=13.4,
        textColor=NAVY,
        spaceBefore=2 * mm,
        spaceAfter=1.5 * mm,
    )
    callout = ParagraphStyle(
        "Callout",
        parent=body,
        fontName="Helvetica-Bold",
        fontSize=9.7,
        leading=12.2,
        textColor=NAVY,
        spaceAfter=0,
    )
    page_title = ParagraphStyle(
        "PageTitle",
        parent=title,
        fontSize=20,
        leading=22,
        spaceAfter=1.5 * mm,
    )
    bullet = ParagraphStyle(
        "Bullet",
        parent=body,
        leftIndent=4 * mm,
        firstLineIndent=-2.8 * mm,
        bulletIndent=0,
        spaceAfter=1.1 * mm,
    )

    doc = SimpleDocTemplate(
        str(OUTPUT),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=17 * mm,
        invariant=1,
        title="Candidate Generation or Constraint Verification?",
        author="Independent LIMIT+ follow-up",
        subject="Independent research brief",
    )
    story = []
    story.append(paragraph("CANDIDATE GENERATION OR<br/>CONSTRAINT VERIFICATION?", title))
    story.append(paragraph("A controlled follow-up on LIMIT+ | Independent research brief", subtitle))
    story.append(
        callout_box(
            f"On LIMIT+'s literal synthetic grammar, an unmodified BM25 top-1,000 pool has "
            f"<b>macro candidate recall {metrics.bm25.candidate_recall:.3f}</b> "
            f"({metrics.micro_pool_coverage:.1%} micro coverage). Text-only Qwen reranking "
            f"raises Recall@100 from <b>{metrics.bm25.recall_at_100:.3f} to "
            f"{metrics.qwen.recall_at_100:.3f}</b>; an oracle-decomposed exact verifier "
            f"reaches <b>{metrics.exact.recall_at_100:.3f}</b>. Candidate generation is "
            "not the sole bottleneck.",
            callout,
            doc.width,
        )
    )
    story.append(Spacer(1, 3 * mm))
    story.append(paragraph("Why this test", heading))
    story.append(
        paragraph(
            "Table 3 evaluates reranking on a gold-complete diagnostic pool containing every "
            "relevant document plus sampled BM25 negatives. This follow-up instead keeps an "
            "unmodified first-stage BM25 top-1,000 pool and separates <b>pool coverage</b> "
            "from <b>constraint verification and ranking</b>.",
            body,
        )
    )
    story.append(paragraph("Protocol and hard gates", heading))
    for item in (
        "Pinned full LIMIT (50,000 documents) and released LIMIT+ (700 queries) by revision and SHA-256.",
        "Independent corpus parser and Boolean executor: exact set equality for 700/700 released qrels.",
        "Paper-compatible BM25Okapi: case-sensitive NLTK tokens, text only, k1=1.5, b=0.75, epsilon=0.25, _id IDs, released NumPy quicksort ties.",
        "Every final run is capped at 100 unique IDs; candidate coverage is reported separately; paired template-stratified bootstrap uses 10,000 draws.",
    ):
        story.append(paragraph(f"- {item}", bullet))
    story.append(Spacer(1, 1.5 * mm))
    story.append(paragraph("Main comparison", heading))
    table_data = [
        ["Method", "R@100", "Norm. R@100", "nDCG@20", "Macro pool R"],
        ["Paper BM25", "0.837", "-", "0.785", "-"],
        [
            "Independent BM25",
            f"{metrics.bm25.recall_at_100:.3f}",
            f"{metrics.bm25.normalized_recall_at_100:.3f}",
            f"{metrics.bm25.ndcg_at_20:.3f}",
            f"{metrics.bm25.candidate_recall:.3f}",
        ],
        [
            "BM25 pool + Qwen",
            f"{metrics.qwen.recall_at_100:.3f}",
            f"{metrics.qwen.normalized_recall_at_100:.3f}",
            f"{metrics.qwen.ndcg_at_20:.3f}",
            f"{metrics.qwen.candidate_recall:.3f}",
        ],
        [
            "BM25 pool + oracle verify",
            f"{metrics.exact.recall_at_100:.3f}",
            f"{metrics.exact.normalized_recall_at_100:.3f}",
            f"{metrics.exact.ndcg_at_20:.3f}",
            f"{metrics.exact.candidate_recall:.3f}",
        ],
        [
            "Hybrid pool + verify",
            f"{metrics.hybrid.recall_at_100:.3f}",
            f"{metrics.hybrid.normalized_recall_at_100:.3f}",
            f"{metrics.hybrid.ndcg_at_20:.3f}",
            f"{metrics.hybrid.candidate_recall:.3f}",
        ],
        [
            "Boolean oracle",
            f"{metrics.oracle.recall_at_100:.3f}",
            f"{metrics.oracle.normalized_recall_at_100:.3f}",
            f"{metrics.oracle.ndcg_at_20:.3f}",
            f"{metrics.oracle.candidate_recall:.3f}",
        ],
    ]
    table = Table(table_data, colWidths=[62 * mm, 24 * mm, 30 * mm, 26 * mm, 27 * mm])
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 3), (-1, 4), PALE),
                ("TEXTCOLOR", (0, 1), (-1, -1), INK),
                ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
                ("ALIGN", (1, 0), (-1, 0), "RIGHT"),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, GRID),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 3 * mm))
    story.append(
        MetricBars(
            doc.width,
            [
                (
                    "Reproduced BM25",
                    metrics.bm25.recall_at_100,
                    metrics.bm25.ndcg_at_20,
                ),
                (
                    "BM25 pool + Qwen",
                    metrics.qwen.recall_at_100,
                    metrics.qwen.ndcg_at_20,
                ),
                (
                    "BM25 pool + oracle verify",
                    metrics.exact.recall_at_100,
                    metrics.exact.ndcg_at_20,
                ),
                (
                    "Hybrid pool + verify",
                    metrics.hybrid.recall_at_100,
                    metrics.hybrid.ndcg_at_20,
                ),
                (
                    "Boolean oracle",
                    metrics.oracle.recall_at_100,
                    metrics.oracle.ndcg_at_20,
                ),
            ],
        )
    )
    story.append(
        paragraph(
            "R = Recall@100; N = nDCG@20. Verification matches BM25 candidate depth, "
            "not compute or information access; the hybrid is secondary and adds up to "
            "three atomic top-1,000 lists. "
            "Raw recall is capped by gold sets larger than 100.",
            small,
        )
    )

    story.append(PageBreak())
    story.append(paragraph("WHAT THE FOLLOW-UP REVEALS", page_title))
    story.append(paragraph("Three findings that refine the bottleneck diagnosis", subtitle))

    findings = [
        (
            "1. Reproduction is tight.",
            f"The released np.argsort(-scores) tie behavior gives "
            f"R@100={metrics.bm25.recall_at_100:.5f} and "
            f"nDCG@20={metrics.bm25.ndcg_at_20:.5f}. All six BM25 metrics match the "
            "typeset PDF (3 d.p.); at four decimals in the arXiv HTML/source, four "
            "match and R@5/R@100 differ by +0.0004/-0.0003 (ours minus paper).",
        ),
        (
            "2. Broad retrieval beats premature composition.",
            f"BM25 top-1,000 has macro candidate recall "
            f"{metrics.bm25.candidate_recall:.3f} "
            f"({metrics.micro_pool_coverage:.1%} micro coverage), while hard AND/OR "
            "composition of truncated positive atomic lists (with exact NOT filtering) "
            f"reaches only R@100={metrics.hard_composition.recall_at_100:.3f}. "
            f"Positive-atom-weighted recall is "
            f"{metrics.atomic_positive_recall_at_1000:.3f} at L=1,000: large postings "
            "cause early pruning.",
        ),
        (
            "3. A residual same-pool ranking/verification gap remains.",
            f"Across all {metrics.queries} queries, Qwen adds "
            f"{metrics.qwen_vs_bm25['recall_at_100'].delta:+.3f} R@100 and "
            f"{metrics.qwen_vs_bm25['ndcg_at_20'].delta:+.3f} nDCG@20 over BM25, "
            f"yet exact ordering adds another "
            f"{metrics.exact_vs_qwen['recall_at_100'].delta:+.3f} and "
            f"{metrics.exact_vs_qwen['ndcg_at_20'].delta:+.3f}. Qwen closes "
            f"{metrics.recall_gap_closed:.0%} of the recall gain but only "
            f"{metrics.ndcg20_gap_closed:.0%} of the early-ranking upper-bound gain.",
        ),
    ]
    for label, text in findings:
        story.append(KeepTogether([paragraph(label, heading), paragraph(text, body)]))

    story.append(paragraph("Template view", heading))
    template_data = [
        ["Template", "BM25 N@20", "Qwen N@20", "Exact N@20", "Macro pool R"]
    ] + [
        [
            row.label,
            f"{row.bm25_ndcg_at_20:.3f}",
            f"{row.qwen_ndcg_at_20:.3f}",
            f"{row.exact_ndcg_at_20:.3f}",
            f"{row.candidate_recall:.3f}",
        ]
        for row in metrics.templates
    ]
    template_table = Table(
        template_data, colWidths=[52 * mm, 29 * mm, 29 * mm, 30 * mm, 32 * mm]
    )
    template_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("BACKGROUND", (0, 5), (-1, 7), PALE),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 1), (-1, -1), 0.35, GRID),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(template_table)
    story.append(Spacer(1, 1 * mm))

    story.append(paragraph("Paired uncertainty", heading))
    qwen_recall = metrics.qwen_vs_bm25["recall_at_100"]
    qwen_ndcg = metrics.qwen_vs_bm25["ndcg_at_20"]
    exact_recall = metrics.exact_vs_qwen["recall_at_100"]
    exact_ndcg = metrics.exact_vs_qwen["ndcg_at_20"]
    bootstrap_data = [
        ["Paired comparison", "Delta", "95% CI", "Positive draws"],
        [
            "Qwen - BM25 R@100",
            f"{qwen_recall.delta:+.4f}",
            f"[{qwen_recall.ci_low:+.4f}, {qwen_recall.ci_high:+.4f}]",
            f"{qwen_recall.probability_positive * metrics.bootstrap_draws:,.0f} / "
            f"{metrics.bootstrap_draws:,}",
        ],
        [
            "Qwen - BM25 nDCG@20",
            f"{qwen_ndcg.delta:+.4f}",
            f"[{qwen_ndcg.ci_low:+.4f}, {qwen_ndcg.ci_high:+.4f}]",
            f"{qwen_ndcg.probability_positive * metrics.bootstrap_draws:,.0f} / "
            f"{metrics.bootstrap_draws:,}",
        ],
        [
            "Exact - Qwen R@100",
            f"{exact_recall.delta:+.4f}",
            f"[{exact_recall.ci_low:+.4f}, {exact_recall.ci_high:+.4f}]",
            f"{exact_recall.probability_positive * metrics.bootstrap_draws:,.0f} / "
            f"{metrics.bootstrap_draws:,}",
        ],
        [
            "Exact - Qwen nDCG@20",
            f"{exact_ndcg.delta:+.4f}",
            f"[{exact_ndcg.ci_low:+.4f}, {exact_ndcg.ci_high:+.4f}]",
            f"{exact_ndcg.probability_positive * metrics.bootstrap_draws:,.0f} / "
            f"{metrics.bootstrap_draws:,}",
        ],
    ]
    bootstrap_table = Table(bootstrap_data, colWidths=[48 * mm, 25 * mm, 54 * mm, 45 * mm])
    bootstrap_table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("BACKGROUND", (0, 0), (-1, 0), PALE),
                ("TEXTCOLOR", (0, 0), (-1, 0), NAVY),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LINEBELOW", (0, 0), (-1, -1), 0.35, GRID),
                ("TOPPADDING", (0, 0), (-1, -1), 2.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
            ]
        )
    )
    story.append(bootstrap_table)
    story.append(Spacer(1, 1 * mm))

    story.append(paragraph("Defensible conclusion", heading))
    story.append(
        callout_box(
            "On LIMIT+'s synthetic literal grammar, candidate generation is not the sole "
            "source of top-100 error. Qwen recovers much of the recall gap, but about half "
            "the early-ranking gap remains. Aggregate gains hide a crossover: Qwen helps "
            "negation but underperforms BM25 on the 100-query three-way-conjunction slice.",
            callout,
            doc.width,
        )
    )
    story.append(Spacer(1, 1 * mm))
    story.append(paragraph("Full neural test", heading))
    story.append(
        paragraph(
            f"Pinned Qwen3-Reranker-4B scored {metrics.pairs_scored:,} "
            "retrieval-derived pairs from query and "
            "document text only: no gold injection or truncation. Under the balanced "
            f"template mix, negation accounts for "
            f"{metrics.negation_share_of_net_ndcg20_gain:.1%} of the signed net macro "
            f"nDCG@20 gain; over 500 non-negation queries the "
            f"{metrics.non_negation_ndcg20_delta:+.3f} mean 95% interval "
            f"{non_negation_interval}. "
            "Operator and cardinality are confounded. Exact verification is an "
            "information-advantaged diagnostic upper bound.",
            body,
        )
    )
    story.append(
        callout_box(
            "Open question: Would retrieval-pool oracle ordering plus operator-specific "
            "neural results help separate candidate ceiling, constraint execution, and model "
            "calibration in a follow-up to Table 3?",
            callout,
            doc.width,
        )
    )
    story.append(
        paragraph(
            "Sources: arXiv:2605.03824; github.com/informagi/Complex-Set-Compositional-IR; "
            "huggingface.co/datasets/orionweller/LIMIT (CC BY 4.0). LIMIT+ has no explicit "
            "root license; this brief redistributes no data.",
            small,
        )
    )

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
