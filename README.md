# Candidate Generation or Constraint Verification?

**An independent, reproducible follow-up to _Reproducing Complex Set-Compositional
Information Retrieval_ (Degenhart et al., 2026).**

This project asks a focused diagnostic question about the released LIMIT+ benchmark:

> Given the released query decomposition, where are relevant documents lost - during
> candidate generation, Boolean constraint verification, or final ranking?

[Two-page research brief](artifacts/limit_plus_independent_followup_brief.pdf) |
[Detailed methods and results](notes/METHODS_AND_RESULTS.md) |
[Full Qwen protocol](notes/QWEN_PROTOCOL.md) |
[Reproduction guide](docs/REPRODUCIBILITY.md)

This is an independent study. It is not affiliated with or endorsed by the authors of
the source paper.

## Headline result

The independent BM25 implementation matches all six LIMIT+ Table 2 values as typeset in
the paper PDF (three decimals). The arXiv HTML/source reports four decimals: four values
agree, while Recall@5 and Recall@100 differ by `+0.0004` and `-0.0003`, respectively
(independent minus paper). An unmodified BM25 top-1,000 pool has macro candidate recall
0.981 (95.5% micro coverage), but BM25 does not consistently place
constraint-satisfying documents near the top. Reranking that unchanged pool with a
pinned text-only neural model helps substantially; exact constraint-aware ordering still
exposes a large same-pool upper-bound gap.

| Method | Recall@100 | ceiling-normalized R@100 | nDCG@20 | Macro candidate recall |
|---|---:|---:|---:|---:|
| Paper BM25 (Table 2) | 0.837 | - | 0.785 | - |
| Independent BM25 | **0.837** | 0.875 | 0.785 | 0.981 |
| BM25 pool + Qwen3-Reranker-4B | **0.904** | 0.955 | **0.886** | 0.981 |
| BM25 pool + oracle-decomposed exact ordering | **0.927** | 0.984 | **0.991** | 0.981 |
| BM25 + atomic expansion + exact ordering | **0.936** | **0.993** | **0.998** | **0.989** |
| Exact Boolean oracle | 0.942 | 1.000 | 1.000 | 1.000 |

Raw Recall@100 cannot reach 1.0 for queries with more than 100 relevant documents. The
ceiling-normalized measure accounts for that structural cutoff.

On the identical unmodified BM25 pool, Qwen improves Recall@100 by `+0.0667` (paired,
template-stratified 95% bootstrap interval `[+0.0561, +0.0774]`) and nDCG@20 by
`+0.1009` (`[+0.0867, +0.1152]`). Exact constraint-aware ordering remains `+0.0239`
Recall and `+0.1052` nDCG@20 higher. Qwen therefore closes about 74% of the verifier's
recall gain, but only 49% of its nDCG@20 gain.

## The aggregate hides an operator crossover

The neural improvement is not uniform across query templates:

| LIMIT+ template | BM25 nDCG@20 | Qwen nDCG@20 | Exact nDCG@20 | Pool recall |
|---|---:|---:|---:|---:|
| `A AND B` | 0.992 | 0.993 | 1.000 | 1.000 |
| `A AND B AND C` | **0.994** | **0.845** | 1.000 | 1.000 |
| `(A AND B) NOT C` | 0.175 | 0.716 | 0.975 | 0.905 |
| `A NOT B` | 0.530 | 0.682 | 0.980 | 0.980 |

Under LIMIT+'s balanced 100-query-per-template construction, the two negation templates
account for 98.2% of the **signed net macro** Qwen-minus-BM25 nDCG@20 improvement. Over
the 500 non-negation queries, the mean gain is only `+0.0025`, and its paired interval
crosses zero. Conversely, Qwen underperforms BM25 on the 100-query
three-way-conjunction slice.

This is an observed benchmark failure mode, not a causal result: operator type and gold
cardinality are confounded by LIMIT+'s construction.

## Experimental design

- **Data integrity:** the 50,000-document LIMIT corpus and 700-query LIMIT+ release are
  pinned by revision and SHA-256.
- **Boolean audit:** an independent parser and executor reproduce all 700 released gold
  sets exactly.
- **Paper-compatible BM25:** NLTK tokenization, `BM25Okapi(k1=1.5, b=0.75,
  epsilon=0.25)`, canonical `_id` document IDs, and the released NumPy quicksort tie
  behavior.
- **Portable validation profile:** a separate `corpus_ordinal` mode resolves equal BM25
  scores by corpus position and is checked end to end on both Ubuntu and Windows. It is
  not used for the reported metrics or Qwen candidate pool.
- **Retrieval-derived neural pool:** Qwen sees only query and document text and reranks the unchanged
  BM25 top-1,000. No gold documents are injected.
- **Strict budgets:** every final ranking contains at most 100 unique IDs; candidate and
  output budgets are logged separately.
- **Uncertainty:** paired query bootstrap, stratified by all seven templates, with 10,000
  draws and fixed seed `20260804`.
- **Auditability:** the Qwen run uses 700 resumable query checkpoints with a pinned model
  revision and a fingerprint over the complete prompt/scoring contract.
- **Release consistency:** the PDF reads the committed CSV/JSON results and fails closed
  if overlapping CPU, Qwen, or per-query evidence disagrees; regression tests freeze the
  headline values.

The exact verifier is deliberately information-advantaged: it uses the released
template plan and attributes parsed from LIMIT+'s literal text. It never reads qrels,
but it is a diagnostic upper bound, not a deployable or information-matched competitor
to Qwen.

The Qwen experiment is a same-pool diagnostic control. It is not a reproduction of the
paper's Table 3 neural configuration or candidate distribution.

## Reproduce the CPU experiment

Python 3.12 is the tested version. The compact results are committed, so cloning the
repository is sufficient to inspect the study.

```bash
git clone https://github.com/RenataLi/limit-plus-independent-followup.git
cd limit-plus-independent-followup
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell or
`source .venv/bin/activate` on Linux, then run:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[brief]"
python scripts/download_data.py --data-dir data/raw
python -m limitplus audit --data-dir data/raw
python -m limitplus run --data-dir data/raw --output-dir results/reproduction_cpu_portable --tie-break corpus_ordinal
python scripts/verify_cpu_reproduction.py --reference-dir results/corpus_ordinal_reference --reproduction-dir results/reproduction_cpu_portable
python scripts/build_brief.py
```

This recommended command validates the complete CPU pipeline against a separate
cross-platform reference. The verifier compares the fresh audit, all aggregate tables,
bootstrap intervals, atomic summary, protocol metadata, and generated output structure;
it ignores only operating-system, Python patch-version, and timing fields. The
symbolic/lexical benchmark takes roughly one minute on the recorded machine.

The headline and Qwen results use `paper_quicksort`, which preserves the source paper's
literal `np.argsort(-scores)` path. Equal-score ordering in that unstable sort can depend
on NumPy's CPU-dispatched SIMD kernel even with the pinned NumPy version. Exact replay of
`results/main` therefore requires a matching runtime/SIMD profile; the recorded run used
NumPy's AVX-512 sorting dispatch. The portable profile tests the same pipeline without
redefining the reported experiment. Tests cover small synthetic corpora and consistency
of the committed result artifacts; they require no raw-data download:

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs these tests and rebuilds the PDF on both Ubuntu and Windows. A
separate Ubuntu/Windows matrix downloads the pinned benchmark, reruns the complete
portable CPU profile, and invokes the same strict verifier. The multi-hour Qwen run
remains an explicit GPU reproduction step rather than a CI job.

See the [reproduction guide](docs/REPRODUCIBILITY.md) for Windows/Linux details and the
full neural command.

## Reproduce the neural evaluation

The recorded run used an RTX 5090, BF16/SDPA, batch size 64, and approximately 19 GB of
VRAM. It scored 700,000 pairs in 2.88 cumulative scoring hours. Install the pinned GPU
environment after downloading and auditing the raw data:

```bash
python -m pip install -r requirements-qwen.txt
python scripts/run_qwen_reranker.py \
  --data-dir data/raw \
  --bm25-run results/reproduction_cpu_paper/runs/full_query_bm25.jsonl.gz \
  --pool-k 1000 \
  --batch-size 64 \
  --max-length 384 \
  --output-dir results/reproduction_qwen
```

Before neural scoring, reproduce the `paper_quicksort` CPU profile as described in the
guide. The reranker validates the exact 700 x 1,000 ordered candidate IDs against the
recorded logical SHA-256 fingerprint
`11eaac068169f7355dfc080620ed138a5866cc9e951513c7c5650eaec8993748` and fails before
loading the model if the SIMD-dependent tie order differs. The pool itself is not
redistributed.

The frozen model revision is
`22e683669bc0f0bd69640a1354a6d0aebcfeede5`; the complete protocol fingerprint is
`283963be9bc46420fd98bb703e3cc4c1ccc5885bdcc554a3de3b2a1caae2e81c`.

## Repository map

- `artifacts/limit_plus_independent_followup_brief.pdf` - two-page reviewer-oriented summary;
- `artifacts/interactive/` - standalone HTML comparisons;
- `notes/METHODS_AND_RESULTS.md` - detailed methods, findings, and limitations;
- `notes/QWEN_PROTOCOL.md` - full neural protocol and operator-level diagnosis;
- `src/limitplus/` - data validation, retrieval, logic, metrics, and pipelines;
- `scripts/` - pinned download, CPU verification, Qwen evaluation, and PDF generation;
- `tests/` - synthetic unit tests and committed-result consistency checks;
- `results/main/` - compact reported paper-quicksort symbolic/lexical result tables;
- `results/corpus_ordinal_reference/` - compact cross-platform CPU validation reference;
- `results/qwen4b_paper_pool1000_full700/` - complete per-query neural metrics and
  frozen summary.

Raw benchmark data, model weights, scored run files, candidate pools, logs, and score
checkpoints are not stored in Git.

## Scope and limitations

- The study evaluates the released synthetic LIMIT+ grammar; it does not establish
  generalization to QUEST, natural-language compositional search, or open-world retrieval.
- Confidence intervals describe variation over the 700 released benchmark queries, not
  a broader population guarantee.
- Only one neural checkpoint and one frozen prompt/scoring protocol are evaluated.
- The exact verifier has privileged structural access and should be interpreted as an
  analytical ceiling.
- The hybrid atomic-expansion result has a larger retrieval budget than the unchanged
  BM25-pool comparison and is reported as a secondary result.

## Data, attribution, and licensing

- Source paper: [Degenhart et al., _Reproducing Complex Set-Compositional Information
  Retrieval_](https://arxiv.org/abs/2605.03824).
- Source implementation and LIMIT+ release:
  [`informagi/Complex-Set-Compositional-IR`](https://github.com/informagi/Complex-Set-Compositional-IR),
  pinned at `0a4105a328474d4a4c58b8e4fc613ec05c59fc22`.
- LIMIT corpus: [`orionweller/LIMIT`](https://huggingface.co/datasets/orionweller/LIMIT),
  CC BY 4.0, pinned at `215834026c13176e520b3bc9d0a055099537ef99`.
- Neural model: [`Qwen/Qwen3-Reranker-4B`](https://huggingface.co/Qwen/Qwen3-Reranker-4B),
  Apache 2.0.

At the pinned upstream revision, the LIMIT+ repository has no root license. This
repository therefore redistributes neither the raw LIMIT+ files nor the source paper.
The independently written code is MIT-licensed; third-party data, model weights, and
papers remain under their respective terms. See
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the complete attribution boundary.
