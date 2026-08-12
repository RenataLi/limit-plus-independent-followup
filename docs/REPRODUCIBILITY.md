# Reproducibility guide

This guide reproduces the symbolic/lexical benchmark and, optionally, the full
Qwen3-Reranker-4B evaluation. The repository already contains compact result tables and
the two-page research brief; no GPU is needed to inspect the reported findings.

## 1. Clone and create an environment

Python 3.12 is the tested version. Python 3.11 or newer is supported by the base package.

```bash
git clone https://github.com/RenataLi/limit-plus-independent-followup.git
cd limit-plus-independent-followup
```

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[brief]"
```

Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[brief]"
```

Run the synthetic and committed-artifact consistency tests, then rebuild the PDF:

```bash
python -m unittest discover -s tests -v
python scripts/build_brief.py
```

## 2. Download and audit the pinned data

The downloader records the upstream revisions and verifies SHA-256 hashes for every
required file. Raw data are intentionally excluded from version control.

```bash
python scripts/download_data.py --data-dir data/raw
python -m limitplus audit --data-dir data/raw
```

The audit must report all 700 released LIMIT+ queries, 50,000 LIMIT documents, and exact
Boolean reconstruction of all 700 qrel sets before any result is accepted.

## 3. Validate the full CPU pipeline portably

Write reproductions to a new directory so the committed reference tables stay immutable:

```bash
python -m limitplus run \
  --data-dir data/raw \
  --output-dir results/reproduction_cpu_portable \
  --tie-break corpus_ordinal
python scripts/verify_cpu_reproduction.py \
  --reference-dir results/corpus_ordinal_reference \
  --reproduction-dir results/reproduction_cpu_portable
```

On Windows PowerShell, replace the backslash continuations with a single line. The run
writes protocol metadata, per-query metrics, aggregates, error accounting, and compressed
rankings under `results/reproduction_cpu_portable/`.

The verifier compares CSVs semantically, so line endings and harmless floating-point
serialization differences do not matter. It checks the complete data audit, all
committed aggregate and bootstrap rows, the atomic summary, every non-runtime protocol
field, and the presence/readability of the generated per-query tables and compressed
rankings. Only operating-system, Python patch-version, and timing fields are allowed to
vary.

The portable validation profile has these frozen aggregate values:

| Metric | Expected value |
|---|---:|
| Recall@100 | 0.8382643545 |
| ceiling-normalized Recall@100 | 0.8759775531 |
| nDCG@20 | 0.7856536047 |
| nDCG@100 | 0.8017155342 |

This profile orders equal BM25 scores by corpus position. It exists to test the full
pipeline deterministically across the tested Windows and Ubuntu CPU environments and is
not the source of the headline or Qwen numbers.

### Exact replay of the reported paper-compatible profile

The reported results preserve the source paper's literal unstable sorting expression:

```bash
python -m limitplus run \
  --data-dir data/raw \
  --output-dir results/reproduction_cpu_paper \
  --tie-break paper_quicksort
python scripts/verify_cpu_reproduction.py \
  --reference-dir results/main \
  --reproduction-dir results/reproduction_cpu_paper
```

NumPy `2.2.6` and NLTK `3.9.1` are frozen, but equal-score ordering from
`np.argsort(-scores)` can still depend on NumPy's CPU-dispatched SIMD sorting kernel.
The scores themselves are unchanged; tied order and candidate membership can differ.
Consequently, exact verification against `results/main` requires a matching runtime and
SIMD profile; the recorded run used NumPy's AVX-512 sorting dispatch. GitHub Actions
instead executes the strict `corpus_ordinal` reproduction on both Ubuntu and Windows.
This validates the full pipeline without silently changing the reported paper-compatible
experiment. The full neural run is not a CI job because it requires a large NVIDIA GPU
and several hours of scoring.

## 4. Reproduce the full neural evaluation

The recorded protocol requires an NVIDIA GPU with CUDA 12.8 and at least 24 GB VRAM.
The RTX 5090 run used about 19 GB peak VRAM and scored 700,000 query-document pairs in
2.88 cumulative scoring hours. CPU-only and Apple Silicon execution are outside the
tested protocol.

Install the pinned neural environment:

```bash
python -m pip install -r requirements-qwen.txt
```

First reproduce the reported paper-compatible CPU profile from Section 3, then rerank
that candidate pool:

```bash
python scripts/run_qwen_reranker.py \
  --data-dir data/raw \
  --bm25-run results/reproduction_cpu_paper/runs/full_query_bm25.jsonl.gz \
  --pool-k 1000 \
  --batch-size 64 \
  --max-length 384 \
  --output-dir results/reproduction_qwen
```

The runner verifies all 700,000 ordered candidate IDs against the recorded logical
SHA-256 fingerprint
`11eaac068169f7355dfc080620ed138a5866cc9e951513c7c5650eaec8993748` before loading the
model. A CPU-specific tie permutation therefore fails closed instead of silently
changing the neural experiment. The candidate pool itself is not redistributed.

Do not add `--per-template` for the complete 700-query run. The runner is resumable:
compatible query checkpoints are skipped, while a changed model, prompt, pool, batch
size, maximum length, or candidate ordering fails closed.

Compare `results/reproduction_qwen/summary.json` with the committed reference
`results/qwen4b_paper_pool1000_full700/summary.json`. Scientific metrics and paired
intervals should match; GPU loading and throughput fields may vary.

## 5. Preserve protocol boundaries

- Use a new output directory for every altered model, prompt, pool depth, token limit,
  tokenizer, or operator strategy.
- Do not inject qrels or gold documents into the retrieval-derived BM25 pool.
- Treat the exact verifier as an information-advantaged analytical ceiling, not a
  deployable baseline.
- Keep raw data, candidate pools, scored diagnostic runs, score checkpoints, model
  caches, and logs out of Git.
- Report changes to dependency versions because BM25 tie ordering is protocol-sensitive.
- Do not substitute the portable corpus-ordinal pool for the recorded Qwen pool; it is a
  separate CPU portability control.
