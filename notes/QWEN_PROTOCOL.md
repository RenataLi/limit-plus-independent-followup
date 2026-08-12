# Qwen3-Reranker-4B full same-pool benchmark

Status: complete, all `700` released LIMIT+ queries and `700,000` query-document pairs.

This is a diagnostic control on an unmodified first-stage BM25 candidate pool, not a
reproduction of the paper's Table 3 Qwen setup or candidate distribution.

## Frozen protocol

- Model: `Qwen/Qwen3-Reranker-4B` (Apache-2.0).
- Revision: `22e683669bc0f0bd69640a1354a6d0aebcfeede5`.
- Candidate set: unchanged recorded paper-quicksort full-query BM25 top-1,000; no gold
  injection. The exact 700 x 1,000 ordered pool is fingerprinted but not redistributed
  (logical SHA-256 `11eaac068169f7355dfc080620ed138a5866cc9e951513c7c5650eaec8993748`).
- Inputs: original query text and complete document text only.
- Instruction: literal satisfaction of the complete AND/OR/NOT information need.
- Scoring: final-token `logit(yes) - logit(no)`.
- Protocol fingerprint: `283963be9bc46420fd98bb703e3cc4c1ccc5885bdcc554a3de3b2a1caae2e81c`.
- Maximum length 384; BF16; PyTorch SDPA; batch size 64; TF32 matrix multiply.
- Hardware: NVIDIA RTX 5090 32 GB.
- Checkpoints: one atomic JSON file per query with strict model, revision, fingerprint,
  batch, length, and exact ordered-pool validation.
- Truncation audit: the independently longest corpus-document/longest-query stress pair
  uses 274 content tokens versus the 336-token content budget. No benchmark pair is
  truncated.

## Full result

| Method on identical pool | Recall@100 | Normalized R@100 | nDCG@20 | nDCG@100 | Pool recall |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.8369 | 0.8746 | 0.7853 | 0.8013 | 0.9812 |
| Qwen3-Reranker-4B | 0.9036 | 0.9548 | 0.8862 | 0.8979 | 0.9812 |
| Exact verifier | 0.9275 | 0.9839 | 0.9914 | 0.9859 | 0.9812 |

Paired, template-stratified 10,000-draw bootstrap over the complete released benchmark:

- Qwen minus BM25 Recall@100: `+0.0667`, 95% CI `[+0.0561, +0.0774]`.
- Qwen minus BM25 normalized Recall@100: `+0.0802`, CI `[+0.0685, +0.0919]`.
- Qwen minus BM25 nDCG@20: `+0.1009`, CI `[+0.0867, +0.1152]`.
- Qwen minus BM25 nDCG@100: `+0.0966`, CI `[+0.0832, +0.1104]`.
- Exact verifier minus Qwen Recall@100: `+0.0239`, CI `[+0.0195, +0.0288]`.
- Exact verifier minus Qwen nDCG@20: `+0.1052`, CI `[+0.0949, +0.1158]`.
- Exact verifier minus Qwen nDCG@100: `+0.0879`, CI `[+0.0799, +0.0962]`.

Every interval above excludes zero; every corresponding bootstrap distribution is
positive in `10,000/10,000` draws.

## Template diagnosis

| Template | BM25 nDCG@20 | Qwen nDCG@20 | Exact nDCG@20 | Pool recall |
|---|---:|---:|---:|---:|
| A | 0.922 | 0.990 | 0.990 | 0.990 |
| A OR B | 0.934 | 0.983 | 0.996 | 0.995 |
| A OR B OR C | 0.951 | 0.993 | 0.999 | 0.998 |
| A AND B | 0.992 | 0.993 | 1.000 | 1.000 |
| A AND B AND C | 0.994 | 0.845 | 1.000 | 1.000 |
| (A AND B) NOT C | 0.175 | 0.716 | 0.975 | 0.905 |
| A NOT B | 0.530 | 0.682 | 0.980 | 0.980 |

Qwen nearly closes the verifier gap for atoms and OR templates. It makes its largest
absolute improvement over BM25 on `(A AND B) NOT C`, yet exact verification remains
another `+0.259` nDCG@20 higher. The residual gap is also `+0.298` for `A NOT B`.
Surprisingly, Qwen degrades the three-way conjunction despite complete pool coverage;
this rules out candidate loss as the explanation for that slice.

The aggregate improvement is highly non-uniform. Under LIMIT+'s balanced
100-query-per-template mix, the two negation templates contribute 98.2% of the signed
net macro nDCG@20 improvement (and 91.8% for nDCG@100). On the 500 non-negation queries,
Qwen minus BM25 nDCG@20 is only `+0.0025`, with a paired interval
`[-0.0104, +0.0158]`. Operator and gold cardinality are confounded by benchmark
construction, so this is an observed failure pattern, not a causal operator effect.

Across all queries, Qwen recovers approximately 74% of the exact verifier's Recall@100
gain over BM25 and 49% of its nDCG@20 gain. The result therefore supports a mixed
bottleneck diagnosis: stronger neural ranking helps substantially, while explicit
constraint satisfaction still accounts for a large share of early-ranking quality.

## Integrity and runtime

- `700/700` checkpoint IDs, exactly `0` through `699`.
- Every checkpoint has 1,000 unique candidates, finite scores, and reconstructs the
  corresponding ordered BM25 pool exactly.
- The generated run has 700 unique queries; the metric file has 2,100 unique rows
  (`700 x 3` methods); aggregate metrics reproduce `summary.json` exactly.
- No traceback, OOM, or CUDA error occurred. Failed Hugging Face update checks in stderr
  were non-fatal; the pinned local revision loaded successfully.
- Aggregate scoring time: 10,368.8 seconds (2.88 hours), 67.51 pairs/s.

## Limits on the claim

- This is one pinned reranker and one fixed instruction, not a model-family comparison.
- The exact verifier has privileged access to oracle decomposition and parsed explicit
  attributes; it is not compute- or information-matched to Qwen.
- LIMIT+ is synthetic and literal. The result does not establish performance on QUEST,
  open-world retrieval, or naturally occurring compositional queries.
