# Candidate Generation or Constraint Verification?

## A controlled follow-up on LIMIT+

**Status:** complete symbolic/lexical experiment and complete 700-query neural reranker
experiment.
**Source:** Degenhart et al., *Reproducing Complex Set-Compositional Information
Retrieval*, SIGIR 2026, Table 2 and Table 3.

### Question

The paper concludes from very strong reranking results on a curated candidate set that
the main bottleneck in set-compositional retrieval is first-stage candidate generation.
That diagnostic experiment gives each reranker every gold document plus five hard and
five easy BM25 negatives. This follow-up instead evaluates an unmodified first-stage
retrieval pool, allowing candidate coverage and within-pool ranking to be measured
separately.

I test a sharper counterfactual:

> If we keep an unmodified BM25 top-1,000 pool and use an exact
> constraint-aware verifier, how much of the remaining error is really caused by
> candidate generation?

### Protocol

1. Pin the full LIMIT corpus and the released 700-query LIMIT+ file by revision and
   SHA-256.
2. Parse corpus attributes independently and execute all seven released Boolean
   templates. Require exact equality with embedded gold for every query before running
   retrieval.
3. Reimplement the paper's BM25 setup: case-sensitive NLTK tokenization, text field only,
   BM25Okapi `k1=1.5`, `b=0.75`, `epsilon=0.25`, `_id` document identifiers, and
   the released `np.argsort(-scores)` quicksort behavior. This unstable tie ordering can
   vary with NumPy's CPU-dispatched SIMD sorting kernel even when scores are identical.
   Stable corpus-order ties are retained as a cross-platform validation profile rather
   than substituted into the primary run.
4. Compare:
   - full-query BM25;
   - hard AND/OR composition of truncated positive-atom BM25 lists, with exact lexical
     filtering for NOT;
   - exact constraint verification over an unmodified BM25 top-1,000 pool;
   - the same pool expanded with atomic top-1,000 lists, followed by verification;
   - exact Boolean oracle.
5. Cap every final ranking at 100 unique IDs. Report candidate coverage separately.
6. Use a paired, template-stratified bootstrap with 10,000 draws.

No method injects gold documents. The verifier uses the template metadata and literal
attribute facts parsed from the corpus; it is therefore an oracle-decomposed symbolic
hybrid, not an automatic semantic system.

### Integrity gates

- 50,000 unique corpus IDs; 1,848 parsed attributes.
- 700 unique query IDs; exactly 100 instances of each template.
- zero unknown qrel IDs and zero duplicate gold IDs.
- canonical template rendering matches 700/700 query strings.
- independently executed AND/OR/NOT sets match 700/700 released gold sets with zero
  false positives and zero false negatives.

### Results

| Method | R@100 | normalized R@100 | nDCG@20 | candidate recall |
|---|---:|---:|---:|---:|
| Paper BM25 | 0.837 | - | 0.785 | - |
| Independent BM25 | 0.837 | 0.875 | 0.785 | 0.981 (top 1,000) |
| Same BM25 pool + Qwen3-Reranker-4B | 0.904 | 0.955 | 0.886 | 0.981 |
| Hard atomic AND/OR composition + exact NOT, L=1,000 | 0.639 | 0.664 | 0.667 | 0.714 |
| BM25 top-1,000 + exact verifier | 0.927 | 0.984 | 0.991 | 0.981 |
| BM25 + atomic expansion + verifier | **0.936** | **0.993** | **0.998** | **0.989** |
| Boolean oracle | 0.942 | 1.000 | 1.000 | 1.000 |

The exact released sorting path gives Recall@100 `0.83693`, nDCG@20 `0.78533`, and
nDCG@100 `0.80129`. At the three-decimal precision typeset in the paper PDF, all six
Table 2 BM25 metrics agree. The arXiv HTML/source reports four decimals: four agree,
while Recall@5 and Recall@100 differ by `+0.0004` and `-0.0003` (independent minus
paper). The stable-tie ablation changes the row, so the paper-compatible path preserves
the released tie behavior.

For exact verification versus BM25 on the identical top-1,000 pool and retrieval budget,
the paired delta is:

- Recall@100: **+0.0906**, 95% CI **[+0.0807, +0.1008]**;
- ceiling-normalized Recall@100: **+0.1092**, CI **[+0.0985, +0.1204]**;
- nDCG@20: **+0.2061**, CI **[+0.1956, +0.2169]**.

All three bootstrap distributions are positive in 10,000/10,000 draws.
The higher-budget hybrid remains a secondary result: it uses the full-query top-1,000
plus up to three positive atomic top-1,000 lists and reaches R@100=0.936.

### What failed, and why that matters

Naively decomposing a query, retrieving each positive atom independently, and
hard-composing the truncated top-L lists (intersection for AND, union for OR, plus an
exact NOT filter) is
worse than full-query BM25 at every tested L. The new per-attribute audit shows the
mechanism directly: among 1,072 used attributes at L=1,000, positive-atom-occurrence-
weighted recall is only 0.704 while its ceiling-normalized counterpart is 0.995.
The median posting has size 2 but the mean is 1,252 and the 90th percentile is 2,687;
common attributes are structurally too large for a 1,000-item atomic cutoff. A document
satisfying a conjunction can therefore be pruned even when the joint query ranks it
well. Decomposition does not remove candidate loss; it can worsen it by pruning early.

By contrast, the full-query BM25 top-1,000 pool has macro candidate recall 0.981 (95.5%
micro coverage). Exact verification of that same pool raises R@100 from 0.837 to 0.927
without adding a single candidate. Atomic expansion recovers another 0.8 points and leaves most
remaining candidate loss in the `A AND B BUT NOT C` template: its pool coverage rises
from 0.905 to 0.937, while other major template groups reach approximately 0.99-1.00.

### Interpretation

The defensible conclusion is narrow:

> On LIMIT+'s synthetic literal grammar, candidate generation is not the sole source of
> top-100 error. BM25 top-1,000 already has near-complete coverage; identifying documents
> that satisfy the complete Boolean constraint inside that noisy pool accounts for a
> large, statistically clear part of top-100 error. Premature hard composition of atomic
> lists is harmful, while retrieve broadly and verify symbolically is near-oracle.

This complements rather than invalidates the paper. The paper correctly shows that a
reranker can succeed when every gold item is supplied. The follow-up evaluates a
different, retrieval-derived candidate distribution on LIMIT+ so that
ranking/verification and candidate generation can be measured separately.

The full neural result provides the missing non-symbolic control. Qwen sees only query
and document text and recovers 0.0667 Recall@100 and 0.1009 nDCG@20 over BM25 on the
unchanged pool. It therefore captures roughly 74% of the exact verifier's recall gain,
but only 49% of its nDCG@20 gain. A substantial constraint-ranking gap remains even
after strong neural reranking.

### Limitations

- Oracle decomposition comes from released template metadata.
- Exact verification exploits the controlled, explicit attribute language of LIMIT+.
- Same-pool verification matches candidate depth, not compute or information access;
  the verifier has privileged access to parsed explicit attributes.
- Cardinality and template type are confounded by dataset construction; subgroup
  differences are descriptive, not causal.
- LIMIT+ is synthetic. Results do not imply a general Boolean parser or verifier for
  natural language and world knowledge.
- Qwen is one pinned reranker and one fixed instruction; the result does not rank model
  families or establish that the residual gap is irreducible by neural methods.
- The reproduction repository has no explicit root license; do not redistribute its
  ready JSONL without clarification.

### Full Qwen3-Reranker-4B experiment

The full neural control reranks the same unmodified BM25 top-1,000 pool with pinned
Qwen3-Reranker-4B, using only query and document text. This is a diagnostic same-pool
control, not a reproduction of the paper's Table 3 model or candidate distribution. The pool, final cutoff, and
evaluation are identical for BM25, Qwen, and exact-verifier ordering; no gold document
is injected.

Protocol details:

- model revision `22e683669bc0f0bd69640a1354a6d0aebcfeede5`;
- protocol fingerprint `283963be9bc46420fd98bb703e3cc4c1ccc5885bdcc554a3de3b2a1caae2e81c`;
- BF16, PyTorch SDPA, batch size 64, maximum length 384;
- all 700 queries and 700,000 query-document pairs;
- zero input truncation under the independently checked 336-token content budget;
- strict per-query checkpoint validation against the exact ordered BM25 pool.

| Identical candidate pool | R@100 | normalized R@100 | nDCG@20 | nDCG@100 |
|---|---:|---:|---:|---:|
| BM25 | 0.8369 | 0.8746 | 0.7853 | 0.8013 |
| Qwen3-Reranker-4B | 0.9036 | 0.9548 | 0.8862 | 0.8979 |
| Exact verifier / oracle ordering | 0.9275 | 0.9839 | 0.9914 | 0.9859 |

Paired, template-stratified 10,000-draw bootstrap over the full benchmark:

- Qwen minus BM25 Recall@100: **+0.0667**, 95% CI **[+0.0561, +0.0774]**;
- Qwen minus BM25 nDCG@20: **+0.1009**, CI **[+0.0867, +0.1152]**;
- exact verifier minus Qwen Recall@100: **+0.0239**, CI **[+0.0195, +0.0288]**;
- exact verifier minus Qwen nDCG@20: **+0.1052**, CI **[+0.0949, +0.1158]**.

Every delta is positive in 10,000/10,000 draws. The template breakdown localizes the
residual problem. Exact minus Qwen nDCG@20 is +0.259 on `(A AND B) NOT C`, +0.298 on
`A NOT B`, and +0.155 on `A AND B AND C`, versus 0.000-0.013 on the simpler positive
templates. Qwen makes the largest improvement over BM25 on `(A AND B) NOT C`, but still
leaves a large gap to exact exclusion-aware ordering. It also degrades the three-way
conjunction from BM25 nDCG@20 0.994 to 0.845, despite complete candidate coverage.

The aggregate neural gain must not be described as uniform. Under LIMIT+'s balanced
100-query-per-template mix, the two negation templates account for 98.2% of the signed
net macro nDCG@20 improvement. Over the 500 non-negation queries, Qwen minus BM25
nDCG@20 is only +0.0025 with a paired interval `[-0.0104, +0.0158]`. Operator type and
gold cardinality are confounded by benchmark construction; this observed crossover
motivates an operator-aware follow-up but does not identify the causal mechanism.

The 700 checkpoints contain finite scores for exactly the unchanged 1,000 candidates
per query and reconstruct the BM25 pool exactly. The full run completed in 2.88
cumulative scoring hours across resumable checkpoints on an RTX 5090 at 67.51 pairs/s.
These results refine the paper's bottleneck diagnosis: candidate generation matters,
but neither BM25 nor a strong text-only reranker exhausts the available
constraint-verification signal inside a retrieval-derived pool.
