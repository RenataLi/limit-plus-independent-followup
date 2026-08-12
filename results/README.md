# Result artifacts

The committed files are compact evaluation artifacts without query or document text,
sufficient to inspect the reported aggregate and per-query metrics.

- `main/` contains the data audit, frozen run manifest, aggregate metrics, paired
  bootstrap intervals, and aggregate atomic-retrieval diagnostics.
- `corpus_ordinal_reference/` contains the compact deterministic reference used by the
  full Ubuntu/Windows CPU pipeline check. It is a portability control, not the source of
  the reported headline values.
- `qwen4b_paper_pool1000_full700/` contains the complete 700-query by three-method metric
  table and frozen protocol summary. The recorded candidate-pool fingerprint is
  documented, but the pool itself is not redistributed.

The CPU pipeline also generates candidate pools, scored run files, per-query symbolic diagnostics,
attribute-level diagnostics, and error-decomposition rows. Those files are excluded
from the public repository: some contain benchmark-derived text whose upstream license
is not explicit, while others are large and deterministically regenerable.

Raw LIMIT/LIMIT+ data, Qwen scores, and model weights are never stored here. Follow
[`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) to regenerate the complete
local output tree.
