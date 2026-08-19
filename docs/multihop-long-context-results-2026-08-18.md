# Qwen3.6 two-hop long-context retrieval — 2026-08-18

## Result

Qwen3.6 35B-A3B UD-Q4_K_XL completed **9/10** exact two-hop retrieval
requests at measured prompt sizes through the 245,760-target tier. Its MTP2
variant also completed **9/10**. Both missed one of three requests at the
131,072-target tier; every other repetition passed, including the single
245,760-target request.

This is a narrow long-context retrieval result, not a general reasoning,
multi-document QA, or production-context-window claim. MTP2 did not reduce
latency for this prompt-dominated, short-answer fixture: summed case wall time
was **566.477 s** versus **527.899 s** for the baseline, a **7.308%**
increase. The two profiles emitted slightly different short completion totals,
so this is not a matched decode-throughput comparison.

## Matched runs

Both frozen plans resolve the benchmark code and suite to commit
`05cff1058fdcb049c8f1e1ee36c121db04a83a3c` (`Add multi-hop long-context
benchmark`). The baseline plan captured an empty Git status. The MTP2 plan
captured the same code revision with a non-empty worktree status, so the pair
is code-matched but only the baseline plan is literally a clean-tree capture.

The profiles use the same Qwen3.6 35B-A3B UD-Q4_K_XL main artifact, one
llama.cpp slot, the 262,144-token served context, temperature zero, no
warm-ups, and a 32-token completion cap. The accelerated profile enables the
artifact's native two-token MTP draft path.

## Per-tier outcomes and timing

Each lower tier has three measured requests; the 245,760-target tier has one.
Token columns are measured totals across those requests. “Case wall” is the
suite's measured wall time for the tier. TTFT is the median per request within
the tier, rather than a throughput estimate.

| Target | Requests | Baseline pass | MTP2 pass | Baseline prompt / completion tokens | MTP2 prompt / completion tokens | Baseline case wall | MTP2 case wall | Baseline median TTFT | MTP2 median TTFT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32,768 | 3 | 3/3 | 3/3 | 99,264 / 45 | 99,269 / 43 | 43.155 s | 45.292 s | 14.080 s | 14.881 s |
| 65,536 | 3 | 3/3 | 3/3 | 197,589 / 41 | 197,562 / 45 | 93.181 s | 98.207 s | 30.828 s | 32.447 s |
| 131,072 | 3 | 2/3 | 2/3 | 394,199 / 46 | 394,198 / 42 | 217.133 s | 234.212 s | 72.016 s | 77.733 s |
| 245,760 | 1 | 1/1 | 1/1 | 246,088 / 15 | 246,082 / 14 | 174.430 s | 188.767 s | 173.845 s | 188.202 s |
| **Total** | **10** | **9/10** | **9/10** | **937,140 / 147** | **937,111 / 144** | **527.899 s** | **566.477 s** | — | — |

The scalar request journal sums are consistent with that direction: combined
TTFT was 524.261 s for baseline and 563.358 s for MTP2, a 7.458% increase.
The tier medians and total wall time both worsen under MTP2. With only three
requests at the lower tiers and one at the upper tier, this shows a direction
for this fixture rather than a stable latency distribution or p95 result.

## MTP activity and interpretation

The accelerated server's cumulative lifetime counters show that MTP was active:
52 draft proposals produced 104 proposed draft tokens, of which 101 were
accepted (**97.115385%**). Accepted-token position counts were 52 and 49 at
positions zero and one; configured depth two was exercised and the reported
mean accepted length was 2.942308. The baseline correctly reported zero draft
and accepted tokens because speculation was not requested.

Those counters cover the complete persisted server lifetime, including the
first post-start request; they cannot be assigned to a context tier or a
specific measured request. Strong draft acceptance therefore does not imply a
prefill win. Here, the visible answers are only 14–46 tokens per tier while
the inputs approach the context limit, so TTFT and total case wall dominate.

## What this test does—and does not—validate

The schema-versioned
[multi-hop suite](../manifests/suites/llamacpp_multihop_long_context.toml)
uses a deterministic, nonce-derived two-relation path placed amid filler and
two decoy paths. The oracle accepts only the exact final visible key after the
source-to-relay-to-final lookup; it does not accept an intermediate relay,
extra prose, or a key returned only in hidden reasoning. This deliberately
tests a stricter retrieval chain than a single-key needle, while retaining a
bounded scalar oracle.

It remains one synthetic construction on one model artifact, runtime,
quantization, machine, temperature, and slot geometry. It does not measure
multi-hop reasoning generally, robustness to realistic documents, retrieval
augmentation, cache reuse, concurrent serving, long-form generation, or other
models. The one failure per profile is retained as a failed exact-oracle
outcome; this report intentionally does not reproduce generated prompts,
answers, reasoning, identifiers, or raw run locations.

## Evidence status

The [tracked scalar evidence archive](../evidence/README.md) now includes
sanitized, checksummed bundles for both measured configurations; its
[index](../evidence/index.json) accounts for their inclusion. The bundles retain
only allowlisted scalar summaries, validation outcomes, request measurements,
telemetry, and public artifact/runtime identities—never prompts, completions,
reasoning, request identifiers, or raw artifacts.

The archive is schema-, topology-, and checksum-verifiable with
`python3 sparkbench.py verify-evidence evidence`. Raw run artifacts remain
untracked.
