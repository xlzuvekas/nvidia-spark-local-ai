# Autoresearch controller post-cutoff hardening — 2026-08-28

## Status and scope

This is the implemented post-cutoff hardening record, not a change to the
admission-expired frozen campaign. Its fourteen cell plans, policy, queue, raw
summary, and frozen executable-harness identity remain immutable. The exact
legacy campaign is content-sealed and refuses run and checkpoint entry points;
its public summary path is read-only and byte-preserving.

The implementation is complete:

- fresh frozen campaigns use schema 3 and require a chained mode-0600
  `admissions.jsonl`;
- summary derivation, evidence snapshots, and evidence export hold the exact
  campaign lock;
- cutoff is a typed terminal reason with an effective `expired` status;
- the exporter publishes a strict scalar-only `autoresearch_campaign` bundle;
  and
- the exact schema-2 legacy campaign is published with
  `sealed_legacy_unjournaled` provenance, zero invented admission records, and
  its exact sealed blocker state.

The final gates passed Python compilation, all 894 repository tests,
deterministic full temporary export, normal verification, and exact staged
verification. Admission/controller hardening landed from `bfe8f86` through
`264ac96`, resume coverage in `47d4fa7`, read-only snapshots in `f5db2b7`,
campaign export in `1af6e83`, and the tracked bundle in `b668306`.

The first campaign-schema-2 run/admission attempt safely stopped before
measurement, without launching a cell, container, or request, but it exposed
four reporting gaps that the implementation now closes:

1. a fresh pre-journal blocker existed only in `summary.json`, and the public
   summarizer could replace it with `planned`;
2. the public summarizer did not own the campaign lock;
3. a cutoff at an active boundary could become a generic measurement
   termination; and
4. the evidence exporter published all cell plans but no typed campaign-level
   admission or cutoff record.

A second pristine invocation after the pair budget expired again launched
nothing and added time, swap, and memory blocker codes to the same legacy
summary. Both observations are preserved in documentation, but neither has a
durable admission journal. Do not manufacture a retroactive admission record
from either one. Fresh schema-3 campaigns now record launch-capable admission
attempts exactly, while the current legacy summary uses the non-destructive
compatibility path.

## Authority model

Keep execution authority deliberately narrow:

| Record | Controls whether work launches? | Purpose |
| --- | --- | --- |
| Live preflight | Yes | Current time, memory, swap, and harness admission |
| Controller journal | Yes | Sole durable authority for campaign execution state |
| Admission journal | No | Append-only scalar provenance for each launch decision |
| `summary.json` | No | Lock-consistent derived cache for humans and CLI output |
| Tracked evidence | No | Deterministic sanitized projection for publication |

Recovery and exact-owned cleanup remain ahead of all new admission checks.
Checkpoint enforcement remains ahead of host admission at stable boundaries.
Neither path may launch inference or be skipped merely because the cutoff has
passed.

## Durable admission journal

Fresh schema-3 campaigns use a mode-0600 `admissions.jsonl` under the frozen
campaign directory. Under the campaign lock, append and durably flush one
record immediately before each new calibration, screen, or confirmation pair
is either admitted or denied. A failed append launches nothing.

Use a strict schema with contiguous sequence numbers, a chained integrity
digest, and exact frozen-campaign/controller-prefix bindings. Each record
contains only these classes of fields:

- schema version, sequence, previous-record digest, and record digest;
- campaign ID, campaign integrity hash, preview hash, and policy hash;
- controller event count and canonical controller-prefix hash;
- target kind, candidate ID, and pair index;
- observed and cutoff timestamps;
- remaining and required-remaining seconds;
- observed and required `MemAvailable`;
- observed and maximum starting swap;
- observed harness hash/count and a harness-match boolean;
- the complete ordered blocker list; and
- `admitted`, `blocked_environment`, or `cutoff` outcome.

Freeze blocker order as harness, time, swap, then memory. Retain every
simultaneous blocker even when a higher-priority audit or pressure condition
determines an active campaign's terminal reason. Do not include paths,
processes, commands, environment, run nonces, request identifiers, prompts,
outputs, reasoning, tool payloads, or logs.

Each `run_campaign` invocation that reaches a new calibration, screen, or
confirmation launch boundary recomputes admission from its current clock,
`/proc/meminfo`, and harness identity. Terminal replay, checkpoint pauses, and
cleanup- or reconciliation-only returns append no new admission because none
can launch inference. A prior record never authorizes a launch. Before any
required append, the implementation strictly verifies the existing chain. A
malformed chain raises `CampaignPlanningError` before reconciliation or
controller mutation and launches nothing; it does not create an audit terminal
transition.

## Exact cutoff behavior

The inclusive boundary is exact: 4,930.0 seconds remaining admits a pair;
anything below it does not. The single typed check runs immediately before the
durable pair-launch boundary; no later generic calibration or search-pair time
check can reclassify cutoff as a measurement failure.

Use an explicit `cutoff` failure kind when cutoff is the decisive blocker for
an already active controller. A fresh campaign whose decisive denial is short
time has effective status `expired`, controller status `planned`, and no
measurement claim. An active campaign whose decisive denial at the next pair
boundary is cutoff appends a terminal cutoff transition and also reports
`expired`. A completed or otherwise terminal controller never resumes.

When cutoff and an audit or pressure blocker occur together, record every code
in the admission entry and apply one frozen precedence. Harness audit, swap,
and memory safety remain stronger causal labels than cutoff. For a fresh
controller, that mixed denial stays `blocked_environment`/`planned`, but cutoff
makes it non-resumable. For an active controller, it becomes `terminated` with
the stronger safety reason rather than `expired`. Do not hide a safety failure
merely because the clock also expired. Ownership failures are handled earlier
during recovery and are not live-admission blocker codes.

## Summary schema and locking

The summarizer is split into:

- a private lock-assuming helper used by `run_campaign`; and
- for normal campaigns, a public `summarize_campaign` that loads the campaign,
  acquires its exact lock, reloads under that lock, derives one snapshot, and
  writes atomically.

This avoids a mixed controller/admission snapshot without making internal run
paths recursively acquire the same non-blocking lock. For normal campaigns,
the summary remains a derived cache built from strict controller/admission
journals and the integrity-verified calibration record; it is never execution
authority. The sealed legacy summary is the explicit read-only compatibility
exception.

Schema numbers are independent: a fresh freeze uses campaign schema 3,
admission records use schema 1, and normal summaries use schema 2. The exact
legacy freeze and summary remain schema 2 and schema 1 respectively.

Normal campaign summaries contain:

- effective `status`;
- `controller_status` and `controller_phase`;
- terminal reason;
- calibration-recorded boolean and next pair index;
- candidate-decision mapping;
- admission count and exact last-admission projection;
- campaign, preview, policy, controller-prefix, and last-admission hashes.

Use these status semantics:

| Situation | Effective status | Controller status | Resumable? |
| --- | --- | --- | --- |
| Frozen, never attempted | `planned` | `planned` | Yes |
| Fresh non-cutoff host/harness denial | `blocked_environment` | `planned` | Yes, after a new live preflight |
| Fresh decisive cutoff denial | `expired` | `planned` | No new work |
| Fresh mixed cutoff + safety denial | `blocked_environment` | `planned` | No; cutoff cannot reopen |
| Active host/safety denial | `terminated` | `terminated` | No |
| Active decisive cutoff denial | `expired` | `terminated` | No |
| Queue completed | `complete` | `complete` | No |

`checkpoint_required` remains an in-memory CLI response. It writes neither
summary nor admission history. Acknowledging it changes only the separate
private checkpoint state.

For the exact sealed legacy campaign, public summarize acquires the existing
safe lock, verifies the complete sealed topology and content, and returns the
exact schema-1 `blocked_environment` summary without rewriting it. It can
create neither `admissions.jsonl` nor controller state; admission records
themselves use schema version 1. The campaign evidence bundle instead publishes
`sealed_legacy_unjournaled` provenance, zero admission records, and the exact
sealed summary's effective blocker state. No historical observation is
reconstructed as a typed admission record.

## Resume and reconciliation invariants

- A fresh non-cutoff environmental denial is resumable only after a new live
  preflight; the prior admission record is observational.
- A fresh cutoff denial never launches because the immutable deadline has
  passed.
- A terminal controller never resumes, regardless of later host state.
- Previously admitted raw-complete cell reconciliation remains permitted after
  cutoff because it launches no inference; schema-3 raw work without a valid
  preceding admission fails closed.
- An incomplete one-use cell is never relaunched; exact-owned recovery either
  proves a terminal raw cell or terminalizes the campaign.
- A checkpoint pause changes no controller, admission, cell, or summary file.
- Fresh host denial stays nonterminal, while the same denial at an active pair
  boundary remains campaign-terminal to avoid inheriting contaminated state.

## Campaign-level scalar evidence

The exporter uses a specialized `autoresearch_campaign` evidence kind rather
than forcing campaign state into one of the fourteen cell bundles. It exports
one campaign directory containing `manifest.json`, `controller.json`,
`admissions.json`, and `checksums.json`.

The manifest contains only frozen public identity and provenance:

- public campaign ID and frozen campaign/preview/policy hashes;
- creation and cutoff timestamps;
- planned cell count, baseline ID, suite ID, and ordered candidate identities;
- frozen harness hash and file count, plus each candidate's axis and delta
  hash;
- frozen schema and admission-journal requirement;
- effective status and provenance mode; and
- strict scalar-only sanitization flags.

The controller projection carries effective/controller status, phase, terminal
reason, calibration state, next pair index, ordered decision history, exact
event counts, and the controller-prefix hash. The admissions projection carries
the complete allowlisted record chain plus its effective tail state.

The admissions projection contains the allowlisted records above, never raw
journal bytes. For normal schema-2 and schema-3 campaigns, source export replays
controller and admission truth under the campaign lock and does not trust or
copy `summary.json`. The exact sealed legacy campaign is the sole exception:
its read-only snapshot verifies and reads the hash-sealed schema-1 summary for
compatibility status and blockers. The exporter preserves the existing
fourteen nonterminal cell bundles and adds one campaign bundle rather than
converting a blocked admission into a synthetic cell result.

Because the current legacy campaign has no admission journal and cannot launch
again, its typed bundle projects controller status `planned`, zero admission
records, and provenance mode `sealed_legacy_unjournaled`. Its effective outcome
is the exact sealed `blocked_environment` compatibility state with the three
sealed blocker codes; this is not a reconstructed live admission record and it
does not relabel the campaign `expired`. The documentation remains the detailed
account of both historical preflight observations.

Source validation rejects journal, prefix-binding, topology, and
missing/extra-cell failures before projection. The published verifier rejects
unknown fields/files, inconsistent scalar relationships, admission-chain/hash
errors, ordering changes, paths, secrets, and private payload surfaces.
Deterministic re-export and exact staged verification remain mandatory.

## Landed implementation

1. `bench/autoresearch_admission.py` owns typed observations, the strict
   chained journal, frozen bindings, and controller-prefix validation.
2. `bench/autoresearch_campaign.py` owns live admission, cutoff mapping,
   lock-consistent schema-2 summaries, the sealed-legacy compatibility path,
   and read-only snapshots.
3. `bench/autoresearch.py` owns categorical `FailureKind.CUTOFF` plus the
   controller event, decision, and terminal grammar; the numeric cutoff
   threshold lives in `bench/autoresearch_campaign.py`.
4. `bench/evidence.py` holds a fixed lock-protected source set while exporting
   the campaign bundle and fourteen cell bundles, and rejects both topology
   races and checksum-refreshed tampering.
5. The CLI retains failure exit status for `expired`, exit code 3 for a
   checkpoint pause, and refuses run or checkpoint entry for the sealed legacy
   campaign.

## Verification coverage

Regression coverage includes:

- blocker preservation across public summarize;
- the exact 4,930.0/4,929.999-second boundary;
- partial calibration and search-pair resumes require fresh duplicate
  admissions;
- fresh and active cutoff behavior;
- simultaneous blocker order and fresh mixed-denial classification;
- admitted and denied record schema, chaining, and frozen bindings;
- a valid stale admission record not deciding live admission, while a malformed
  chain prevents the required append and any launch;
- public summarize lock contention and internal non-recursive summarization;
- checkpoint pause writing no summary or admission record;
- previously admitted raw-complete reconciliation after cutoff with no new
  cell launch;
- deterministic campaign evidence export and verification;
- preservation of fourteen cell bundles while campaign count increases by
  one; and
- staged secret/path scans and refreshed-checksum tamper rejection.

The completed gate ran full unittest discovery and Python compilation.
Evidence coverage also includes offline fixtures, deterministic temporary
re-export, normal verification, exact staged verification, lock-contention and
source-set race injection, refreshed-checksum tamper rejection, and scalar
privacy scans.
