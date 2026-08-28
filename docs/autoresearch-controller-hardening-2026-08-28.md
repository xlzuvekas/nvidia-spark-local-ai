# Autoresearch controller post-cutoff hardening — 2026-08-28

## Status and scope

This is an implementation plan, not a change to the active frozen campaign.
Do not edit the campaign-bound Python or manifest tree before the 07:00 MST
cutoff or an explicit decision to abandon that campaign. Its fourteen cell
plans, policy, queue, and executable harness identity remain immutable.

The first schema-2 run/admission attempt safely stopped before measurement,
without launching a cell, container, or request, but it exposed four reporting
gaps:

1. a fresh pre-journal blocker exists only in `summary.json`, and the public
   summarizer can replace it with `planned`;
2. the public summarizer does not own the campaign lock;
3. a cutoff at an active boundary becomes a generic measurement termination;
   and
4. the evidence exporter publishes all cell plans but no typed campaign-level
   admission or cutoff record.

The current blocker is preserved in the existing summary and documentation.
Do not manufacture a retroactive admission record from it. The design below
records future launch attempts exactly and gives the current legacy summary a
non-destructive compatibility path.

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

Add a mode-0600 `admissions.jsonl` under the frozen campaign directory. Under
the campaign lock, append and durably flush one record immediately before each
new calibration, screen, or confirmation pair is either admitted or denied.
A failed append launches nothing.

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

Every `run_campaign` invocation must recompute admission from its current
clock, `/proc/meminfo`, and harness identity. It must never consult a valid old
admission record or `summary.json` to decide the live result. Before appending
the new observation, however, it must strictly verify the existing chain. A
malformed chain makes the required append fail and therefore launches nothing
as an audit failure. Prior records can never authorize a launch.

## Exact cutoff behavior

Preserve the inclusive boundary: exactly 4,930.0 seconds remaining admits a
pair; anything below it does not. Centralize that check immediately before the
durable pair-launch boundary. Remove or route the later generic time checks in
calibration and search through the same typed decision so time cannot be
reclassified as a measurement failure.

Use an explicit `cutoff` failure kind when cutoff is the decisive blocker for
an already active controller. A fresh campaign whose decisive denial is short
time has effective status `expired`, controller status `planned`, and no
measurement claim. An active campaign whose decisive denial at the next pair
boundary is cutoff appends a terminal cutoff transition and also reports
`expired`. A completed or otherwise terminal controller never resumes.

When cutoff and an audit or pressure blocker occur together, record every code
in the admission entry and apply one frozen terminal precedence. Harness
audit, swap, and memory safety remain stronger causal terminal labels than
cutoff. A combined denial therefore remains `terminated` with the stronger
reason rather than `expired`; do not hide a safety failure merely because the
clock also expired. Ownership failures are handled earlier during recovery and
are not live-admission blocker codes.

## Summary schema and locking

Split the current summarizer into:

- a private lock-assuming helper used by `run_campaign`; and
- a public `summarize_campaign` that loads the campaign, acquires its exact
  lock, reloads under that lock, derives one snapshot, and writes atomically.

This avoids a mixed controller/admission snapshot without making internal run
paths recursively acquire the same non-blocking lock. The summary remains a
cache and must be derived from strict journals rather than replayed as state.

Move the campaign summary to schema version 2 with at least:

- effective `status`;
- `controller_status` and `controller_phase`;
- terminal reason;
- calibration-recorded boolean and next pair index;
- ordered candidate decisions;
- admission count and exact last-admission projection;
- campaign, preview, policy, controller-prefix, and last-admission hashes.

Use these status semantics:

| Situation | Effective status | Controller status | Resumable? |
| --- | --- | --- | --- |
| Frozen, never attempted | `planned` | `planned` | Yes |
| Fresh host/harness denial | `blocked_environment` | `planned` | Yes, after a new live preflight |
| Fresh decisive cutoff denial | `expired` | `planned` | No new work |
| Active host/safety denial | `terminated` | `terminated` | No |
| Active decisive cutoff denial | `expired` | `terminated` | No |
| Queue completed | `complete` | `complete` | No |

`checkpoint_required` remains an in-memory CLI response. It writes neither
summary nor admission history. Acknowledging it changes only the separate
private checkpoint state.

For the current legacy campaign only, if both journals are absent and the
existing exact schema-1 summary says `blocked_environment`, public summarize
must preserve that summary rather than downgrade it or invent observation
fields. The next genuine launch attempt may create the first schema-2
admission record. The historical blocker remains only in the legacy raw
summary and documentation unless a new live observation reproduces it; it
cannot enter typed evidence retroactively.

## Resume and reconciliation invariants

- A fresh environmental denial is resumable only after a new live preflight;
  the prior admission record is observational.
- A fresh cutoff denial never launches because the immutable deadline has
  passed.
- A terminal controller never resumes, regardless of later host state.
- Raw-complete cell reconciliation remains permitted after cutoff because it
  launches no inference.
- An incomplete one-use cell is never relaunched; exact-owned recovery either
  proves a terminal raw cell or terminalizes the campaign.
- A checkpoint pause changes no controller, admission, cell, or summary file.
- Fresh host denial stays nonterminal, while the same denial at an active pair
  boundary remains campaign-terminal to avoid inheriting contaminated state.

## Campaign-level scalar evidence

Add a specialized `autoresearch_campaign` evidence kind rather than forcing
campaign state into one of the fourteen cell bundles. Export one campaign
directory containing a strict manifest, sanitized admission projection, and
checksums.

The manifest should contain only:

- public campaign ID and frozen campaign/preview/policy hashes;
- creation and cutoff timestamps;
- planned cell count and ordered published cell IDs;
- effective/controller status and phase, terminal reason, and calibration
  boolean;
- next pair index and ordered candidate decisions;
- controller event count and prefix hash;
- admission count and latest admission hash; and
- strict scalar-only sanitization flags.

The admissions projection contains the allowlisted records above, never the
raw journal bytes. Source export must replay controller and admission truth
under the campaign lock; it must not trust or copy `summary.json`. Preserve the
existing fourteen nonterminal cell bundles and add one campaign bundle, rather
than converting a blocked admission into a synthetic cell result.

Until the current legacy campaign receives a new live admission observation,
its typed bundle can project only controller status `planned`, zero admissions,
and no admission outcome. It must not infer `blocked_environment` from the
legacy summary. The documentation remains the only tracked account of that
historical preflight.

The exporter and verifier must reject unknown fields/files, changed ordering,
bad hashes, broken controller-prefix bindings, missing or extra cell IDs,
paths, secrets, and private payload surfaces. Deterministic re-export and exact
staged verification remain mandatory.

## Implementation map after cutoff

1. In `bench/autoresearch_campaign.py`, introduce the typed live-admission
   result, strict admission journal, locked summary split, schema-2 derivation,
   and explicit cutoff mapping. Keep `campaign_admission` as a compatibility
   wrapper if external tests use its blocker tuple.
2. In `bench/autoresearch.py`, add the categorical cutoff failure kind and its
   replay validation.
3. In `bench/evidence.py`, return validated campaign descriptors alongside
   cell run directories, then add specialized campaign export and verification
   before generic campaign dispatch.
4. Update the CLI only where status-to-exit mapping changes: `expired` returns
   failure, while checkpoint pause retains its distinct exit code.
5. Refresh the protocol documentation and tracked scalar evidence only after
   the implementation, fixtures, deterministic export, and staged verifier all
   pass.

Commit and push each passing logical slice separately: admission/summary,
cutoff semantics, campaign evidence, then refreshed documentation/evidence.

## Regression matrix

At minimum, add tests for:

- blocker preservation across public summarize;
- the exact 4,930.0/4,929.999-second boundary;
- fresh host denial followed by a clean resume;
- fresh and active cutoff behavior;
- simultaneous blocker order and terminal precedence;
- admitted and denied record schema, chaining, and frozen bindings;
- a valid stale admission record not deciding live admission, while a malformed
  chain prevents the required append and any launch;
- public summarize lock contention and internal non-recursive summarization;
- checkpoint pause writing no summary or admission record;
- raw-complete reconciliation after cutoff with no new cell launch;
- deterministic campaign evidence export and verification;
- preservation of fourteen cell bundles while campaign count increases by
  one; and
- staged secret/path scans and refreshed-checksum tamper rejection.

Run full unittest discovery and Python compilation. Evidence changes also
require an offline fixture export, deterministic temporary re-export, normal
verification, exact staged verification, and a final frozen-harness decision:
the old campaign is already past cutoff before any executable input changes.
