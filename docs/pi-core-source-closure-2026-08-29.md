# Pi-core frozen source closure and cache audit — 2026-08-29

## Result

The project has completed a deliberately narrow, offline **source-closure and
cache-audit** checkpoint for a prospective Pi-core wrapper. It freezes the
input package metadata needed for a later isolated materialization and checks
that its selected artifacts are already present in the local content-addressed
cache. It is not an executable dependency tree.

| Audited scalar | Result |
| --- | ---: |
| Frozen package records | 222 |
| Unique selected package artifacts | 207 |
| Hash-verified compressed artifact bytes | 13,404,593 |

All 207 selected artifacts were deduplicated by their pinned SHA-512 integrity
records and verified against the corresponding cached bytes. The difference
between package records and artifacts is expected: more than one installation
path may reference the same immutable package artifact.

## Source-closure scope boundary

The source-closure/cache-audit checkpoint did **not** run Pi, Node, npm, a
package installer, a network request, a model server, inference, or any agent
runtime. It did not load an entrypoint, use an ambient installation, or access
ambient user configuration, extension, session, or agent-workspace state.

Accordingly, the frozen source closure is **not** an `npm ci` lock claim, a
normalized `node_modules` prefix, a static import admission, a Pi/cowork agent
admission, or a benchmark. It provides no evidence about tool correctness,
reasoning behavior, context capacity, cache behavior, wall time, latency, or
throughput.

## Scripts-disabled materialization smoke

The repository now has a separately tested `pi-core-prefix-materialize`
command for the next supply-chain step. It requires an explicitly supplied
external owner-private `0700` parent and uses no default output location. A
one-off run against the real cached closure completed, was independently
re-inspected, and was then removed. It did **not** create a retained or admitted
prefix.

| Temporary-materialization scalar | Result |
| --- | ---: |
| Package records | 222 |
| Hash-verified source artifacts | 207 |
| Materialized regular files | 13,828 |
| Materialized tree entries | 15,563 |
| Materialized tree bytes | 75,042,106 |
| Materialized tree digest | `sha256:aebaccc9fa0c58d9ef15a8b718b08f700d2564cbcc31b518c492a6e993964ac8` |

The independent immutable-tree inspection reproduced all of those counts and
the digest before the owner-private temporary directory was deleted. The smoke
did not run Pi, Node, npm, lifecycle scripts, a network client, a container, a
model server, inference, or an agent runtime. It supplies no behavioral,
performance, tool-correctness, or benchmark result.

For every selected installation record, the command copies the mutable npm
cache blob through no-follow descriptors to a private `0400` staging file
while checking its exact SHA-512 digest. It then parses only that private copy;
it never invokes npm, Node, Pi, a network client, a container, or inference.
The extractor accepts one safe archive root (including supported `node` and
`retry` alternate roots), preserves the frozen nested install layout, validates
package-manifest name/version including npm aliases, rejects links, sparse or
special members, embedded `node_modules`, conflicting duplicate paths,
traversal, and overwrites, and normalizes output files/directories to
`0444`/`0555`.

The output is published only through Linux `renameat2(RENAME_NOREPLACE)` into
an absent deterministic child name. On a failure before publication, the
private staging tree is removed descriptor-relatively when safe; failure to
clean is retained as an owner-private stale staging directory rather than
triggering broad pathname deletion. The output's scalar tree identity is
calculated with the existing fd-relative immutable-tree inspector. This is
preparation for a later explicit prefix admission, not that admission itself.
The `0700` parent is the same-UID trust boundary: it excludes other users, not
an intentionally hostile process already running as the materializer owner.

## What this enables—and what it does not

The 222-record closure prevents a future implementation from silently expanding
its dependency set through an ambient package tree or a network resolver. The
207 cached artifacts and the temporary smoke demonstrate that fully offline
construction is feasible, but a retained campaign prefix remains a separate
safety gate.

Before Pi-core can be used even for an unscored static wrapper check, the
materializer must re-verify every selected artifact before extraction, and a
newly materialized retained prefix must still receive a frozen-prefix admission.
At a minimum that admission must:

- preserve the frozen dependency layout without resolver fallback or package
  installation;
- prohibit lifecycle scripts and network access;
- reject unexpected files, unsafe ownership or permissions, and symlink escape;
  and
- freeze and verify the resulting regular-file inventory, tree digest, and
  approved entrypoint boundary.

That resulting immutable prefix would still need its own static import and
wrapper admission. Separately, the prospective server and Pi/cowork agent path
remain subject to their dedicated parser, payload, tool-semantics, long-context,
cache-isolation, memory, and swap gates. Only after those are complete can a
fresh-lifetime coding or cowork benchmark be proposed.

## Relationship to the Pi/cowork plan

This record narrows one supply-chain prerequisite in the existing
[Pi/cowork harness plan](pi-cowork-harness-plan-2026-08-28.md). It does not
change that plan's isolation model, agent-runner boundary, experimental
topology, scoring rules, or evidence policy.
