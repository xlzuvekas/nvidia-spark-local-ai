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

## Strict scope boundary

This checkpoint did **not** run Pi, Node, npm, a package installer, a network
request, a model server, inference, or any agent runtime. It did not unpack or
materialize a dependency prefix, load an entrypoint, use an ambient installation,
or access ambient user configuration, extension, session, or agent-workspace
state.

Accordingly, the frozen source closure is **not** an `npm ci` lock claim, a
normalized `node_modules` prefix, a static import admission, a Pi/cowork agent
admission, or a benchmark. It provides no evidence about tool correctness,
reasoning behavior, context capacity, cache behavior, wall time, latency, or
throughput.

## What this enables—and what it does not

The 222-record closure prevents a future implementation from silently expanding
its dependency set through an ambient package tree or a network resolver. The
207 cached artifacts make a fully offline construction feasible to evaluate,
but the construction itself remains a separate safety gate.

Before Pi-core can be used even for an unscored static wrapper check, a future
implementation must use a tarball-direct, scripts-disabled materializer. At a
minimum it must:

- re-verify every selected artifact before extraction;
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
