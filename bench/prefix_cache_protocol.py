"""Fixed, auditable schedule for native llama.cpp prompt-KV measurements."""

from __future__ import annotations


PREFIX_CACHE_PROTOCOL = "llamacpp-same-slot-prompt-kv-v2"
PREFIX_CACHE_SUITE_ID = "llamacpp-prefix-cache"
PREFIX_CACHE_CONTEXT_TOKENS = 262_144
PREFIX_CACHE_PREFIX_TARGETS = {
    "llamacpp-prefix-cache-8192": 8_192,
    "llamacpp-prefix-cache-32768": 32_768,
}

# Cache profiles deliberately pin every llama.cpp argument that is not owned by
# SparkBench's launcher.  The launcher appends its own model, host, context,
# parallelism, and metrics controls after ``model.args``; accepting arbitrary
# frozen-plan arguments here would let a stale or modified plan override those
# owned controls.  Keep this tuple in execution order so the frozen-plan gate
# can require an exact, auditable protocol shape.
PREFIX_CACHE_LLAMACPP_COMMON_ARGS = (
    "--n-gpu-layers",
    "all",
    "--flash-attn",
    "on",
    "--fit",
    "off",
    "--batch-size",
    "8192",
    "--ubatch-size",
    "512",
    "--cache-type-k",
    "q8_0",
    "--cache-type-v",
    "q8_0",
    "--jinja",
    "--reasoning",
    "off",
)

# ``None`` deliberately means omit the request override and exercise the
# profile-level ``--no-cache-prompt`` control.  The cache-on profile explicitly
# forces both cold observations off, then omits the treatment override so its
# explicit ``--cache-prompt`` profile control causes the third-position reuse.
PREFIX_CACHE_STEPS: dict[str, tuple[tuple[str, bool | None, str], ...]] = {
    "off": (
        ("forced-cold-a", None, "profile-default"),
        ("forced-cold-b", None, "profile-default"),
        ("forced-cold-c", None, "profile-default"),
    ),
    "on": (
        ("forced-cold-a", False, "force-off"),
        ("forced-cold-b", False, "force-off"),
        ("warm-prefix-hit", None, "profile-default"),
    ),
}


def prefix_cache_steps(mode: str) -> tuple[tuple[str, bool | None, str], ...]:
    """Return the frozen three-request control/treatment block for ``mode``."""

    try:
        return PREFIX_CACHE_STEPS[mode]
    except KeyError as error:
        raise ValueError(f"unsupported prefix-cache mode: {mode!r}") from error


def prefix_cache_llamacpp_args(mode: str) -> tuple[str, ...]:
    """Return the only user-supplied llama.cpp arguments in a cache profile."""

    if mode == "on":
        return PREFIX_CACHE_LLAMACPP_COMMON_ARGS + ("--cache-prompt",)
    if mode == "off":
        return PREFIX_CACHE_LLAMACPP_COMMON_ARGS + ("--no-cache-prompt",)
    raise ValueError(f"unsupported prefix-cache mode: {mode!r}")


def prefix_cache_conditions(mode: str) -> tuple[str, ...]:
    """Return protocol condition labels in their required request order."""

    return tuple(condition for condition, _, _ in prefix_cache_steps(mode))
