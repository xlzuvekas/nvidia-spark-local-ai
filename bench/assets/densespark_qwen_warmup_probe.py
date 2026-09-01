"""Hash-pinned diagnostics for the vLLM Qwen Triton startup warmup.

This file is meant to be copied into a diagnostic image as
``sitecustomize.py``. Python imports that module in both the API process and
the spawned EngineCore process, unlike an entrypoint-only monkeypatch. The
probe is inert unless ``SPARKBENCH_QWEN_WARMUP_PROBE=1`` is present.

The permanent serving profile must not enable a skip. Skips only isolate a
startup failure and can defer JIT work to the first real request. The optional
rank-four state arm changes only the dummy tensor used by the third warmup
helper so that its rank matches the state tensor used during model execution.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import functools
import hashlib
import importlib
import inspect
import logging
import os
from pathlib import Path
import time
from typing import Any, NoReturn


LOGGER = logging.getLogger("sparkbench.densespark_qwen_warmup_probe")
TARGET_MODULE = "vllm.model_executor.warmup.qwen_triton_warmup"
TARGET_SOURCE_SHA256 = (
    "2b08d94662e7b04ce61c0f7a818e0cd1768fe7602a89df04ec6148f62fe3acdb"
)
ACTIVATION_VARIABLE = "SPARKBENCH_QWEN_WARMUP_PROBE"
SKIP_VARIABLE = "SPARKBENCH_QWEN_WARMUP_SKIP"
RANK4_VARIABLE = "SPARKBENCH_QWEN_WARMUP_RANK4_STATE"
INSTALL_SENTINEL = "_sparkbench_qwen_warmup_probe_v1"
EXIT_CONFIGURATION_ERROR = 78

WARMUP_HELPERS = (
    "_warm_causal_conv1d_fwd_kernel",
    "_warm_fused_post_conv_kernel",
    "_warm_fused_sigmoid_gating_delta_rule_update_kernel",
)
SIGMOID_HELPER = WARMUP_HELPERS[-1]


def _binary_flag(
    name: str,
    environ: Mapping[str, str],
    *,
    default: bool = False,
) -> bool:
    raw = environ.get(name)
    if raw is None or raw == "":
        return default
    if raw == "0":
        return False
    if raw == "1":
        return True
    raise RuntimeError(f"{name} must be exactly 0 or 1")


def _requested_skips(environ: Mapping[str, str]) -> frozenset[str]:
    raw = environ.get(SKIP_VARIABLE, "")
    requested = frozenset(part.strip() for part in raw.split(",") if part.strip())
    unknown = requested.difference(WARMUP_HELPERS)
    if unknown:
        raise RuntimeError(f"unknown Qwen warmup helpers: {sorted(unknown)!r}")
    return requested


def _requested_mode(
    environ: Mapping[str, str],
) -> tuple[frozenset[str], bool]:
    skips = _requested_skips(environ)
    rank4_state = _binary_flag(RANK4_VARIABLE, environ)
    if rank4_state and SIGMOID_HELPER in skips:
        raise RuntimeError(
            "the rank-four Qwen state arm and sigmoid-helper skip are mutually "
            "exclusive"
        )
    return skips, rank4_state


def _target_source_digest(module: Any) -> str:
    source_name = getattr(module, "__file__", None)
    if not isinstance(source_name, str) or not source_name.endswith(".py"):
        raise RuntimeError("Qwen warmup module did not resolve to a Python source file")
    source = Path(source_name)
    if source.is_symlink():
        raise RuntimeError("Qwen warmup source must not be a symlink")
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise RuntimeError("Qwen warmup source could not be read") from error
    return hashlib.sha256(payload).hexdigest()


def _require_exact_target(module: Any) -> None:
    digest = _target_source_digest(module)
    if digest != TARGET_SOURCE_SHA256:
        raise RuntimeError(
            "Qwen warmup source digest mismatch: "
            f"expected {TARGET_SOURCE_SHA256}, observed {digest}"
        )
    for name in WARMUP_HELPERS:
        helper = getattr(module, name, None)
        if not callable(helper):
            raise RuntimeError(f"Qwen warmup helper is missing or not callable: {name}")
        parameters = tuple(inspect.signature(helper).parameters)
        if parameters != ("device", "config"):
            raise RuntimeError(
                f"Qwen warmup helper signature drifted for {name}: {parameters!r}"
            )


def _device_from_call(args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> Any:
    if args:
        return args[0]
    if "device" in kwargs:
        return kwargs["device"]
    raise RuntimeError("Qwen warmup helper call did not provide its device")


def _synchronize(device: Any) -> None:
    # Import lazily so an activation/configuration failure remains inside the
    # fail-closed sitecustomize boundary below.
    import torch

    torch.accelerator.synchronize(device)


def _logged_synchronize(name: str, device: Any, phase: str) -> None:
    LOGGER.warning(
        "SPARKBENCH_QWEN_WARMUP_SYNC_BEGIN helper=%s phase=%s pid=%d",
        name,
        phase,
        os.getpid(),
    )
    try:
        _synchronize(device)
    except BaseException:
        LOGGER.exception(
            "SPARKBENCH_QWEN_WARMUP_EXCEPTION helper=%s phase=%s pid=%d",
            name,
            phase,
            os.getpid(),
        )
        raise
    LOGGER.warning(
        "SPARKBENCH_QWEN_WARMUP_SYNC_END helper=%s phase=%s pid=%d",
        name,
        phase,
        os.getpid(),
    )


def _rank4_state_shape(config: Any) -> tuple[int, int, int, int]:
    hv = int(config.hv)
    value_dim = int(config.v)
    key_dim = int(config.k)
    stride = int(config.state_stride_token)
    if min(hv, value_dim, key_dim, stride) <= 0:
        raise RuntimeError("Qwen warmup state dimensions and stride must be positive")
    expected_stride = hv * value_dim * key_dim
    if stride != expected_stride:
        raise RuntimeError(
            "Qwen warmup state stride does not match its rank-four shape: "
            f"expected {expected_stride}, observed {stride}"
        )
    return (1, hv, value_dim, key_dim)


def _warm_fused_sigmoid_with_rank4_state(device: Any, config: Any) -> None:
    """Run the pinned third helper with only its dummy-state rank repaired."""

    import torch
    from vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating import (
        fused_sigmoid_gating_delta_rule_update,
    )

    state_shape = _rank4_state_shape(config)
    LOGGER.warning(
        "SPARKBENCH_QWEN_WARMUP_RANK4_STATE shape=%s stride_token=%d pid=%d",
        state_shape,
        int(config.state_stride_token),
        os.getpid(),
    )
    q = torch.empty((1, 1, config.h, config.k), dtype=config.conv_dtype, device=device)
    key = torch.empty_like(q)
    value = torch.empty(
        (1, 1, config.hv, config.v), dtype=config.conv_dtype, device=device
    )
    alpha = torch.empty(
        (1, 1, config.hv), dtype=config.conv_dtype, device=device
    )
    beta = torch.empty_like(alpha)
    state = torch.empty(state_shape, dtype=config.state_dtype, device=device)
    if state.ndim != 4 or int(state.stride(0)) != int(config.state_stride_token):
        raise RuntimeError("rank-four Qwen warmup state allocation drifted")
    cu_seqlens = torch.tensor([0, 1], dtype=torch.int32, device=device)
    ssm_state_indices = torch.zeros((1, 1), dtype=torch.int32, device=device)

    fused_sigmoid_gating_delta_rule_update(
        A_log=config.a_log,
        a=alpha,
        b=beta,
        dt_bias=config.dt_bias,
        q=q,
        k=key,
        v=value,
        beta=1.0,
        threshold=20.0,
        initial_state=state,
        inplace_final_state=True,
        cu_seqlens=cu_seqlens,
        ssm_state_indices=ssm_state_indices,
        use_qk_l2norm_in_kernel=True,
        is_kda=False,
    )


def _wrap_helper(
    name: str,
    original: Callable[..., Any],
    *,
    skip: bool,
    replacement: Callable[..., Any] | None = None,
) -> Callable[..., Any]:
    selected = replacement if replacement is not None else original

    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        device = _device_from_call(args, kwargs)
        # Establish a clean boundary before attributing any pending CUDA work to
        # this helper. A skipped helper still performs this pre-boundary sync.
        _logged_synchronize(name, device, "pre")
        if skip:
            LOGGER.warning(
                "SPARKBENCH_QWEN_WARMUP_SKIP helper=%s pid=%d",
                name,
                os.getpid(),
            )
            return None

        started = time.monotonic()
        LOGGER.warning(
            "SPARKBENCH_QWEN_WARMUP_BEGIN helper=%s pid=%d",
            name,
            os.getpid(),
        )
        try:
            result = selected(*args, **kwargs)
        except BaseException:
            LOGGER.exception(
                "SPARKBENCH_QWEN_WARMUP_EXCEPTION helper=%s phase=call pid=%d",
                name,
                os.getpid(),
            )
            raise
        _logged_synchronize(name, device, "post")
        LOGGER.warning(
            "SPARKBENCH_QWEN_WARMUP_END helper=%s elapsed_s=%.6f pid=%d",
            name,
            time.monotonic() - started,
            os.getpid(),
        )
        return result

    return wrapped


def install(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Install the exact probe in the current Python process."""

    effective_environ = os.environ if environ is None else environ
    skips, rank4_state = _requested_mode(effective_environ)
    module = importlib.import_module(TARGET_MODULE)
    _require_exact_target(module)
    requested_receipt = {
        "source_sha256": TARGET_SOURCE_SHA256,
        "skips": tuple(sorted(skips)),
        "rank4_state": rank4_state,
    }
    existing = getattr(module, INSTALL_SENTINEL, None)
    if existing is not None:
        if existing != requested_receipt:
            raise RuntimeError("Qwen warmup probe was already installed in another mode")
        return dict(existing)

    originals = {name: getattr(module, name) for name in WARMUP_HELPERS}
    for name, original in originals.items():
        replacement = None
        if rank4_state and name == SIGMOID_HELPER:
            replacement = _warm_fused_sigmoid_with_rank4_state
        setattr(
            module,
            name,
            _wrap_helper(
                name,
                original,
                skip=name in skips,
                replacement=replacement,
            ),
        )
    setattr(module, INSTALL_SENTINEL, requested_receipt)
    LOGGER.warning(
        "SPARKBENCH_QWEN_WARMUP_PROBE_INSTALLED source_sha256=%s skips=%s "
        "rank4_state=%s pid=%d",
        TARGET_SOURCE_SHA256,
        ",".join(sorted(skips)) or "none",
        int(rank4_state),
        os.getpid(),
    )
    return dict(requested_receipt)


def _flush_log_handlers() -> None:
    for candidate in (LOGGER, logging.getLogger()):
        for handler in candidate.handlers:
            try:
                handler.flush()
            except Exception:
                pass


def _exit_process(status: int) -> NoReturn:
    os._exit(status)


def activate_from_sitecustomize(
    environ: Mapping[str, str] | None = None,
    *,
    exit_process: Callable[[int], NoReturn] = _exit_process,
) -> bool:
    """Activate under ``sitecustomize``, terminating on any requested drift."""

    effective_environ = os.environ if environ is None else environ
    activation_raw = effective_environ.get(ACTIVATION_VARIABLE)
    mode_requested = bool(effective_environ.get(SKIP_VARIABLE)) or bool(
        effective_environ.get(RANK4_VARIABLE)
    )
    if activation_raw in (None, "", "0") and not mode_requested:
        return False
    try:
        if not _binary_flag(ACTIVATION_VARIABLE, effective_environ):
            raise RuntimeError(
                "Qwen warmup mode was requested while the probe was disabled"
            )
        install(effective_environ)
    except BaseException:
        logging.basicConfig(level=logging.INFO)
        LOGGER.exception(
            "SPARKBENCH_QWEN_WARMUP_PROBE_FATAL exit_status=%d pid=%d",
            EXIT_CONFIGURATION_ERROR,
            os.getpid(),
        )
        _flush_log_handlers()
        exit_process(EXIT_CONFIGURATION_ERROR)
        raise AssertionError("exit_process unexpectedly returned")
    return True


if __name__ == "sitecustomize":
    activate_from_sitecustomize()
