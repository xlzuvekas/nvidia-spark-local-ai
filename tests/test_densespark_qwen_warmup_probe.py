from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
PROBE_PATH = ROOT / "bench" / "assets" / "densespark_qwen_warmup_probe.py"
DOCKERFILE_PATH = (
    ROOT / "patches" / "vllm" / "Dockerfile.densespark-qwen-warmup-probe"
)


def _load_probe() -> ModuleType:
    spec = importlib.util.spec_from_file_location("densespark_qwen_warmup_probe", PROBE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load DenseSpark Qwen warmup probe")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fake_target(source: Path) -> ModuleType:
    module = ModuleType("fake_qwen_warmup")
    module.__file__ = str(source)

    def causal(device: object, config: object) -> None:
        return None

    def post_conv(device: object, config: object) -> None:
        return None

    def sigmoid(device: object, config: object) -> None:
        return None

    module._warm_causal_conv1d_fwd_kernel = causal
    module._warm_fused_post_conv_kernel = post_conv
    module._warm_fused_sigmoid_gating_delta_rule_update_kernel = sigmoid
    return module


class DenseSparkQwenWarmupProbeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.probe = _load_probe()

    def test_skip_allowlist_and_rank4_arm_fail_closed(self) -> None:
        helper = self.probe.SIGMOID_HELPER
        self.assertEqual(
            self.probe._requested_mode({self.probe.SKIP_VARIABLE: helper}),
            (frozenset({helper}), False),
        )
        with self.assertRaisesRegex(RuntimeError, "unknown Qwen warmup helpers"):
            self.probe._requested_mode({self.probe.SKIP_VARIABLE: "not-a-helper"})
        with self.assertRaisesRegex(RuntimeError, "mutually exclusive"):
            self.probe._requested_mode(
                {
                    self.probe.SKIP_VARIABLE: helper,
                    self.probe.RANK4_VARIABLE: "1",
                }
            )
        with self.assertRaisesRegex(RuntimeError, "exactly 0 or 1"):
            self.probe._requested_mode({self.probe.RANK4_VARIABLE: "true"})

    def test_derived_image_contract_pins_probe_target_and_entrypoint(self) -> None:
        dockerfile = DOCKERFILE_PATH.read_text(encoding="utf-8")
        probe_digest = hashlib.sha256(PROBE_PATH.read_bytes()).hexdigest()
        instructions = tuple(
            line.strip() for line in dockerfile.splitlines() if line.strip()
        )
        self.assertEqual(
            instructions[0],
            "FROM local/densespark:qwen38-27b-v1.2-0abecc3",
        )
        self.assertFalse(any(line.startswith("ARG ") for line in instructions))
        self.assertNotIn("${", dockerfile)
        self.assertIn(probe_digest, dockerfile)
        self.assertIn(self.probe.TARGET_SOURCE_SHA256, dockerfile)
        self.assertIn(
            "sha256:"
            "d8d02859a49ebf452d9e20b5fbc0790cd4c38fe9a1f5184096b06e3cc6a751d1",
            dockerfile,
        )
        self.assertIn("ENTRYPOINT [\"vllm\"]", dockerfile)
        self.assertIn("SPARKBENCH_QWEN_WARMUP_PROBE=1", dockerfile)
        for digest in (
            "d42cdc95d8d221b49693a46119c714fee3f290282bdfefa63f92f9725f1b20ea",
            "53eaae681b5a0327465b28b7b1983303335db852ac9667ae05faa3682d8c6b8c",
            "000ab8996af9788fdb8843a6a3b91833e7a14c8acc0e1ea073a536330f64cb6f",
        ):
            self.assertIn(digest, dockerfile)

    def test_wrapper_syncs_before_and_after_helper(self) -> None:
        events: list[object] = []

        def original(device: object, config: object) -> str:
            events.append(("call", device, config))
            return "result"

        wrapped = self.probe._wrap_helper("helper", original, skip=False)
        with patch.object(
            self.probe,
            "_synchronize",
            side_effect=lambda device: events.append(("sync", device)),
        ):
            self.assertEqual(wrapped("cuda:0", "config"), "result")
        self.assertEqual(
            events,
            [
                ("sync", "cuda:0"),
                ("call", "cuda:0", "config"),
                ("sync", "cuda:0"),
            ],
        )

    def test_skipped_helper_still_establishes_pre_sync_boundary(self) -> None:
        events: list[object] = []

        def original(device: object, config: object) -> None:
            events.append("unexpected-call")

        wrapped = self.probe._wrap_helper("helper", original, skip=True)
        with patch.object(
            self.probe,
            "_synchronize",
            side_effect=lambda device: events.append(("sync", device)),
        ):
            wrapped(device="cuda:0", config="config")
        self.assertEqual(events, [("sync", "cuda:0")])

    def test_helper_exception_is_logged_and_reraised(self) -> None:
        sync_count = 0

        def synchronize(device: object) -> None:
            nonlocal sync_count
            sync_count += 1

        def original(device: object, config: object) -> None:
            raise ValueError("synthetic helper failure")

        wrapped = self.probe._wrap_helper("helper", original, skip=False)
        with (
            patch.object(self.probe, "_synchronize", side_effect=synchronize),
            self.assertLogs(
                "sparkbench.densespark_qwen_warmup_probe", level="WARNING"
            ) as captured,
            self.assertRaisesRegex(ValueError, "synthetic helper failure"),
        ):
            wrapped("cuda:0", "config")
        self.assertEqual(sync_count, 1)
        self.assertTrue(any("phase=call" in line for line in captured.output))

    def test_source_digest_mismatch_prevents_install(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "qwen_triton_warmup.py"
            source.write_text("# drifted source\n", encoding="utf-8")
            target = _fake_target(source)
            with patch.object(
                self.probe.importlib,
                "import_module",
                return_value=target,
            ):
                with self.assertRaisesRegex(RuntimeError, "source digest mismatch"):
                    self.probe.install({})
        self.assertFalse(hasattr(target, self.probe.INSTALL_SENTINEL))

    def test_install_is_exact_and_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "qwen_triton_warmup.py"
            source.write_text("# synthetic fixture\n", encoding="utf-8")
            target = _fake_target(source)
            originals = tuple(getattr(target, name) for name in self.probe.WARMUP_HELPERS)
            with (
                patch.object(
                    self.probe.importlib,
                    "import_module",
                    return_value=target,
                ),
                patch.object(
                    self.probe,
                    "_target_source_digest",
                    return_value=self.probe.TARGET_SOURCE_SHA256,
                ),
            ):
                first = self.probe.install({})
                installed = tuple(
                    getattr(target, name) for name in self.probe.WARMUP_HELPERS
                )
                second = self.probe.install({})
            self.assertEqual(first, second)
            self.assertEqual(first["source_sha256"], self.probe.TARGET_SOURCE_SHA256)
            self.assertTrue(
                all(after is not before for before, after in zip(originals, installed))
            )
            self.assertEqual(
                installed,
                tuple(getattr(target, name) for name in self.probe.WARMUP_HELPERS),
            )

    def test_rank4_state_shape_matches_runtime_layout(self) -> None:
        config = SimpleNamespace(hv=48, v=128, k=128, state_stride_token=786_432)
        self.assertEqual(self.probe._rank4_state_shape(config), (1, 48, 128, 128))
        config.state_stride_token += 1
        with self.assertRaisesRegex(RuntimeError, "state stride"):
            self.probe._rank4_state_shape(config)

    def test_rank4_arm_passes_a_runtime_shaped_state_to_pinned_op(self) -> None:
        class FakeTensor:
            def __init__(self, shape: tuple[int, ...]) -> None:
                self.shape = shape
                self.ndim = len(shape)

            def stride(self, dimension: int) -> int:
                if dimension != 0:
                    raise AssertionError("fixture only models stride(0)")
                stride = 1
                for extent in self.shape[1:]:
                    stride *= extent
                return stride

        fake_torch = ModuleType("torch")
        fake_torch.int32 = object()
        fake_torch.empty = lambda shape, **kwargs: FakeTensor(tuple(shape))
        fake_torch.empty_like = lambda tensor: FakeTensor(tensor.shape)
        fake_torch.tensor = lambda value, **kwargs: FakeTensor((len(value),))
        fake_torch.zeros = lambda shape, **kwargs: FakeTensor(tuple(shape))

        calls: list[dict[str, object]] = []
        fused_module_name = (
            "vllm.third_party.flash_linear_attention.ops.fused_sigmoid_gating"
        )
        fused_module = ModuleType(fused_module_name)
        fused_module.fused_sigmoid_gating_delta_rule_update = (
            lambda **kwargs: calls.append(kwargs)
        )
        module_tree: dict[str, ModuleType] = {"torch": fake_torch}
        for name in (
            "vllm",
            "vllm.third_party",
            "vllm.third_party.flash_linear_attention",
            "vllm.third_party.flash_linear_attention.ops",
        ):
            package = ModuleType(name)
            package.__path__ = []
            module_tree[name] = package
        module_tree[fused_module_name] = fused_module

        config = SimpleNamespace(
            h=16,
            hv=48,
            k=128,
            v=128,
            conv_dtype="bfloat16",
            state_dtype="bfloat16",
            state_stride_token=786_432,
            a_log=object(),
            dt_bias=object(),
        )
        with patch.dict(sys.modules, module_tree):
            self.probe._warm_fused_sigmoid_with_rank4_state("cuda:0", config)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["initial_state"].shape, (1, 48, 128, 128))
        self.assertEqual(calls[0]["ssm_state_indices"].shape, (1, 1))
        self.assertTrue(calls[0]["inplace_final_state"])
        self.assertFalse(calls[0]["is_kda"])

    def test_sitecustomize_activation_exits_on_configuration_drift(self) -> None:
        statuses: list[int] = []

        class ExitCalled(RuntimeError):
            pass

        def exit_process(status: int) -> None:
            statuses.append(status)
            raise ExitCalled

        with self.assertRaises(ExitCalled):
            self.probe.activate_from_sitecustomize(
                {
                    self.probe.ACTIVATION_VARIABLE: "1",
                    self.probe.SKIP_VARIABLE: "not-a-helper",
                },
                exit_process=exit_process,
            )
        self.assertEqual(statuses, [self.probe.EXIT_CONFIGURATION_ERROR])

    def test_mode_cannot_be_requested_while_probe_is_disabled(self) -> None:
        statuses: list[int] = []

        class ExitCalled(RuntimeError):
            pass

        def exit_process(status: int) -> None:
            statuses.append(status)
            raise ExitCalled

        with self.assertRaises(ExitCalled):
            self.probe.activate_from_sitecustomize(
                {
                    self.probe.ACTIVATION_VARIABLE: "0",
                    self.probe.RANK4_VARIABLE: "1",
                },
                exit_process=exit_process,
            )
        self.assertEqual(statuses, [self.probe.EXIT_CONFIGURATION_ERROR])


if __name__ == "__main__":
    unittest.main()
