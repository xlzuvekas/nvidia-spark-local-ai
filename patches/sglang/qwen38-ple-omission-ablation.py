#!/usr/bin/env python3
"""Add an explicit, fail-closed Qwen3.8 PLE omission ablation.

This patch is intentionally not a serving optimization.  It permits a matched
experiment that removes the trained PLE layer from the model graph and skips
exactly the checkpoint tensors belonging to that layer.  The resulting model
has different semantics and must be identified as an ablation by its manifest.
"""

from __future__ import annotations

import sys


INIT_ANCHOR = """    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, \"rope_parameters\", None) or getattr("""
INIT_REPLACEMENT = """    ) -> None:
        sparkbench_omit_ple = getattr(config, \"sparkbench_omit_ple\", False)
        if type(sparkbench_omit_ple) is not bool:
            raise ValueError(\"sparkbench_omit_ple must be boolean\")
        self._sparkbench_ple_omitted = sparkbench_omit_ple
        if self._sparkbench_ple_omitted:
            text_config = getattr(config, \"text_config\", config)
            if list(getattr(text_config, \"ple_layer_ids\", ())) != [2]:
                raise ValueError(
                    \"SparkBench PLE omission requires the pinned [2] layer layout\"
                )
            text_config.ple_layer_ids = []
            logger.warning(
                \"SparkBench semantic ablation enabled: trained PLE layer omitted\"
            )
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, \"rope_parameters\", None) or getattr("""

STATE_ANCHOR = """        ple_cache_seen_shards: Set[int] = set()
        skipped_visual_count = 0

        for name, loaded_weight in weights:
            if \"rotary_emb.inv_freq\" in name:"""
STATE_REPLACEMENT = """        ple_cache_seen_shards: Set[int] = set()
        ple_omitted_checkpoint_weights: Set[str] = set()
        skipped_visual_count = 0

        for name, loaded_weight in weights:
            if self._sparkbench_ple_omitted and \".ple.\" in name:
                if name in ple_omitted_checkpoint_weights:
                    raise ValueError(f\"duplicate omitted PLE checkpoint tensor: {name}\")
                ple_omitted_checkpoint_weights.add(name)
                continue
            if \"rotary_emb.inv_freq\" in name:"""

FINAL_ANCHOR = """        if _ple_cache_is_readonly():
            expected_ple_shards = set(range(128))"""
FINAL_REPLACEMENT = """        if self._sparkbench_ple_omitted:
            ple_prefix = \"model.language_model.layers.1.ple\"
            expected_ple_weights = {
                f\"{ple_prefix}.conv1d.weight\",
                f\"{ple_prefix}.key_proj.weight\",
                f\"{ple_prefix}.norm_conv.weight\",
                f\"{ple_prefix}.norm_key.weight\",
                f\"{ple_prefix}.norm_query.weight\",
                f\"{ple_prefix}.ple_embedding.layer_multipliers\",
                f\"{ple_prefix}.ple_embedding.ngram_embedding.weight_scale\",
                f\"{ple_prefix}.ple_embedding.ngram_heads_offsets\",
                f\"{ple_prefix}.ple_embedding.ngram_heads_vocab_sizes\",
                f\"{ple_prefix}.value_proj.weight\",
                *{
                    f\"{ple_prefix}.ple_embedding.ngram_embedding.shard_{index}.weight\"
                    for index in range(128)
                },
            }
            if ple_omitted_checkpoint_weights != expected_ple_weights:
                missing = sorted(expected_ple_weights - ple_omitted_checkpoint_weights)
                extra = sorted(ple_omitted_checkpoint_weights - expected_ple_weights)
                raise ValueError(
                    f\"omitted PLE checkpoint tensor set mismatch: {missing=} {extra=}\"
                )
            if ple_modules:
                raise ValueError(
                    \"omitted PLE configuration unexpectedly constructed a PLE module\"
                )

        if _ple_cache_is_readonly():
            if self._sparkbench_ple_omitted:
                raise ValueError(\"PLE omission cannot reuse the mapped PLE cache\")
            expected_ple_shards = set(range(128))"""


def transform(source: str) -> str:
    if "ple_omitted_checkpoint_weights" in source:
        return source
    anchors = (
        (INIT_ANCHOR, INIT_REPLACEMENT, "conditional-generation initializer"),
        (STATE_ANCHOR, STATE_REPLACEMENT, "PLE loader state"),
        (FINAL_ANCHOR, FINAL_REPLACEMENT, "PLE loader finalization"),
    )
    for anchor, replacement, label in anchors:
        if source.count(anchor) != 1:
            raise ValueError(f"expected one {label} anchor")
        source = source.replace(anchor, replacement, 1)
    return source


def main(path: str) -> int:
    try:
        with open(path, encoding="utf-8") as stream:
            source = stream.read()
        patched = transform(source)
    except (OSError, UnicodeError, ValueError) as error:
        print("ERROR:", error)
        return 1
    if patched == source:
        print("ALREADY PATCHED:", path)
        return 0
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(patched)
    print("PATCHED:", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
