from __future__ import annotations

import runpy
import sys
from pathlib import Path
from types import ModuleType
import unittest
from unittest.mock import AsyncMock, Mock, patch


REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "bench" / "harbor_pinned_agents.py"


class _FakeOpenCode:
    async def run(self, instruction: str, environment: object, context: object) -> None:
        del instruction, environment, context
        failure = getattr(self, "parent_failure", None)
        if failure is not None:
            raise failure


class _FakeQwenCode:
    async def run(self, instruction: str, environment: object, context: object) -> None:
        del instruction, environment, context


def _module(name: str, **values: object) -> ModuleType:
    module = ModuleType(name)
    module.__dict__.update(values)
    return module


def _load_pinned_opencode() -> type:
    modules = {
        "harbor": _module("harbor"),
        "harbor.agents": _module("harbor.agents"),
        "harbor.agents.installed": _module("harbor.agents.installed"),
        "harbor.agents.installed.opencode": _module(
            "harbor.agents.installed.opencode", OpenCode=_FakeOpenCode
        ),
        "harbor.agents.installed.qwen_code": _module(
            "harbor.agents.installed.qwen_code", QwenCode=_FakeQwenCode
        ),
        "harbor.environments": _module("harbor.environments"),
        "harbor.environments.base": _module(
            "harbor.environments.base", BaseEnvironment=object
        ),
        "harbor.models": _module("harbor.models"),
        "harbor.models.agent": _module("harbor.models.agent"),
        "harbor.models.agent.context": _module(
            "harbor.models.agent.context", AgentContext=object
        ),
    }
    with patch.dict(sys.modules, modules):
        namespace = runpy.run_path(str(SOURCE))
    pinned = namespace["PinnedOpenCode"]
    if not isinstance(pinned, type):
        raise AssertionError("PinnedOpenCode did not load as a class")
    return pinned


class PinnedOpenCodeCleanupTests(unittest.IsolatedAsyncioTestCase):
    @staticmethod
    def _agent() -> object:
        agent = _load_pinned_opencode()()
        agent._network_probe = AsyncMock(return_value={"schema_version": 1})
        agent._write_network_admission = Mock()
        return agent

    async def test_success_deletes_only_the_two_ephemeral_state_roots(self) -> None:
        agent = self._agent()
        agent.exec_as_agent = AsyncMock(return_value=None)

        await agent.run("instruction", object(), object())

        agent.exec_as_agent.assert_awaited_once_with(
            unittest.mock.ANY,
            command=(
                "set -euo pipefail; rm -rf --one-file-system -- "
                "/logs/agent/opencode/xdg-data /logs/agent/opencode/xdg-state"
            ),
            timeout_sec=15,
        )
        agent._write_network_admission.assert_called_once_with()

    async def test_cleanup_failure_propagates_after_success(self) -> None:
        agent = self._agent()
        agent.exec_as_agent = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        with self.assertRaisesRegex(RuntimeError, "cleanup failed"):
            await agent.run("instruction", object(), object())

        agent._write_network_admission.assert_called_once_with()

    async def test_primary_failure_survives_cleanup_failure(self) -> None:
        agent = self._agent()
        agent.parent_failure = LookupError("primary failed")
        agent.exec_as_agent = AsyncMock(side_effect=RuntimeError("cleanup failed"))

        with self.assertRaisesRegex(LookupError, "primary failed"):
            await agent.run("instruction", object(), object())

        agent._write_network_admission.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
