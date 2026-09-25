"""Opt in through the canonical runner with -- --run-copilot-runtime-smoke.

RUN_COPILOT_RUNTIME_SMOKE=1 also enables the test when the enclosing runner
preserves it. The Hermes canonical runner strips non-allowlisted env variables,
so its supported invocation uses the pytest option instead.
"""

import asyncio
import os

from copilot import CopilotClient, RuntimeConnection
import pytest

from hermes_copilot_sdk.client import _cleanup, _deny_permission


async def test_real_empty_runtime_exposes_no_native_tools(tmp_path, request):
    if not (request.config.getoption("--run-copilot-runtime-smoke")
            or os.environ.get("RUN_COPILOT_RUNTIME_SMOKE") == "1"):
        pytest.skip("Opt in with --run-copilot-runtime-smoke; requires a cached pinned runtime")
    directory = tmp_path / "runtime"
    directory.mkdir()
    env = {key: os.environ[key] for key in (
        "PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT",
        "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP",
    ) if key in os.environ}
    client = CopilotClient(
        connection=RuntimeConnection.for_stdio(),
        mode="empty", base_directory=str(directory), working_directory=str(directory),
        use_logged_in_user=False, env=env,
    )
    session = None
    try:
        async with asyncio.timeout(30):
            await client.start()
            session = await client.create_session(
                available_tools=[], on_permission_request=_deny_permission,
                system_message={"mode": "replace", "content": "Protocol-only test; never run inference."},
                working_directory=str(directory), config_directory=str(directory),
                enable_config_discovery=False, skip_custom_instructions=True,
                enable_on_demand_instruction_discovery=False,
                enable_file_hooks=False, enable_host_git_operations=False,
                enable_skills=False, enable_session_store=False,
                enable_session_telemetry=False, memory={"enabled": False},
                infinite_sessions={"enabled": False}, streaming=False,
            )
            metadata = await session.rpc.tools.get_current_metadata()
            assert metadata.tools == []
    finally:
        try:
            await asyncio.wait_for(_cleanup(client, session), timeout=10)
        except BaseException:
            await asyncio.wait_for(client.force_stop(), timeout=5)
            raise
