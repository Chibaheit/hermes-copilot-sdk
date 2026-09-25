"""A fresh, tool-disabled SDK runtime for each Hermes completion."""

from __future__ import annotations

import asyncio
import logging
import math
import threading
from concurrent.futures import Future
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from .bridge import allowed_tools, completion, render_request

logger = logging.getLogger(__name__)
_CLEANUP_TIMEOUT = 5.0


def _seconds(value, default=120.0) -> float:
    if value is None:
        value = default
    if not isinstance(value, (int, float)):
        values = [getattr(value, key, None) for key in ("read", "write", "connect", "pool")]
        components = [v for v in values if isinstance(v, (int, float))]
        if not components:
            raise ValueError("copilot-sdk timeout must be numeric or an HTTP timeout with finite components")
        value = max(components)
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("copilot-sdk timeout must be a positive finite number")
    return value


def _deny_permission(request, invocation):
    from copilot.rpc import PermissionDecisionReject

    return PermissionDecisionReject(feedback="Native tools are disabled; Hermes owns tool execution and approval.")


class _AsyncChunks:
    def __init__(self, chunks):
        self._chunks = iter(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class CopilotSDKClient:
    HERMES_SKIP_TRANSPORT_WRAP = True
    HERMES_SKIP_ASYNC_WRAP = True

    def __init__(self, *, api_key=None, base_url=None, timeout=None, default_headers=None, **kwargs):
        self.api_key = api_key or "copilot-sdk"
        self.base_url = base_url or "copilot-sdk://runtime"
        self._default_headers = dict(default_headers or {})
        self._timeout = timeout
        self.is_closed = False
        self._lock = threading.Lock()
        self._active: dict[asyncio.Task, asyncio.AbstractEventLoop] = {}
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        with self._lock:
            if self.is_closed:
                return
            self.is_closed = True
            for task, loop in self._active.items():
                loop.call_soon_threadsafe(task.cancel)

    def _sync(self, operation, **kwargs):
        from agent.memory_provider import spawn_context_thread

        outcome = Future()

        def run():
            try:
                outcome.set_result(asyncio.run(self._run(operation, **kwargs)))
            except BaseException as exc:
                outcome.set_exception(exc)

        worker = spawn_context_thread(target=run, name="hermes-copilot-sdk", daemon=True)
        worker.start()
        try:
            return outcome.result()
        except KeyboardInterrupt:
            self.close()
            raise

    def list_models(self, *, timeout=30.0) -> list[str]:
        return self._sync("models", timeout=timeout)

    def _create(self, **kwargs):
        # Hermes reuses the same opted-out client for its async auxiliary path.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return self._sync("completion", **kwargs)
        return self._acreate(**kwargs)

    async def _acreate(self, **kwargs):
        result = await self._run("completion", **kwargs)
        return _AsyncChunks(result) if kwargs.get("stream") else result

    async def _run(self, operation, *, model=None, messages=None, tools=None, tool_choice=None,
                   stream=False, timeout=None, reasoning_effort=None, response_format=None, **kwargs):
        try:
            from copilot import CopilotClient, RuntimeConnection
        except ImportError as exc:
            raise RuntimeError("Install hermes-copilot-sdk in the same Python environment as Hermes") from exc
        from agent.secret_scope import current_secret_scope, get_secret
        from hermes_cli.config import load_config
        from hermes_constants import get_hermes_home, get_process_hermes_home
        from tools.environments.local import served_profile_child_env

        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("copilot-sdk requires an asyncio task")
        with self._lock:
            if self.is_closed:
                raise RuntimeError("Copilot SDK client is closed")
            self._active[task] = asyncio.get_running_loop()
        try:
            config = load_config().get("copilot_sdk", {})
            if not isinstance(config, dict):
                raise ValueError("copilot_sdk config must be a mapping")
            budget = _seconds(timeout if timeout is not None else self._timeout,
                              default=_seconds(config.get("timeout_seconds")))
            home = get_hermes_home()
            scope = current_secret_scope()
            if scope is None and home.resolve() != Path(get_process_hermes_home()).resolve():
                raise RuntimeError("copilot-sdk requires a bound secret scope for a routed Hermes profile")
            token = (
                get_secret("COPILOT_GITHUB_TOKEN")
                if scope is None or "COPILOT_GITHUB_TOKEN" in scope else None
            )
            if not token:
                raise RuntimeError(
                    "copilot-sdk requires COPILOT_GITHUB_TOKEN in this Hermes profile's .env "
                    "(with Copilot access); interactive login and other profiles are not used"
                )
            if operation == "completion":
                if not isinstance(model, str) or not model.strip():
                    raise ValueError("copilot-sdk requires an explicit model id")
                if response_format not in (None, {"type": "text"}):
                    raise ValueError("copilot-sdk does not implement structured response_format")
                allowed_tools(tools, tool_choice)
                system, prompt = render_request(messages or [], tools, tool_choice)
            storage = home / "copilot-sdk"
            storage.mkdir(parents=True, exist_ok=True, mode=0o700)
            env = served_profile_child_env(target_home=home, inherit_credentials=False)
            # The SDK receives only its explicit, scope-resolved token over RPC.
            for key in tuple(env):
                if key.startswith("COPILOT_") or key in ("GH_TOKEN", "GITHUB_TOKEN"):
                    env.pop(key)
            cli_path = config.get("cli_path")
            if cli_path is not None and (not isinstance(cli_path, str) or not Path(cli_path).is_file()):
                raise ValueError("copilot_sdk.cli_path must be an existing executable file")
            with TemporaryDirectory(prefix="request-", dir=storage) as directory:
                client = CopilotClient(
                    connection=RuntimeConnection.for_stdio(path=cli_path),
                    mode="empty", base_directory=directory, working_directory=directory,
                    github_token=token, use_logged_in_user=False, env=env,
                )
                session = None
                try:
                    async with asyncio.timeout(budget):
                        await client.start()
                        if operation == "models":
                            models = await client.list_models()
                            return [item.id for item in models]
                        session = await client.create_session(
                            model=model, available_tools=[], on_permission_request=_deny_permission,
                            system_message={"mode": "replace", "content": system},
                            working_directory=directory, config_directory=directory,
                            enable_config_discovery=False, skip_custom_instructions=True,
                            enable_on_demand_instruction_discovery=False,
                            enable_file_hooks=False, enable_host_git_operations=False,
                            enable_skills=False, enable_session_store=False,
                            enable_session_telemetry=False, memory={"enabled": False},
                            infinite_sessions={"enabled": False}, streaming=False,
                            reasoning_effort=reasoning_effort,
                        )
                        response = await session.send_and_wait(
                            prompt, agent_mode="interactive", timeout=budget,
                        )
                        if response is None or not isinstance(response.data.content, str):
                            raise RuntimeError("Copilot SDK returned no assistant message")
                        return completion(response.data.content, model=model, tools=tools,
                                          tool_choice=tool_choice, usage=None, stream=stream)
                finally:
                    await _cleanup(client, session)
        finally:
            with self._lock:
                self._active.pop(task, None)


async def _cleanup(client, session):
    """Bound teardown even when the send RPC or runtime has stopped responding."""
    if session is not None:
        try:
            async with asyncio.timeout(_CLEANUP_TIMEOUT):
                await session.abort()
                await session.disconnect()
                await client.delete_session(session.session_id)
        except Exception:
            logger.warning("Copilot SDK session cleanup failed; stopping its owned runtime", exc_info=True)
    try:
        async with asyncio.timeout(_CLEANUP_TIMEOUT):
            errors = await client.stop()
        if errors:
            logger.warning("Copilot SDK graceful stop reported errors: %s", errors)
            await asyncio.wait_for(client.force_stop(), timeout=_CLEANUP_TIMEOUT)
    except Exception:
        logger.warning("Copilot SDK graceful stop failed; forcing owned runtime shutdown", exc_info=True)
        await asyncio.wait_for(client.force_stop(), timeout=_CLEANUP_TIMEOUT)
