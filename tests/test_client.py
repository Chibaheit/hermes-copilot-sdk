import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import copilot
from copilot.generated.session_events import Data, SessionEvent, SessionEventType
from copilot.rpc import PermissionDecisionReject
import httpx
import pytest

from hermes_copilot_sdk.client import CopilotSDKClient


SDK_CLIENT = copilot.CopilotClient
SDK_SESSION = copilot.CopilotSession
MESSAGES = [{"role": "user", "content": "Say hello"}]
TOOL = {"type": "function", "function": {
    "name": "read_file", "parameters": {"type": "object"}}}
ACTION = '<tool_call>{"id":"read-1","type":"function","function":{"name":"read_file","arguments":"{}"}}</tool_call>'


@dataclass
class Transport:
    text: str = "Hello from Copilot"
    failure: str | None = None
    blocked: str | None = None
    stop_errors: bool = False
    delays: dict = field(default_factory=dict)
    expected_waiters: int = 1
    waiters: int = 0
    instances: list = field(default_factory=list)
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    release: asyncio.Event = field(default_factory=asyncio.Event)

    async def step(self, name):
        if name in self.delays:
            await asyncio.sleep(self.delays[name])
        if self.failure == name:
            raise RuntimeError(f"synthetic {name} failure")
        if self.blocked == name:
            self.waiters += 1
            if self.waiters == self.expected_waiters:
                self.entered.set()
            await self.release.wait()


@pytest.fixture
def transport(monkeypatch):
    state = Transport()

    class Session:
        def __init__(self, owner):
            self.owner = owner
            self.session_id = uuid4().hex

        async def send_and_wait(self, prompt, **kwargs):
            inspect.signature(SDK_SESSION.send_and_wait).bind(self, prompt, **kwargs)
            self.owner.prompt, self.owner.send_kwargs = prompt, kwargs
            self.owner.events.append("send")
            await state.step("send")
            if state.text is None:
                return None
            return SessionEvent(
                id=uuid4(), timestamp=datetime.now(timezone.utc),
                type=SessionEventType.ASSISTANT_MESSAGE, data=Data(content=state.text),
            )

        async def abort(self):
            self.owner.events.append("abort")
            await state.step("abort")

        async def disconnect(self):
            self.owner.events.append("disconnect")
            await state.step("disconnect")

    class Client:
        def __init__(self, **kwargs):
            inspect.signature(SDK_CLIENT).bind(**kwargs)
            self.kwargs, self.events = kwargs, []
            self.session = Session(self)
            self.session_kwargs = None
            state.instances.append(self)
            assert Path(kwargs["base_directory"]).is_dir()

        async def start(self):
            self.events.append("start")
            await state.step("start")

        async def create_session(self, **kwargs):
            inspect.signature(SDK_CLIENT.create_session).bind(self, **kwargs)
            self.session_kwargs = kwargs
            self.events.append("create")
            await state.step("create")
            return self.session

        async def delete_session(self, session_id):
            assert session_id == self.session.session_id
            self.events.append("delete")
            await state.step("delete")

        async def list_models(self):
            self.events.append("models")
            return [SimpleNamespace(id="model-A"), SimpleNamespace(id="model-B")]

        async def stop(self):
            self.events.append("stop")
            await state.step("stop")
            return [RuntimeError("synthetic shutdown error")] if state.stop_errors else []

        async def force_stop(self):
            self.events.append("force_stop")

    monkeypatch.setattr(copilot, "CopilotClient", Client)
    return state


def create(client, **kwargs):
    return client.chat.completions.create(model="model-A", messages=MESSAGES, **kwargs)


def assert_cleaned(runtime, *, session=True):
    assert "stop" in runtime.events
    if session:
        assert runtime.events[-4:] == ["abort", "disconnect", "delete", "stop"]
    assert not Path(runtime.kwargs["base_directory"]).exists()


def test_sync_plain_and_model_listing_lifecycle(profiles, transport):
    with profiles[0].activate(), CopilotSDKClient() as client:
        answer = create(client)
        assert answer.choices[0].message.content == transport.text
        assert answer.choices[0].finish_reason == "stop"
        assert answer.usage is None
        assert client.list_models() == ["model-A", "model-B"]
    assert client.is_closed
    assert len(transport.instances) == 2
    assert_cleaned(transport.instances[0])
    assert_cleaned(transport.instances[1], session=False)
    assert transport.instances[1].session_kwargs is None


@pytest.mark.parametrize("stream", [False, True])
@pytest.mark.parametrize("text", ["Hello", ACTION])
async def test_async_text_tools_and_buffered_stream(profiles, transport, text, stream):
    transport.text = text
    with profiles[0].activate(), CopilotSDKClient() as client:
        response = await create(client, stream=stream, tools=[TOOL])
        if stream:
            buffered = [chunk async for chunk in response]
            assert all(getattr(chunk, "usage", None) is None for chunk in buffered)
            chunks = [chunk for chunk in buffered if chunk.choices]
            assert chunks[-1].choices[0].finish_reason == ("tool_calls" if text == ACTION else "stop")
            if text == ACTION:
                calls = [call for chunk in chunks for call in (chunk.choices[0].delta.tool_calls or [])]
                assert calls[0].id == "read-1"
                assert calls[0].function.arguments == "{}"
            else:
                assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == text
        else:
            assert response.usage is None
            message = response.choices[0].message
            if text == ACTION:
                assert message.tool_calls[0].id == "read-1"
                assert message.tool_calls[0].function.name == "read_file"
            else:
                assert message.content == text
    assert_cleaned(transport.instances[0])


def test_sync_buffered_tool_stream(profiles, transport):
    transport.text = ACTION
    with profiles[0].activate(), CopilotSDKClient() as client:
        chunks = [chunk for chunk in create(client, stream=True, tools=[TOOL]) if chunk.choices]
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert any(chunk.choices[0].delta.tool_calls for chunk in chunks)


def test_real_profile_a_b_a_and_sdk_options(profiles, transport, monkeypatch):
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "synthetic-launch-token")
    monkeypatch.setenv("COPILOT_CLI_PATH", "must-not-be-inherited")
    monkeypatch.setenv("GITHUB_TOKEN", "synthetic-launch-github")
    monkeypatch.setenv("GH_TOKEN", "synthetic-launch-gh")
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-other-provider")
    with CopilotSDKClient() as client:
        for profile in (profiles[0], profiles[1], profiles[0]):
            with profile.activate():
                from hermes_cli.config import load_config
                from agent.secret_scope import get_secret

                assert get_secret("COPILOT_GITHUB_TOKEN") == profile.token
                assert load_config()["copilot_sdk"]["timeout_seconds"] == profile.timeout
                create(client, reasoning_effort="high")
            runtime = transport.instances[-1]
            opts, session = runtime.kwargs, runtime.session_kwargs
            assert opts["mode"] == "empty"
            assert opts["github_token"] == profile.token
            assert opts["use_logged_in_user"] is False
            assert opts["working_directory"] == opts["base_directory"]
            assert Path(opts["base_directory"]).parent == profile.home / "copilot-sdk"
            assert opts["env"]["HERMES_HOME"] == str(profile.home)
            assert not any(key.startswith("COPILOT_") or key in (
                "GITHUB_TOKEN", "GH_TOKEN", "OPENAI_API_KEY") for key in opts["env"])
            assert session["working_directory"] == session["config_directory"] == opts["base_directory"]
            assert session["available_tools"] == []
            assert not session.get("tools")
            assert session["skip_custom_instructions"] is True
            for flag in ("enable_config_discovery", "enable_on_demand_instruction_discovery",
                         "enable_file_hooks", "enable_host_git_operations", "enable_skills",
                         "enable_session_store", "enable_session_telemetry", "streaming"):
                assert session[flag] is False
            assert session["memory"] == {"enabled": False}
            assert session["infinite_sessions"] == {"enabled": False}
            assert session["system_message"]["mode"] == "replace"
            assert session["reasoning_effort"] == "high"
            decision = session["on_permission_request"]({}, {})
            assert isinstance(decision, PermissionDecisionReject)
            assert runtime.send_kwargs == {"agent_mode": "interactive", "timeout": profile.timeout}
            assert json.loads(runtime.prompt) == MESSAGES
            assert_cleaned(runtime)
    directories = [runtime.kwargs["base_directory"] for runtime in transport.instances]
    assert len(set(directories)) == 3


@pytest.mark.parametrize("phase", ["start", "create", "send"])
async def test_deadline_bounds_startup_session_and_send(profiles, transport, phase):
    transport.blocked = phase
    with profiles[0].activate(), CopilotSDKClient() as client:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(create(client, timeout=0.05), timeout=3)
    assert transport.entered.is_set()
    assert_cleaned(transport.instances[0], session=phase == "send")
    if phase == "start":
        assert transport.instances[0].session_kwargs is None


async def test_startup_and_send_share_one_outer_deadline(profiles, transport):
    transport.delays = {"start": 0.6, "send": 0.6}
    with profiles[0].activate(), CopilotSDKClient() as client:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(create(client, timeout=1.0), timeout=4)
    assert_cleaned(transport.instances[0], session="create" in transport.instances[0].events)


@pytest.mark.parametrize("phase", ["start", "create", "send"])
async def test_runtime_errors_propagate_and_always_stop(profiles, transport, phase):
    transport.failure = phase
    with profiles[0].activate(), CopilotSDKClient() as client:
        with pytest.raises(RuntimeError, match=f"synthetic {phase} failure"):
            await create(client)
    assert_cleaned(transport.instances[0], session=phase == "send")


@pytest.mark.parametrize("text", [None, "", " \n "])
async def test_empty_response_is_not_success(profiles, transport, text):
    transport.text = text
    with profiles[0].activate(), CopilotSDKClient() as client:
        with pytest.raises(RuntimeError, match="no assistant"):
            await create(client)
    assert_cleaned(transport.instances[0])


@pytest.mark.parametrize("phase", ["abort", "disconnect", "delete", "stop", "stop-errors"])
async def test_cleanup_failures_still_stop_owned_runtime(profiles, transport, phase):
    if phase == "stop-errors":
        transport.stop_errors = True
    else:
        transport.failure = phase
    with profiles[0].activate(), CopilotSDKClient() as client:
        assert (await create(client)).choices[0].message.content == transport.text
    runtime = transport.instances[0]
    assert "stop" in runtime.events
    if phase in ("stop", "stop-errors"):
        assert runtime.events[-1] == "force_stop"
    assert not Path(runtime.kwargs["base_directory"]).exists()


async def test_hung_cleanup_forces_stop(profiles, transport, monkeypatch):
    import hermes_copilot_sdk.client as module
    monkeypatch.setattr(module, "_CLEANUP_TIMEOUT", 0.05)
    transport.blocked = "stop"
    with profiles[0].activate(), CopilotSDKClient() as client:
        answer = await asyncio.wait_for(create(client), timeout=3)
    assert answer.choices[0].message.content == transport.text
    assert transport.instances[0].events[-1] == "force_stop"


async def test_close_cancels_active_call_and_rejects_reuse(profiles, transport):
    transport.blocked = "send"
    with profiles[0].activate():
        client = CopilotSDKClient()
        pending = asyncio.create_task(create(client))
        await asyncio.wait_for(transport.entered.wait(), timeout=3)
        client.close()
        client.close()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(pending, timeout=3)
        with pytest.raises(RuntimeError, match="closed"):
            await create(client)
    assert len(transport.instances) == 1
    assert_cleaned(transport.instances[0])


async def test_repeated_close_does_not_cancel_in_progress_cleanup(profiles, transport):
    transport.blocked = "send"
    with profiles[0].activate():
        client = CopilotSDKClient()
        pending = asyncio.create_task(create(client))
        try:
            await asyncio.wait_for(transport.entered.wait(), timeout=3)
            transport.entered.clear()
            transport.waiters = 0
            transport.blocked = "abort"
            client.close()
            await asyncio.wait_for(transport.entered.wait(), timeout=3)
            client.close()
            transport.release.set()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(pending, timeout=3)
        finally:
            transport.release.set()
            client.close()
    assert_cleaned(transport.instances[0])


async def test_concurrent_profiles_have_separate_sessions_and_tokens(profiles, transport):
    transport.blocked = "send"
    transport.expected_waiters = 2
    client = CopilotSDKClient()

    async def request(profile):
        with profile.activate():
            return await create(client)

    tasks = [asyncio.create_task(request(profile)) for profile in profiles]
    try:
        await asyncio.wait_for(transport.entered.wait(), timeout=3)
        transport.release.set()
        answers = await asyncio.wait_for(asyncio.gather(*tasks), timeout=3)
    finally:
        client.close()
    assert [answer.choices[0].message.content for answer in answers] == [transport.text] * 2
    runtimes = transport.instances
    assert {runtime.kwargs["github_token"] for runtime in runtimes} == {p.token for p in profiles}
    assert len({runtime.session.session_id for runtime in runtimes}) == 2
    assert len({runtime.kwargs["base_directory"] for runtime in runtimes}) == 2
    for runtime in runtimes:
        assert_cleaned(runtime)


async def test_close_cancels_every_concurrent_session(profiles, transport):
    transport.blocked = "send"
    transport.expected_waiters = 2
    client = CopilotSDKClient()
    with profiles[0].activate():
        tasks = [asyncio.create_task(create(client)) for _ in range(2)]
        try:
            await asyncio.wait_for(transport.entered.wait(), timeout=3)
            client.close()
            results = await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=3)
        finally:
            client.close()
    assert all(isinstance(result, asyncio.CancelledError) for result in results)
    for runtime in transport.instances:
        assert_cleaned(runtime)


async def test_sync_request_error_does_not_close_client_or_cancel_sibling(profiles, transport):
    transport.blocked = "send"
    with profiles[0].activate(), CopilotSDKClient() as client:
        sibling = asyncio.create_task(create(client))
        try:
            await asyncio.wait_for(transport.entered.wait(), timeout=3)
            with pytest.raises(ValueError, match="response_format"):
                await asyncio.to_thread(
                    create, client, response_format={"type": "json_schema"},
                )
            assert not client.is_closed
            transport.release.set()
            response = await asyncio.wait_for(sibling, timeout=3)
            assert response.choices[0].message.content == transport.text
        finally:
            transport.release.set()
            client.close()
            await asyncio.gather(sibling, return_exceptions=True)
    assert len(transport.instances) == 1
    assert_cleaned(transport.instances[0])


@pytest.mark.parametrize("options", [
    {"model": ""},
    {"response_format": {"type": "json_schema"}},
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "image"}}]}]},
    {"tools": [], "tool_choice": "required"},
])
async def test_unsupported_requests_never_start_runtime(profiles, transport, options):
    request = {"model": "model-A", "messages": MESSAGES, **options}
    with profiles[0].activate(), CopilotSDKClient() as client:
        with pytest.raises(ValueError):
            await client.chat.completions.create(**request)
    assert transport.instances == []


@pytest.mark.parametrize("timeout", [
    0, -1, float("nan"), float("inf"), "not-a-timeout", "1.0", {}, object(),
])
async def test_invalid_timeout_starts_no_runtime(profiles, transport, timeout):
    with profiles[0].activate(), CopilotSDKClient() as client:
        with pytest.raises(ValueError, match="timeout"):
            await create(client, timeout=timeout)
    assert transport.instances == []


@pytest.mark.parametrize("override,expected", [(None, 7.0), (9.0, 9.0)])
async def test_http_timeout_components_and_request_override(profiles, transport, override, expected):
    timeout = httpx.Timeout(connect=2.0, read=7.0, write=3.0, pool=1.0)
    with profiles[0].activate(), CopilotSDKClient(timeout=timeout) as client:
        await create(client, timeout=override)
    assert transport.instances[0].send_kwargs["timeout"] == expected
    assert_cleaned(transport.instances[0])


@pytest.mark.parametrize("multiplex", [False, True])
async def test_missing_scoped_token_never_falls_back_to_launch_credentials(
        profiles, transport, monkeypatch, multiplex):
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "synthetic-launch-token")
    (profiles[1].home / ".env").write_text("GITHUB_TOKEN=synthetic-unrelated-token\n", encoding="utf-8")
    with profiles[1].activate(), CopilotSDKClient() as client:
        assert "COPILOT_GITHUB_TOKEN" not in secret_scope.current_secret_scope()
        with pytest.raises(RuntimeError, match="COPILOT_GITHUB_TOKEN"):
            await create(client)
    assert transport.instances == []


@pytest.mark.parametrize("multiplex", [False, True])
async def test_routed_home_without_bound_scope_cannot_use_launch_token(
        profiles, transport, monkeypatch, multiplex):
    from agent import secret_scope
    monkeypatch.setattr(secret_scope, "_MULTIPLEX_ACTIVE", multiplex)
    monkeypatch.setenv("COPILOT_GITHUB_TOKEN", "synthetic-launch-token")
    with profiles[1].activate(), CopilotSDKClient() as client:
        scope = secret_scope.set_secret_scope(None)
        try:
            with pytest.raises(RuntimeError, match="bound secret scope"):
                await create(client)
        finally:
            secret_scope.reset_secret_scope(scope)
    assert transport.instances == []
