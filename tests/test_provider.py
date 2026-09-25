from types import SimpleNamespace

import pytest

from hermes_copilot_sdk.client import CopilotSDKClient


@pytest.fixture
def fresh_registry(monkeypatch):
    import providers
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", False)
    return providers


def test_real_runtime_resolver_reaches_main_provider_client(profiles, fresh_registry):
    with profiles[0].activate():
        from hermes_cli.runtime_provider import resolve_runtime_provider
        from agent.agent_runtime_helpers import _provider_supplied_client

        runtime = resolve_runtime_provider(
            requested="copilot-sdk",
        )
        assert runtime["provider"] == "copilot-sdk"
        assert runtime["base_url"].startswith("copilot-sdk:")
        client = _provider_supplied_client(
            SimpleNamespace(provider=runtime["provider"]),
            {"api_key": runtime["api_key"], "base_url": runtime["base_url"]},
        )
        assert isinstance(client, CopilotSDKClient)
        client.close()


@pytest.mark.parametrize("async_mode", [False, True])
def test_real_auxiliary_resolver_preserves_native_client(profiles, fresh_registry, async_mode):
    with profiles[0].activate():
        from agent.auxiliary_client import resolve_provider_client

        client, model = resolve_provider_client(
            "copilot-sdk", model="model-A", async_mode=async_mode,
        )
        assert isinstance(client, CopilotSDKClient)
        assert model == "model-A"
        assert client.HERMES_SKIP_ASYNC_WRAP and client.HERMES_SKIP_TRANSPORT_WRAP
        client.close()


def test_async_auxiliary_task_uses_real_profile_config(profiles, fresh_registry):
    from agent.auxiliary_client import get_async_text_auxiliary_client

    with profiles[1].activate():
        client, model = get_async_text_auxiliary_client(
            "compression",
            main_runtime={"provider": "copilot-sdk", "model": "model-B",
                          "base_url": "copilot-sdk://runtime", "api_key": "copilot-sdk"},
        )
        assert isinstance(client, CopilotSDKClient)
        assert model == "model-B"
        client.close()
