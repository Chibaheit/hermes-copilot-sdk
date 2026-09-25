from importlib.metadata import entry_points

import pytest
import yaml

from hermes_copilot_sdk.client import CopilotSDKClient


@pytest.mark.parametrize("settings,enabled", [
    ({}, False),
    ({"enabled": []}, False),
    ({"enabled": ["copilot-sdk"]}, True),
    ({"enabled": ["copilot-sdk"], "disabled": ["copilot-sdk"]}, False),
])
def test_installed_entry_point_obeys_real_hermes_opt_in(profiles, monkeypatch, settings, enabled):
    import providers
    monkeypatch.setattr(providers, "_REGISTRY", {})
    monkeypatch.setattr(providers, "_ALIASES", {})
    monkeypatch.setattr(providers, "_PROVIDER_LIST_CACHE", None)
    monkeypatch.setattr(providers, "_discovered", False)
    candidates = entry_points(group="hermes_agent.plugins", name="copilot-sdk")
    assert len(candidates) == 1, "Install this project editable before running integration tests"
    profile = profiles[0]
    (profile.home / "config.yaml").write_text(
        yaml.safe_dump({"plugins": settings}), encoding="utf-8")
    with profile.activate():
        discovered = providers.get_provider_profile("copilot-sdk")
    assert (discovered is not None) is enabled
    if enabled:
        assert discovered.auth_type == "external_process"
        assert discovered.supports_vision is False
        assert discovered.supports_vision_tool_messages is False
        assert discovered.supports_health_check is False
        with discovered.create_client() as client:
            assert isinstance(client, CopilotSDKClient)
