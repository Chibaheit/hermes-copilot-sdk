"""Explicitly enabled, pip-discovered Hermes model provider."""


def register() -> None:
    from providers import register_provider
    from .profile import CopilotSDKProfile

    register_provider(CopilotSDKProfile())
