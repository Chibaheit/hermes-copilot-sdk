"""Provider registration uses Hermes' existing external-process routing."""

import sys

from providers.base import ProviderProfile


class CopilotSDKProfile(ProviderProfile):
    def __init__(self):
        super().__init__(
            name="copilot-sdk",
            display_name="GitHub Copilot SDK (experimental)",
            description="Copilot SDK text-tool bridge; Hermes owns tool execution",
            base_url="copilot-sdk://runtime",
            auth_type="external_process",
            # The SDK provisions its matching runtime; no interactive CLI is required.
            process_command=sys.executable,
            supports_health_check=False,
            supports_model_listing=False,
            supports_vision=False,
            supports_vision_tool_messages=False,
            unsupported_response_formats=("json_object", "json_schema"),
        )

    def create_client(self, **client_kwargs):
        from .client import CopilotSDKClient

        return CopilotSDKClient(**client_kwargs)

    def fetch_models(self, *, timeout=8.0, **kwargs):
        from .client import CopilotSDKClient

        with CopilotSDKClient() as client:
            return client.list_models(timeout=timeout)
