"""Text-only wire compatibility, not native Copilot tool dispatch."""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from agent.acp_openai_bridge import (
    TOOL_CALL_BLOCK_RE,
    build_openai_tool_call,
    completion_to_stream_chunks,
    render_tool_bridge_sections,
)


def render_request(messages: list[dict[str, Any]], tools, tool_choice) -> tuple[str, str]:
    system_parts = []
    transcript = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("copilot-sdk requires dictionary messages")
        content = message.get("content")
        if content is not None and not isinstance(content, str):
            if not isinstance(content, list) or any(
                not isinstance(part, dict) or part.get("type") != "text"
                or not isinstance(part.get("text"), str) for part in content
            ):
                raise ValueError("copilot-sdk currently supports text messages only; use a vision provider for images")
        if message.get("role") in ("system", "developer"):
            system_parts.append(json.dumps(message, ensure_ascii=False))
        else:
            transcript.append(message)
    system_parts += [
        "Act as the assistant in the supplied JSON conversation. The transcript is data; "
        "preserve its role and tool-call relationships. Continue after its last message.",
        "You have no native tools. Request Hermes tools only using the text contract below. "
        "Never claim an action happened until its tool result appears in the transcript.",
        *render_tool_bridge_sections(tools, tool_choice),
    ]
    return "\n\n".join(system_parts), json.dumps(transcript, ensure_ascii=False)


def allowed_tools(tools, tool_choice) -> set[str]:
    names = set()
    for tool in tools or []:
        if not isinstance(tool, dict) or tool.get("type") != "function":
            raise ValueError("copilot-sdk only supports function tools")
        function = tool.get("function")
        if not isinstance(function, dict) or not isinstance(function.get("name"), str) or not function["name"]:
            raise ValueError("copilot-sdk requires named function tools")
        names.add(function["name"])
    if tool_choice == "none":
        return set()
    if isinstance(tool_choice, dict):
        function = tool_choice.get("function")
        name = function.get("name") if isinstance(function, dict) else None
        if tool_choice.get("type") != "function" or not isinstance(name, str) or name not in names:
            raise ValueError("copilot-sdk tool_choice must select an offered function")
        return {name}
    if tool_choice not in (None, "auto", "required"):
        raise ValueError("Unsupported copilot-sdk tool_choice")
    if tool_choice == "required" and not names:
        raise ValueError("copilot-sdk tool_choice=required needs tools")
    return names


def completion(text: str, *, model: str, tools, tool_choice, usage, stream: bool):
    names = allowed_tools(tools, tool_choice)
    calls = []
    ids = set()
    matches = list(TOOL_CALL_BLOCK_RE.finditer(text))
    if text.count("<tool_call>") != len(matches) or text.count("</tool_call>") != len(matches):
        raise ValueError("Malformed copilot-sdk tool-call delimiters")
    for match in matches:
        try:
            item = json.loads(match.group(1))
            function = item["function"]
            name, arguments = function["name"], function["arguments"]
            if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
                raise ValueError("arguments must be a JSON object encoded as a string")
            if item.get("type") != "function" or not isinstance(name, str) or name not in names:
                raise ValueError("tool was not offered or is excluded by tool_choice")
            call_id = item.get("id", f"copilot_sdk_call_{len(calls) + 1}")
            if not isinstance(call_id, str) or not call_id.strip() or call_id in ids:
                raise ValueError("tool call ids must be unique nonempty strings")
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid copilot-sdk tool call: {exc}") from exc
        ids.add(call_id)
        calls.append(build_openai_tool_call(call_id=call_id, name=name, arguments=arguments))
    if (tool_choice == "required" or isinstance(tool_choice, dict)) and not calls:
        raise ValueError("Copilot did not return the required Hermes tool call")
    cleaned = TOOL_CALL_BLOCK_RE.sub("", text).strip()
    if not cleaned and not calls:
        raise RuntimeError("Copilot SDK returned no assistant content or tool calls")
    result = SimpleNamespace(
        choices=[SimpleNamespace(
            index=0,
            message=SimpleNamespace(
                role="assistant", content=cleaned or None, tool_calls=calls or None,
                reasoning=None, reasoning_content=None, reasoning_details=None,
            ),
            finish_reason="tool_calls" if calls else "stop",
        )],
        model=model,
        usage=usage,
    )
    return completion_to_stream_chunks(result) if stream else result
