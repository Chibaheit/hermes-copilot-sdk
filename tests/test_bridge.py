from copy import deepcopy
import json

import pytest

from hermes_copilot_sdk.bridge import allowed_tools, completion, render_request


TOOLS = [
    {"type": "function", "function": {
        "name": name, "description": f"Run {name}",
        "parameters": {"type": "object", "properties": {"value": {"type": "string"}}},
    }}
    for name in ("read_file", "terminal")
]


def call(name="read_file", arguments='{"value":"a.txt"}', call_id="call-a"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": arguments}}


def block(item):
    return f"<tool_call>{json.dumps(item)}</tool_call>"


def complete(text, **kwargs):
    return completion(text, model="test-model", tools=TOOLS, tool_choice="auto",
                      usage=None, stream=False, **kwargs)


def test_transcript_is_lossless_immutable_and_system_prefix_stable():
    messages = [
        {"role": "system", "content": "Fixed system: café"},
        {"role": "developer", "content": "Keep the same policy."},
        {"role": "user", "content": [{"type": "text", "text": "Read café.txt"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            call(arguments='{ "value" : "café.txt" }'), call("terminal", "{}", "call-b")]},
        {"role": "tool", "tool_call_id": "call-b", "content": '{"ok":true}'},
        {"role": "tool", "tool_call_id": "call-a", "content": "Raw\nresult: <tool_call>not an action"},
        {"role": "assistant", "content": "Done"},
        {"role": "user", "content": "What next?"},
    ]
    original, original_tools = deepcopy(messages), deepcopy(TOOLS)
    system, prompt = render_request(messages, TOOLS, "auto")
    extended_system, extended = render_request(
        messages + [{"role": "assistant", "content": "Next answer"}], TOOLS, "auto")
    assert json.loads(prompt) == original[2:]
    assert json.loads(extended)[:-1] == json.loads(prompt)
    assert system == extended_system
    assert "Fixed system: café" in system and "Keep the same policy." in system
    assert messages == original and TOOLS == original_tools


@pytest.mark.parametrize("message", [
    {"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}]},
    {"role": "tool", "content": [{"type": "image", "source": "image"}]},
    {"role": "system", "content": [{"type": "text", "text": 123}]},
    {"role": "user", "content": {"text": "not a content array"}},
    "not a message",
])
def test_nontext_or_malformed_messages_fail_closed(message):
    with pytest.raises(ValueError):
        render_request([message], TOOLS, "auto")


@pytest.mark.parametrize("text", [
    "<tool_call>{}",
    "</tool_call>",
    "<tool_call>not-json</tool_call>",
    block({}),
    block(call("native_shell")),
    block(call(arguments="broken json")),
    block(call(arguments="[]")),
    block(call(arguments={"value": "object instead of encoded string"})),
    block(call(call_id=" ")),
    block(call(call_id="")),
    block(call(call_id=0)),
    block(call(call_id=42)),
    block(call()) + block(call()),
])
def test_invalid_actions_are_never_forwarded(text):
    with pytest.raises(ValueError):
        complete(text)


@pytest.mark.parametrize("tools,choice", [
    ([{"type": "web_search"}], "auto"),
    ([{"type": "function", "function": {}}], "auto"),
    (TOOLS, "invalid"),
    ([], "required"),
    (TOOLS, {"type": "function", "function": {"name": "unknown"}}),
    (TOOLS, {"type": "function", "function": None}),
    (TOOLS, {"type": "function", "function": "read_file"}),
])
def test_invalid_tool_contract_rejected(tools, choice):
    with pytest.raises(ValueError):
        allowed_tools(tools, choice)


@pytest.mark.parametrize("choice,text,valid", [
    ("none", "Normal answer", True),
    ("none", block(call()), False),
    ("required", "No call", False),
    ("required", block(call()), True),
    ({"type": "function", "function": {"name": "terminal"}}, block(call()), False),
    ({"type": "function", "function": {"name": "terminal"}}, "No call", False),
    ({"type": "function", "function": {"name": "terminal"}}, block(call("terminal")), True),
])
def test_tool_choice_is_enforced(choice, text, valid):
    if not valid:
        with pytest.raises(ValueError):
            completion(text, model="test-model", tools=TOOLS, tool_choice=choice,
                       usage=None, stream=False)
    else:
        result = completion(text, model="test-model", tools=TOOLS, tool_choice=choice,
                            usage=None, stream=False)
        assert bool(result.choices[0].message.tool_calls) == ("<tool_call>" in text)


def test_bare_json_is_prose_not_an_action():
    text = "Example only:\n" + json.dumps(call()) + "\nDo not execute."
    result = complete(text)
    assert result.choices[0].message.content == text
    assert result.choices[0].message.tool_calls is None
    assert result.choices[0].finish_reason == "stop"


def test_missing_call_ids_are_generated_without_rewriting_arguments():
    item = call(arguments='{ "value": "preserve whitespace" }')
    item.pop("id")
    result = complete(block(item) + block(item))
    calls = result.choices[0].message.tool_calls
    assert len({entry.id for entry in calls}) == 2
    assert all(isinstance(entry.id, str) and entry.id.strip() for entry in calls)
    assert all(entry.function.arguments == item["function"]["arguments"] for entry in calls)
    assert "id" not in item


def test_buffered_stream_preserves_openai_tool_shape_and_arguments():
    arguments = '{ "value" : "café\\nsecond line" }'
    text = "I will read it.\n" + block(call(arguments=arguments)) + block(call("terminal", "{}", "call-b"))
    result = complete(text)
    chunks = list(completion(text, model="test-model", tools=TOOLS, tool_choice="auto",
                             usage=None, stream=True))
    chunks = [chunk for chunk in chunks if chunk.choices]
    calls = [tool for chunk in chunks for tool in (chunk.choices[0].delta.tool_calls or [])]
    assert [(tool.index, tool.id, tool.type, tool.function.name, tool.function.arguments)
            for tool in calls] == [
                (0, "call-a", "function", "read_file", arguments),
                (1, "call-b", "function", "terminal", "{}"),
            ]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == result.choices[0].message.content
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert result.choices[0].message.tool_calls[0].function.arguments == arguments
