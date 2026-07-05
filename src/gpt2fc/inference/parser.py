import json
import re


def extract_functioncall(text):
    """Extract the JSON payload of a `<functioncall> {...}` block as a dict.

    Glaive formats function calls with the arguments object wrapped in single
    quotes — `{"name": "f", "arguments": '{"k": "v"}'}` — which is not valid
    JSON. The payload span is found by brace-depth counting (regexes fail on
    nested objects); if plain json.loads fails, the single-quoted arguments
    blob is unwrapped into raw JSON and parsing is retried.

    Returns None when no parseable function call is present.
    """
    start = text.find("<functioncall>")
    if start == -1:
        return None
    json_start = text.find("{", start)
    if json_start == -1:
        return None

    depth = 0
    span = None
    for i, ch in enumerate(text[json_start:], json_start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                span = text[json_start:i + 1]
                break
    if span is None:
        return None

    try:
        return json.loads(span)
    except json.JSONDecodeError:
        pass

    unwrapped = re.sub(r"'(\{.*\})'", r"\1", span, flags=re.DOTALL)
    try:
        return json.loads(unwrapped)
    except json.JSONDecodeError:
        return None
