from gpt2fc.inference.parser import extract_functioncall


def test_plain_json():
    text = '###ASSISTANT: <functioncall> {"name": "get_random_fact", "arguments": {}}'
    assert extract_functioncall(text) == {"name": "get_random_fact", "arguments": {}}


def test_glaive_single_quoted_arguments():
    text = '###ASSISTANT: <functioncall> {"name": "get_movie_details", "arguments": \'{"title": "Inception"}\'}'
    assert extract_functioncall(text) == {
        "name": "get_movie_details",
        "arguments": {"title": "Inception"},
    }


def test_nested_arguments():
    text = '<functioncall> {"name": "f", "arguments": \'{"outer": {"inner": 1}, "k": [1, 2]}\'}'
    fc = extract_functioncall(text)
    assert fc == {"name": "f", "arguments": {"outer": {"inner": 1}, "k": [1, 2]}}


def test_apostrophe_in_value():
    text = '<functioncall> {"name": "f", "arguments": \'{"title": "Ender\'s Game"}\'}'
    fc = extract_functioncall(text)
    assert fc == {"name": "f", "arguments": {"title": "Ender's Game"}}


def test_no_functioncall():
    assert extract_functioncall("###ASSISTANT: Sure, the answer is 42.") is None


def test_unbalanced_braces():
    assert extract_functioncall('<functioncall> {"name": "f"') is None


def test_malformed_json():
    assert extract_functioncall("<functioncall> {name: broken}") is None


def test_trailing_text_after_call():
    text = '<functioncall> {"name": "f", "arguments": {}} some trailing text {"x": 1}'
    assert extract_functioncall(text) == {"name": "f", "arguments": {}}
