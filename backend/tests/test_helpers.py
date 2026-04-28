from app.graph.nodes._helpers import parse_json, render_docs_brief, render_history


def test_parse_json_plain():
    assert parse_json('{"a": 1}') == {"a": 1}


def test_parse_json_fenced():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_with_garbage():
    text = "여기 결과입니다:\n{\"intent\": \"question\", \"resolved_query\": \"hi\"}\n끝."
    assert parse_json(text) == {"intent": "question", "resolved_query": "hi"}


def test_parse_json_empty():
    assert parse_json("") == {}
    assert parse_json("not json at all") == {}


def test_render_history_empty():
    assert "없음" in render_history([])


def test_render_docs_brief():
    docs = [{"title": "T", "url": "http://x", "content": "abc" * 100}]
    out = render_docs_brief(docs)
    assert "[1]" in out
    assert "T" in out
    assert "http://x" in out
