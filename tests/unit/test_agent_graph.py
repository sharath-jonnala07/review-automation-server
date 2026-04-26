from app.agent.graph import _should_summarize


def test_should_summarize_allows_two_clusters() -> None:
    decision = _should_summarize({"clusters": [object(), object()]})

    assert decision == "summarize"


def test_should_summarize_stops_without_clusters() -> None:
    decision = _should_summarize({"clusters": []})

    assert decision == "__end__"