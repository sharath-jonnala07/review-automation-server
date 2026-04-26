from app.agent.graph import _should_publish_gmail


def test_should_publish_gmail_without_docs_when_email_ready() -> None:
    decision = _should_publish_gmail({"dry_run": False, "email_html": "<p>ready</p>"})

    assert decision == "publish_gmail"


def test_should_not_publish_gmail_for_dry_run() -> None:
    decision = _should_publish_gmail({"dry_run": True, "email_html": "<p>ready</p>"})

    assert decision == "__end__"