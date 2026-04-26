from types import SimpleNamespace

import pytest

from app.mcp_client.docs_ops import DocsPublisher


@pytest.mark.anyio
async def test_publish_uses_configured_doc_id(monkeypatch: pytest.MonkeyPatch) -> None:
    publisher = DocsPublisher(SimpleNamespace())

    async def fail_resolve_document(_product: str, title: str | None = None) -> str:
        raise AssertionError("resolve_document should not be called when doc_id is configured")

    async def return_false(_doc_id: str, _anchor: str) -> bool:
        return False

    async def append_section(doc_id: str, requests: list[dict[str, object]], anchor: str) -> str:
        assert doc_id == "doc-123"
        assert requests == [{"insertText": {}}]
        assert anchor == "pulse-groww-2026-W17"
        return "heading-456"

    monkeypatch.setattr(publisher, "resolve_document", fail_resolve_document)
    monkeypatch.setattr(publisher, "check_anchor", return_false)
    monkeypatch.setattr(publisher, "append_section", append_section)

    doc_id, heading_id = await publisher.publish(
        product="groww",
        doc_requests=[{"insertText": {}}],
        anchor="pulse-groww-2026-W17",
        doc_id="doc-123",
    )

    assert doc_id == "doc-123"
    assert heading_id == "heading-456"