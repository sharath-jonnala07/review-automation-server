"""Build Google Docs batchUpdate request trees from PulseSummary."""

import structlog

from app.core.models import PulseSummary

logger = structlog.get_logger()


def _insert_text_request(text: str, index: int) -> dict[str, object]:
    """Create an insertText request."""
    return {
        "insertText": {
            "location": {"index": index},
            "text": text,
        }
    }


def _update_paragraph_style_request(
    start_index: int, end_index: int, named_style: str
) -> dict[str, object]:
    """Create an updateParagraphStyle request."""
    return {
        "updateParagraphStyle": {
            "range": {
                "startIndex": start_index,
                "endIndex": end_index,
            },
            "paragraphStyle": {
                "namedStyleType": named_style,
            },
            "fields": "namedStyleType",
        }
    }


def _create_bullets_request(start_index: int, end_index: int) -> dict[str, object]:
    """Create a createParagraphBullets request."""
    return {
        "createParagraphBullets": {
            "range": {
                "startIndex": start_index,
                "endIndex": end_index,
            },
            "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
        }
    }


def _update_text_style_request(
    start_index: int,
    end_index: int,
    bold: bool | None = None,
    italic: bool | None = None,
) -> dict[str, object]:
    """Create an updateTextStyle request."""
    style: dict[str, bool] = {}
    if bold is not None:
        style["bold"] = bold
    if italic is not None:
        style["italic"] = italic

    return {
        "updateTextStyle": {
            "range": {
                "startIndex": start_index,
                "endIndex": end_index,
            },
            "textStyle": style,
            "fields": ",".join(style.keys()),
        }
    }


class DocsTreeBuilder:
    """Builds a Google Docs batchUpdate request tree from a PulseSummary."""

    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []
        self._index = 1  # Docs body starts at index 1

    def _append_text(self, text: str) -> tuple[int, int]:
        """Append text and return (start, end) indices."""
        start = self._index
        self.requests.append(_insert_text_request(text, start))
        self._index += len(text)
        end = self._index
        return start, end

    def _newline(self) -> None:
        """Append a newline."""
        self._index += 1
        self.requests.append(_insert_text_request("\n", self._index - 1))

    def build(self, summary: PulseSummary, anchor: str) -> list[dict[str, object]]:
        """Convert PulseSummary to Docs batchUpdate requests.

        Args:
            summary: The pulse summary to render
            anchor: Idempotency anchor string (e.g. pulse-groww-2026-W16)

        Returns:
            List of batchUpdate request dicts
        """
        # Heading 1: Section title with anchor
        title = f"Weekly Pulse - {summary.product} - {anchor}\n"
        start, end = self._append_text(title)
        self.requests.append(
            _update_paragraph_style_request(start, end, "HEADING_1")
        )
        self._newline()

        # Stats line
        stats_text = (
            f"Period: {summary.window.start} to {summary.window.end} "
            f"({summary.stats.total_reviews} reviews, "
            f"avg rating {summary.stats.avg_rating})\n"
        )
        self._append_text(stats_text)
        self._newline()

        # Top Themes heading
        themes_heading = "Top Themes\n"
        start, end = self._append_text(themes_heading)
        self.requests.append(
            _update_paragraph_style_request(start, end, "HEADING_2")
        )

        # Themes as bullets
        for theme in summary.top_themes:
            theme_text = f"{theme.label} - {theme.description} ({theme.sentiment}, {theme.review_count} reviews)\n"
            theme_start, theme_end = self._append_text(theme_text)
            self.requests.append(
                _create_bullets_request(theme_start, theme_end)
            )

        self._newline()

        # Quotes heading
        quotes_heading = "Real User Quotes\n"
        start, end = self._append_text(quotes_heading)
        self.requests.append(
            _update_paragraph_style_request(start, end, "HEADING_2")
        )

        # Quotes as italic text
        for quote in summary.quotes:
            quote_text = f'"{quote.text}"\n'
            q_start, q_end = self._append_text(quote_text)
            self.requests.append(
                _update_text_style_request(q_start, q_end, italic=True)
            )

        self._newline()

        # Action Ideas heading
        actions_heading = "Action Ideas\n"
        start, end = self._append_text(actions_heading)
        self.requests.append(
            _update_paragraph_style_request(start, end, "HEADING_2")
        )

        # Actions as bullets
        for action in summary.action_ideas:
            action_text = f"{action.title} - {action.description}\n"
            a_start, a_end = self._append_text(action_text)
            self.requests.append(
                _create_bullets_request(a_start, a_end)
            )

        self._newline()

        # Google Docs batchUpdate does not support insertHorizontalRule.
        self._append_text("----------\n")

        logger.info(
            "Built Docs request tree",
            requests=len(self.requests),
            product=summary.product,
        )
        return self.requests


def build_doc_requests(summary: PulseSummary, anchor: str) -> list[dict[str, object]]:
    """Convenience function to build Docs requests."""
    builder = DocsTreeBuilder()
    return builder.build(summary, anchor)
