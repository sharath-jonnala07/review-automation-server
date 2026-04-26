"""Branded types and literal types for type safety."""

from typing import Literal, NewType

# Branded IDs — prevent mixing up different string IDs
RunId = NewType("RunId", str)
ReviewId = NewType("ReviewId", str)
ThemeId = NewType("ThemeId", str)
ProductKey = NewType("ProductKey", str)

# Literals
ReviewSource = Literal["appstore", "playstore"]
Sentiment = Literal["negative", "mixed", "positive"]
RunStatus = Literal[
    "pending",
    "ingesting",
    "clustering",
    "summarizing",
    "rendering",
    "publishing",
    "completed",
    "failed",
]
