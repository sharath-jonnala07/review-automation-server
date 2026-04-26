"""LLM client wrapper with retries, structured output, and cost tracking."""

from dataclasses import dataclass
import re
from typing import TypeVar

import structlog
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from app.config import get_settings
from app.core.exceptions import SummarizationError
from app.summarization.cost_tracker import CostTracker

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

AUTH_FAILURE_SNIPPETS = (
    "invalid_api_key",
    "incorrect api key",
    "invalid api key",
    "authentication",
    "unauthorized",
    "error code: 401",
)
RATE_LIMIT_SNIPPETS = (
    "rate_limit_exceeded",
    "rate limit reached",
    "error code: 429",
    "tokens per day",
    "requests per minute",
)
NEGATIVE_HINTS = {
    "bug",
    "crash",
    "delay",
    "difficult",
    "error",
    "fail",
    "frustrated",
    "hang",
    "issue",
    "lag",
    "missing",
    "not",
    "poor",
    "problem",
    "slow",
    "stuck",
    "worst",
}
POSITIVE_HINTS = {
    "easy",
    "excellent",
    "fast",
    "good",
    "great",
    "helpful",
    "love",
    "smooth",
}
KEYPHRASE_STOPWORDS = {
    "app",
    "apps",
    "application",
    "best",
    "customer",
    "groww",
    "issue",
    "notes",
    "problem",
    "ui",
    "use",
    "using",
}


@dataclass(frozen=True)
class _LLMBackend:
    """Resolved LLM backend candidate."""

    name: str
    kind: str
    api_key: str | None = None
    base_url: str | None = None


class LLMClient:
    """Typed LLM client with structured output and cost tracking."""

    def __init__(
        self,
        model: str | None = None,
        temperature: float | None = None,
        cost_tracker: CostTracker | None = None,
    ) -> None:
        settings = get_settings()
        self._settings = settings
        self.model = model or settings.llm_model
        self.temperature = temperature or settings.llm_temperature
        self.cost_tracker = cost_tracker or CostTracker()
        self._backends = self._build_backends()
        self._clients: dict[str, ChatOpenAI] = {}
        self._active_backend = self._backends[0].name if self._backends else None

        if not self._backends:
            raise SummarizationError(
                "No LLM backend is configured. Set GROQ_API_KEY, OPENAI_API_KEY, "
                "or enable LLM_ALLOW_HEURISTIC_FALLBACK for local development."
            )

    @property
    def backend_name(self) -> str:
        """Return the backend configured or most recently used."""
        if self._active_backend is not None:
            return self._active_backend
        return self._backends[0].name

    def _build_backends(self) -> list[_LLMBackend]:
        """Resolve remote and local backend candidates in preference order."""
        settings = self._settings
        backends: list[_LLMBackend] = []

        if settings.llm_api_key:
            provider_name = settings.llm_provider if settings.llm_provider != "auto" else "custom"
            base_url = settings.llm_base_url
            if base_url is None and provider_name == "openai":
                base_url = settings.openai_base_url
            if base_url is None and provider_name == "groq":
                base_url = settings.groq_base_url
            backends.append(
                _LLMBackend(
                    name=provider_name,
                    kind="remote",
                    api_key=settings.llm_api_key,
                    base_url=base_url,
                )
            )
        elif settings.llm_provider == "groq":
            if settings.groq_api_key:
                backends.append(
                    _LLMBackend(
                        name="groq",
                        kind="remote",
                        api_key=settings.groq_api_key,
                        base_url=settings.groq_base_url,
                    )
                )
        elif settings.llm_provider == "openai":
            if settings.openai_api_key:
                backends.append(
                    _LLMBackend(
                        name="openai",
                        kind="remote",
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                    )
                )
        else:
            model_name = self.model.strip().lower()
            prefer_openai = model_name.startswith(("gpt", "o1", "o3", "o4"))

            if prefer_openai and settings.openai_api_key:
                backends.append(
                    _LLMBackend(
                        name="openai",
                        kind="remote",
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                    )
                )

            if settings.groq_api_key:
                backends.append(
                    _LLMBackend(
                        name="groq",
                        kind="remote",
                        api_key=settings.groq_api_key,
                        base_url=settings.groq_base_url,
                    )
                )

            if not prefer_openai and settings.openai_api_key:
                backends.append(
                    _LLMBackend(
                        name="openai",
                        kind="remote",
                        api_key=settings.openai_api_key,
                        base_url=settings.openai_base_url,
                    )
                )

        if settings.heuristic_llm_enabled:
            backends.append(_LLMBackend(name="heuristic", kind="heuristic"))

        return backends

    def _get_remote_llm(self, backend: _LLMBackend) -> ChatOpenAI:
        """Return a cached remote chat client."""
        cache_key = f"{backend.name}:{backend.base_url or 'default'}:{self.model}"
        if cache_key not in self._clients:
            self._clients[cache_key] = ChatOpenAI(
                model=self.model,
                temperature=self.temperature,
                api_key=backend.api_key,  # type: ignore[arg-type]
                base_url=backend.base_url,
            )
        return self._clients[cache_key]

    def _is_auth_failure(self, error: Exception) -> bool:
        """Return whether an exception points to bad credentials or auth."""
        message = str(error).lower()
        return any(snippet in message for snippet in AUTH_FAILURE_SNIPPETS)

    def _is_rate_limit_failure(self, error: Exception) -> bool:
        """Return whether an exception points to a temporary quota or rate cap."""
        message = str(error).lower()
        return any(snippet in message for snippet in RATE_LIMIT_SNIPPETS)

    def _raise_if_heuristic_auth_fallback_is_disabled(
        self,
        error: Exception,
        remaining_backends: list[_LLMBackend],
    ) -> None:
        """Prevent silent heuristic output when a configured remote backend rejects auth."""
        if self._settings.llm_allow_heuristic_auth_fallback:
            return
        if any(backend.kind == "remote" for backend in remaining_backends):
            return
        if any(backend.kind == "heuristic" for backend in remaining_backends):
            raise SummarizationError(
                "The configured remote LLM rejected authentication, so the run was stopped "
                "before using heuristic fallback. Add a valid GROQ_API_KEY or OPENAI_API_KEY, "
                "or set LLM_ALLOW_HEURISTIC_AUTH_FALLBACK=true only for local demos."
            ) from error

    def _parse_reviews(self, reviews_text: str) -> list[str]:
        """Extract review bullets from prompt variables."""
        lines = [
            line[2:].strip()
            for line in reviews_text.splitlines()
            if line.strip().startswith("- ")
        ]
        if lines:
            return lines
        return [segment.strip() for segment in reviews_text.split("\n\n") if segment.strip()]

    def _pick_label(self, keyphrases_text: str) -> str:
        """Derive a stable human label from cluster keyphrases."""
        for phrase in [item.strip() for item in keyphrases_text.split(",") if item.strip()]:
            tokens = [
                token
                for token in re.split(r"[^a-z0-9]+", phrase.lower())
                if token and token not in KEYPHRASE_STOPWORDS
            ]
            if tokens:
                return " ".join(tokens[:4]).title()
        return "Customer Feedback"

    def _classify_sentiment(self, reviews: list[str]) -> str:
        """Classify sentiment with simple lexical heuristics."""
        review_text = " ".join(reviews).lower()
        negative_score = sum(review_text.count(token) for token in NEGATIVE_HINTS)
        positive_score = sum(review_text.count(token) for token in POSITIVE_HINTS)
        if negative_score > positive_score:
            return "negative"
        if positive_score > negative_score:
            return "positive"
        return "neutral"

    def _heuristic_structured_call(
        self,
        output_schema: type[T],
        variables: dict[str, str],
    ) -> T:
        """Provide deterministic local outputs for dev and auth fallback."""
        field_names = set(output_schema.model_fields)

        if {"label", "description", "sentiment"}.issubset(field_names):
            reviews = self._parse_reviews(variables.get("reviews", ""))
            keyphrases = variables.get("keyphrases", "")
            label = self._pick_label(keyphrases)
            phrases = ", ".join(item.strip() for item in keyphrases.split(",")[:3] if item.strip())
            description = (
                f"Customers repeatedly mention {label.lower()} in recent reviews"
                + (f", especially around {phrases.lower()}." if phrases else ".")
            )
            return output_schema.model_validate(
                {
                    "label": label,
                    "description": description,
                    "sentiment": self._classify_sentiment(reviews),
                }
            )

        if "quotes" in field_names:
            reviews = self._parse_reviews(variables.get("reviews", ""))
            quotes = list(dict.fromkeys(sorted(reviews, key=len, reverse=True)))[:3]
            return output_schema.model_validate(
                {"quotes": quotes or ["No representative quote available."]}
            )

        if "actions" in field_names:
            themes = []
            for line in variables.get("themes", "").splitlines():
                match = re.match(r"\d+\.\s+(.+)", line.strip())
                if match:
                    themes.append(match.group(1).strip())
            actions = [
                {
                    "title": f"Investigate {theme.lower()}",
                    "description": (
                        f"Review the recurring feedback behind {theme.lower()} and "
                        "turn the highest-frequency issues into a concrete fix plan."
                    ),
                }
                for theme in themes
            ]
            if not actions:
                actions.append(
                    {
                        "title": "Review recurring feedback",
                        "description": "Consolidate the highest-volume themes into the next product review cycle.",
                    }
                )
            return output_schema.model_validate({"actions": actions})

        raise SummarizationError(
            f"Heuristic LLM fallback does not support schema {output_schema.__name__}."
        )

    async def ensure_ready(self) -> str:
        """Verify that at least one configured backend can serve the run."""
        if self._backends[0].kind == "heuristic" and len(self._backends) == 1:
            self._active_backend = self._backends[0].name
            return self.backend_name

        await self.chat(
            system_prompt="Reply with READY.",
            user_prompt="READY",
            max_retries=1,
        )
        return self.backend_name

    async def structured_call(
        self,
        prompt_template: str,
        output_schema: type[T],
        variables: dict[str, str],
        *,
        max_retries: int = 3,
    ) -> T:
        """Make a structured LLM call with validation."""
        last_error: Exception | None = None

        for backend_index, backend in enumerate(self._backends):
            if backend.kind == "heuristic":
                logger.info("Using heuristic LLM fallback", backend=backend.name)
                self._active_backend = backend.name
                return self._heuristic_structured_call(output_schema, variables)

            parser = PydanticOutputParser(pydantic_object=output_schema)
            prompt = ChatPromptTemplate.from_template(
                prompt_template,
                partial_variables={"format_instructions": parser.get_format_instructions()},
            )
            chain = prompt | self._get_remote_llm(backend) | parser

            for attempt in range(max_retries):
                try:
                    result = await chain.ainvoke(variables)
                    self.cost_tracker.charge(
                        self.model,
                        prompt_tokens=0,
                        completion_tokens=0,
                    )
                    self._active_backend = backend.name
                    return result  # type: ignore[no-any-return]
                except Exception as error:
                    last_error = error
                    logger.warning(
                        "LLM structured call failed",
                        attempt=attempt + 1,
                        backend=backend.name,
                        max_retries=max_retries,
                        error=str(error),
                    )
                    if self._is_auth_failure(error):
                        logger.warning(
                            "LLM backend authentication failed, trying fallback",
                            backend=backend.name,
                        )
                        self._raise_if_heuristic_auth_fallback_is_disabled(
                            error,
                            self._backends[backend_index + 1 :],
                        )
                        break
                    if self._is_rate_limit_failure(error):
                        logger.warning(
                            "LLM backend rate-limited, trying fallback",
                            backend=backend.name,
                        )
                        break
                    if attempt == max_retries - 1:
                        raise SummarizationError(
                            f"Failed to get valid structured output after {max_retries} attempts: {error}"
                        ) from error

        raise SummarizationError(
            f"Failed to get valid structured output after fallback attempts: {last_error}"
        )

    async def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_retries: int = 3,
    ) -> str:
        """Make a simple chat completion."""
        messages = [("system", system_prompt), ("user", user_prompt)]
        last_error: Exception | None = None

        for backend_index, backend in enumerate(self._backends):
            if backend.kind == "heuristic":
                self._active_backend = backend.name
                return "READY"

            llm = self._get_remote_llm(backend)
            for attempt in range(max_retries):
                try:
                    response = await llm.ainvoke(messages)
                    self._active_backend = backend.name
                    return str(response.content)
                except Exception as error:
                    last_error = error
                    logger.warning(
                        "LLM chat call failed",
                        attempt=attempt + 1,
                        backend=backend.name,
                        max_retries=max_retries,
                        error=str(error),
                    )
                    if self._is_auth_failure(error):
                        logger.warning(
                            "LLM backend authentication failed, trying fallback",
                            backend=backend.name,
                        )
                        self._raise_if_heuristic_auth_fallback_is_disabled(
                            error,
                            self._backends[backend_index + 1 :],
                        )
                        break
                    if self._is_rate_limit_failure(error):
                        logger.warning(
                            "LLM backend rate-limited, trying fallback",
                            backend=backend.name,
                        )
                        break
                    if attempt == max_retries - 1:
                        raise SummarizationError(
                            f"LLM chat failed after {max_retries} attempts: {error}"
                        ) from error

        raise SummarizationError(f"LLM chat failed after fallback attempts: {last_error}")
