from __future__ import annotations

import asyncio
import os

from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI, RateLimitError


class LLMServiceError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class LLMClient:
    def __init__(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY. Set it in .env, .env.example, or the server environment."
            )

        base_url = os.getenv("OPENAI_BASE_URL")
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)

        self.default_model = os.getenv("OPENAI_MODEL")
        if not self.default_model:
            raise RuntimeError("Missing OPENAI_MODEL. Set it in .env or the environment.")

    async def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> str:
        last_error: Exception | None = None
        selected_model = model or self.default_model
        extra_body = None
        if selected_model.startswith("deepseek-v4"):
            extra_body = {
                "thinking": {
                    "type": os.getenv("DEEPSEEK_THINKING", "disabled"),
                }
            }
        elif selected_model.startswith("mimo-") or selected_model.startswith("xiaomi/mimo-"):
            extra_body = {
                "thinking": {
                    "type": os.getenv("MIMO_THINKING", "disabled"),
                }
            }

        # Query planning is an optional accelerator, not a reason to hold a search
        # request for minutes. Keep retries conservative; callers can fall back to
        # deterministic rules when a provider is unavailable.
        try:
            max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))
        except ValueError:
            max_attempts = 2
        max_attempts = max(1, min(max_attempts, 5))
        try:
            retry_max_delay = float(os.getenv("LLM_RETRY_MAX_DELAY_SECONDS", "3"))
        except ValueError:
            retry_max_delay = 3.0
        retry_max_delay = max(0.0, min(retry_max_delay, 30.0))
        try:
            request_timeout = float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "30"))
        except ValueError:
            request_timeout = 30.0
        request_timeout = max(3.0, min(request_timeout, 90.0))
        for attempt in range(max_attempts):
            try:
                response = await self.client.chat.completions.create(
                    model=selected_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=request_timeout,
                    extra_body=extra_body,
                )
                break
            except (APIConnectionError, APITimeoutError, RateLimitError) as error:
                last_error = error
                await asyncio.sleep(min(retry_max_delay, 2**attempt))
            except APIStatusError as error:
                last_error = error
                if error.status_code < 500:
                    raise LLMServiceError(
                        f"Upstream model request failed: HTTP {error.status_code}. {error.message}",
                        error.status_code,
                    ) from error
                await asyncio.sleep(min(retry_max_delay, 2**attempt))
        else:
            if isinstance(last_error, APIStatusError):
                raise LLMServiceError(
                    f"Upstream model service is temporarily unavailable: HTTP {last_error.status_code}. Please try again later.",
                    last_error.status_code,
                ) from last_error
            raise LLMServiceError(
                "Upstream model service timed out or is temporarily unavailable. Please try again later.",
                503,
            ) from last_error

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMServiceError(
                "The upstream model output hit the token limit and was truncated. "
                "Try fast mode, narrow the topic, or increase max_tokens.",
                502,
            )

        content = choice.message.content
        if not content:
            raise LLMServiceError("The upstream model returned empty content. Please try again later.", 502)
        return content.strip()
