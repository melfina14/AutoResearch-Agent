
import time
from google import genai

from app.config import GEMINI_API_KEY, GEMINI_MODEL


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Check your .env configuration."
    )

client = genai.Client(api_key=GEMINI_API_KEY)


def generate(prompt: str, max_retries: int = 4) -> str:
    """
    Generate text with Gemini, retrying temporary API failures.

    Retries temporary server overloads and rate limits.
    Authentication and other permanent errors are reported clearly.
    """

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("The prompt cannot be empty.")

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )

            result = response.text

            if not result or not result.strip():
                raise RuntimeError(
                    "Gemini returned an empty response. Please try again."
                )

            return result.strip()

        except Exception as exc:
            last_error = exc
            error_text = str(exc).lower()

            temporary_error = any(
                marker in error_text
                for marker in (
                    "503",
                    "unavailable",
                    "high demand",
                    "429",
                    "resource_exhausted",
                    "rate limit",
                    "500",
                    "502",
                    "504",
                    "internal server error",
                    "deadline exceeded",
                    "timed out",
                    "temporarily",
                )
            )

            if not temporary_error:
                raise RuntimeError(
                    f"Gemini API request failed: {exc}"
                ) from exc

            if attempt < max_retries - 1:
                wait_seconds = 2 ** (attempt + 1)
                print(
                    f"Gemini is temporarily unavailable. "
                    f"Retrying in {wait_seconds} seconds "
                    f"({attempt + 1}/{max_retries - 1})..."
                )
                time.sleep(wait_seconds)

    raise RuntimeError(
        f"Gemini remained unavailable after {max_retries} attempts. "
        f"Please wait a few minutes and try again. "
        f"Last error: {last_error}"
    ) from last_error
