"""Utility for sending Telegram bot messages from the web layer.

Uses httpx directly instead of python-telegram-bot's Bot class, which
conflicts with uvicorn's event loop when instantiated per-request.
"""
from dotenv import load_dotenv
load_dotenv()

import httpx
import os
import asyncio
from typing import Awaitable, Callable, TypeVar

T = TypeVar("T")

# Retry only on network errors
_RETRYABLE_ERRORS = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError, httpx.NetworkError)

def _is_retryable(exc: Exception) -> bool:
    """Check if an exception is a transient network error worth retrying."""
    if isinstance(exc, _RETRYABLE_ERRORS):
        return True
    exc_module = type(exc).__module__ or ""
    if exc_module.startswith("httpx") or exc_module.startswith("httpcore"):
        return True
    return False

async def _retry_on_network_error(
    coro_factory: Callable[..., Awaitable[T]],
    *args,
    max_retries: int = 5,
    base_delay: float = 2.0,
    **kwargs,
) -> T:
    """Call a coroutine, retrying on transient network errors with exponential backoff.

    Args:
        coro_factory: An async callable (e.g. ``update.message.reply_text``).
        *args: Positional arguments forwarded to the callable.
        max_retries: Maximum number of retry attempts (default 5).
        base_delay: Initial backoff delay in seconds, doubles each attempt.
        **kwargs: Keyword arguments forwarded to the callable.

    Returns:
        The return value of the callable.

    Raises:
        The last exception if all retries are exhausted.
    """
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return await coro_factory(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt >= max_retries:
                raise
            delay = base_delay * (2 ** attempt)
            # Do not print exc directly: its message embeds the request URL (bot token).
            print(f"Network error ({type(exc).__name__}). Retrying in {delay:.1f}s (attempt {attempt + 1}/{max_retries})...")
            await asyncio.sleep(delay)

    # Should be unreachable; appease type checkers and belt-and-suspenders.
    raise last_exc  # type: ignore[misc]


async def _send_request(payload: dict, url: str) -> httpx.Response:
    """Post the payload, retrying transient network errors."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        return await _retry_on_network_error(client.post, url, json=payload)


def send_bot_message(text: str, reply_markup=None) -> None:
    """Send a plain-text message via the bot. Best-effort: failures print instead of raising."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    missing = [
        name for name, val in (("TELEGRAM_BOT_TOKEN", token), ("TELEGRAM_CHAT_ID", chat_id))
        if not val
    ]
    if missing:
        print(f"{', '.join(missing)} not set; skipping Telegram notification.")
        return

    payload: dict = {
        "chat_id": str(chat_id),
        "text": text,
    }
    if reply_markup is not None:
        payload["reply_markup"] = reply_markup.to_dict()

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    try:
        response = asyncio.run(_send_request(payload, url)) # Run it synchronously for script callers
        response.raise_for_status()
        print("Telegram notification sent.")
    except httpx.HTTPStatusError as exc:
        print(f"Telegram rejected the message: {exc.response.text}")
    except Exception as exc:
        print(f"Failed to send Telegram message: {type(exc).__name__}")

if __name__ == "__main__":
    send_bot_message("Test message from telegram_service.py")