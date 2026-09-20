import traceback

def format_error(err: Exception) -> str:
    """Return a compact error description suitable for a Telegram message.

    Keeps only the tail of the traceback (the error line plus the last
    frames): full PyTorch traces are easily 30-80 frames and would exceed
    Telegram's 4096-character message limit, which makes the notification
    itself fail.
    """
    tb_tail = "\n".join(traceback.format_exc().splitlines()[-25:])
    body = f"{type(err).__name__}: {err}\n\nTraceback (tail):\n{tb_tail}"
    # Cap below 4096 so callers can prepend a short header without
    # exceeding Telegram's message limit.
    return body[:3700]
