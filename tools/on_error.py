import traceback

def format_error(err: Exception) -> str:
    """Return a string describing the error, suitable for sending to a bot."""
    return f"Error: {err}, Type: {type(err).__name__}\nMessage: {str(err)}\nTraceback: {traceback.format_exc()}"
