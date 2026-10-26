"""Debug utility for conditional logging."""

# Global debug flag - can be set via set_debug() or directly
_DEBUG = False


def set_debug(enabled: bool) -> None:
    """Set the global debug flag."""
    global _DEBUG
    _DEBUG = enabled


def is_debug() -> bool:
    """Check if debug mode is enabled."""
    return _DEBUG


def debug_print(*args, **kwargs) -> None:
    """Print only if debug mode is enabled."""
    if _DEBUG:
        print(*args, **kwargs)
