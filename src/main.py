from __future__ import annotations

from unibot.api.app import create_app
from unibot.cli import build_parser, main

__all__ = ["build_parser", "create_app", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
