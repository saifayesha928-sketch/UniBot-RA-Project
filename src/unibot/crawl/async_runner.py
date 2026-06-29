from __future__ import annotations

import asyncio
import threading
from collections.abc import Coroutine
from typing import Any, Callable, TypeVar

T = TypeVar("T")
_JOIN_TIMEOUT_SECONDS = 30.0


def run_sync(coro_factory: Callable[[], Coroutine[Any, Any, T]], *, timeout: float = _JOIN_TIMEOUT_SECONDS) -> T:
    """Run an async coroutine from synchronous code, safe inside or outside an event loop.

    If no event loop is running, uses ``asyncio.run()``.
    If an event loop is already running (e.g. inside an ``async`` caller),
    executes the coroutine in a worker thread with its own event loop.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop is None:
        return asyncio.run(coro_factory())

    result_container: list[T] = []
    exception_container: list[BaseException] = []

    def _run_in_thread() -> None:
        try:
            result_container.append(asyncio.run(coro_factory()))
        except BaseException as exc:
            exception_container.append(exc)

    thread = threading.Thread(target=_run_in_thread, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        raise TimeoutError("run_sync worker thread did not finish before timeout")

    if exception_container:
        raise exception_container[0]
    return result_container[0]
