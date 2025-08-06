"""
Progress tracking utilities for long-running operations.

This module provides progress bars and indicators for operations like:
- Batch processing multiple prompts
- Running benchmarks
- Downloading datasets
- Fine-tuning models
"""

from __future__ import annotations  # Enable Python 3.9+ union syntax

import queue
import sys
import time
from contextlib import contextmanager
from threading import Event, Thread
from typing import Any, Iterable, Iterator, Optional


class ProgressBar:
    """A simple progress bar for terminal output."""

    def __init__(
        self,
        total: int,
        desc: str = "",
        width: int = 50,
        show_percentage: bool = True,
        show_time: bool = True,
        file=sys.stderr,
    ):
        """
        Initialize a progress bar.

        Args:
            total: Total number of items to process
            desc: Description to show before the bar
            width: Width of the progress bar in characters
            show_percentage: Whether to show percentage complete
            show_time: Whether to show elapsed time
            file: File object to write to (default: stderr)
        """
        self.total = total
        self.desc = desc
        self.width = width
        self.show_percentage = show_percentage
        self.show_time = show_time
        self.file = file
        self.current = 0
        self.start_time = time.time()
        self._last_update = 0

    def update(self, n: int = 1) -> None:
        """Update the progress bar by n steps."""
        self.current = min(self.current + n, self.total)
        self._render()

    def set_description(self, desc: str) -> None:
        """Update the description text."""
        self.desc = desc
        self._render()

    def _render(self) -> None:
        """Render the progress bar to the terminal."""
        # Throttle updates to avoid excessive rendering
        now = time.time()
        if now - self._last_update < 0.1 and self.current < self.total:
            return
        self._last_update = now

        # Calculate progress
        if self.total > 0:
            progress = self.current / self.total
        else:
            progress = 0

        # Build the bar
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)

        # Build the output string
        parts = []
        if self.desc:
            parts.append(f"{self.desc}: ")
        parts.append(f"[{bar}]")

        if self.show_percentage:
            parts.append(f" {progress * 100:.1f}%")

        if self.show_time:
            elapsed = time.time() - self.start_time
            if elapsed > 0 and self.current > 0:
                rate = self.current / elapsed
                if self.current < self.total:
                    eta = (self.total - self.current) / rate
                    parts.append(f" ETA: {self._format_time(eta)}")
                parts.append(f" ({self._format_time(elapsed)})")

        # Write to terminal
        line = "".join(parts)
        self.file.write(f"\r{line:<80}")
        self.file.flush()

        # Add newline when complete
        if self.current >= self.total:
            self.file.write("\n")
            self.file.flush()

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to a readable string."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            return f"{seconds / 60:.1f}m"
        else:
            return f"{seconds / 3600:.1f}h"

    def close(self) -> None:
        """Ensure the progress bar is complete and add a newline."""
        if self.current < self.total:
            self.current = self.total
            self._render()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


class Spinner:
    """An indeterminate progress spinner for operations without known duration."""

    FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

    def __init__(self, desc: str = "Processing", file=sys.stderr):
        """
        Initialize a spinner.

        Args:
            desc: Description to show next to the spinner
            file: File object to write to
        """
        self.desc = desc
        self.file = file
        self._stop_event = Event()
        self._thread = None

    def start(self) -> None:
        """Start the spinner animation."""
        if self._thread is not None:
            return

        self._stop_event.clear()
        self._thread = Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        """Run the spinner animation in a background thread."""
        frame_idx = 0
        while not self._stop_event.is_set():
            frame = self.FRAMES[frame_idx % len(self.FRAMES)]
            self.file.write(f"\r{frame} {self.desc}")
            self.file.flush()
            frame_idx += 1
            time.sleep(0.1)

        # Clear the line when stopping
        self.file.write(f"\r{' ' * (len(self.desc) + 3)}\r")
        self.file.flush()

    def stop(self) -> None:
        """Stop the spinner animation."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join()
        self._thread = None

    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


def track_progress(
    iterable: Iterable[Any], total: int | None = None, desc: str = "", disable: bool = False
) -> Iterator[Any]:
    """
    Wrap an iterable with a progress bar.

    Args:
        iterable: The iterable to track
        total: Total number of items (if not provided, tries len())
        desc: Description for the progress bar
        disable: If True, don't show progress bar

    Yields:
        Items from the iterable

    Example:
        ```python
        for item in track_progress(items, desc="Processing"):
            process(item)
        ```
    """
    if disable:
        yield from iterable
        return

    # Try to get total if not provided
    if total is None:
        try:
            total = len(iterable)
        except (TypeError, AttributeError):
            # If we can't get length, use a spinner instead
            with Spinner(desc=desc):
                yield from iterable
            return

    # Use progress bar
    with ProgressBar(total=total, desc=desc) as pbar:
        for item in iterable:
            yield item
            pbar.update(1)


@contextmanager
def progress_context(desc: str = "Processing", spinner: bool = False):
    """
    Context manager for showing progress during an operation.

    Args:
        desc: Description of the operation
        spinner: If True, use spinner instead of waiting message

    Example:
        ```python
        with progress_context("Loading model", spinner=True):
            model = load_large_model()
        ```
    """
    if spinner:
        with Spinner(desc=desc):
            yield
    else:
        print(f"{desc}...", file=sys.stderr)
        start = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - start
            print(f"{desc}... done ({elapsed:.1f}s)", file=sys.stderr)


class BatchProgressTracker:
    """Track progress across multiple batches of work."""

    def __init__(self, total_batches: int, items_per_batch: int | None = None):
        """
        Initialize batch progress tracker.

        Args:
            total_batches: Total number of batches
            items_per_batch: Number of items per batch (if known)
        """
        self.total_batches = total_batches
        self.items_per_batch = items_per_batch
        self.current_batch = 0
        self.batch_bar = ProgressBar(total_batches, desc="Batches")
        self.item_bar = None

    def start_batch(self, batch_num: int, batch_size: int | None = None):
        """Start tracking a new batch."""
        self.current_batch = batch_num
        self.batch_bar.set_description(f"Batch {batch_num + 1}/{self.total_batches}")

        if batch_size or self.items_per_batch:
            size = batch_size or self.items_per_batch
            self.item_bar = ProgressBar(size, desc=f"  Items", width=40)

    def update_item(self, n: int = 1):
        """Update progress within current batch."""
        if self.item_bar:
            self.item_bar.update(n)

    def finish_batch(self):
        """Mark current batch as complete."""
        if self.item_bar:
            self.item_bar.close()
            self.item_bar = None
        self.batch_bar.update(1)

    def close(self):
        """Close all progress bars."""
        if self.item_bar:
            self.item_bar.close()
        self.batch_bar.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


# Convenience functions for common use cases


def show_progress(enabled: bool = True):
    """
    Decorator to show progress for a function.

    Example:
        ```python
        @show_progress()
        def process_data(items):
            for item in track_progress(items):
                # Process item
                pass
        ```
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            if enabled:
                with progress_context(f"Running {func.__name__}", spinner=True):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def parallel_progress(tasks: list, max_workers: int = 4, desc: str = "Processing"):
    """
    Execute tasks in parallel with progress tracking.

    Args:
        tasks: List of callables to execute
        max_workers: Maximum number of parallel workers
        desc: Description for progress bar

    Returns:
        List of results from each task

    Example:
        ```python
        tasks = [lambda: process(item) for item in items]
        results = parallel_progress(tasks, max_workers=8)
        ```
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = [None] * len(tasks)

    with ProgressBar(len(tasks), desc=desc) as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(task): idx for idx, task in enumerate(tasks)}

            # Process completed tasks
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = e
                pbar.update(1)

    return results
