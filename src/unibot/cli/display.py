from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from urllib.parse import urlsplit

from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table
from rich.text import Text


_MAX_RECENT = 5

_UPDATE_CYCLE_PHASES = ("crawling", "reconciliation", "generation_build", "audit")

_BUILD_GENERATION_PHASES = (
    "plan",
    "build",
    "contextualize",
    "embed",
    "index",
    "activate",
)

_ERROR_SUGGESTIONS: dict[str, str] = {
    "fetch_failed": "check network or retry with --limit",
    "extraction_failed": "source HTML may have changed structure",
    "scope_processing_failed": "possible data conflict; review audit results",
}


@dataclass
class _RecentEvent:
    url: str
    ok: bool
    detail: str


class RichProgressDisplay:
    """Rich Live display implementing CycleProgressCallback."""

    def __init__(
        self,
        *,
        console: Console,
        generation_label: str,
        title: str = "Update Cycle",
        phase_order: tuple[str, ...] = _UPDATE_CYCLE_PHASES,
    ) -> None:
        self._console = console
        self._generation_label = generation_label
        self._title = title
        self._phase_order = phase_order
        self._start_time = monotonic()

        # Counters
        self.fetched_count: int = 0
        self.failed_count: int = 0
        self.discovered_count: int = 0
        self.total_records: int = 0

        # State
        self._current_url: str = ""
        self._current_step: str = ""
        self._current_phase: str = ""
        self._phase_total: int | None = None
        self._phase_done_count: int = 0
        self._recent: list[_RecentEvent] = []
        self._all_sources: list[_RecentEvent] = []
        self._phase_start_times: dict[str, float] = {}
        self._phase_durations: dict[str, float] = {}
        self._gen_step: str = ""
        self._gen_detail: str = ""
        self._gen_completed: int = 0
        self._gen_total: int = 0

        # Rich components
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(compact=True, elapsed_when_finished=True),
            console=console,
        )
        self._task_id: TaskID | None = None

    def on_phase_start(self, phase: str, total: int | None = None) -> None:
        self._current_phase = phase
        self._phase_total = total
        self._phase_done_count = 0
        self._phase_start_times[phase] = monotonic()
        if total is not None:
            self._task_id = self._progress.add_task(
                f"[cyan]{_phase_label(phase)}", total=total
            )

    def on_source_start(self, source_url: str, step: str) -> None:
        self._current_url = source_url
        self._current_step = step

    def on_source_done(self, source_url: str, records: int) -> None:
        self.fetched_count += 1
        self.total_records += records
        self._phase_done_count += 1
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=self._phase_done_count)
        self._add_recent(source_url, ok=True, detail=f"{records} records")
        self._all_sources.append(_RecentEvent(url=source_url, ok=True, detail=f"{records} records"))

    def on_source_failed(self, source_url: str, error: str) -> None:
        self.failed_count += 1
        self._phase_done_count += 1
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=self._phase_done_count)
        self._add_recent(source_url, ok=False, detail=error)
        self._all_sources.append(_RecentEvent(url=source_url, ok=False, detail=error))

    def on_sources_discovered(self, count: int) -> None:
        self.discovered_count += count
        if self._task_id is not None and self._phase_total is not None:
            self._phase_total += count
            self._progress.update(self._task_id, total=self._phase_total)

    def on_phase_done(self, phase: str) -> None:
        if phase in self._phase_start_times:
            self._phase_durations[phase] = monotonic() - self._phase_start_times[phase]
        if self._task_id is not None:
            self._progress.stop_task(self._task_id)
            self._task_id = None
        self._current_phase = ""

    def on_generation_step(self, step: str, detail: str) -> None:
        self._gen_step = step
        self._gen_detail = detail
        self._gen_completed = 0
        self._gen_total = 0

    def on_generation_progress(self, completed: int, total: int) -> None:
        self._gen_completed = completed
        self._gen_total = total
        if self._task_id is not None:
            self._progress.update(self._task_id, completed=completed, total=total)

    def build_display(self) -> Group:
        """Build the composite renderable for Rich Live."""
        parts: list[RenderableType] = []

        # Header
        header = Text.assemble(
            (f"{self._title}: ", "bold"),
            (self._generation_label, "bold cyan"),
        )
        parts.append(Panel(header, border_style="blue", expand=True))

        # Phase stepper (always visible)
        parts.append(self._build_phase_stepper())

        # Phase + progress
        if self._current_phase:
            if self._task_id is not None:
                parts.append(self._progress)
            elif self._gen_step:
                gen_info = Text.assemble(
                    ("  ", ""),
                    (self._gen_detail or self._gen_step, "cyan"),
                )
                if self._gen_total > 0:
                    gen_info.append(
                        f"  ({self._gen_completed}/{self._gen_total})", style="dim"
                    )
                parts.append(gen_info)

        if self._phase_order is not _BUILD_GENERATION_PHASES:
            # Current source (crawl-specific)
            if self._current_url:
                current = Text.assemble(
                    ("  Current: ", "dim"),
                    (_truncate_url(self._current_url), "cyan"),
                    ("\n  Status:  ", "dim"),
                    (self._current_step.replace("_", " ").title(), ""),
                )
                parts.append(current)

            # Counters
            rate_text = ""
            if self.fetched_count > 0 and "crawling" in self._phase_start_times:
                elapsed = monotonic() - self._phase_start_times["crawling"]
                if elapsed > 0:
                    rate = self.fetched_count / elapsed * 60
                    rate_text = f"  ({rate:.1f}/min)"

            counters = Text.assemble(
                ("\n  ", ""),
                ("\u2713", "green"),
                (f" Fetched {self.fetched_count:>4}", ""),
                (rate_text, "dim"),
                ("    ", ""),
                ("\u2717", "red"),
                (f" Failed {self.failed_count:>4}    ", ""),
                ("\u25c6", "blue"),
                (f" Discovered {self.discovered_count:>4}", ""),
                ("\n", ""),
            )
            parts.append(counters)

            # Recent events
            if self._recent:
                recent_text = Text()
                recent_text.append("  Recent:\n", style="dim")
                for event in self._recent:
                    if event.ok:
                        recent_text.append("    \u2713 ", style="green")
                    else:
                        recent_text.append("    \u2717 ", style="red")
                    recent_text.append(f"{_truncate_url(event.url)} ", style="")
                    recent_text.append(f"\u2014 {event.detail}\n", style="dim")
                parts.append(recent_text)

        return Group(*parts)

    def build_summary_table(self, *, title: str | None = None) -> Table:
        """Build the final summary table shown after completion."""
        elapsed = monotonic() - self._start_time
        minutes, seconds = divmod(int(elapsed), 60)

        effective_title = title if title is not None else f"{self._title} Complete"

        table = Table(
            title=effective_title,
            border_style="green",
            show_header=False,
            padding=(0, 2),
        )
        table.add_column("Key", style="bold")
        table.add_column("Value")
        table.add_row("Generation", self._generation_label)
        if self._phase_order is _BUILD_GENERATION_PHASES:
            table.add_row("Chunks embedded", f"{self._gen_completed}/{self._gen_total}")
            table.add_row("Vectors indexed", str(self._gen_completed))
        else:
            total_crawled = self.fetched_count + self.failed_count
            table.add_row("Sources crawled", f"{self.fetched_count}/{total_crawled}")
            table.add_row("Failed", str(self.failed_count))
            table.add_row("Records extracted", str(self.total_records))
            table.add_row("Discovered", f"{self.discovered_count} new sources")
        table.add_row("Duration", f"{minutes}m {seconds:02d}s")

        # Per-phase timing breakdown
        if self._phase_durations:
            table.add_section()
            for phase in self._phase_order:
                if phase in self._phase_durations:
                    d = self._phase_durations[phase]
                    pm, ps = divmod(int(d), 60)
                    table.add_row(
                        f"  {_phase_label(phase)}",
                        f"{pm}m {ps:02d}s",
                        style="dim",
                    )

        return table

    def build_source_details_table(self) -> Table:
        """Build per-source breakdown table shown after the summary."""
        table = Table(
            title="Source Details",
            border_style="dim",
            padding=(0, 1),
        )
        table.add_column("URL", style="cyan", no_wrap=True, max_width=60)
        table.add_column("Records", justify="right", style="green")
        table.add_column("Status", justify="center")

        # Show failed sources first, then successful
        failed = [s for s in self._all_sources if not s.ok]
        succeeded = [s for s in self._all_sources if s.ok]

        for source in failed:
            suggestion = _ERROR_SUGGESTIONS.get(source.detail, "")
            detail_text = source.detail
            if suggestion:
                detail_text = f"{source.detail} \u2014 {suggestion}"
            table.add_row(
                _truncate_url(source.url),
                "\u2014",
                f"[red]\u2717 {detail_text}[/red]",
            )
        for source in succeeded:
            table.add_row(
                _truncate_url(source.url),
                source.detail.split()[0],
                "[green]\u2713[/green]",
            )

        return table

    def _build_phase_stepper(self) -> Text:
        """Build a persistent phase list showing completed/current/pending status."""
        stepper = Text()
        total = len(self._phase_order)
        for i, phase in enumerate(self._phase_order, 1):
            label = _phase_label(phase)
            if phase in self._phase_durations:
                duration = self._phase_durations[phase]
                minutes, seconds = divmod(int(duration), 60)
                stepper.append(f"  \u2713 [{i}/{total}] {label}", style="green")
                stepper.append(f" {minutes}m {seconds:02d}s\n", style="dim")
            elif phase == self._current_phase:
                elapsed = ""
                if phase in self._phase_start_times:
                    e = monotonic() - self._phase_start_times[phase]
                    m, s = divmod(int(e), 60)
                    elapsed = f" {m}m {s:02d}s"
                stepper.append(f"  \u203a [{i}/{total}] {label}", style="bold yellow")
                stepper.append(f"{elapsed}\n", style="dim")
            else:
                stepper.append(f"    [{i}/{total}] {label}\n", style="dim")
        return stepper

    def _add_recent(self, url: str, *, ok: bool, detail: str) -> None:
        self._recent.append(_RecentEvent(url=url, ok=ok, detail=detail))
        if len(self._recent) > _MAX_RECENT:
            self._recent.pop(0)


def _phase_label(phase: str) -> str:
    labels = {
        # Update cycle phases
        "crawling": "Crawling Sources",
        "reconciliation": "Reconciling Sources",
        "generation_build": "Building Generation",
        "audit": "Running Audit",
        # Build generation phases
        "plan": "Planning Generation",
        "build": "Building Chunks",
        "contextualize": "Contextualizing Chunks",
        "embed": "Embedding Chunks",
        "index": "Indexing to Qdrant",
        "activate": "Activating Generation",
    }
    return labels.get(phase, phase.replace("_", " ").title())


def _truncate_url(url: str, max_len: int = 60) -> str:
    path = urlsplit(url).path or url
    if len(path) > max_len:
        return path[: max_len - 3] + "..."
    return path
