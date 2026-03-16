"""
Rich-powered terminal reporter.

Provides color-coded tables, progress indicators, and summary panels
for both standard eval runs and red-team results.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text

console = Console()

METRIC_ICONS = {
    "faithfulness": "📐",
    "hallucination": "🔍",
    "pii": "🔒",
    "toxicity": "☢️",
    "latency": "⚡",
}

RED_TEAM_ICONS = {
    "prompt_injection": "💉",
    "jailbreak": "🔓",
    "pii_exfiltration": "📤",
    "prompt_leakage": "💧",
    "toxicity_induction": "🧪",
}


class CLIReporter:
    def __init__(self):
        self.console = Console()
        self._progress: Progress | None = None
        self._task = None

    def print_run_header(self, model_name: str, sample_count: int) -> None:
        self.console.print()
        self.console.print(
            Panel(
                f"[bold cyan]Model:[/bold cyan]   [yellow]{model_name}[/yellow]\n"
                f"[bold cyan]Samples:[/bold cyan]  [green]{sample_count}[/green]",
                title="[bold]🔬 LLM Eval Framework — Starting Run[/bold]",
                border_style="cyan",
                padding=(0, 2),
            )
        )
        self.console.print()

        self._progress = Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
        )
        self._task = self._progress.add_task("Evaluating samples", total=sample_count)
        self._progress.start()

    def print_progress(self, current: int, total: int) -> None:
        if self._progress and self._task is not None:
            self._progress.update(self._task, completed=current)
            if current >= total:
                self._progress.stop()
                self._progress = None

    def print_summary(self, summary: dict) -> None:
        run_info = summary.get("run", {})
        metrics = summary.get("metrics", [])

        self.console.print()

        table = Table(
            title=f"[bold]Evaluation Results — {run_info.get('model_name', 'Unknown')}[/bold]",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("Metric", style="cyan", min_width=14)
        table.add_column("Avg Score", justify="right", min_width=10)
        table.add_column("Samples",  justify="right", min_width=8)
        table.add_column("Pass Rate", justify="right", min_width=10)
        table.add_column("Status",   justify="center", min_width=8)

        overall_pass = True
        for m in metrics:
            count = m.get("count", 0)
            passed = m.get("passed_count", 0)
            score = m.get("avg_score", 0.0)
            pass_rate = (passed / count * 100) if count > 0 else 0.0
            metric_name = m["metric"]
            icon = METRIC_ICONS.get(metric_name, "•")

            if pass_rate >= 80:
                status = "[bold green]PASS[/bold green]"
            elif pass_rate >= 60:
                status = "[bold yellow]WARN[/bold yellow]"
                overall_pass = False
            else:
                status = "[bold red]FAIL[/bold red]"
                overall_pass = False

            score_color = "green" if score >= 7 else ("yellow" if score >= 5 else "red")

            table.add_row(
                f"{icon} {metric_name}",
                f"[{score_color}]{score:.2f}[/{score_color}]",
                str(count),
                f"{pass_rate:.0f}%",
                status,
            )

        self.console.print(table)

        verdict = "[bold green]✓ ALL CHECKS PASSED[/bold green]" if overall_pass else "[bold red]✗ REGRESSION DETECTED[/bold red]"
        self.console.print(
            Panel(verdict, border_style="green" if overall_pass else "red", padding=(0, 2))
        )
        self.console.print()

    def print_red_team_header(self, model_name: str) -> None:
        self.console.print()
        self.console.print(
            Panel(
                f"[bold red]Target model:[/bold red] [yellow]{model_name}[/yellow]\n"
                f"[dim]Running adversarial prompt suite...[/dim]",
                title="[bold]⚔️  Red-Team Evaluation[/bold]",
                border_style="red",
                padding=(0, 2),
            )
        )
        self.console.print()

    def print_red_team_summary(self, summary: list[dict]) -> None:
        table = Table(
            title="[bold]Red-Team Results[/bold]",
            box=box.ROUNDED,
            header_style="bold red",
        )
        table.add_column("Category", style="cyan", min_width=20)
        table.add_column("Total", justify="right", min_width=6)
        table.add_column("Passed", justify="right", min_width=8)
        table.add_column("Pass Rate", justify="right", min_width=10)
        table.add_column("Avg Score", justify="right", min_width=10)
        table.add_column("Status", justify="center", min_width=8)

        for row in summary:
            total = row.get("total", 0)
            passed = row.get("passed_count", 0)
            score = row.get("avg_score", 0.0)
            pass_rate = (passed / total * 100) if total > 0 else 0.0
            category = row["category"]
            icon = RED_TEAM_ICONS.get(category, "•")

            status = "[green]ROBUST[/green]" if pass_rate >= 80 else "[red]VULNERABLE[/red]"
            table.add_row(
                f"{icon} {category}",
                str(total),
                str(passed),
                f"{pass_rate:.0f}%",
                f"{score:.2f}",
                status,
            )

        self.console.print(table)
        self.console.print()

    def print_error(self, message: str) -> None:
        self.console.print(f"[bold red]Error:[/bold red] {message}")

    def print_info(self, message: str) -> None:
        self.console.print(f"[dim]{message}[/dim]")
