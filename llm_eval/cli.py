"""
CLI entrypoint — llm-eval

Commands:
  llm-eval run        --config configs/eval.yaml [--model gpt-4o]
  llm-eval report     --run-id <id>
  llm-eval red-team   --config configs/eval.yaml [--model gpt-4o]
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from dotenv import load_dotenv
from rich.console import Console

# Load .env if present
load_dotenv()

app = typer.Typer(
    name="llm-eval",
    help="🔬 LLM Evaluation Framework — benchmark and red-team your LLMs",
    add_completion=False,
)
console = Console()

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "WARNING"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@app.command()
def run(
    config: str = typer.Option(
        "configs/eval.yaml",
        "--config", "-c",
        help="Path to eval.yaml config file",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Run only this model (name must match config). Defaults to all models in config.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config and dataset without making API calls",
    ),
) -> None:
    """Run the evaluation suite against one or all configured models."""
    from .config import load_config
    from .runner import EvalRunner

    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        raise typer.Exit(1)

    if dry_run:
        console.print(f"[green]Config loaded:[/green] {len(cfg.models)} model(s), {len(cfg.evaluators)} evaluator(s)")
        console.print("[dim]--dry-run: skipping API calls[/dim]")
        raise typer.Exit(0)

    runner = EvalRunner(cfg)
    runner.setup()

    models_to_run = [m for m in cfg.models if model is None or m.name == model]
    if not models_to_run:
        console.print(f"[red]No model named '{model}' in config. Available: {[m.name for m in cfg.models]}[/red]")
        raise typer.Exit(1)

    run_ids = []
    for model_cfg in models_to_run:
        try:
            run_id = runner.run(model_cfg.name)
            run_ids.append(run_id)
        except Exception as e:
            console.print(f"[bold red]Run failed for {model_cfg.name}:[/bold red] {e}")
            logging.exception(e)

    if run_ids:
        console.print(f"\n[dim]Results saved. Run IDs: {', '.join(r[:8] for r in run_ids)}[/dim]")


@app.command()
def report(
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Show report for a specific run ID. Defaults to latest.",
    ),
    config: str = typer.Option(
        "configs/eval.yaml",
        "--config", "-c",
    ),
    post_to_github: bool = typer.Option(
        False,
        "--post-github",
        help="Post the report as a GitHub PR comment (requires env vars)",
    ),
) -> None:
    """Generate a report for a completed eval run."""
    from .config import load_config
    from .storage.sqlite_store import SQLiteStore
    from .reporting.cli_report import CLIReporter
    from .reporting.pr_comment import generate_markdown_report, post_pr_comment

    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    store = SQLiteStore(cfg.output.sqlite_path)
    resolved_id = run_id or store.get_latest_run_id()

    if not resolved_id:
        console.print("[yellow]No eval runs found. Run 'llm-eval run' first.[/yellow]")
        raise typer.Exit(1)

    summary = store.get_run_summary(resolved_id)
    red_team = store.get_red_team_summary(resolved_id)

    reporter = CLIReporter()
    reporter.print_summary(summary)
    if red_team:
        reporter.print_red_team_summary(red_team)

    if post_to_github:
        post_pr_comment(summary, red_team or None)
    else:
        md = generate_markdown_report(summary, red_team or None)
        console.print("\n[dim]--- Markdown report (copy-paste for PR) ---[/dim]")
        console.print(md)


@app.command(name="red-team")
def red_team_cmd(
    config: str = typer.Option(
        "configs/eval.yaml",
        "--config", "-c",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model", "-m",
        help="Target model name. Defaults to first model in config.",
    ),
) -> None:
    """Run adversarial red-team evaluation against a model."""
    from .config import load_config
    from .runner import EvalRunner

    try:
        cfg = load_config(config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(1)

    if not cfg.red_team.enabled:
        console.print("[yellow]Red-team is disabled in config (red_team.enabled: false)[/yellow]")
        raise typer.Exit(0)

    runner = EvalRunner(cfg)
    runner.setup()

    target = model or cfg.models[0].name
    try:
        run_id = runner.run_red_team(target)
        console.print(f"\n[dim]Red-team run ID: {run_id[:8]}...[/dim]")
    except Exception as e:
        console.print(f"[bold red]Red-team failed:[/bold red] {e}")
        logging.exception(e)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
