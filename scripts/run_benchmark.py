#!/usr/bin/env python3
"""
One-shot full benchmark runner.

Runs the complete eval suite (all models × all evaluators + red-team) and
generates a combined report. Use this for the initial demo data run.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --config configs/eval.yaml --no-red-team
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure project root is on the path when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.panel import Panel

from llm_eval.config import load_config
from llm_eval.runner import EvalRunner
from llm_eval.storage.sqlite_store import SQLiteStore

console = Console()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Eval Framework — Full Benchmark Runner")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to eval.yaml")
    parser.add_argument("--no-red-team", action="store_true", help="Skip red-team evaluation")
    parser.add_argument("--model", default=None, help="Run only this model")
    args = parser.parse_args()

    console.print(
        Panel(
            "[bold cyan]LLM Eval Framework — Full Benchmark[/bold cyan]\n"
            f"Config: [yellow]{args.config}[/yellow]",
            border_style="cyan",
        )
    )

    try:
        cfg = load_config(args.config)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        return 1

    runner = EvalRunner(cfg)
    runner.setup()

    models_to_run = cfg.models if args.model is None else [m for m in cfg.models if m.name == args.model]

    all_run_ids: list[str] = []
    store = SQLiteStore(cfg.output.sqlite_path)

    # ── Eval runs ────────────────────────────────────────────────────────────
    for model_cfg in models_to_run:
        console.print(f"\n[bold]Running eval for:[/bold] [cyan]{model_cfg.name}[/cyan]")
        try:
            run_id = runner.run(model_cfg.name)
            all_run_ids.append(run_id)
            console.print(f"[green]✓[/green] Eval complete — run ID: [dim]{run_id[:8]}...[/dim]")
        except Exception as e:
            console.print(f"[red]✗ Eval failed for {model_cfg.name}: {e}[/red]")
            logging.exception(e)

    # ── Red-team runs ────────────────────────────────────────────────────────
    if not args.no_red_team and cfg.red_team.enabled:
        for model_cfg in models_to_run:
            console.print(f"\n[bold]Red-teaming:[/bold] [cyan]{model_cfg.name}[/cyan]")
            try:
                rt_run_id = runner.run_red_team(model_cfg.name)
                console.print(f"[green]✓[/green] Red-team complete — run ID: [dim]{rt_run_id[:8]}...[/dim]")
            except Exception as e:
                console.print(f"[red]✗ Red-team failed for {model_cfg.name}: {e}[/red]")

    # ── Combined summary ─────────────────────────────────────────────────────
    if all_run_ids:
        console.print("\n[bold]Final Summaries:[/bold]")
        for run_id in all_run_ids:
            summary = store.get_run_summary(run_id)
            model = summary["run"].get("model_name", "unknown")
            console.print(f"\n[cyan]{model}[/cyan]:")
            for m in summary.get("metrics", []):
                count = m.get("count", 0)
                passed = m.get("passed_count", 0)
                pass_rate = (passed / count * 100) if count > 0 else 0
                status = "[green]PASS[/green]" if pass_rate >= 80 else "[red]FAIL[/red]"
                console.print(f"  {m['metric']:15s} {m['avg_score']:5.2f}/10  {pass_rate:3.0f}%  {status}")

    console.print(f"\n[dim]Results written to: {cfg.output.sqlite_path}[/dim]")
    console.print("[dim]Start dashboard: streamlit run dashboard/app.py[/dim]")

    return 0


if __name__ == "__main__":
    sys.exit(main())
