"""
CLI entry point for running the Social Arbitrage Sub-Agent standalone.

Usage:
    python -m openclaw.social_arb.cli run          # Run the orchestrator
    python -m openclaw.social_arb.cli run --cycles 5  # Run 5 cycles then stop
    python -m openclaw.social_arb.cli status        # Show signal log stats
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from openclaw.social_arb.config import config
from openclaw.social_arb.orchestrator import Orchestrator
from openclaw.social_arb.signals.signal_schema import SocialArbSignal
from openclaw.social_arb.storage.signal_log import SignalLog
from openclaw.social_arb.utils.logger import get_logger

logger = get_logger(__name__)
console = Console()


async def signal_printer(signal: SocialArbSignal) -> None:
    """Pretty-print signals to the terminal."""
    direction_color = "green" if signal.direction.value == "long" else "red"
    safe_emoji = "[bold green]SAFE[/]" if signal.is_safe_to_trade() else "[bold yellow]CAUTION[/]"

    table = Table(title=f"Signal: {signal.ticker}", show_header=False, padding=(0, 1))
    table.add_column("Field", style="bold cyan")
    table.add_column("Value")
    table.add_row("Direction", f"[{direction_color}]{signal.direction.value.upper()}[/]")
    table.add_row("Confidence", f"{signal.confidence:.1%}")
    table.add_row("Sources", ", ".join(signal.sources))
    table.add_row("Edge Decay", signal.edge_decay_estimate)
    table.add_row("Position Size", f"{signal.suggested_position_pct}%")
    table.add_row("Stop Loss", f"{signal.stop_loss_pct}%")
    table.add_row("Take Profit", f"{signal.take_profit_pct}%")
    table.add_row("Bot Risk", f"{signal.bot_risk_score:.1%}")
    table.add_row("Hype Risk", f"{signal.hype_risk_score:.1%}")
    table.add_row("Safety", safe_emoji)

    console.print(Panel(table, border_style="green" if signal.is_safe_to_trade() else "yellow"))


async def run_orchestrator(max_cycles: int | None = None) -> None:
    """Start the orchestrator with a pretty signal printer callback."""
    console.print(
        Panel(
            "[bold]OpenClaw Social Arbitrage Sub-Agent[/]\n"
            f"Monitoring: {', '.join(config.subreddits)}\n"
            f"Anomaly threshold: z > {config.anomaly.z_score_threshold}\n"
            f"Min sources: {config.anomaly.min_cross_sources}",
            title="Starting",
            border_style="blue",
        )
    )

    orchestrator = Orchestrator(signal_callback=signal_printer)

    try:
        await orchestrator.run(max_cycles=max_cycles)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutting down...[/]")
        orchestrator.stop()


def show_status() -> None:
    """Show signal log statistics."""
    signal_log = SignalLog()
    count = signal_log.get_signal_count()
    recent = signal_log.read_signals(limit=10)

    console.print(f"\n[bold]Total signals logged:[/] {count}")

    if recent:
        table = Table(title="Recent Signals")
        table.add_column("Ticker", style="cyan")
        table.add_column("Direction")
        table.add_column("Confidence")
        table.add_column("Sources")
        table.add_column("Time")

        for sig in recent:
            direction = sig.get("direction", "?")
            color = "green" if direction == "long" else "red"
            table.add_row(
                sig.get("ticker", "?"),
                f"[{color}]{direction.upper()}[/]",
                f"{sig.get('confidence', 0):.1%}",
                ", ".join(sig.get("sources", [])),
                sig.get("detected_at", "?")[:19],
            )
        console.print(table)
    else:
        console.print("[dim]No signals logged yet.[/]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenClaw Social Arbitrage Sub-Agent"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the orchestrator")
    run_parser.add_argument(
        "--cycles",
        type=int,
        default=None,
        help="Max cycles to run (default: unlimited)",
    )

    # Status command
    subparsers.add_parser("status", help="Show signal log stats")

    args = parser.parse_args()

    if args.command == "run":
        asyncio.run(run_orchestrator(max_cycles=args.cycles))
    elif args.command == "status":
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
