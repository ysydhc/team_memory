"""Prompt Arena CLI — manual control over the arena system.

Usage:
    python -m team_memory.daemon.quality.arena.cli run [--rounds N] [--no-eval]
    python -m team_memory.daemon.quality.arena.cli eval [--batch-size N]
    python -m team_memory.daemon.quality.arena.cli seed [--force]
    python -m team_memory.daemon.quality.arena.cli status
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys

from team_memory.project_resolver import load_dotenv_if_exists

load_dotenv_if_exists(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("arena.cli")


async def cmd_run(dsn: str, rounds: int, no_eval: bool) -> None:
    from team_memory.daemon.quality.arena.runner import ArenaRunner

    runner = ArenaRunner(dsn=dsn)
    try:
        for i in range(1, rounds + 1):
            print(f"\n--- Round {i}/{rounds} ---")
            record = await runner.run_round()
            if record is None:
                print("Round skipped (not enough data or prompts).")
                continue

            print(f"  {record.prompt_a_id} vs {record.prompt_b_id} -> {record.meta_verdict}")

            if not no_eval:
                print("  Running production eval...")
                summary = await runner.run_production_eval()
                print(
                    f"  Eval: evaluated={summary['evaluated']} "
                    f"skipped={summary.get('skipped', 0)} "
                    f"errors={summary.get('errors', 0)}"
                )

            if i < rounds:
                await asyncio.sleep(2.0)
    finally:
        await runner.close()


async def cmd_eval(dsn: str, batch_size: int) -> None:
    from team_memory.daemon.quality.arena.runner import ArenaRunner

    runner = ArenaRunner(dsn=dsn)
    try:
        summary = await runner.run_production_eval(batch_size=batch_size)
        print("\n=== Production Eval Results ===")
        print(f"Champion:   {summary.get('champion_id', 'N/A')} (ELO {summary.get('champion_elo', 'N/A')})")
        print(f"Evaluated:  {summary['evaluated']}")
        print(f"Skipped:    {summary.get('skipped', 0)}")
        print(f"Errors:     {summary.get('errors', 0)}")
        if summary.get("avg_score") is not None:
            print(f"Avg score:  {summary['avg_score']:.3f}")
        if summary.get("error"):
            print(f"Error:      {summary['error']}")
    finally:
        await runner.close()


async def cmd_seed(force: bool) -> None:
    from team_memory.daemon.quality.arena.prompt_generator import PromptGenerator
    from team_memory.daemon.quality.arena.storage import ArenaStorage

    storage = ArenaStorage()
    existing = storage.get_active_prompts()

    if existing and not force:
        print(f"Already have {len(existing)} active prompts. Use --force to regenerate.")
        return

    generator = PromptGenerator()
    print("Generating seed prompts...")
    seeds = await generator.generate_seed_prompts()
    storage.save_prompts(seeds)
    print(f"Generated {len(seeds)} seed prompts:")
    for p in seeds:
        print(f"  {p.id}  v{p.version}  ELO={p.elo:.0f}")


def cmd_status() -> None:
    from team_memory.daemon.quality.arena.storage import ArenaStorage

    storage = ArenaStorage()
    prompts = storage.load_prompts()
    rounds = storage.load_rounds()
    epoch = storage.get_current_epoch()
    converged = storage.is_converged()

    champion = storage.get_champion()

    print("\n=== Prompt Arena Status ===")
    print(f"Epoch:      {epoch}")
    print(f"Rounds:     {len(rounds)}")
    print(f"Converged:  {'Yes' if converged else 'No'}")

    if not prompts:
        print("\nNo prompts registered. Run 'seed' to generate initial prompts.")
        return

    sorted_prompts = sorted(prompts, key=lambda p: p.elo, reverse=True)
    champion_id = champion.id if champion else None

    print("\nELO Rankings:")
    for rank, p in enumerate(sorted_prompts, 1):
        flags: list[str] = []
        if p.id == champion_id:
            flags.append("champion")
        if p.retired:
            flags.append("retired")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""

        print(f"  {rank}. {p.id}  ELO={p.elo:.0f}  W={p.wins} L={p.losses} D={p.draws}{flag_str}")

    if rounds:
        recent = rounds[-5:]
        print(f"\nRecent rounds (last {len(recent)}):")
        for r in recent:
            print(f"  {r.round_id}  {r.prompt_a_id} vs {r.prompt_b_id}  -> {r.meta_verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prompt Arena CLI — manage arena competitions and evaluations")
    parser.add_argument("--db-url", default=None, help="Database URL (or set TEAM_MEMORY_DB_URL)")

    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run arena round(s)")
    run_parser.add_argument("--rounds", type=int, default=1, help="Number of rounds to run (default: 1)")
    run_parser.add_argument("--no-eval", action="store_true", help="Skip production eval after each round")

    eval_parser = subparsers.add_parser("eval", help="Run production evaluation only")
    eval_parser.add_argument("--batch-size", type=int, default=20, help="Max entries to evaluate (default: 20)")

    seed_parser = subparsers.add_parser("seed", help="Generate initial seed prompts")
    seed_parser.add_argument("--force", action="store_true", help="Regenerate even if prompts already exist")

    subparsers.add_parser("status", help="Show arena status and ELO rankings")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "status":
        cmd_status()
        return

    if args.command == "seed":
        asyncio.run(cmd_seed(force=args.force))
        return

    dsn = (args.db_url or os.environ.get("TEAM_MEMORY_DB_URL") or "").strip()
    if not dsn:
        parser.error("Database URL required: pass --db-url or set TEAM_MEMORY_DB_URL.")

    if args.command == "run":
        asyncio.run(cmd_run(dsn=dsn, rounds=args.rounds, no_eval=args.no_eval))
    elif args.command == "eval":
        asyncio.run(cmd_eval(dsn=dsn, batch_size=args.batch_size))


if __name__ == "__main__":
    main()
