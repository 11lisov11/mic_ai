from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.plots_ai import plot_ident_and_learning


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Plot identification and AI learning summary.")
    p.add_argument("--ident", required=True, help="Path to ident JSON (from demo_ai_with_ident).")
    p.add_argument("--episodes", required=True, help="Path to episodes JSON (from demo_ai_with_ident).")
    p.add_argument("--output", required=True, help="Path to save PNG.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    plot_ident_and_learning(args.ident, args.episodes, args.output)


if __name__ == "__main__":
    main()
