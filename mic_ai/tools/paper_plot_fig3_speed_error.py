from __future__ import annotations

import argparse
from typing import List

from mic_ai.tools.paper_plots_ai_vs_foc import make_plots, prepare_data


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Fig. 3 (speed error) comparing AI vs FOC.")
    p.add_argument("--env-config", default="config/env_demo_true_motor1.py")
    p.add_argument("--ai-checkpoint", default="outputs/demo_ai/checkpoints/motor1/last_actor.pth")
    p.add_argument("--episode-steps", type=int, default=200)
    p.add_argument("--episodes-per-stage", type=int, default=25)
    p.add_argument("--window", type=int, default=5, help="Rolling mean window for AI curves.")
    p.add_argument("--out-dir", default="outputs/paper_figures")
    p.add_argument("--voltage-scale", type=float, default=1.25, help="Per-unit voltage_scale used for AI eval.")
    p.add_argument("--disable-noise", action="store_true", help="Disable measurement noise in AI env for eval.")
    p.add_argument("--force-eval", action="store_true", help="Recompute baseline and AI eval logs even if JSON files exist.")
    p.add_argument("--no-captions", action="store_true", help="Do not write captions_ru.txt / captions_en.txt.")
    return p.parse_args(argv)


def main() -> None:
    args = parse_args()
    prepared = prepare_data(
        env_config=args.env_config,
        ai_checkpoint=args.ai_checkpoint,
        out_dir=args.out_dir,
        episode_steps=int(args.episode_steps),
        episodes_per_stage=int(args.episodes_per_stage),
        window=int(args.window),
        voltage_scale=float(args.voltage_scale),
        disable_noise=bool(args.disable_noise),
        force_eval=bool(args.force_eval),
    )
    captions_path = make_plots(
        prepared.out_dir,
        prepared.ai,
        prepared.foc,
        window=int(args.window),
        episodes_per_stage=int(prepared.episodes_per_stage),
        n_stages=int(prepared.n_stages),
        stage_omega_ref_rad_s=prepared.stage_omega_ref_rad_s,
        figures=[3],
        write_captions=not bool(args.no_captions),
    )
    print(f"Saved Fig. 3 to {prepared.out_dir}")
    if captions_path is not None:
        print(f"Saved captions to {captions_path}")


if __name__ == "__main__":
    main()

