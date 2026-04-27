#!/usr/bin/env python3
"""Generate M-106 JIGSAWS surgical action segmentation samples.

Usage:
    python examples/generate.py --num-samples 3
    python examples/generate.py --num-samples 800 --output data/questions
    python examples/generate.py            # uses argparse default = 800

Bootstrap on EC2 calls this without --num-samples (defaults to 800), so the
pipeline iterates over all 103 episodes with 800 as the cap.
"""
import os
os.environ.setdefault("PYTHONUNBUFFERED", "1")
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import TaskPipeline, TaskConfig


def main():
    parser = argparse.ArgumentParser(
        description="Generate M-106 JIGSAWS surgical action segmentation dataset"
    )
    parser.add_argument("--num-samples", type=int, default=800,
                        help="Cap on number of episodes to process (default: 800).")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/questions")
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--frame-size", type=int, default=480)
    parser.add_argument("--banner-height", type=int, default=60)
    parser.add_argument("--max-frames", type=int, default=240)
    parser.add_argument("--s3-bucket", type=str, default="med-vr-datasets")
    parser.add_argument("--s3-prefix", type=str, default="M-106/JIGSAWS/")
    args = parser.parse_args()

    print("Generating M-106 JIGSAWS surgical action segmentation dataset...",
          flush=True)

    config = TaskConfig(
        num_samples=args.num_samples,
        output_dir=Path(args.output),
        start_index=args.start_index,
        fps=args.fps,
        frame_size=args.frame_size,
        banner_height=args.banner_height,
        max_frames=args.max_frames,
        s3_bucket=args.s3_bucket,
        s3_prefix=args.s3_prefix,
    )

    pipeline = TaskPipeline(config)
    samples = pipeline.run()

    layout = f"{config.output_dir}/{config.domain}_task/"
    print(f"Wrote {len(samples)} samples to {layout}", flush=True)


if __name__ == "__main__":
    main()
