from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from cli import add_training_arguments, build_config, configure_logging
from trainer import Trainer


def main():
    parser = add_training_arguments(argparse.ArgumentParser())
    args = parser.parse_args()
    timestamp = datetime.now().strftime("%m%d-%H%M%S")
    task_dir = Path(args.output_dir) / f"DAGNN_{args.source}_to_{args.target}_{timestamp}"
    task_dir.parent.mkdir(parents=True, exist_ok=True)
    configure_logging(task_dir.parent / f"{task_dir.name}.log")
    trainer = Trainer(build_config(args, args.source, args.target, task_dir))
    trainer.setup()
    summary = trainer.train()
    print(summary)


if __name__ == "__main__":
    main()
