import argparse
from pathlib import Path
from typing import List

from dprt.datasets import prepare
from dprt.utils.config import load_config
from dprt.utils.misc import set_seed


def main(src: List[Path], cfg: Path, dst: Path) -> None:
    """Data preparation for subsequent model training or evaluation.

    Args:
        src_list (List[Path]): List of source directory paths to the raw dataset folders.
        cfg (Path): Path to the configuration file.
        dst (Path): Destination directory to save the processed dataset files.
    """
    try:
        # Load dataset configuration
        config = load_config(cfg)
    except Exception as e:
        print(f"Error loading configuration file: {e}")
        return

    try:
        # Set global random seed
        set_seed(config["computing"]["seed"])
    except KeyError:
        print("Seed not found in configuration file.")
        return

    # Prepare dataset
    preperator = prepare(config["dataset"], config)

    preperator.prepare(src, dst, config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DPRT data preprocessing")

    default_src = [
        Path("/data/Samsung 8TB 1"),
        Path("/data/Samsung 500GB"),
        Path("/data/Samsung 8TB 2"),
    ]

    parser.add_argument(
        "--src",
        type=Path,
        nargs="+",
        default=default_src,
        help="Paths to the raw dataset folders.",
    )

    parser.add_argument(
        "--cfg",
        type=Path,
        default=Path("/app/config/kradar.json"),
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("/data/Samsung 8TB 2/kradar/processed"),
        help="Path to save the processed dataset.",
    )
    args = parser.parse_args()

    main(src=args.src, cfg=args.cfg, dst=args.dst)
