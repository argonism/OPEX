from argparse import ArgumentParser, Namespace
import json
from typing import Union
from logging import getLogger

import matplotlib.pyplot as plt
from pathlib import Path

logger = getLogger(__name__)


def load_intact_result(result_path: Path) -> dict:
    path = Path(result_path)
    return json.loads(path.read_text())


def validate_analyze_results(target_path: Path, compare_path: Path) -> None:
    (
        model_name,
        _,
        percentage,
        base_rank,
        ymd,
        hms,
    ) = target_path.parent.stem.split("_")
    (
        bm25_name,
        _,
        bm25_percentage,
        bm25_base_rank,
        _,
        _,
    ) = compare_path.parent.stem.split("_")
    if base_rank != bm25_base_rank:
        raise Exception(
            f"analyze result base ranks are not match! {base_rank} and {bm25_base_rank}"
        )
    return (model_name, bm25_name)


def compare(
    target_result_path: Path,
    bm25_result_path: Path,
    output_dir: Path,
) -> None:
    model_name, bm25_name = validate_analyze_results(
        target_result_path, bm25_result_path
    )

    out_filename = f"{target_result_path.parent.stem}/{bm25_result_path.parent.stem}"
    output_path = Path(output_dir).joinpath(out_filename)
    output_path.mkdir(parents=True, exist_ok=True)

    target_result = load_intact_result(target_result_path)
    bm25_result = load_intact_result(bm25_result_path)

    taget_average_ranking_shift = sum(target_result["new_ranks"]) / len(
        target_result["new_ranks"]
    )
    bm25_average_ranking_shift = sum(bm25_result["new_ranks"]) / len(
        bm25_result["new_ranks"]
    )
    print(f"taget_average_ranking_shift: {taget_average_ranking_shift:.3f}")
    print(f"bm25_average_ranking_shift: {bm25_average_ranking_shift:.3f}")

    fig = plt.figure(figsize=(24, 7))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    ax1.hist(target_result["new_ranks"], bins=100)
    ax1.set_title(
        f"{model_name} \n average_ranking_shift: {taget_average_ranking_shift:.3f}"
    )
    ax2.hist(bm25_result["new_ranks"], bins=100)
    ax2.set_title(
        f"{bm25_name} \n average_ranking_shift: {bm25_average_ranking_shift:.3f}"
    )

    save_fig_path = output_path.joinpath("new_ranks_distribution.png")
    fig.savefig(save_fig_path)


def parse_args() -> Namespace:
    parser = ArgumentParser(description="intact bm25_comparison")
    parser.add_argument("target_result")
    parser.add_argument("bm25_result")
    parser.add_argument("-o", "--output_dir", default="intact_compare/")
    args, other = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    output_dir = Path(__file__).parent.joinpath(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    compare(Path(args.target_result), Path(args.bm25_result), output_dir)
