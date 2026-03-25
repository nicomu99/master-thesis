"""Unified CLI entry point for evaluation utilities."""
import argparse

from .ifbench_utils import run_ifbench
from .preparation_utils import prepare_df


def main() -> None:
    parser = argparse.ArgumentParser(prog="python3 -m evaluate.entrypoint")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prep_parser = subparsers.add_parser("prepare", help="Prepare evaluation dataframes.")
    prep_parser.add_argument(
        "--dataset-name",
        default="mmlu-pro",
        help=(
            "Name of the dataset to load via the evaluator. Should be the same as the corresponding"
            "dataset names in the config_dataset.json file. By default, this is \"mmlu-pro\", "
            "\"flores\" and \"MATH\". Defaults to %(default)s."
        ),
    )
    prep_parser.add_argument(
        "--df-output-root",
        default="data/evaluation",
        help="Folder where the consolidated csv will be written. Defaults to %(default)s.",
    )

    ifbench_parser = subparsers.add_parser("ifbench", help="Run IFBench evaluation.")
    ifbench_parser.add_argument(
        "--eval-root-env",
        default="IFBENCH_EVAL_ROOT",
        help="Name of the env var pointing to the IFBench evaluation script root. Defaults to %(default)s.",
    )
    ifbench_parser.add_argument(
        "--dataset-name",
        default="IFBench",
        help=(
            "Name of the dataset to load via the evaluator. Make sure this name is equal to the one given to the "
            "IFBench dataset in the config_dataset.json file. Defaults to %(default)s."
        ),
    )
    ifbench_parser.add_argument(
        "--df-output-root",
        default="data/evaluation",
        help="Folder where the consolidated IFBench csv will be written. Defaults to %(default)s.",
    )

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_df(dataset_name=args.dataset_name, output_folder=args.df_output_root)
    elif args.command == "ifbench":
        run_ifbench(
            eval_root_env=args.eval_root_env,
            dataset_name=args.dataset_name,
            df_output_root=args.df_output_root,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
