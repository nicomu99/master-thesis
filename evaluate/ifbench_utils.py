import json
import argparse

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from inference.evaluator import Evaluator
from inference.utils import QUESTION_COLUMN, DATA_PATH


def prepare_input(
    output_folder: str = "temp/ifbench_out",
    dataset_name: str = "IFBench"
):
    """Generate JSONL files for the official IFBench evaluation pipeline.

    This function reads completions from the directory specified by
    `DATA_PATH`, loads the dataset for each model subdirectory, and writes one
    JSONL file per persona configuration in the format the pipeline assumes.

    Args:
        output_folder: Directory where the generated JSONL files will be stored.
            The directory is created if it does not already exist.
        dataset_name: Name of the dataset to load through the evaluator.

    Raises:
        RuntimeError: If the directory referenced by `DATA_PATH` does not exist.
    """
    load_dotenv()

    data_folder = DATA_PATH
    if not data_folder.exists():
        raise RuntimeError("Data folder does not exist. Please make sure that inference has finished.")

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)
    for model_folder in data_folder.iterdir():
        model_folder_name = model_folder.name
        if not model_folder.is_dir():
            continue

        evaluator = Evaluator(dataset_path=model_folder_name)
        _, df = evaluator.get_data(dataset_name)

        persona_registry = evaluator.get_persona_registry()
        persona_configs = persona_registry.get_configs()

        df["key"] = pd.to_numeric(df["key"])
        df = df.sort_values("key").reset_index(drop=True)
        for persona_cfg in persona_configs:
            persona_name = persona_cfg.name
            answer_column = persona_cfg.answer_column
            if answer_column not in df.columns:
                print(f"Skipping missing persona column: {persona_name}")
                continue

            output_path = output_folder / f"{model_folder_name}_{persona_name}.jsonl"
            with output_path.open("w", encoding="utf-8") as f:
                for row in df.itertuples(index=False):
                    response = getattr(row, answer_column)
                    question = getattr(row, QUESTION_COLUMN)
                    entry = {
                        "prompt": question.strip(),
                        "response": response,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Wrote {output_path}")


def prepare_df():
    pass


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="python3 -m evaluate.ifbench_utils")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # prepare_input() arguments
    parser_prepare_input = subparsers.add_parser(
        "prepare_input",
        help="Create the JSONL input files for the official IFBench evaluation script."
    )
    parser_prepare_input.add_argument(
        "-o", "--output-folder",
        default="temp/ifbench_out",
        help="Directory where generated JSONL files will be written. Defaults to %(default)s."
    )
    parser_prepare_input.add_argument(
        "-d", "--dataset-name",
        default="IFBench",
        help=(
            "Name of the dataset to load via the evaluator. Make sure this name is equal to the one given to the "
            "IFBench dataset in the config_dataset.json file. Defaults to %(default)s."
        )
    )

    args = parser.parse_args()
    if args.command == "prepare_input":
        prepare_input(args.output_folder, args.dataset_name)


if __name__ == "__main__":
    main()
