import argparse

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from inference.evaluator import Evaluator
from inference.utils import QUESTION_COLUMN, DATA_PATH

from .extract_answers import extract_answers

from inference.utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def prepare_df(
    dataset_name: str = "mmlu-pro",
    output_folder: str = "data/evaluation"
):
    """

    Args:
        dataset_name (str): Name of the dataset to load through the evaluator. Defaults to "mmlu-pro".
        output_folder (str, optional): Path to the folder where the resulting
            CSV file will be saved. Defaults to "data/evaluation".

    Raises:
        RuntimeError: If the directory referenced by `DATA_PATH` does not exist.
    """
    log.info("Starting dataframe preparation")
    load_dotenv()

    data_folder = DATA_PATH
    if not data_folder.exists():
        raise RuntimeError("Data folder does not exist. Please make sure that inference has finished.")

    data = []
    for model_folder in data_folder.iterdir():
        model_folder_name = model_folder.name
        if not model_folder.is_dir():
            continue

        evaluator = Evaluator(dataset_path=model_folder_name)
        _, df = evaluator.get_data(dataset_name)

        persona_registry = evaluator.get_persona_registry()
        answer_columns = persona_registry.get_answer_columns()
        extracted_columns = [col.replace("_answer", "") for col in answer_columns]

        df = extract_answers(df, answer_columns, "mmlu")
        df["model"] = model_folder_name
        for col in extracted_columns:
            df[col] = (df["answer"] == df[col]).astype(int)

        df = pd.melt(
            df,
            id_vars=["static_id", "model", "category"],
            value_vars=extracted_columns,
            var_name="persona",
            value_name="score"
        )
        data.append(df)

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    result = pd.concat(data, ignore_index=True)
    csv_path = output_path / "mmlu-pro.csv"
    result.to_csv(csv_path)
    log.info("Dataframe preparation finished, created %s", csv_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-name",
        default="mmlu-pro",
        help=(
            "Name of the dataset to load via the evaluator. Make sure this name is equal to the one given to the "
            "MMLU-Pro dataset in the config_dataset.json file. Defaults to %(default)s."
        )
    )
    parser.add_argument(
        "--df-output-root",
        default="data/evaluation",
        help="Folder where the consolidated IFBench csv will be written. Defaults to %(default)s."
    )

    args = parser.parse_args()
    prepare_df(args.dataset_name, args.df_output_root)


if __name__ == "__main__":
    main()
