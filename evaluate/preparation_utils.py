import argparse

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from inference.evaluator import Evaluator
from inference.utils import DATA_PATH

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
        id_vars = ["static_id", "model"]
        if dataset_name == "flores":
            scoring_columns = persona_registry.get_judgment_columns()
            extracted_columns = [col.replace("_judgment", "") for col in scoring_columns]
            id_vars.append("iso_639_3")
        elif dataset_name == "mmlu-pro":
            scoring_columns = persona_registry.get_answer_columns()
            extracted_columns = [col.replace("_answer", "") for col in scoring_columns]
            id_vars.append("category")
        else:
            scoring_columns = persona_registry.get_answer_columns()
            extracted_columns = [col.replace("_answer", "") for col in scoring_columns]
            id_vars.append("Subject")

        df = extract_answers(df, scoring_columns, dataset_name)
        df["model"] = model_folder_name
        for col in extracted_columns:
            df[col] = (df["answer"] == df[col]).astype(int)

        df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=extracted_columns,
            var_name="persona",
            value_name="score"
        )
        data.append(df)

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    result = pd.concat(data, ignore_index=True)
    csv_path = output_path / f"{dataset_name}.csv"
    result.to_csv(csv_path)
    log.info("Dataframe preparation finished, created %s", csv_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-name",
        default="mmlu-pro",
        help=(
            "Name of the dataset to load via the evaluator. Should be  "
            "MMLU-Pro, Flores+ or MATH dataset in the config_dataset.json file. Defaults to %(default)s."
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
