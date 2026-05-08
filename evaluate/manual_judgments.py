"""CLI program for the creation of manual judgments."""
from typing import Literal

import re
import argparse
import platform
import subprocess
from pathlib import Path

import pandas as pd

from evaluate import extract_answer_flores, extract_answer_alpaca
from inference.persona_registry import PersonaRegistry

from inference.utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


def _create_sample_df(
    dataset_id: Literal["flores", "alpaca"],
    save_file: Path
):
    model_dir = Path("data/inference")
    if not model_dir.is_dir():
        raise RuntimeError("Model path does not exist.")

    judgment_samples = []
    for model_path in model_dir.iterdir():
        if not model_path.is_dir():
            continue
        model_file = model_path / f"{dataset_id}.parquet"
        if not model_file.exists():
            log.warning("%s could not be found, skipping.", model_file)
            continue
        df = pd.read_parquet(model_file)

        if dataset_id == "flores":
            df = df[df["iso_639_3"] == "deu"]

        if dataset_id == "flores":
            df = df.dropna()
            sample = df.sample(10, random_state=42)
        else:
            df = df.dropna(thresh=16)
            alpaca_ids = [
                "alpaca_110", "alpaca_559", "alpaca_586", "alpaca_247", "alpaca_192",
                "alpaca_589", "alpaca_725", "alpaca_101", "alpaca_507", "alpaca_742",
            ]
            sample = df[df["static_id"].isin(alpaca_ids)].copy()
        sample.loc[:, "model"] = model_path.stem
        judgment_samples.append(sample)
    if not judgment_samples:
        log.warning("No samples found, skipping.")
        return
    judgment_df: pd.DataFrame = pd.concat(judgment_samples)
    judgment_df.to_parquet(save_file)


def _remove_think_block(
    completion: str
):
    return re.sub(r"<think>.*?</think>", "", completion, flags=re.DOTALL)


def clear_cli():
    """Clears the CLI."""
    command = "cls" if platform.system() == "Windows" else "clear"
    subprocess.run(command, shell=True, check=False)


def create_manual_judgments(
    dataset_id: Literal["flores", "alpaca"]
) -> None:
    """CLI program that lets the user judge samples by hand.

    The program iterates through each sample, one by one, letting the user choose which
    sample they prefer more.

    Args:
        dataset_id (Literal["flores", "alpaca"]): The chosen dataset to process.
    """
    judgment_sample_file = Path(f"data/judgments/{dataset_id}_judgment_samples.parquet")
    _create_sample_df(dataset_id, judgment_sample_file)
    if not judgment_sample_file.exists():
        raise RuntimeError("Judgment sample file could not be created")
    judgment_sample_df = pd.read_parquet(judgment_sample_file)

    persona_registry = PersonaRegistry()
    persona_configs = persona_registry.get_configs()
    reference_column = persona_registry.get_reference_config().answer_column

    judgments_file = Path(f"data/judgments/{dataset_id}_manual_judgments.parquet")
    if judgments_file.exists():
        manual_judgments_df = pd.read_parquet(judgments_file)
    else:
        columns = ["static_id", "model", "persona", "score"]
        if dataset_id == "flores":
            columns.append("iso_639_3")
        manual_judgments_df = pd.DataFrame(columns=columns)

    processed_samples = 0
    judgment_sample_df = judgment_sample_df.sort_values(reference_column, key=lambda x: x.str.len(), ascending=False)
    for row_dict in judgment_sample_df.to_dict(orient="records"):
        for persona in persona_configs:
            processed_samples += 1
            static_id = row_dict["static_id"]
            persona_name = persona.name.replace("_persona", "")
            model = row_dict["model"]

            answer_column = persona.answer_column
            if answer_column not in judgment_sample_df.columns:
                continue

            exists = (
                (manual_judgments_df["static_id"] == static_id) &
                (manual_judgments_df["model"] == model) &
                (manual_judgments_df["persona"] == persona_name)
            ).any()
            if exists:
                continue

            print(f"Sample {processed_samples}/{len(judgment_sample_df) * len(persona_configs)}")
            print(f"Static ID: ", static_id, ", model: ", model, "reference length: ", len(row_dict[reference_column]))
            print("Question: ", row_dict["question"], "\n")
            print("\033[32mReference:\033[0m\n", _remove_think_block(row_dict[reference_column]).strip(), "\n")
            print("\033[32mAnswer:\033[0m\n", _remove_think_block(row_dict[answer_column]).strip(), "\n")

            while True:
                judgment = input("0 reference wins; 1 tie; 2 persona wins: ")
                if judgment in {"0", "1", "2"}:
                    judgment = int(judgment)
                    break
                print("Please enter 0, 1, or 2.")

            judgment_dict = {
                "static_id": static_id,
                "model": model,
                "persona": persona_name,
                "score": judgment
            }
            if dataset_id == "flores":
                judgment_dict["iso_639_3"] = row_dict["iso_639_3"]
            manual_judgments_df.loc[len(manual_judgments_df)] = judgment_dict
            manual_judgments_df.to_parquet(judgments_file)
            clear_cli()
    log.info("All samples were processed.")


def calculate_agreement():
    """Creates agreement dataframes."""
    persona_registry = PersonaRegistry()
    judgment_columns = persona_registry.get_judgment_columns()

    judgment_path = Path("data/judgments")
    if not judgment_path.is_dir():
        raise RuntimeError("Judgment path does not exist")
    model_path = Path("data/inference")
    if not model_path.is_dir():
        raise RuntimeError("Model path does not exist")

    for dataset_id, extract_fn in [("flores", extract_answer_flores), ("alpaca", extract_answer_alpaca)]:
        manual_path = judgment_path / f"{dataset_id}_manual_judgments.parquet"
        if not manual_path.exists():
            log.warning("Manual judgments for %s not found, skipping.", dataset_id)
            continue
        manual_df = pd.read_parquet(manual_path)
        samples_ids = manual_df["static_id"].unique()

        llm_df = []
        for model_dir in model_path.iterdir():
            if not model_dir.is_dir():
                continue
            model_name = model_dir.stem

            model_file = model_dir / f"{dataset_id}.parquet"
            if not model_file.exists():
                log.warning("%s not found, skipping.", model_file)
                continue
            model_df = pd.read_parquet(model_file)
            judgment_samples = model_df[model_df["static_id"].isin(samples_ids)]

            id_vars = ["static_id"]
            if dataset_id == "flores":
                id_vars.append("iso_639_3")

            judgment_samples = judgment_samples.melt(
                id_vars=id_vars, value_vars=judgment_columns,
                var_name="persona", value_name="score"
            )
            judgment_samples["score"] = judgment_samples.apply(
                lambda x: extract_fn(x, "score"), axis="columns")
            judgment_samples["persona"] = judgment_samples["persona"].str.replace("_judgment", "")
            judgment_samples["model"] = model_name
            llm_df.append(judgment_samples)
        if not llm_df:
            log.warning("No agreement data collected for %s, skipping.", dataset_id)
            continue
        llm_df = pd.concat(llm_df, ignore_index=True)
        key_cols = ["static_id", "model", "persona"]
        if dataset_id == "flores":
            key_cols.append("iso_639_3")

        agreement_df = pd.merge(
            llm_df, manual_df, suffixes=("_llm", "_manual"),
            left_on=key_cols, right_on=key_cols,
        )
        agreement_df["agreement"] = agreement_df["score_llm"] == agreement_df["score_manual"]
        agreement_file = judgment_path / f"{dataset_id}_agreement.csv"
        agreement_df.to_csv(agreement_file, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual judgment CLI tool")
    parser.add_argument(
        "--dataset-name",
        default="flores",
        choices=["flores", "alpaca"],
        help=(
            "Name of the dataset to load. Defaults to %(default)s."
        ),
    )
    parser.add_argument(
        "command", nargs="?", choices=["judgments", "agreement"],
        default="judgments", help="Command to run"
    )
    args = parser.parse_args()

    if args.command == "agreement":
        calculate_agreement()
    else:
        create_manual_judgments(args.dataset_name)
