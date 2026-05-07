from typing import Literal

import re
import argparse
import platform
import subprocess
from pathlib import Path

import pandas as pd
from inference.persona_registry import PersonaRegistry


def _create_sample_df(
    dataset_id: Literal["flores", "alpaca"],
    save_file: Path
):
    model_dir = Path("data/inference")

    judgment_samples = []
    for model_path in model_dir.iterdir():
        df = pd.read_parquet(f"{model_path}/{dataset_id}.parquet")

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


def main(
    dataset_id: Literal["flores", "alpaca"]
):
    judgment_sample_file = Path(f"data/judgments/{dataset_id}_judgment_samples.parquet")
    # if not judgment_sample_file.exists():
    _create_sample_df(dataset_id, judgment_sample_file)
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
    print("All samples were processed.")


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
    args = parser.parse_args()
    main(args.dataset_name)
