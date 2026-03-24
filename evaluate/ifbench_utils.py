"""Utility functions to prepare for the IFBench evaluation script, and to read its output in a structured way."""
import json
import os
import argparse
import subprocess
from collections import defaultdict

import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

from inference.evaluator import Evaluator
from inference.utils import QUESTION_COLUMN, DATA_PATH

from inference.utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def prepare_input(
    output_folder: str | Path = "temp/ifbench",
    dataset_name: str = "IFBench"
):
    """Generate JSONL files for the official IFBench evaluation pipeline.

    This function reads completions from the directory specified by
    `DATA_PATH`, loads the dataset for each model subdirectory, and writes one
    JSONL file per persona configuration in the format the pipeline assumes.

    Args:
        output_folder (str |Path): Directory where the generated JSONL files will be stored.
            The directory is created if it does not already exist. Defaults to "temp/ifbench".
        dataset_name (str): Name of the dataset to load through the evaluator. Defaults to "IFBench".

    Raises:
        RuntimeError: If the directory referenced by `DATA_PATH` does not exist.
    """
    log.info("Starting with input preparation")
    load_dotenv()

    data_folder = DATA_PATH
    if not data_folder.exists():
        raise RuntimeError("Data folder does not exist. Please make sure that inference has finished.")

    output_folder = Path(output_folder)
    output_folder.mkdir(exist_ok=True)

    input_path = output_folder / "ifbench_input.jsonl"
    input_written = False
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

        if not input_written:
            # Create the reference file only once
            with input_path.open("w", encoding="utf-8") as f:
                for row in df.itertuples(index=False):
                    instruction_id_list = getattr(row, "instruction_id_list")
                    kwargs = getattr(row, "kwargs")
                    if hasattr(instruction_id_list, "tolist"):
                        instruction_id_list = instruction_id_list.tolist()
                    if hasattr(kwargs, "tolist"):
                        kwargs = kwargs.tolist()
                    for param in kwargs:
                        for k, v in param.items():
                            if isinstance(v, float):
                                param[k] = int(v)
                    entry = {
                        "key": getattr(row, "key"),
                        "prompt": getattr(row, QUESTION_COLUMN).strip(),
                        "instruction_id_list": instruction_id_list,
                        "kwargs": kwargs,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            input_written = True
            log.info("Wrote %s", input_path)
        for persona_cfg in persona_configs:
            persona_name = persona_cfg.name
            answer_column = persona_cfg.answer_column
            if answer_column not in df.columns:
                log.warning("Skipping missing persona column: %s", persona_name)
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

            log.debug("Wrote %s", output_path)
    log.info("Input preparation finished")


def run_ifbench(
    eval_root_env: str = "IFBENCH_EVAL_ROOT",
    dataset_name: str = "IFBench",
    responses_root: str = "temp/ifbench",
    output_root: str = "temp/ifbench/raw",
    df_output_root: str = "data/evaluation",
):
    """Run the IFBench evaluation pipeline

    This function generates the required IFBench input file, evaluates all
    response JSONL files in the given folder using the official script, and
    consolidates the results into a single CSV in long-format.

    Args:
        eval_root_env (str): Environment variable pointing to the IFBench
            evaluation script root. Defaults to "IFBENCH_EVAL_ROOT".
        dataset_name (str): Name of the dataset to load through the evaluator.
            Defaults to "IFBench".
        responses_root (str): Folder containing model response JSONL files.
            Defaults to "temp/ifbench".
        output_root (str): Folder where the IFBench evaluation outputs will be
            written. Defaults to "temp/ifbench/raw".
        df_output_root (str): Folder where the consolidated CSV will be written.
            Defaults to "data/evaluation".

    Raises:
        RuntimeError: If the IFBench evaluation script root is not configured,
            or if the generated input file is missing.
    """
    log.info("Starting IFBench evaluation run")
    load_dotenv()
    value = os.getenv(eval_root_env)
    if value is None:
        raise RuntimeError(
            f"Environment variable {eval_root_env} was not found or is empty. Please set it to the "
            "IFBench evaluation script root."
        )
    eval_root = Path(value).expanduser()
    if not eval_root.exists():
        raise RuntimeError(
            f"Could not locate IFBench evaluation script. Set {eval_root_env} to its root folder."
        )

    responses_root = Path(responses_root)
    output_root = Path(output_root)
    output_root.mkdir(exist_ok=True, parents=True)

    prepare_input(output_folder=responses_root, dataset_name=dataset_name)
    benchmark_file = responses_root / "ifbench_input.jsonl"
    if not benchmark_file.exists():
        raise RuntimeError(f"Expected IFBench input file at {benchmark_file}.")
    log.info("Using IFBench input file at %s", benchmark_file)

    for response_file in responses_root.glob("*.jsonl"):
        if response_file == benchmark_file:
            continue
        try:
            cmd = [
                "python3", "-m", "run_eval",
                f"--input_data={benchmark_file.resolve()}",
                f"--input_response_data={response_file.resolve()}",
                f"--output_dir={output_root.resolve()}",
            ]
            log.info("Running IFBench eval on %s", response_file.name)
            subprocess.run(cmd, cwd=eval_root)
        except Exception as e:
            log.error(e)

    prepare_df(input_folder=str(output_root), output_folder=df_output_root)
    log.info("IFBench evaluation finished")


def prepare_df(
    input_folder: str | Path = "temp/ifbench/raw",
    output_folder: str = "data/evaluation"
):
    """Convert raw IFBench evaluation outputs into a consolidated CSV file.

    This function reads JSONL files generated by the IFBench evaluation script,
    extracts model names, personas, and instruction-following scores, and
    aggregates them into a single pandas DataFrame in long format. Only the
    loose evaluation is considered.

    The DataFrame is saved as "ifbench.csv" in the specified output folder.

    Args:
        input_folder (str | Path): Path to the folder containing raw IFBench
            JSONL output files. Defaults to "temp/ifbench/raw".
        output_folder (str, optional): Path to the folder where the resulting
            CSV file will be saved. Defaults to "data/evaluation".

    Raises:
        ValueError: If the input folder does not exist.
    """
    input_folder = Path(input_folder)
    if not input_folder.exists():
        raise ValueError("Input folder could not be found. Please make sure it exists.")

    data = defaultdict(list)
    for file in input_folder.iterdir():
        file_name = file.stem
        if "_loose" not in file_name:
            # only use loose output for evaluation
            continue

        file_name = file_name.replace("_persona-eval_results_loose", "").split("_")
        model, persona = file_name[0], "_".join(file_name[1:])
        with file.open("r", encoding="utf-8") as f:
            count = 0
            for line in f:
                obj = json.loads(line)
                data["static_id"].append(f"IFBench_{count}")
                data["model"].append(model)
                data["persona"].append(persona)
                data["score"].append(int(obj["follow_all_instructions"]))
                count += 1

    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(data)
    csv_path = output_path / "ifbench.csv"
    df.to_csv(csv_path)
    log.info("Wrote %s", csv_path)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(prog="python3 -m evaluate.ifbench_utils")
    parser.add_argument(
        "--eval-root-env",
        default="IFBENCH_EVAL_ROOT",
        help="Name of the env var pointing to the IFBench evaluation script root. Defaults to %(default)s."
    )
    parser.add_argument(
        "--dataset-name",
        default="IFBench",
        help=(
            "Name of the dataset to load via the evaluator. Make sure this name is equal to the one given to the "
            "IFBench dataset in the config_dataset.json file. Defaults to %(default)s."
        )
    )
    parser.add_argument(
        "--responses-root",
        default="temp/ifbench",
        help="Folder containing model response JSONL files. Defaults to %(default)s."
    )
    parser.add_argument(
        "--output-root",
        default="temp/ifbench/raw",
        help="Folder where IFBench evaluation outputs will be written. Defaults to %(default)s."
    )
    parser.add_argument(
        "--df-output-root",
        default="data/evaluation",
        help="Folder where the consolidated IFBench csv will be written. Defaults to %(default)s."
    )

    args = parser.parse_args()
    run_ifbench(
        args.eval_root_env,
        args.dataset_name,
        args.responses_root,
        args.output_root,
        args.df_output_root,
    )


if __name__ == "__main__":
    main()
