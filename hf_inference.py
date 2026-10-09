"""Local HF inference."""
import gc
import re
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from datasets import Dataset
from huggingface_hub import login
from transformers.pipelines.pt_utils import KeyDataset
from transformers import pipeline, GenerationConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer

from inference.evaluator import Evaluator
from inference.batch_request_handler import BatchRequestHandler
from inference.utils import TEMP_PATH, DATA_PATH

CHECKPOINT_KEYS = ["static_id", "persona"]
DONE_MARKER = "hf_inference_done.json"
DATASET_IDS = ["mmlu-pro", "MATH", "flores", "IFBench", "alpaca"]


# def merge_and_write(
#     dataframe: pd.DataFrame,
#     subset_df: pd.DataFrame,
#     file_name: str,
#     id_column: str = "static_id"
# ):
#     """Merges two dataframes.
#
#     This function updates the entries in dataframe with the corresponding values in subset_df. Any
#     entries in dataframe will be overwritten.
#
#     Args:
#         dataframe (pd.DataFrame): DataFrame object. The entries in this dataframe will be overwritten.
#         subset_df (pd.DataFrame): Subset dataframe that will be used to update dataframe.
#         file_name (str): File name of the written dataframe.
#         id_column (str): Column to use as keys for the updates.
#     """
#     for col in subset_df.columns:
#         if col not in dataframe.columns:
#             dataframe[col] = None
#
#     dataframe.set_index(id_column, inplace=True)
#     subset_df = subset_df.set_index(id_column)
#
#     # Any value in dataframe will be overwritten by subset_df
#     dataframe.update(subset_df)
#     dataframe.reset_index(inplace=True)
#
#     # df_file = f"{drive_path}/{file_name}.parquet"
#     df_file = f"data/{file_name}.parquet"
#     dataframe.to_parquet(df_file)


def get_checkpoint_path(
    dataset_id: str,
    dataset_path: str,
    model_string: str
) -> Path:
    """Returns the checkpoint file for one dataset of one inference run.

    The file name contains both the data folder and the model, so that jobs for different models never
    share a checkpoint and a checkpoint is never picked up by a different model.

    Args:
        dataset_id (str): Dataset identifier.
        dataset_path (str): Data folder of the run.
        model_string (str): Model identifier.

    Returns:
        Path: Path to the checkpoint file.
    """
    run_id = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{dataset_path}_{model_string}")
    return TEMP_PATH / "hf_inference" / f"{dataset_id}_{run_id}.jsonl"


def load_checkpoint(checkpoint_path: Path) -> pd.DataFrame:
    """Reads the completions that were already generated.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.

    Returns:
        pd.DataFrame: One row per finished static_id and answer column.
    """
    rows = []
    if checkpoint_path.exists():
        with checkpoint_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # A job killed while writing leaves a partial line, that sample is generated again
                    continue
    checkpoint_df = pd.DataFrame(rows, columns=[*CHECKPOINT_KEYS, "completion"])
    return checkpoint_df.drop_duplicates(CHECKPOINT_KEYS, keep="last")


def open_checkpoint(checkpoint_path: Path):
    """Opens the checkpoint file for appending.

    Args:
        checkpoint_path (Path): Path to the checkpoint file.

    Returns:
        A text file object in append mode.
    """
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    ends_with_partial_line = False
    if checkpoint_path.exists() and checkpoint_path.stat().st_size > 0:
        with checkpoint_path.open("rb") as f:
            f.seek(-1, 2)
            ends_with_partial_line = f.read(1) != b"\n"

    file = checkpoint_path.open("a", encoding="utf-8")
    if ends_with_partial_line:
        file.write("\n")
    return file


def drop_answered(
    long_df: pd.DataFrame,
    dataset_df: pd.DataFrame
) -> pd.DataFrame:
    """Removes the samples that already have an answer in the dataset.

    Args:
        long_df (pd.DataFrame): Long format samples, one row per static_id and answer column.
        dataset_df (pd.DataFrame): The dataset the samples were created from.

    Returns:
        pd.DataFrame: The samples whose answer is still missing.
    """
    answer_cols = [col for col in long_df["answer_col"].unique() if col in dataset_df.columns]
    answered_df = pd.melt(
        dataset_df[["static_id", *answer_cols]],
        id_vars=["static_id"],
        value_vars=answer_cols,
        var_name="answer_col",
        value_name="answer"
    ).dropna(subset=["answer"])
    long_index = pd.MultiIndex.from_frame(long_df[["static_id", "answer_col"]])
    is_answered = long_index.isin(pd.MultiIndex.from_frame(answered_df[["static_id", "answer_col"]]))
    return long_df[~is_answered].reset_index(drop=True)


def prepare_prompt(
    batch: dict,
    fn_tokenizer: PreTrainedTokenizer
) -> dict:
    """Applies the tokenizers chat template to a batch of inputs.

    Args:
        batch (dict): A batch of inputs.
        fn_tokenizer (fn_tokenizer): The tokenizer.

    Returns:
        dict: Inputs with applied chat template.
    """
    kwargs = dict(
        tokenize=False,
        add_generation_prompt=True)
    # The switch is a variable of the chat template, not a parameter of apply_chat_template
    if "enable_thinking" in str(fn_tokenizer.chat_template):
        kwargs["enable_thinking"] = False

    prompts_list = []
    for system, user in zip(batch["persona"], batch["prompt"]):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        prompts_list.append(
            fn_tokenizer.apply_chat_template(
                messages,
                **kwargs
            )
        )
    return {"prompt": prompts_list}


def main(
    dataset_path: str,
    model_string: str,
    force: bool = False,
    datasets: list[str] | None = None,
    only_missing: bool = False,
):
    """Main inference pipeline.

    Args:
        dataset_path (str): Dataset identifier of the dataset to use.
        model_string (str): Model identifier. Must be the same as the repository
            name on HF.
        force (bool): If true, generations are created again even if the data folder is marked as
            completed. Defaults to False.
        datasets (list[str] | None): Datasets to generate for. Defaults to all datasets.
        only_missing (bool): If true, only the samples without an answer in the data folder are generated,
            all existing answers are kept. Defaults to False.
    """
    datasets = datasets or DATASET_IDS
    marker_path = DATA_PATH / dataset_path / DONE_MARKER
    was_completed = marker_path.exists()
    if was_completed:
        if not force:
            raise SystemExit(
                f"All generations in {marker_path.parent} are already completed: {marker_path.read_text()}\n"
                "Use --force to generate them again."
            )
        # The folder holds a mix of old and new generations until the forced run has finished
        marker_path.unlink()

    tokenizer = AutoTokenizer.from_pretrained(model_string, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(
        model_string,
        device_map="auto",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        dtype=torch.bfloat16,
    )
    pipe.tokenizer.pad_token_id = pipe.tokenizer.eos_token_id

    evaluator = Evaluator(dataset_path)
    persona_registry = evaluator.get_persona_registry()
    personas = persona_registry.get_names()

    checkpoint_paths = []

    # Load dataset
    for dataset_id in datasets:
        _, dataset_df = evaluator.get_data(dataset_id)
        qs_type = evaluator.get_question_type(dataset_id)
        dataset_df = dataset_df.copy()

        # Bulk create question prompts
        prompts = dataset_df.apply(
            BatchRequestHandler.create_question_prompt, axis=1,
            question_type=qs_type)
        dataset_df.loc[:, "prompt"] = prompts

        # Sort the prompts by length to minimize padding overhead
        prompts_tokenized = pipe.tokenizer(prompts.to_list())
        dataset_df.loc[:, "length"] = [len(p) for p in prompts_tokenized["input_ids"]]
        dataset_df = dataset_df.sort_values(by="length", ascending=False)

        # Long format allows using HF's batch inference pipeline
        long_df = pd.melt(
            dataset_df,
            id_vars=["static_id", "prompt"],
            value_vars=personas,
            var_name="persona_col",
            value_name="persona"
        )
        print(f"{len(long_df)} total samples to process.")

        long_df["answer_col"] = long_df["persona_col"].str.replace("persona", "answer")
        if only_missing:
            long_df = drop_answered(long_df, dataset_df)
            print(f"{len(long_df)} samples without an answer.")
            if len(long_df) == 0:
                continue
        long_index = pd.MultiIndex.from_frame(long_df[["static_id", "answer_col"]])

        # Apply the chat template here
        dataset_hf = Dataset.from_pandas(long_df, preserve_index=False)
        dataset_hf = dataset_hf.map(
            prepare_prompt,
            batched=True,
            fn_kwargs={"fn_tokenizer": tokenizer},
            num_proc=4
        )

        gen_cfg = GenerationConfig(
            max_new_tokens=2048,
            do_sample=False,
            top_p=1.0,
            temperature=1.0,
            pad_token_id=pipe.tokenizer.eos_token_id
        )
        checkpoint_path = get_checkpoint_path(dataset_id, dataset_path, model_string)
        checkpoint_paths.append(checkpoint_path)
        for batch_size in [16, 8, 4, 3, 2, 1]:
            # Read the checkpoint before every attempt, so that samples finished by an earlier job or by an
            # attempt that ran out of memory are not generated twice
            checkpoint_df = load_checkpoint(checkpoint_path)
            is_done = long_index.isin(pd.MultiIndex.from_frame(checkpoint_df[CHECKPOINT_KEYS]))
            remaining = np.flatnonzero(~is_done)
            print(f"{is_done.sum()} samples already processed, {len(remaining)} remaining samples.")
            if len(remaining) == 0:
                break

            remaining_df = long_df.iloc[remaining]
            print(f"Testing batch size {batch_size}")
            try:
                with open_checkpoint(checkpoint_path) as checkpoint_file:
                    # noinspection PyTypeChecker
                    for static_id, answer_col, out in tqdm(
                            zip(
                                remaining_df["static_id"],
                                remaining_df["answer_col"],
                                pipe(
                                    KeyDataset(dataset_hf.select(remaining), "prompt"),
                                    batch_size=batch_size, return_full_text=False,
                                    generation_config=gen_cfg
                                )
                            ), total=len(remaining)
                    ):
                        row = {
                            "static_id": static_id,
                            "persona": answer_col,
                            "completion": out[0]["generated_text"]
                        }
                        checkpoint_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                        checkpoint_file.flush()
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()
        else:
            raise RuntimeError(f"Ran out of memory for {dataset_id} even with batch size 1.")

        response_df = load_checkpoint(checkpoint_path)
        response_index = pd.MultiIndex.from_frame(response_df[CHECKPOINT_KEYS])
        response_df = response_df[response_index.isin(long_index)]
        if len(response_df) != len(long_df):
            raise RuntimeError(
                f"Expected {len(long_df)} completions for {dataset_id}, but found {len(response_df)} "
                f"in {checkpoint_path}."
            )

        # pivot back from long to wide
        response_df = response_df.pivot(index="static_id", columns="persona", values="completion")
        response_df = response_df.reset_index()

        # merge and save
        evaluator.save_data(dataset_id, response_df)

    # The checkpoints are kept until the whole run has finished, so that a restarted job skips finished datasets
    for checkpoint_path in checkpoint_paths:
        checkpoint_path.unlink()

    # A run on a subset of the datasets only completes a folder that was already completed before
    if not was_completed and set(datasets) != set(DATASET_IDS):
        return
    marker = {"model": model_string, "completed_at": datetime.now().isoformat(timespec="seconds")}
    marker_path.write_text(json.dumps(marker), encoding="utf-8")


if __name__ == "__main__":
    load_dotenv()
    login()

    parser = argparse.ArgumentParser(
        description="Inference script for HF open weight models.")
    parser.add_argument(
        "--model-path",
        help="The dataset to use for inference.",
        type=str,
        required=True)
    parser.add_argument(
        "--model",
        help="The model used for inference.",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument(
        "--force",
        help="Generate again, even if the data folder is marked as completed.",
        action="store_true")
    parser.add_argument(
        "--datasets",
        help="The datasets to generate for. Defaults to all datasets.",
        nargs="+",
        choices=DATASET_IDS,
        default=None)
    parser.add_argument(
        "--only-missing",
        help="Only generate the samples without an answer in the data folder, existing answers are kept.",
        action="store_true")
    args = parser.parse_args()

    main(args.model_path, args.model, args.force, args.datasets, args.only_missing)
