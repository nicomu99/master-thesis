# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#     "vllm==0.11.0",
#     "transformers>=4.56,<5",
#     "anthropic",
#     "datasets>=4.1.1",
#     "google-genai",
#     "huggingface-hub",
#     "numpy",
#     "openai~=1.109.1",
#     "pandas~=2.3.2",
#     "pyarrow",
#     "python-dotenv~=1.1.1",
#     "tqdm",
#     "torch==2.8.0",
#     "torchaudio==2.8.0",
#     "torchvision==0.23.0",
#     "xformers==0.0.32.post1",
# ]
#
# [tool.uv.sources]
# torch = { index = "pytorch" }
# torchaudio = { index = "pytorch" }
# torchvision = { index = "pytorch" }
# xformers = { index = "pytorch" }
#
# [[tool.uv.index]]
# name = "pytorch"
# url = "https://download.pytorch.org/whl/cu129"
# explicit = true
# ///
"""Local vLLM inference.

Produces the same data as hf_inference.py, but generates the completions with vLLM instead of the
transformers pipeline. Checkpoints and the done marker are shared with hf_inference.py.

vLLM cannot share an environment with the rest of the project: no release supports both transformers 5.2
and a torch version that a pre-compiled flash-attn wheel exists for. The dependencies are therefore
declared in the script header and `uv run vllm_inference.py` (without `python`) runs it in its own
environment, locked in vllm_inference.py.lock.
"""
import os
import json
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from hf_inference import (
    CHECKPOINT_KEYS,
    DATASET_IDS,
    DONE_MARKER,
    drop_answered,
    get_checkpoint_path,
    load_checkpoint,
    open_checkpoint,
    prepare_prompt,
)
from inference.evaluator import Evaluator
from inference.batch_request_handler import BatchRequestHandler
from inference.utils import DATA_PATH

# Number of prompts handed to vLLM at once, the completions of a chunk are checkpointed together
CHUNK_SIZE = 1024


def main(
    dataset_path: str,
    model_string: str,
    force: bool = False,
    datasets: list[str] | None = None,
    only_missing: bool = False,
    gpus: int = 1,
    max_model_len: int | None = None,
    gpu_memory_utilization: float = 0.9,
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
        gpus (int): Number of GPUs the model is split across. Defaults to 1.
        max_model_len (int | None): Context length (prompt and completion) vLLM reserves memory for.
            Defaults to the context length of the model.
        gpu_memory_utilization (float): Fraction of the GPU memory vLLM may use. Defaults to 0.9.
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

    tokenizer = AutoTokenizer.from_pretrained(model_string)
    llm_kwargs = dict(
        model=model_string,
        dtype="bfloat16",
        tensor_parallel_size=gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        # Ignore the sampling defaults of the model's generation_config.json, its stop tokens are still used
        generation_config="vllm",
    )
    if max_model_len is not None:
        llm_kwargs["max_model_len"] = max_model_len
    if gpus > 1:
        # The faster all-reduce implementations share GPU memory between the worker processes, which the
        # CUDA driver refuses on the cluster. Plain NCCL works everywhere.
        os.environ["VLLM_ALLREDUCE_USE_SYMM_MEM"] = "0"
        llm_kwargs["disable_custom_all_reduce"] = True
    llm = LLM(**llm_kwargs)

    # Greedy decoding, same as do_sample=False in hf_inference.py
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.0,
        top_p=1.0,
    )

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

        checkpoint_path = get_checkpoint_path(dataset_id, dataset_path, model_string)
        checkpoint_paths.append(checkpoint_path)

        checkpoint_df = load_checkpoint(checkpoint_path)
        is_done = long_index.isin(pd.MultiIndex.from_frame(checkpoint_df[CHECKPOINT_KEYS]))
        remaining = np.flatnonzero(~is_done)
        print(f"{is_done.sum()} samples already processed, {len(remaining)} remaining samples.")

        with open_checkpoint(checkpoint_path) as checkpoint_file:
            for start in tqdm(range(0, len(remaining), CHUNK_SIZE)):
                chunk_df = long_df.iloc[remaining[start:start + CHUNK_SIZE]]

                # Apply the chat template here
                chat_prompts = prepare_prompt(
                    {"persona": chunk_df["persona"], "prompt": chunk_df["prompt"]},
                    fn_tokenizer=tokenizer
                )["prompt"]
                # The chat template already contains the special tokens. The transformers pipeline does not
                # add them a second time either.
                token_ids = tokenizer(chat_prompts, add_special_tokens=False)["input_ids"]

                outputs = llm.generate(
                    [{"prompt_token_ids": ids} for ids in token_ids],
                    sampling_params,
                    use_tqdm=False
                )
                for static_id, answer_col, out in zip(chunk_df["static_id"], chunk_df["answer_col"], outputs):
                    row = {
                        "static_id": static_id,
                        "persona": answer_col,
                        "completion": out.outputs[0].text
                    }
                    checkpoint_file.write(json.dumps(row, ensure_ascii=False) + "\n")
                checkpoint_file.flush()

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


# vLLM starts worker processes that import this module again, so the entry point has to be guarded
if __name__ == "__main__":
    load_dotenv()
    # huggingface_hub reads HF_TOKEN on its own. Without it, login() asks for a token, which only works in
    # an interactive session and not in a batch job.
    if not os.environ.get("HF_TOKEN"):
        login()

    parser = argparse.ArgumentParser(
        description="Inference script for HF open weight models, served with vLLM.")
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
    parser.add_argument(
        "--gpus",
        help="Number of GPUs the model is split across.",
        type=int,
        default=1)
    parser.add_argument(
        "--max-model-len",
        help="Context length vLLM reserves memory for. Defaults to the context length of the model.",
        type=int,
        default=None)
    parser.add_argument(
        "--gpu-memory-utilization",
        help="Fraction of the GPU memory vLLM may use.",
        type=float,
        default=0.9)
    args = parser.parse_args()

    main(
        args.model_path, args.model, args.force, args.datasets, args.only_missing,
        args.gpus, args.max_model_len, args.gpu_memory_utilization)
