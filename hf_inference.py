"""Local HF inference."""
import gc
import inspect
import argparse
from pathlib import Path
from collections import defaultdict

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
    if "enable_thinking" in inspect.signature(fn_tokenizer.apply_chat_template).parameters:
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
):
    """Main inference pipeline.

    Args:
        dataset_path (str): Dataset identifier of the dataset to use.
        model_string (str): Model identifier. Must be the same as the repository
            name on HF.
    """
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

    # Load dataset
    for dataset_id in ["MATH"]:  # ["mmlu-pro", "MATH", "flores", "IFBench", "alpaca"]:
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

        temp_file_path = Path(f"temp_{dataset_id}_{dataset_path}.parquet")
        if temp_file_path.exists():
            temp_df = pd.read_parquet(temp_file_path)
            print(f"{len(temp_df)} samples already processed.")

            # TAKEN FROM:
            # https://stackoverflow.com/questions/33282119/pandas-filter-dataframe-by-another-dataframe-by-row-elements
            temp_df = temp_df.rename(columns={"persona": "persona_col"})
            temp_df["persona_col"] = temp_df["persona_col"].str.replace("answer", "persona")
            keys = ["static_id", "persona_col"]
            idx_long_df = long_df.set_index(keys).index
            idx_temp_df = temp_df.set_index(keys).index
            print(idx_long_df[:3], idx_temp_df[:3])
            long_df = long_df[~idx_long_df.isin(idx_temp_df)]
            print(f"{len(temp_df)} samples already processed, {len(long_df)} remaining samples.")

        dataset_hf = Dataset.from_pandas(long_df)

        # Apply the chat template here
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
        for batch_size in [16, 8, 4, 3, 2, 1]:
            print(f"Testing batch size {batch_size}")
            try:
                responses = defaultdict(list)
                # noinspection PyTypeChecker
                for static_id, persona, out in tqdm(
                        zip(
                            long_df["static_id"],
                            long_df["persona_col"],
                            pipe(
                                KeyDataset(dataset_hf, "prompt"),
                                batch_size=batch_size, return_full_text=False,
                                generation_config=gen_cfg
                            )
                        ), total=len(dataset_hf)
                ):
                    row = {
                        "static_id": static_id,
                        "persona": persona.replace("persona", "answer"),
                        "completion": out[0]["generated_text"]
                    }

                    responses["static_id"].append(row["static_id"])
                    responses["persona"].append(row["persona"])
                    responses["completion"].append(row["completion"])

                    row_df = pd.DataFrame([row])
                    if temp_file_path.exists():
                        temp_df = pd.read_parquet(temp_file_path)
                        temp_df = pd.concat([temp_df, row_df])
                    else:
                        temp_df = row_df
                    temp_df.to_parquet(temp_file_path)

                # pivot back from long to wide
                if temp_file_path.exists():
                    response_df = pd.read_parquet(temp_file_path)
                else:
                    response_df = pd.DataFrame(responses)
                response_df = response_df.pivot(index="static_id", columns="persona", values="completion")
                response_df = response_df.reset_index()

                # merge and save
                evaluator.save_data(dataset_id, response_df)
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect()
                torch.cuda.empty_cache()


if __name__ == "__main__":
    load_dotenv()
    login()

    parser = argparse.ArgumentParser(
        description="Inference script for HF open weight models.")
    parser.add_argument(
        "--dataset",
        help="The dataset to use for inference.",
        type=str,
        default=None)
    parser.add_argument(
        "--model",
        help="The model used for inference.",
        type=str,
        default="meta-llama/Llama-3.2-3B-Instruct")
    args = parser.parse_args()

    main(args.dataset, args.model)
