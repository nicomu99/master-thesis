"""Local HF inference."""
import gc
import argparse
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


def merge_and_write(
    dataframe: pd.DataFrame,
    subset_df: pd.DataFrame,
    file_name: str,
    id_column: str = "static_id"
):
    """Merges two dataframes.

    This function updates the entries in dataframe with the corresponding values in subset_df. Any
    entries in dataframe will be overwritten.

    Args:
        dataframe (pd.DataFrame): DataFrame object. The entries in this dataframe will be overwritten.
        subset_df (pd.DataFrame): Subset dataframe that will be used to update dataframe.
        file_name (str): File name of the written dataframe.
        id_column (str): Column to use as keys for the updates.
    """
    for col in subset_df.columns:
        if col not in dataframe.columns:
            dataframe[col] = None

    dataframe.set_index(id_column, inplace=True)
    subset_df = subset_df.set_index(id_column)

    # Any value in dataframe will be overwritten by subset_df
    dataframe.update(subset_df)
    dataframe.reset_index(inplace=True)

    # df_file = f"{drive_path}/{file_name}.parquet"
    df_file = f"data/{file_name}.parquet"
    dataframe.to_parquet(df_file)


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
    prompts_list = []
    for system, user in zip(batch["persona"], batch["prompt"]):
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        prompts_list.append(
            fn_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        )
    return {"prompt": prompts_list}


def main(
    dataset_id: str,
    model_string: str,
):
    """Main inference pipeline.

    Args:
        dataset_id (str): Dataset identifier of the dataset to use.
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

    dataset_path = model_string.split("/")[-1]
    dataset_path = dataset_path.replace(".", "_")
    dataset_path = dataset_path.lower()

    evaluator = Evaluator(dataset_path)
    persona_registry = evaluator.get_persona_registry()
    personas = persona_registry.get_names()

    # Load dataset
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
    dataset_hf = Dataset.from_pandas(long_df)

    # Apply the chat template here
    dataset_hf = dataset_hf.map(
        prepare_prompt,
        batched=True,
        fn_kwargs={"fn_tokenizer": tokenizer},
        num_proc=4
    )

    gen_cfg = GenerationConfig(
        max_new_tokens=1024,
        do_sample=False,
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
                responses["static_id"].append(static_id)
                responses["persona"].append(persona.replace("persona", "answer"))
                responses["completion"].append(out[0]["generated_text"])

            # pivot back from long to wide
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
