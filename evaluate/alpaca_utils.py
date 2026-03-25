import json
from pathlib import Path
from dotenv import load_dotenv

from inference.evaluator import Evaluator
from inference.utils import DATA_PATH


def prepare_jsonl(dataset_name: str = "alpaca"):
    """Create JSONL files for the official IFBench evaluation script."""
    data_folder = DATA_PATH
    if not data_folder.exists():
        raise RuntimeError("Data folder does not exist. Please make sure that inference has finished.")

    output_folder = Path("temp/alpaca_out")
    output_folder.mkdir(exist_ok=True)

    for model_folder in data_folder.iterdir():
        model_folder_name = model_folder.name
        if not model_folder.is_dir():
            continue

        evaluator = Evaluator(dataset_path=model_folder_name)
        _, df = evaluator.get_data(dataset_name)

        persona_registry = evaluator.get_persona_registry()
        persona_cols = persona_registry.get_answer_columns()

        for persona_col in persona_cols:
            if persona_col not in df.columns:
                print(f"Skipping missing persona column: {persona_col}")
                continue

            records = []
            for _, row in df.iterrows():
                output_text = row[persona_col]
                if output_text is None:
                    continue

                records.append(
                    {
                        "instruction": row["question"],
                        "output": output_text,
                    }
                )

            out_path = output_folder / f"{model_folder_name}_{persona_col}.json"
            with out_path.open("w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            print(f"Wrote {out_path}")


if __name__ == "__main__":
    load_dotenv()
    prepare_jsonl()
