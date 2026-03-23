import json
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

from inference.evaluator import Evaluator


def prepare_jsonl():
    """Create JSONL files for the official IFBench evaluation script."""
    data_folder = Path("data")
    output_folder = Path("temp/ifbench_out")
    output_folder.mkdir(exist_ok=True)

    for model_folder in data_folder.iterdir():
        path_name = model_folder.name
        if "backup" in path_name:
            continue

        evaluator = Evaluator(dataset_path=path_name)
        _, df = evaluator.get_data("IFBench")

        persona_registry = evaluator.get_persona_registry()
        persona_cols = persona_registry.get_answer_columns()

        df["key"] = pd.to_numeric(df["key"])
        df = df.sort_values("key").reset_index(drop=True)
        for persona_col in persona_cols:
            if persona_col not in df.columns:
                print(f"Skipping missing persona column: {persona_col}")
                continue

            output_path = output_folder / f"{path_name}={persona_col}.jsonl"

            with output_path.open("w", encoding="utf-8") as f:
                for row in df.itertuples(index=False):
                    response = getattr(row, persona_col)

                    # skip missing responses if needed
                    if response is None:
                        continue

                    entry = {
                        "key": row.key,
                        "prompt": row.question.strip(),
                        "response": response,
                    }
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

            print(f"Wrote {output_path}")


if __name__ == "__main__":
    load_dotenv()
    prepare_jsonl()
