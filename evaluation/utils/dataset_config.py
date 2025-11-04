from dataclasses import dataclass
from .enums import QuestionType


@dataclass
class HFConfig:
    """Huggingface metadata with information about datasets from the HF hub.

    Attributes:
        huggingface_id (str): HF hub identifier of the dataset.
        split (str): Split parameter controlling which portion should be loaded from HF hub.
        load_name (str | None): Parameter load_name of the load_dataset function from the HF datasets library.
    """
    huggingface_id: str
    split: str
    load_name: str | None = None


@dataclass
class DatasetConfig:
    """Dataset specific configuration class.

    Attributes:
        dataset_id (str): String identifier of the dataset.
        hf_config (HFConfig): Huggingface hub specific metadata associated with this dataset.
        question_type (QuestionType): Type of the question. Can be "open_question", "mc_question" and "summarization".
        question_column (str): Column in the dataset containing the question of the sample.
        category_column (str | None): If a dataset contains several tasks, this attribute denotes which column contains
            the identifier of each task. The name field in the TaskConfig class is used to identify rows that belong
            together.
        answer_column (str | None): If the dataset contains multiple-choice samples, this attribute specifies which
            column of the dataset contains the possible choices.
    """
    dataset_id: str
    hf_config: HFConfig
    question_type: QuestionType
    question_column: str
    category_column: str | None = None
    answer_column: str | None = None
