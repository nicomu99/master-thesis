from typing import Optional

from dataclasses import dataclass


@dataclass
class DatasetConfig:
    """Dataset specific configuration class.
    
    Attributes:
        dataset_id (str): String identifier of the dataset.
        huggingface_id (str): Huggingface hub identifier of the dataset.
        split (str): Split parameter controlling which portion should be loaded from huggingface.
        question_type (str): Type of the question. Can be "open_question", "mc_question" and "summarization".
        question_column (str): Column in the dataset containing the question of the sample.
        answer_column (str | None): If the dataset contains multiple-choice samples, this attribute specifies which
            column of the dataset contains the possible choices.
        task_column (str | None): If a dataset contains several tasks, this attribute denotes which column contains the
            identifier of each task. The name field in the TaskConfig class is used to identify rows that belong 
            together.
        load_name (str | None): Parameter load_name of the load_dataset function from the HF datasets library.
    """
    dataset_id: str
    huggingface_id: str
    split: str
    question_type: str
    question_column: str
    answer_column: Optional[str]
    task_column: Optional[str]
    load_name: Optional[str]

    def __post_init__(self):
        if self.task_column == "":
            self.task_column = None
        if self.answer_column == "":
            self.answer_column = None
        if self.load_name == "":
            self.load_name = None
