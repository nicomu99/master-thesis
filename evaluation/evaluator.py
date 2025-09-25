from typing import Dict, Optional, Iterable, List, Iterator

import json
from pathlib import Path
from collections import defaultdict

import pandas as pd
from openai import OpenAI
from datasets import load_dataset, disable_progress_bar, Dataset
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from dataset_config import DatasetConfig
from task_config import TaskConfig
from prompt_templates import *

from log_conf import get_logger

log = get_logger(__name__)

disable_progress_bar()

class Evaluator:
    def __init__(self):
        self.config_path = Path("config.json")
        self.dataset_path = Path("data")

        self.dataset_ids: List[str] = []
        self.datasets: Dict[str, pd.DataFrame] = dict()
        self.dataset_configs: Dict[str, DatasetConfig] = dict()
        self.task_configs: Dict[str, List[TaskConfig]] = defaultdict(list)

        self.persona_types = [
            "base_persona", "static_short_persona", "static_long_persona",
            "dynamic_short_persona", "dynamic_long_persona"
        ]
        self.persona_prompt_templates = {
            "static_short_persona":     STATIC_SHORT_TEMPLATE,
            "static_long_persona":      STATIC_LONG_TEMPLATE,
            "dynamic_short_persona":    DYNAMIC_SHORT_TEMPLATE,
            "dynamic_long_persona":     DYNAMIC_LONG_TEMPLATE
        }

        self.client = OpenAI()

    @staticmethod
    def _select_dataset_names(
            dataset_names: Iterable[str],
            include_datasets: Optional[Iterable[str]] = None,
            exclude_datasets: Optional[Iterable[str]] = None
    ) -> List[str]:
        """Filters dataset_names.

        Filters out any datasets not present in include_datasets, if the parameter is not `None`. If include_datasets
        is not passed, all datasets remain in consideration. If any datasets are present in exclude_datasets, they
        also will be deleted from consideration.

        Args:
            dataset_names: Dataset keys.
            include_datasets: Dataset keys that should be kept in consideration. If `None`, all will be kept.
            exclude_datasets: Dataset keys that should not be kept. If `None`, no keys will be deleted.

        Returns:
            List[str]: A refined list with dataset keys.

        """
        dataset_names = list(dataset_names)
        if include_datasets:
            dataset_names = [d for d in dataset_names if d in include_datasets]
        if exclude_datasets:
            dataset_names = [d for d in dataset_names if d not in exclude_datasets]
        return dataset_names

    def _read_or_download_dataset(
            self,
            dataset_id: str,
            huggingface_id: str,
            dataset_split: str,
            dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Reads or downloads a dataset.

        If a dataset with the given dataset_id already exists in the local file system, the dataset is loaded directly.
        If it can not be found, it is first downloaded from the huggingface hub and saved locally.

        Args:
            dataset_id: The designated dataset id, taken from dataset_config.py. The id is created manually.
            huggingface_id: The corresponding repository on the huggingface hub.
            dataset_split: Data splits that should be downloaded from the hub.
            dataset_name: Data set name to be downloaded.

        Returns:
            Returns a pandas DataFrame containing the samples of the specified dataset.

        """
        dataset_file = Path(f"{self.dataset_path}/{dataset_id}.parquet")

        if not dataset_file.is_file():
            log.info(f"Dataset {dataset_id} could not be found locally, commencing with download")
            # Download dataset
            if dataset_name:
                dataset = load_dataset(huggingface_id, split=dataset_split, name=dataset_name)
            else:
                dataset = load_dataset(huggingface_id, split=dataset_split)
            
            assert isinstance(dataset, Dataset), f"Error while loading {dataset_id}: Wrong dataset type {type(dataset)}, should be Dataset."
            dataset.to_parquet(dataset_file)

        df = pd.read_parquet(dataset_file)
        return df

    def _iter_dataset_ids(self, desc: str) -> Iterator[str]:
        """Helper function for iterating the dataset.

        Iterates and yields over the dataset ids. The function makes sure the logging output does not interfere with
        tqdm and vice versa.

        Args:
            desc: Description to be shown in the tqdm progress bar.

        Yields:
            str: The next dataset ID from the dataset_ids list.

        """
        dataset_iterator = tqdm(self.dataset_ids, desc=desc)
        with logging_redirect_tqdm(loggers=[log]):
            for dataset_id in dataset_iterator:
                dataset_iterator.set_description(f"{desc} {dataset_id}")
                yield dataset_id

    def load_and_save_datasets(
            self,
            include_datasets: Optional[Iterable[str]] = None,
            exclude_datasets: Optional[Iterable[str]] = None
    ):
        """Loads the dataset configurations and samples into memory.

        The function downloads any datasets not present in the local file system and stores them to disk. If
        include_datasets is given, only datasets in this list will be loaded. If exclude datasets is given, these
        datasets will be ignored.

        Args:
            include_datasets: Dataset ids that should be loaded. If not specified, all datasets will be loaded.
            exclude_datasets: Dataset ids that should be excluded from loading. If not specified, all will be loaded.
        """

        log.info("Starting dataset preparation")

        with open("config.json", "r") as f:
            configs = json.load(f)
        raw_configs = configs.get("datasets", {})

        dataset_ids = raw_configs.keys()
        self.dataset_ids = self._select_dataset_names(dataset_ids, include_datasets, exclude_datasets)

        for dataset_id in self._iter_dataset_ids("Loading dataset"):
            raw_dataset_config = raw_configs[dataset_id]
            config = DatasetConfig(dataset_id=dataset_id, **raw_dataset_config)
            self.dataset_configs[dataset_id] = config
            self.datasets[dataset_id] = self._read_or_download_dataset(
                dataset_id,
                config.huggingface_id,
                config.split,
                config.load_name
            )

            for task in raw_dataset_config["tasks"]:
                task_config = TaskConfig(dataset_id=dataset_id, **task)
                self.task_configs[dataset_id].append(task_config)

    @staticmethod
    def _add_empty_column(
            dataset: pd.DataFrame,
            column_name: str
    ) -> pd.DataFrame:
        """Helper function for adding columns to pandas dataframes

        Adds a new column column_name to the dataframe, if it does not exist yet.

        Args:
            dataset: The dataset for which a new column should be added.
            column_name: The name of the newly created column

        Returns:
            pd.DataFrame: Dataframe with newly created column.
        """

        if column_name not in dataset.columns:
            dataset[column_name] = None
        return dataset

    def get_openai_api_response(
            self,
            template: str,
            prompt_cache_key: str,
            **kwargs
    ) -> str:
        """Helper function for getting output from the OpenAI API.

        The function fills the template with the given keyword arguments and sends the prompt to the
        OpenAI API.

        Args:
            template: A string template with placeholders.
            prompt_cache_key: A key to be used for caching API calls.
            **kwargs: Keyword arguments to be inserted into ``template``. Must match the placeholders.

        Returns:
            str: The generated text returned by the API.
        """
        prompt = template.format(**kwargs)
        response = self.client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            prompt_cache_key=prompt_cache_key
        )

        return response.output_text

    def create_static_personas(
            self,
            task_config: TaskConfig,
            dataset: pd.DataFrame,
            dataset_config: DatasetConfig
    ) -> pd.DataFrame:
        """Creates static personas on task level.

        The function lets an LLM client generate personas that are shared by all samples of the same task in three
        different, increasing length formats. The shortest length is defined manually. The other two formats are
        generated iteratively, using the previous persona as a prefix.

        Args:
            task_config: A task configuration with information about the task.
            dataset: A DataFrame holding samples for which the personas should be generated.
            dataset_config: A DataConfig object holding information about ``dataset``.

        Returns:
            pd.DataFrame: Update dataframe containing the generated personas of the specified task.
        """
        task_name = task_config.name
        task_column = dataset_config.task_column
        persona = task_config.static_persona

        # Task name "general" denotes tasks where no subcategory is chosen from the dataset
        row_mask = slice(None) if task_name == "general" else dataset[task_column] == task_name

        if dataset.loc[row_mask, "base_persona"].isnull().any():
            # Only insert if some rows have an empty base persona
            log.debug(f"Inserting base personas")
            dataset.loc[row_mask, "base_persona"] = persona

        for static_persona_type in ["static_short_persona", "static_long_persona"]:
            if not dataset.loc[row_mask, static_persona_type].isnull().any():
                # Keep value in case if short persona gets skipped
                log.debug(f"Personas already exist {static_persona_type}")
                persona = dataset.loc[row_mask, static_persona_type].unique()[0]
                continue

            log.debug(f"Creating persona {static_persona_type}")
            prompt_template = self.persona_prompt_templates[static_persona_type]
            client_kwargs = {
                "task_type": task_config.name,
                "persona_string": persona
            }
            persona = self.get_openai_api_response(prompt_template, static_persona_type, **client_kwargs)
            dataset.loc[row_mask, static_persona_type] = persona
        return dataset

    def _write_dataframe(
            self,
            dataset_id: str,
            dataset: pd.DataFrame
    ):
        """Writes the dataframe to the data folder.

        Args:
            dataset_id: A string identifier that will be used as the name of the file.
            dataset: A dataframe object that should be written to a file.
        """

        dataset_file = Path(f"{self.dataset_path}/{dataset_id}.parquet")
        dataset.to_parquet(dataset_file)

    def generate_personas(self):
        """Helper script for generating personas.

        Iterates all task configurations and creates personas for them.
        """
        # Assume data is loaded already
        log.info("Generating static personas")

        # First, for each dataset category, create the static personas
        for dataset_id in self._iter_dataset_ids("Processing"):
            log.debug(f"Generating personas for dataset {dataset_id}")

            dataset = self.datasets[dataset_id]
            dataset_config = self.dataset_configs[dataset_id]

            # Add new column, if it does not exist yet
            for persona_type in self.persona_types:
                dataset = self._add_empty_column(dataset, persona_type)

            for task_config in self.task_configs[dataset_id]:
                dataset = self.create_static_personas(task_config, dataset, dataset_config)
                self._write_dataframe(dataset_id, dataset)
                    # Create the task specific personas
        log.info("Finished generating static personas")
        # Then, create personas specific to the sample
        # if data_config.task_column != "":
        #     # Some datasets do not have subtasks, in which case we do not have to filter
        #     task_data = task_data.filter(lambda x: x[data_config] == task.name)
        # Save everything to disk

if __name__ == "__main__":
    evaluator = Evaluator()
    evaluator.load_and_save_datasets()
    evaluator.generate_personas()


