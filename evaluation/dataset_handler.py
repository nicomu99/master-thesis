from __future__ import annotations

from typing import List, Dict, Optional, Iterable, Literal, overload, Generator

import json
from pathlib import Path
from collections import defaultdict
from dataclasses import fields

import pandas as pd
from datasets import disable_progress_bar, load_dataset, Dataset
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from evaluation.task_config import TaskConfig
from evaluation.dataset_config import DatasetConfig
from evaluation.log_conf import get_logger, logging

log = get_logger(__name__)
disable_progress_bar()


class DatasetHandler:
    """Handles dataset loading and saving.
    """
    def __init__(self) -> None:
        self.dataset_path = Path("data")

        self.dataset_ids: List[str] = []
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataset_configs: Dict[str, DatasetConfig] = {}

    @overload
    def iter_datasets(self, desc: str, logger: logging.Logger, kind: Literal["ids"]) -> Generator[str]: ...

    @overload
    def iter_datasets(
            self,
            desc: str,
            logger: logging.Logger,
            kind: Literal["items"]
    ) -> Generator[tuple[str, DatasetConfig, pd.DataFrame]]: ...

    def iter_datasets(
        self,
        desc: str,
        logger: logging.Logger,
        kind: Literal["ids", "items"] = "ids"
    ) -> Generator[tuple[str, DatasetConfig, DataFrame] | str, None]:

        """Helper function for iterating the dataset.

        Iterates and yields over the dataset ids. The function makes sure the logging output does not interfere with
        tqdm and vice versa.

        Args:
            desc: Description to be shown in the tqdm progress bar.
            logger: The logger to use the redirect of the tqdm bar on.
            kind: Values to return. If 'items', dataset ids, data configuration and dataframes will be returned. Else
                only the dataset ids.

        Yields:
            str: The next dataset ID from the dataset_ids list.

        """
        dataset_iterator = tqdm(self.dataset_ids, desc=desc)
        with logging_redirect_tqdm(loggers=[logger]):
            for dataset_id in dataset_iterator:
                dataset_iterator.set_description(f"{desc} {dataset_id}")
                if kind == "items":
                    yield dataset_id, self.dataset_configs[dataset_id], self.dataframes[dataset_id]
                else:
                    yield dataset_id

    def load(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None
    ) -> Dict[str, List[TaskConfig]]:
        """Loads the task configurations, dataset configurations and samples into memory.

        The function downloads any datasets not present in the local file system and stores them to disk. If
        include_datasets is given, only datasets in this list will be loaded. If exclude datasets is given, these
        datasets will be ignored.

        Args:
            include_datasets: Dataset ids that should be loaded. If not specified, all datasets will be loaded.
            exclude_datasets: Dataset ids that should be excluded from loading. If not specified, all will be loaded.

        Returns:
            dict: Dictionary with dataset id as keys and list of task configuration as values
        """

        log.info("Starting dataset preparation")
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        with open("config.json", "r", encoding="utf-8") as f:
            configs = json.load(f)
        raw_configs = configs.get("datasets", {})

        dataset_ids = raw_configs.keys()
        self._select_dataset_ids(dataset_ids, include_datasets, exclude_datasets)

        task_configs = defaultdict(list)
        for dataset_id in self.iter_datasets("Loading dataset", log, kind="ids"):
            raw_dataset_config = raw_configs[dataset_id]

            dataset_config = {
                k: v for k, v in raw_dataset_config.items() if k in {f.name for f in fields(DatasetConfig)}
            }
            config = DatasetConfig(dataset_id=dataset_id, **dataset_config)
            self.dataset_configs[dataset_id] = config
            self.dataframes[dataset_id] = self._read_or_download_dataset(
                dataset_id,
                config.huggingface_id,
                config.split,
                config.load_name
            )

            for task in raw_dataset_config["tasks"]:
                task_id = f"{config.dataset_id}_{task["name"]}"
                task_config = TaskConfig(task_id=task_id, dataset_id=dataset_id, **task)
                task_configs[dataset_id].append(task_config)

        return task_configs

    def write_dataframe(
        self,
        dataset_id: str
    ):
        """Writes a dataframe to disk.

        Args:
            dataset_id (str): String identifier of the data frame.
        """
        dataframe_file = Path(f"{self.dataset_path}/{dataset_id}.parquet")
        self.dataframes[dataset_id].to_parquet(dataframe_file)

    def _select_dataset_ids(
        self,
        dataset_ids: Iterable[str],
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ):
        """Filters dataset_names.

        Filters out any datasets not present in include_datasets, if the parameter is not `None`.
        If include_datasets is not passed, all datasets remain in consideration. If any datasets
        are present in exclude_datasets, they also will be deleted from consideration.

        Args:
            dataset_ids: Dataset keys.
            include_datasets: Dataset keys that should be kept in consideration. If `None`,
            all will be kept.
            exclude_datasets: Dataset keys that should not be kept. If `None`, no keys will be
                                deleted.

        Returns:
            List[str]: A refined list with dataset keys.

        """
        dataset_ids = list(dataset_ids)
        if include_datasets:
            dataset_ids = [d for d in dataset_ids if d in include_datasets]
        if exclude_datasets:
            dataset_ids = [d for d in dataset_ids if d not in exclude_datasets]
        self.dataset_ids = dataset_ids

    def _read_or_download_dataset(
            self,
            dataset_id: str,
            huggingface_id: str,
            dataset_split: str,
            dataset_name: Optional[str] = None
    ) -> pd.DataFrame:
        """Reads or downloads a dataset.

        If a dataset with the given dataset_id already exists in the local file system, the dataset
        is loaded directly. If it can not be found, it is first downloaded from the huggingface hub
        and saved locally.

        Args:
            dataset_id: The designated dataset id, taken from dataset_config.py.
                        The id is created manually.
            huggingface_id: The corresponding repository on the huggingface hub.
            dataset_split: Data splits that should be downloaded from the hub.
            dataset_name: Data set name to be downloaded.

        Returns:
            Returns a pandas DataFrame containing the samples of the specified dataset.

        """
        dataset_file = Path(f"{self.dataset_path}/{dataset_id}.parquet")

        if not dataset_file.is_file():
            log.info("Dataset %s could not be found locally, commencing with download", dataset_file)
            # Download dataset
            if dataset_name:
                dataset = load_dataset(huggingface_id, split=dataset_split, name=dataset_name)
            else:
                dataset = load_dataset(huggingface_id, split=dataset_split)

            assert isinstance(dataset, Dataset), \
                f"Error while loading {dataset_id}: Wrong dataset type {type(dataset)}, should be Dataset."
            dataframe = dataset.to_pandas()

            assert isinstance(dataframe, pd.DataFrame), \
                f"Error while loading {dataset_id}: Wrong dataset type {type(dataframe)}, should be pd.DataFrame."

            dataframe.insert(0, "static_id", [f"{dataset_id}_{i}" for i in range(len(dataframe))])
            dataframe.to_parquet(dataset_file)

        df = pd.read_parquet(dataset_file)
        return df
