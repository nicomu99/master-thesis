from typing import List, Dict, Optional, Iterable, Union

import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset, Dataset

from .utils import DatasetConfig
from .utils import decode_dataclass
from .utils import QUESTION_COLUMN, GROUND_TRUTH_COLUMN, STATIC_ID_COLUMN
from .utils import logging


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class DatasetHandler:
    """Handles dataset loading and saving."""

    def __init__(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None,
    ) -> None:
        self.dataset_path = Path("data")

        self.dataset_ids: List[str] = []
        self.dataframes: Dict[str, pd.DataFrame] = {}
        self.dataset_configs: Dict[str, DatasetConfig] = {}
        self._load(include_datasets, exclude_datasets)

    def _load(
        self,
        include_datasets: Optional[Iterable[str]] = None,
        exclude_datasets: Optional[Iterable[str]] = None
    ):
        """Loads the dataset configurations and samples.

        The function downloads any datasets not present in the local file system and stores them to disk. If
        include_datasets is given, only datasets in this list will be loaded. If exclude datasets is given, these
        datasets will be ignored.

        Args:
            include_datasets (Optional[Iterable[str]], optional): Dataset ids that should be loaded. If not specified,
                all datasets will be loaded. Defaults to None.
            exclude_datasets (Optional[Iterable[str]], optional): Dataset ids that should be excluded from loading.
                If not specified, all will be loaded. Defaults to None.
        """

        log.info("Preparing datasets")
        self.dataset_path.mkdir(parents=True, exist_ok=True)

        with open("config_dataset.json", "r", encoding="utf-8") as f:
            raw_configs = json.load(f)

            dataset_ids = raw_configs.keys()
            self._select_dataset_ids(dataset_ids, include_datasets, exclude_datasets)

            for dataset_id in self.dataset_ids:
                raw_dataset_config = raw_configs[dataset_id]

                dataset_config = decode_dataclass({"dataset_id": dataset_id, **raw_dataset_config}, DatasetConfig)
                self.dataset_configs[dataset_id] = dataset_config
                self.dataframes[dataset_id] = self._download_dataset(
                    dataset_id,
                    dataset_config
                )

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
            dataset_ids (Iterable[str]): Dataset keys.
            include_datasets (Optional[Iterable[str]], optional): Dataset keys that should be kept in consideration.
                If not specified, all will be kept. Defaults to None.
            exclude_datasets (Optional[Iterable[str]], optional): Dataset keys that should not be kept. If not
                specified, no keys will be deleted. Defaults to None.
        """
        dataset_ids = list(dataset_ids)
        if include_datasets:
            dataset_ids = [d for d in dataset_ids if d in include_datasets]
        if exclude_datasets:
            dataset_ids = [d for d in dataset_ids if d not in exclude_datasets]
        self.dataset_ids = dataset_ids

    def _download_dataset(
            self,
            dataset_id: str,
            dataset_config: DatasetConfig
    ) -> pd.DataFrame:
        """Reads or downloads a dataset.

        If a dataset with the given dataset_id already exists in the local file system, the dataset
        is loaded directly. If it can not be found, it is first downloaded from the huggingface hub
        and saved locally.

        Args:
            dataset_id (str): The designated dataset id, taken from dataset_config.py.
                        The id is created manually.
            dataset_config (DatasetConfig): Configuration parameters for the dataset.

        Returns:
            Returns a pandas DataFrame containing the samples of the specified dataset.
        """
        log.info("Loading %s", dataset_id)

        dataset_file = Path(f"{self.dataset_path}/{dataset_id}.parquet")
        if not dataset_file.is_file():
            log.info("Dataset %s could not be found locally, commencing with download", dataset_file)

            # Download dataset
            hf_config = dataset_config.hf_config
            hf_id = hf_config.huggingface_id
            load_name = hf_config.load_name
            dataset_split = hf_config.split
            if load_name:
                dataset = load_dataset(hf_id, split=dataset_split, name=load_name)
            else:
                dataset = load_dataset(hf_id, split=dataset_split)

            assert isinstance(dataset, Dataset), \
                f"Error while loading {dataset_id}: Wrong dataset type {type(dataset)}, should be Dataset."
            dataframe = dataset.to_pandas()

            assert isinstance(dataframe, pd.DataFrame), \
                f"Error while loading {dataset_id}: Wrong dataset type {type(dataframe)}, should be pd.DataFrame."

            rename_columns = {dataset_config.question_column: QUESTION_COLUMN}
            if dataset_config.answer_column:
                rename_columns[dataset_config.answer_column] = GROUND_TRUTH_COLUMN
            dataframe.rename(columns=rename_columns, inplace=True)

            dataframe.insert(0, STATIC_ID_COLUMN, [f"{dataset_id}_{i}" for i in range(len(dataframe))])
            dataframe.to_parquet(dataset_file)
        else:
            dataframe = pd.read_parquet(dataset_file)

        return dataframe

    def write_dataframe(self, dataset_id: str):
        """Writes a dataframe to disk.

        Args:
            dataset_id (str): String identifier of the data frame.
        """
        dataframe_file = Path(f"{self.dataset_path}/{dataset_id}.parquet")
        self.dataframes[dataset_id].to_parquet(dataframe_file)

    def get_config(self, dataset_id: str) -> DatasetConfig:
        """Returns the dataset configuration of a dataset.

        Args:
            dataset_id (str): String identifier of the dataset.

        Returns:
            DatasetConfig: Dataset configuration class.
        """
        return self.dataset_configs[dataset_id]

    def get_task_dataframe(
        self,
        dataset_id: str,
        task_name: Optional[str]
    ) -> pd.DataFrame:
        """Returns a dataframe with task samples.

        Args:
            dataset_id (str): String identifier of the dataset.
            task_name (Optional[str]): Name of the task. If a dataset contains several tasks, this value is used to
                pick correct samples.

        Returns:
            pd.DataFrame: Dataframe with task samples.
        """
        dataframe = self.dataframes[dataset_id]
        category_column = self.dataset_configs[dataset_id].category_column
        if category_column:
            return dataframe[dataframe[category_column] == task_name]
        return dataframe

    def merge_and_write(
        self,
        dataset_id: str,
        subset_df: Union[pd.DataFrame, List[Dict]]
    ):
        """Merges a subset dataframe to the initial one by the static_id column.

        This function merges the subset into the parent dataframe. If any columns in the subset are not present in the
        parent, they will be created. The merged dataframe will be saved to disk.

        Args:
            dataset_id (str): String identifier of the parent dataframe.
            subset_df (pd.DataFrame | List[Dict]): Subset dataframe that will be merged to the parent.
        """
        dataframe = self.dataframes[dataset_id]
        if isinstance(subset_df, List):
            subset_df = pd.DataFrame(subset_df)

        dataframe.set_index(STATIC_ID_COLUMN, inplace=True)
        subset_df = subset_df.set_index(STATIC_ID_COLUMN)
        for col in subset_df.columns:
            if col not in dataframe.columns:
                dataframe[col] = None

        dataframe.update(subset_df)
        dataframe.reset_index(inplace=True)

        self.write_dataframe(dataset_id)
