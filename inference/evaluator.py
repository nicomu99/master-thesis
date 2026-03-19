from typing import Iterable

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from .dataset_handler import DatasetHandler
from .communication_handler import CommunicationHandler
from .persona_registry import PersonaRegistry
from .utils import columns_full, load_dataclass_dict, save_dataclass_dict, load_task_config, QuestionType
from .utils import TaskInfo, BatchInfo, TRANSLATION_JUDGE_TEMPLATE, STATIC_ID_COLUMN, QUESTION_COLUMN

from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Evaluator:
    """Main inference class."""

    def __init__(
        self,
        dataset_path: str | None = None,
        openai_model: str | None = None,
        include_datasets: Iterable[str] | None = None,
        exclude_datasets: Iterable[str] | None = None,
    ):
        self.dataset_path = Path("data")
        if dataset_path is not None:
            self.dataset_path /= Path(dataset_path)
        self.dataset_handler = DatasetHandler(
            self.dataset_path,
            include_datasets,
            exclude_datasets
        )

        self.communication_handler = CommunicationHandler(
            self.dataset_path, openai_model)

        self.persona_registry = PersonaRegistry()

        self.config_path = "config_task.json"

        self.task_info_file = self.dataset_path / "task_info.json"
        self.task_infos: dict[str, TaskInfo] = {}
        self._load()

    def _load(self):
        task_configs = load_task_config(self.config_path)
        dataset_configs = self.dataset_handler.dataset_configs

        if self.task_info_file.exists():
            task_infos = load_dataclass_dict(
                self.task_info_file, TaskInfo,
                key_field=TaskInfo.get_key_field())

            self.task_infos = {
                task_id: task_infos.get(task_id, info)
                for task_id, info in task_configs.items()
                if info.dataset_id in dataset_configs}
        else:
            self.task_infos = task_configs

    def _save(self):
        save_dataclass_dict(
            self.task_info_file, self.task_infos,
            key_field=TaskInfo.get_key_field())

    def get_tasks(self) -> list[str]:
        """Returns a list with all task identifiers.

        Returns:
            list[str]: A list of string identifiers of tasks.
        """
        return list(self.task_infos)

    def get_judgment_tasks(self) -> list[str]:
        """Returns a list with all task identifiers of tasks that require judgments.

        Returns:
            list[str]: A list of string identifiers of tasks.
        """
        task_ids = [
            tid for tid, task_info in self.task_infos.items()
            if task_info.need_judgment
        ]
        return task_ids

    def get_personas_pending_tasks(self) -> list[str]:
        """Returns a list with all tasks that have missing personas.

        Returns:
            list[str]: A list of string identifiers of tasks that have missing personas.
        """
        return [k for k, v in self.task_infos.items() if v.is_personas_pending()]

    def get_answer_pending_tasks(self) -> list[str]:
        """Returns a list with all tasks that have missing answers.

        Returns:
            list[str]: A list of string identifiers of tasks that have missing answers.
        """
        return [k for k, v in self.task_infos.items() if v.is_answers_pending()]

    def get_judgment_pending_tasks(self) -> list[str]:
        """Returns a list with all tasks that have missing judgment inference.

        Returns:
            list[str]: A list of string identifiers of tasks that have missing judgment inference.
        """
        return [k for k, v in self.task_infos.items() if v.is_judgment_pending()]

    def reset_all_tasks(self):
        """Reset all tasks statuses to the default value."""
        for task_info in self.task_infos.values():
            task_info.reset()
        self._save()

    def reset_tasks_to_answers(self, task_ids: list[str]):
        """Resets the specified tasks to the answer pending stage.

        Args:
            task_ids (list[str]): List of tasks to update.
        """
        answer_columns = self.persona_registry.get_answer_columns()
        for tid in task_ids:
            tinfo = self.task_infos[tid]
            dataset_id = tinfo.dataset_id
            self.dataset_handler.clear_columns(dataset_id, tinfo.category_name, answer_columns)
            tinfo.reset_to_answers()
        self._save()

    def reset_tasks_to_judgments(self, task_ids: list[str]):
        """Resets the specified tasks to the judgment pending stage.

        Args:
            task_ids (list[str]): List of tasks to update.
        """
        judgment_columns = self.persona_registry.get_judgment_columns()
        for tid in task_ids:
            tinfo = self.task_infos[tid]
            if not tinfo.need_judgment:
                continue
            dataset_id = tinfo.dataset_id
            self.dataset_handler.clear_columns(dataset_id, tinfo.category_name, judgment_columns)
            tinfo.reset_to_judgments()
        self._save()

    def _static_persona_helper(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
        persona_templates: dict[str, str],
    ):
        if columns_full(task_df, list(persona_templates)):
            return task_df
        persona = self.persona_registry.get_base_persona_string()

        personas = {}
        for persona_name, prompt_template in persona_templates.items():
            client_kwargs = {
                "task_type": task_info.field, "persona_string": persona}
            persona = self.communication_handler.get_api_response(
                prompt_template, "openai", **client_kwargs)

            personas[persona_name] = persona
        task_df = task_df.assign(**personas)
        return task_df

    def _generate_static_personas(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame
    ) -> pd.DataFrame:
        persona_templates = self.persona_registry.get_static_templates()
        return self._static_persona_helper(task_info, task_df, persona_templates)

    def _generate_teacher_personas(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame,
    ) -> pd.DataFrame:
        persona_templates = self.persona_registry.get_teacher_static_templates()
        return self._static_persona_helper(task_info, task_df, persona_templates)

    def _generate_empty_personas(
        self,
        task_df: pd.DataFrame
    ) -> pd.DataFrame:
        if columns_full(task_df, self.persona_registry.get_empty_names()):
            return task_df
        empty_personas = self.persona_registry.get_empty_templates()
        task_df = task_df.assign(**empty_personas)
        return task_df

    def _generate_dynamic_personas(
        self,
        task_info: TaskInfo,
        task_df: pd.DataFrame
    ) -> int:
        if columns_full(task_df, self.persona_registry.get_dynamic_names()):
            return 0
        print("Generating dynamic personas...")
        request_count = self.communication_handler.send_persona_batch(
            task_info, task_df, self.persona_registry.get_dynamic_configs())
        return request_count

    def generate_personas(
        self,
        task_id: str
    ) -> int:
        """Generates static personas and sends requests for dynamic personas.

        Args:
            task_id (str): String identifier of the task.

        Returns:
            int: Number of requests sent to the LLM API.
        """
        task_info = self.task_infos[task_id]
        task_df = self.dataset_handler.get_task_df_from_info(task_info)

        task_df = self._generate_empty_personas(task_df)
        task_df = self._generate_static_personas(task_info, task_df)
        task_df = self._generate_teacher_personas(task_info, task_df)
        self.dataset_handler.merge_and_write(task_info.dataset_id, task_df)

        result = self._generate_dynamic_personas(task_info, task_df)
        if not result:
            return 0

        task_info.update_status(skip=result == 0)
        self._save()
        return result

    def generate_all_static_personas(self):
        """Creates static personas on task level for all tasks."""
        for tid, task_info in self.task_iterator(desc="Processing"):
            task_df = self.dataset_handler.get_task_df_from_info(task_info)
            task_df = self._generate_empty_personas(task_df)
            task_df = self._generate_static_personas(task_info, task_df)
            task_df = self._generate_teacher_personas(task_info, task_df)
            self.dataset_handler.merge_and_write(tid, task_df)
        self._save()

    def send_answer_requests(
        self,
        task_id: str
    ) -> int:
        """Sends answer completion requests to the LLM API.

        Args:
            task_id (str): String identifier of the task.

        Returns:
            int: Number of requests sent to the LLM API.
        """
        task_info = self.task_infos[task_id]
        dataset_config = self.dataset_handler.get_config(task_info.dataset_id)
        persona_configs = self.persona_registry.get_configs()
        task_df = self.dataset_handler.get_task_df_from_info(task_info)

        result = self.communication_handler.send_answer_batch(
            task_info, task_df, dataset_config.question_type, persona_configs)
        if result is False:
            return 0

        task_info.update_status(skip=result == 0)
        self._save()
        return result

    @staticmethod
    def _remove_think_block(text: str) -> str:
        return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

    def _generate_judgments_sync(
        self,
        task_id: str,
    ):
        """Generate judgments synchronously one-by-one.

        Args:
            task_id (str): String identifier of the task.
        """
        task_info = self.task_infos[task_id]
        task_df = self.dataset_handler.get_task_df_from_info(task_info)
        persona_configs = self.persona_registry.get_configs()
        reference_config = self.persona_registry.get_reference_config()
        prompt_template = TRANSLATION_JUDGE_TEMPLATE

        # Iterate over each row in the dataframe and send judgments for each persona
        for idx, row in tqdm(task_df.iterrows()):
            row_dict = row.to_dict()

            for config in persona_configs:
                judgment_column = config.judgment_column

                if (
                    config.name == reference_config.name or
                    judgment_column in row_dict and row_dict[judgment_column] is not None
                ):
                    continue

                static_id = row_dict[STATIC_ID_COLUMN]
                sample_number = int(static_id.split("_")[-1])
                translations = [
                    self._remove_think_block(row_dict[reference_config.answer_column]),
                    self._remove_think_block(row_dict[config.answer_column])
                ]
                if sample_number % 2 == 0:
                    translations.reverse()

                client_kwargs = {
                    "reference": row_dict[QUESTION_COLUMN],
                    "translation_1": translations[0],
                    "translation_2": translations[1]
                }
                judgment = self.communication_handler.get_api_response(
                    prompt_template, "genai", **client_kwargs)
                task_df.loc[idx, judgment_column] = judgment
                self.dataset_handler.merge_and_write(task_info.dataset_id, task_df)
        task_info.update_status(skip=True)
        self._save()

    def generate_all_judgments(self):
        """Generate all judgments"""
        judgment_tasks = self.get_judgment_pending_tasks()
        for tid in judgment_tasks:
            print(f"Processing {tid}")
            self._generate_judgments_sync(tid)

    def send_judgment_requests(
        self,
        task_id: str
    ) -> int:
        """Sends judgment inference requests to the LLM API.

        Args:
            task_id (str): String identifier of the task.

        Returns:
            int: Number of requests sent to the LLM API.
        """
        task_info = self.task_infos[task_id]
        task_df = self.dataset_handler.get_task_df_from_info(task_info)
        persona_configs = self.persona_registry.get_configs()
        reference_config = self.persona_registry.get_reference_config()

        result = self.communication_handler.send_judgment_batch(
            task_info, task_df, persona_configs, reference_config, "genai")
        if result is False:
            return 0

        task_info.update_status(skip=result == 0)
        self._save()
        return result

    def check_batch_statuses(self) -> dict[str, BatchInfo]:
        """Prints the batch statuses.

        Returns:
            dict[str, BatchInfo]: Dictionary with batch information of active batches.
        """
        active_batches = self.communication_handler.check_batch_statuses()
        for bid, batch_info in active_batches.items():
            if batch_info.is_error() and bid in self.task_infos:
                self.task_infos[bid].decrement_status()
        return active_batches

    def fetch_batch_responses(self):
        """Fetches batch responses and saves them to disk."""
        retrieved_batches = self.communication_handler.download_batch_files(self.task_infos.keys())

        for batch_info in retrieved_batches:
            tid = batch_info.task_id
            task_info = self.task_infos[tid]
            dataset_id = task_info.dataset_id
            if tid not in self.task_infos:
                log.warning("Batch for inactive task %s found. Skipping.", tid)
                continue

            response = self.communication_handler.read_batch_file(batch_info)
            if isinstance(response, pd.DataFrame):
                self.dataset_handler.merge_and_write(dataset_id, response)
                batch_info.finish_batch()

            if response is not False:
                task_info.update_status(increment=batch_info.is_retrieved())
        self.communication_handler.save()
        self._save()

    def task_iterator(
        self,
        desc: str,
    ) -> Iterable[tuple[str, TaskInfo]]:
        """Iterates all tasks with a tqdm decorator.

        Args:
            desc (str): Description used by tqdm.

        Returns:
            Iterable[tuple[str, TaskInfo]]: Tuples with the task id and task info.
        """
        task_iterator = tqdm(self.task_infos.items(), desc=desc)
        with logging_redirect_tqdm(loggers=[log]):
            for task_id, task_info in task_iterator:
                task_iterator.set_description(f"{desc} {task_id}")
                yield task_id, task_info

    def _get_dataset_tasks(self, dataset_id: str) -> list[TaskInfo]:
        return [t for t in self.task_infos.values() if t.dataset_id == dataset_id]

    def get_dataset_names(self) -> set[str]:
        """Returns the unique dataset names.

        Returns:
            set[str]: A set with dataset names.
        """
        return {t.dataset_id for t in self.task_infos.values()}

    def quit(self):
        """Ends the program."""
        self._save()
        self.communication_handler.save()

    def get_data(
        self,
        dataset_id: str
    ) -> tuple[list[str], pd.DataFrame]:
        """Returns all data instances of active tasks for a given dataset.

        Args:
            dataset_id (str): Dataset identifier.

        Returns:
            tuple[list[str], pd.DataFrame]: _description_
        """
        dataset_tasks = self._get_dataset_tasks(dataset_id)
        task_ids = [t.task_id for t in dataset_tasks]
        task_names = [t.category_name for t in dataset_tasks if t.category_name is not None]
        return task_ids, self.dataset_handler.get_data(dataset_id, task_names)

    def save_data(
        self,
        dataset_id: str,
        subset_df: pd.DataFrame
    ):
        """Save the subset dataframe.

        Args:
            dataset_id (str): String identifier of the dataset.
            subset_df (pd.DataFrame): Subset dataframe that will be merged into the dataset.
        """
        self.dataset_handler.merge_and_write(dataset_id, subset_df)

    def get_question_type(
        self,
        dataset_id: str
    ) -> QuestionType:
        """Returns the question type present in a dataset.

        Args:
            dataset_id (str): Dataset identifier.

        Returns:
            QuestionType: Question type associated with a dataset.
        """
        dataset_config = self.dataset_handler.get_config(dataset_id)
        return dataset_config.question_type

    def get_persona_registry(self) -> PersonaRegistry:
        """Returns the persona registry.

        Returns:
            PersonaRegistry: The registry containing all available personas.
        """
        return self.persona_registry
