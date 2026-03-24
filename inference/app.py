import argparse
import subprocess
import platform
from dotenv import load_dotenv

from datasets import disable_progress_bar, disable_progress_bars

from .evaluator import Evaluator
from .utils import TEMP_PATH, DATA_PATH
from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class App:
    """Small CLI application to control the program."""

    def __init__(self, dataset_path: str | None, openai_model: str):
        self.run = True
        self.evaluator = Evaluator(dataset_path, openai_model)

        self.commands = {
            "q": ("Quit program", self.quit_program),
            "p": ("Generate personas", self.generate_personas),
            "ps": ("Generate static personas for all tasks", self.generate_static_personas),
            "t": ("Send task requests", self.generate_answers),
            "j": ("Send judgment requests", self.generate_judgment_evaluation),
            "js": ("Generate judgments synchronously for all tasks", self.generate_judgments),
            "s": ("Check batch statuses", self.check_batch_statuses),
            "f": ("Fetch batch responses", self.evaluator.fetch_batch_responses),
            "c": ("Check task statuses", self.check_task_statuses),
            "r": ("Reset tasks", self.reset_tasks),
            "ra": ("Reset tasks to answers pending", self.reset_to_answers),
            "rj": ("Reset tasks to judgment pending", self.reset_to_judgments),
            "clear": ("Clears the CLI", self.clear_cli),
            "h": ("Show this help", self.show_help),
        }

    def reset_tasks(self):
        """Reset all tasks statuses to the default value."""
        self.evaluator.reset_all_tasks()

    def reset_to_answers(self):
        """Lets the user pick which tasks to reset to the answer pending status."""
        task_ids = self.evaluator.get_tasks()
        chosen_task_ids = self._print_and_evaluate_input(task_ids)
        self.evaluator.reset_tasks_to_answers(chosen_task_ids)

    def reset_to_judgments(self):
        """Lets the user pick which tasks to reset to the judgment pending status."""
        task_ids = self.evaluator.get_judgment_tasks()
        chosen_task_ids = self._print_and_evaluate_input(task_ids)
        self.evaluator.reset_tasks_to_judgments(chosen_task_ids)

    def main(self) -> None:
        """Main loop that listens for user input."""
        self.show_help()
        while self.run:
            user_input = input("\nEnter next command (type h for help): ").strip()

            if user_input in self.commands:
                _, func = self.commands[user_input]
                func()
            else:
                print("Command unknown. Type \"h\" for help.")

    @staticmethod
    def _print_and_evaluate_input(task_ids: list[str]) -> list[str]:
        user_input_map = {
            f"{idx + 1}": tid for idx, tid in enumerate(task_ids)}

        if len(user_input_map) < 1:
            print("No tasks to process.")
            return []

        for k, v in user_input_map.items():
            print(f"     ({k}) {v}")

        print("Several tasks can be chosen by writing the numbers separated with a whitespace.")
        user_input = input("Choose task: ")

        valid_tasks = []
        for task in user_input.split(" "):
            if task not in user_input_map:
                print(f"Task selector {task} unknown. Skipping.")
                continue
            valid_tasks.append(user_input_map[task])

        return valid_tasks

    def generate_personas(self) -> None:
        """Lets the user pick which task to generate personas for."""
        task_ids = self.evaluator.get_personas_pending_tasks()
        chosen_task_ids = self._print_and_evaluate_input(task_ids)

        total_request_count = 0
        for tid in chosen_task_ids:
            print(f"Generating personas for {tid}")
            request_count = self.evaluator.generate_personas(tid)
            total_request_count += request_count
        print(f"{total_request_count} requests sent.")

    def generate_static_personas(self) -> None:
        """Generate static personas for all tasks."""
        print("Generating static personas for all tasks")
        self.evaluator.generate_all_static_personas()

    def generate_answers(self) -> None:
        """Lets the user pick for which task to generate answers."""
        task_ids = self.evaluator.get_answer_pending_tasks()
        chosen_task_ids = self._print_and_evaluate_input(task_ids)

        total_request_count = 0
        for tid in chosen_task_ids:
            print(f"Sending request for {tid}")
            request_count = self.evaluator.send_answer_requests(tid)
            total_request_count += request_count
        print(f"{total_request_count} requests sent.")

    def generate_judgment_evaluation(self) -> None:
        """Lets the user pick a task, for which LLM-as-a-judge requests should be sent."""
        task_ids = self.evaluator.get_judgment_pending_tasks()
        chosen_task_ids = self._print_and_evaluate_input(task_ids)

        total_request_count = 0
        for tid in chosen_task_ids:
            print(f"Sending request for {tid}")
            request_count = self.evaluator.send_judgment_requests(tid)
            total_request_count += request_count
        print(f"{total_request_count} requests sent.")

    def generate_judgments(self):
        """Generate LLM-as-a-judge judgments for all tasks."""
        self.evaluator.generate_all_judgments()

    def quit_program(self):
        """Quits the program."""
        self.evaluator.quit()
        self.run = False

    def check_task_statuses(self):
        """Prints task status information."""
        print("Checking task statuses...")
        task_infos = self.evaluator.task_infos.copy()

        table_content = [
            [f"{"TaskID":<30}", f"{"TaskStatus":<30}", f"{"NeedJudge":<15}"],
            [f"{"─" * 29:<30}", f"{"─" * 29:<30}", f"{"─" * 14:<15}"]
        ]

        for task_id, task_info in task_infos.items():
            if len(task_id) > 25:
                task_id = f"{task_id[:22]}..."

            status = task_info.status
            if task_info.is_finished():
                status = "\033[32m" + f"{status:<30}" + "\033[0m"
            else:
                status = "\033[33m" + f"{status:<30}" + "\033[0m"

            need_judgment = "True" if task_info.need_judgment else "False"

            table_row = [f"{task_id:<30}", status, f"{need_judgment:<15}"]
            table_content.append(table_row)
        self._print_table(table_content)

    @staticmethod
    def clear_cli():
        """Clears the CLI."""
        command = "cls" if platform.system() == "Windows" else "clear"
        subprocess.run(command, shell=True, check=False)

    @staticmethod
    def _print_table(table_content: list[list[str]]):
        for row in table_content:
            row_output = " ".join(row)
            print(row_output)

    @staticmethod
    def _get_progress_message(progress: tuple[int, int, int]) -> str:
        if progress[1] > 0 or progress[2] > 0:
            return (
                f"Progress: {progress[1]} out of {progress[0]} finished; "
                f"{progress[2]} requests failed.")
        if progress[0] > 0:
            return f"Processing {progress} requests."
        return "No progress found."

    def check_batch_statuses(self):
        """Fetches batch statuses from the API and prints them to the terminal."""
        print("Checking batch statuses...\n")
        batch_infos = self.evaluator.check_batch_statuses().copy()
        table_content = [
            [f"{"TaskID":<30}", f"{"BatchType":<15}", f"{"BatchStatus":<15}", f"{"RemoteAPI":<10}", "Message"],
            [f"{"─" * 29:<30}", f"{"─" * 14:<15}", f"{"─" * 14:<15}", f"{"─" * 10:<10}", f"{"─" * 10}"]
        ]

        for batch_info in batch_infos.values():
            task_id = batch_info.task_id
            if len(task_id) > 25:
                task_id = f"{task_id[:22]}..."

            status = batch_info.status
            if batch_info.is_error():
                status = "\033[31m" + f"{status:<15}" + "\033[0m"
            elif batch_info.is_completed() or batch_info.is_retrieved():
                status = "\033[32m" + f"{status:<15}" + "\033[0m"
            else:
                status = "\033[33m" + f"{status:<15}" + "\033[0m"

            table_row = [f"{task_id:<30}", f"{batch_info.batch_type:<15}", status, f"{batch_info.client:<15}", ""]
            table_row[3] = self._get_progress_message(batch_info.get_progress())
            table_content.append(table_row)
        self._print_table(table_content)

    def show_help(self):
        """Prints the help menu."""
        print("\nCommands: ")
        for command, (description, _) in self.commands.items():
            print(f"    {command}: {description}")


if __name__ == "__main__":
    TEMP_PATH.mkdir(exist_ok=True, parents=True)
    DATA_PATH.mkdir(exist_ok=True, parents=True)
    load_dotenv()
    disable_progress_bar()
    disable_progress_bars()

    parser = argparse.ArgumentParser(
        description="CLI Application with the intention to communicate to LLM APIs.")
    parser.add_argument(
        "--path",
        help="The data path, used for reading and writing.",
        type=str,
        default=None
    )
    parser.add_argument(
        "--model",
        help="The model used for inference.",
        type=str,
        default="gpt-5.0-nano"
    )
    args = parser.parse_args()

    app = App(args.path, args.model)
    app.main()
