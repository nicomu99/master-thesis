from typing import List

import subprocess
import platform
from dotenv import load_dotenv

from datasets import disable_progress_bar, disable_progress_bars

from .evaluator import Evaluator
from .utils import TEMP_PATH
from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class App:
    """Small CLI application to control the program."""

    def __init__(self):
        self.run = True
        self.evaluator = Evaluator()

        self.commands = {
            "q": ("Quit program", self.quit_program),
            "p": ("Generate personas", self.generate_personas),
            "t": ("Send task requests", self.generate_answers),
            "j": ("Send judge requests", self.generate_judge_answers),
            "s": ("Check batch statuses", self.check_batch_statuses),
            "f": ("Fetch batch responses", self.evaluator.fetch_batch_responses),
            "c": ("Check task statuses", self.check_task_statuses),
            "clear": ("Clears the CLI", self.clear_cli),
            "h": ("Show this help", self.show_help),
        }

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

    def generate_personas(self) -> None:
        """Lets the user pick which task to generate personas for."""
        print("Due to API queue limits, only one request should be sent at the same time.")
        missing_map = {
            f"{i + 1}": m for i, m in enumerate(self.evaluator.get_personas_pending_tasks())
        }

        if len(missing_map) < 1:
            print("No tasks with missing personas. Skipping.")
            return

        for k, v in missing_map.items():
            print(f"     ({k}) {v}")

        user_input = input("Choose task: ")
        if user_input not in missing_map:
            print("Task unknown. Please try again.")
            return

        task_id = missing_map[user_input]
        print(f"Generating personas for {task_id}")
        request_count = self.evaluator.generate_personas(task_id)
        print(f"{request_count} requests sent.")

    def generate_answers(self) -> None:
        """Lets the user pick for which task to generate answers."""
        print("Due to API queue limits, only one request should be sent at the same time.")
        missing_map = {
            f"{i + 1}": m for i, m in enumerate(self.evaluator.get_answer_pending_tasks())
        }

        if len(missing_map) < 1:
            print("No tasks with missing answers.")
            return

        for k, v in missing_map.items():
            print(f"     ({k}) {v}")

        user_input = input("Choose task: ")
        if user_input not in missing_map:
            print("Task unknown. Please try again.")
            return

        task_id = missing_map[user_input]
        print(f"Sending request for {task_id}")
        request_count = self.evaluator.send_task_requests(task_id)
        print(f"{request_count} requests sent.")

    def generate_judge_answers(self) -> None:
        """Lets the user pick a task, for which LLM-as-a-judge requests should be sent."""
        missing_map = {
            f"{i + 1}": m for i, m in enumerate(self.evaluator.get_judge_pending_tasks())
        }

        if len(missing_map) < 1:
            print("No tasks with missing answers.")
            return

        for k, v in missing_map.items():
            print(f"     ({k}) {v}")

        user_input = input("Choose task: ")
        if user_input not in missing_map:
            print("Task unknown. Please try again.")
            return

        task_id = missing_map[user_input]
        print(f"Sending request for {task_id}")
        request_count = self.evaluator.send_judge_requests(task_id)
        print(f"{request_count} requests sent.")

    def quit_program(self):
        """Quits the program."""
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

            need_judge = "True" if task_info.need_judge else "False"

            table_row = [f"{task_id:<30}", status, f"{need_judge:<15}"]
            table_content.append(table_row)
        self._print_table(table_content)

    def clear_cli(self):
        """Clears the CLI."""
        command = "cls" if platform.system() == "Windows" else "clear"
        subprocess.run(command, shell=True, check=False)

    @staticmethod
    def _print_table(table_content: List[List[str]]):
        for row in table_content:
            row_output = " ".join(row)
            print(row_output)

    def check_batch_statuses(self):
        """Fetches batch statuses from the API and prints them to the terminal."""
        print("Checking batch statuses...\n")
        batch_infos = self.evaluator.check_batch_statuses().copy()
        table_content = [
            [f"{"TaskID":<30}", f"{"BatchType":<15}", f"{"BatchStatus":<15}", "Message"],
            [f"{"─" * 29:<30}", f"{"─" * 14:<15}", f"{"─" * 14:<15}", f"{"─" * 10}"]
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

            table_row = [f"{task_id:<30}", f"{batch_info.batch_type:<15}", status, ""]
            if len(batch_info.remote_messages) > 0:
                table_row[3] = " ".join(batch_info.remote_messages)
            if batch_info.progress_message is not None:
                table_row[3] += batch_info.progress_message
            table_content.append(table_row)
        self._print_table(table_content)

    def show_help(self):
        """Prints the help menu."""
        print("\nCommands: ")
        for command, (description, _) in self.commands.items():
            print(f"    {command}: {description}")


if __name__ == "__main__":
    TEMP_PATH.mkdir(exist_ok=True, parents=True)
    load_dotenv()
    disable_progress_bar()
    disable_progress_bars()

    app = App()
    app.main()
