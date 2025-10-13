import subprocess
import platform

from .evaluator import Evaluator
from .utils import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class App:
    """Small CLI application to control the program."""

    def __init__(self):
        self.run = True
        self.evaluator = Evaluator(include_datasets=["mmlu-pro"])

        self.commands = {
            "q": ("Quit program", self.quit_program),
            "p": ("Generate personas", self.generate_personas),
            "t": ("Send task requests", self.generate_answers),
            "s": ("Check batch statuses", self.evaluator.check_batch_statuses),
            "f": ("Fetch batch responses", self.evaluator.fetch_batch_responses),
            "c": ("Check task statuses", self.check_task_statuses),
            "clear": ("Clears the CLI", self.clear_cli),
            # "a": ("Rerun menu", self.ask_task_resend),
            "h": ("Show this help", self.show_help),
        }

    def main(self) -> None:
        """Main loop that listens for user input."""
        self.show_help()
        while self.run:
            user_input = input("Enter next command (type h for help): ").strip()

            if user_input in self.commands:
                _, func = self.commands[user_input]
                func()
            else:
                print("Command unknown. Type \"h\" for help.")

    def generate_personas(self) -> None:
        """Lets the user pick which task to generate personas for."""
        print("Due to API queue limits, only one request should be sent at the same time.")
        missing_map = {
            f"{i + 1}": m for i, m in enumerate(self.evaluator.get_unfinished_tasks_personas())
        }

        if len(missing_map) < 1:
            print("No tasks with missing personas. Skipping.")
            return

        for k, v in missing_map.items():
            print(f"     ({k}) {v}")

        user_input = input("Choose task: ")
        if user_input not in missing_map:
            print("Task unknown. Please try again.")
        else:
            self.evaluator.generate_personas(missing_map[user_input])

    def generate_answers(self) -> None:
        """Lets the user pick for which task to generate answers."""
        print("Due to API queue limits, only one request should be sent at the same time.")
        missing_map = {
            f"{i + 1}": m for i, m in enumerate(self.evaluator.get_unfinished_tasks_answers())
        }

        if len(missing_map) < 1:
            print("No tasks with missing answers.")
            return

        for k, v in missing_map.items():
            print(f"     ({k}) {v}")

        user_input = input("Choose task: ")
        if user_input not in missing_map:
            print("Task unknown. Please try again.")
        else:
            self.evaluator.send_task_requests(missing_map[user_input])

    def quit_program(self):
        """Quits the program."""
        self.run = False

    def check_task_statuses(self):
        """Prints task status information."""
        print("Printing task statuses:")
        task_configs = self.evaluator.task_configs
        for idx, (task_id, task_config) in enumerate(task_configs.items()):
            print(f"    {f'({idx + 1})':>4} Task {task_id:25} status is     {task_config.status}")

    def clear_cli(self):
        """Clears the CLI."""
        command = "cls" if platform.system() == "Windows" else "clear"
        subprocess.run(command, shell=True, check=False)

    def ask_task_resend(self):
        """Lets the user pick tasks to resend."""

    def show_help(self):
        """Prints the help menu."""
        print("\nCommands: ")
        for command, (description, _) in self.commands.items():
            print(f"    {command}: {description}")


if __name__ == "__main__":
    app = App()
    app.main()
