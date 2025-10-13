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

    def ask_task_resend(self):
        """Lets the user pick tasks to resend."""

        batch_infos = self.evaluator.get_batch_infos()
        for task_id, batch_info in batch_infos.items():
            log.info("Task %s has status %s", task_id, batch_info.status)
            user_input = input("Rerun task y|[n]?: ").strip()

            if user_input == "y":
                self.evaluator.update_batch_info(task_id, "send")

    def show_help(self):
        """Prints the help menu."""

        print("\nCommands: ")
        for command, (description, _) in self.commands.items():
            print(f"    {command}: {description}")


if __name__ == "__main__":
    app = App()
    app.main()
