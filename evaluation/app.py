from .evaluator import Evaluator

from .log_conf import get_logger

log = get_logger(__name__)


class App:
    """Small CLI application to control the program."""

    def __init__(self):
        self.run = True
        self.evaluator = Evaluator(include_datasets=["mmlu-pro"])

        self.commands = {
            "q": ("Quit program", self.quit_program),
            "p": ("Generate personas", self.evaluator.generate_personas),
            "t": ("Send task requests", self.evaluator.send_task_requests),
            "s": ("Check batch statuses", self.evaluator.check_batch_statuses),
            "f": ("Fetch batch responses", self.evaluator.fetch_batch_responses),
            "a": ("Rerun menu", self.ask_task_resend),
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
