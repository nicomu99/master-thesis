from .evaluator import Evaluator
from .llm_client import LLMClient


class App:
    """Small CLI application to control the program."""

    def __init__(self):
        self.run = True
        self.llm_client = LLMClient()
        self.evaluator = Evaluator(self.llm_client, include_datasets=["mmlu", "gsm8k"])

        self.commands = {
            "q": ("Quit program", self.quit_program),
            "p": ("Generate personas", self.evaluator.generate_personas),
            "t": ("Send task requests", self.evaluator.send_task_requests),
            "s": ("Check batch statuses", self.llm_client.check_batch_statuses),
            "f": ("Fetch batch responses", self.llm_client.fetch_batch_responses),
            "h": ("Show this help", self.show_help),
        }

    def main(self) -> None:
        """Main loop that listens for user input."""

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

    def show_help(self):
        """Prints the help menu."""

        print("\nCommands: ")
        for command, (description, _) in self.commands.items():
            print(f"    {command}: {description}")


if __name__ == "__main__":
    app = App()
    app.main()
