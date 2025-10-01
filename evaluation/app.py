from .evaluator import Evaluator
from .llm_client import LLMClient


class App:
    def __init__(self):
        self.run = True
        self.llm_client = LLMClient()
        self.evaluator = Evaluator(self.llm_client, include_datasets=["mmlu"])

        self.commands = {
            "q": ("Quit program", self.quit_program),
            "p": ("Generate personas", self.evaluator.generate_personas),
            "t": ("Send task requests", self.evaluator.send_task_requests),
            "s": ("Check batch statuses", self.llm_client.check_batch_statuses),
            "f": ("Fetch batch responses", self.llm_client.fetch_batch_responses),
            "h": ("Show this help", self.show_help),
        }

    def main(self) -> None:
        """_summary_
        """

        while self.run:
            user_input = input("Enter next command (type h for help): ").strip()

            if user_input in self.commands:
                _, func = self.commands[user_input]
                func()
            else:
                print("Command unknown. Type \"h\" for help.")

    def quit_program(self):
        self.run = False

    def show_help(self):
        print("\nCommands: ")
        for command, (description, _) in self.commands.items():
            print(f"    {command}: {description}")


if __name__ == "__main__":
    app = App()
    app.main()
