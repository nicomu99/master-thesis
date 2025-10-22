from enum import Enum

from .prompt_templates import (
    OPEN_QUESTION_TEMPLATE,
    MC_QUESTION_TEMPLATE,
    SUMMARIZATION_TEMPLATE,
    MATH_TEMPLATE,
    TRANSLATION_TEMPLATE,
)


class StringEnum(Enum):
    """Little helper enum subclass for easier printing.

    This helper allows printing an enum as its value.

    Example:
        class Color(StringEnum):
            RED = "red"

        print(Color.RED) # prints "red"
    """
    def __str__(self):
        return self.value


class BatchType(StringEnum):
    """Enumeration of available batch types.

    Attributes:
        PERSONAS: Batch containing persona-related data.
        ANSWERS: Batch containing model-generated answers.
    """
    PERSONAS = "personas"
    ANSWERS = "answers"
    JUDGE = "judge"


class QuestionType(StringEnum):
    MC = "mc"
    OPEN = "open"
    SUMMARIZATION = "summarization"
    MATH = "math"
    TRANSLATION = "translation"
    """The question type is a categorical field for datasets.

    Attributes:
        MC: Closed format questions with provided answer options.
        OPEN: Open format questions with no answer options.
        SUMMARIZATION: Summarization tasks.
    """

    @property
    def template(self) -> str:
        """Returns the question template.

        Returns:
            str: Question template.
        """
        templates = {
            "mc": MC_QUESTION_TEMPLATE,
            "open": OPEN_QUESTION_TEMPLATE,
            "summarization": SUMMARIZATION_TEMPLATE,
            "math": MATH_TEMPLATE,
            "translation": TRANSLATION_TEMPLATE,
        }
        return templates[self.value]


class TaskStatus(StringEnum):
    PERSONAS_PENDING = "personas_pending"
    PERSONAS_REQUESTED = "personas_requested"
    ANSWERS_PENDING = "answers_pending"
    ANSWERS_REQUESTED = "answers_requested"
    JUDGE_PENDING = "judge_pending"
    JUDGE_REQUESTED = "judge_requested"
    FINISHED = "finished"
    """Task status information.

    Attributes:
        PERSONAS_PENDING: The task is in its beginning state. No persona completions have been generated yet.
        PERSONAS_REQUESTED: A request has been sent to generate persona strings.
        ANSWERS_PENDING: Personas have been merged to the task dataframe and answers can be generated.
        ANSWERS_REQUESTED: A request has been sent for question answering.
        FINISHED: Answers have been merged to the task dataframe. The task has finished.
    """


class PersonaCategory(StringEnum):
    """Enumeration of possible persona categories.

    Attributes:
        STATIC: Refers to static personas, that do not change between samples of a task.
        DYNAMIC: Refers to personas that are specific to task samples, i.e. each sample has its own persona.
    """
    STATIC = "static"
    DYNAMIC = "dynamic"


class BatchStatus(StringEnum):
    SEND = "send"
    SENT = "sent"
    FAILED = "failed"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    RETRIEVED = "retrieved"
    ERROR = "error"
    """
        Possibly outdated process status, indicating what step the batch is currently going through.
        Can be "send", "sent", "failed", "in_progress", "completed", "error", or "retrieved".

        If the status is "send", a request file has already been created, but the batch has not been sent to the
        API. The statuses "failed", "in_progress" and "completed" correspond to the steps of the API.
        If the batch can be retrieved, the status is "completed", this changes to "retrieved" or "error" after the
        response has been fetched. Defaults to "sent"
    """
