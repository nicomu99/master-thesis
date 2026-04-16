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
        JUDGMENT: Batch containing LLM-as-a-judge requests.
    """
    PERSONAS = "personas"
    ANSWERS = "answers"
    JUDGMENT = "judgment"


class QuestionType(StringEnum):
    """The question type is a categorical field for datasets.

    Attributes:
        MC: Closed format questions with provided answer options.
        OPEN: Open format questions with no answer options.
        SUMMARIZATION: Summarization tasks.
        MATH: Open format math reasoning questions.
        TRANSLATION: Translation tasks.
    """
    MC = "mc"
    OPEN = "open"
    SUMMARIZATION = "summarization"
    MATH = "math"
    TRANSLATION = "translation"

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
    """Task status information.

    Attributes:
        PERSONAS_PENDING: The task is in its beginning state. No persona completions have been generated yet.
        PERSONAS_REQUESTED: A request has been sent to generate persona strings.
        ANSWERS_PENDING: Personas have been merged to the task dataframe and answers can be generated.
        ANSWERS_REQUESTED: A request has been sent for question answering.
        JUDGMENT_PENDING: If the task requires LLM-as-a-judge, the status turns to judgment pending.
        JUDGMENT_REQUESTED: A request has been sent for judgment inference.
        FINISHED: Answers have been merged to the task dataframe. The task has finished.
    """
    PERSONAS_PENDING = "personas_pending"
    PERSONAS_REQUESTED = "personas_requested"
    ANSWERS_PENDING = "answers_pending"
    ANSWERS_REQUESTED = "answers_requested"
    JUDGMENT_PENDING = "judgment_pending"
    JUDGMENT_REQUESTED = "judgment_requested"
    FINISHED = "finished"


class PersonaCategory(StringEnum):
    """Enumeration of possible persona categories.

    Attributes:
        EMPTY: Refers to the no persona baselines.
        STATIC: Refers to static personas, that do not change between samples of a task.
        STATIC_TEACHER: Refers to the static teacher persona.
        DYNAMIC: Refers to personas that are specific to task samples, i.e. each sample has its own persona.
    """
    EMPTY = "empty"
    STATIC = "static"
    STATIC_TEACHER = "static_teacher"
    DYNAMIC = "dynamic"


class BatchStatus(StringEnum):
    """Enumeration of possible processing states for a batch.

    Represents the lifecycle of a batch as it moves through local processing and
    API interaction steps. The status is used to track progress, errors, and completion.

    Attributes:
        SENT: The batch has been sent successfully and is awaiting processing.
        FAILED: The batch submission failed before or during API processing.
        VALIDATING: The batch is being validated by the API.
        IN_PROGRESS: The batch is currently being processed by the API.
        COMPLETED: The batch has finished processing and is ready to be retrieved.
        RETRIEVED: The results have been successfully fetched.
        ERROR: An error occurred during or after retrieval (e.g., corrupted output).
    """
    SENT = "sent"
    FAILED = "failed"
    VALIDATING = "validating"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    RETRIEVED = "retrieved"
    ERROR = "error"
    CANCELLED = "cancelled"
