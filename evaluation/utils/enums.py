from enum import Enum


class StringEnum(Enum):
    def __str__(self):
        return self.value


class BatchType(StringEnum):
    PERSONAS = "personas"
    ANSWERS = "answers"


class QuestionType(StringEnum):
    OPEN = "open"
    MC = "mc"
    SUMMARIZATION = "summarization"


class TaskStatus(StringEnum):
    PERSONAS_PENDING = "personas_pending"
    PERSONAS_REQUESTED = "personas_requested"
    ANSWERS_PENDING = "answers_pending"
    ANSWERS_REQUESTED = "answers_requested"
    FINISHED = "finished"
