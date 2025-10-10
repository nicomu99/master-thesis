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
