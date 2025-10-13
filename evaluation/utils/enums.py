from enum import Enum


class StringEnum(Enum):
    """Little helper enum subclass for easier printing.
    
    This helper allows printing an enum as its value. 
    
    Example:
        class Color(StringEnum):
            RED = "red
        
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


class QuestionType(StringEnum):
    """The question type is a categorical field for datasets.
    
    Attributes:
        MC: Closed format questions with provided answer options.
        OPEN: Open format questions with no answer options.
        SUMMARIZATION: Summarization tasks.
    """
    MC = "mc"
    OPEN = "open"
    SUMMARIZATION = "summarization"


class TaskStatus(StringEnum):
    """Task status information.
    
    Attributes:
        PERSONAS_PENDING: The task is in its beginning state. No persona completions have been generated yet.
        PERSONAS_REQUESTED: A request has been sent to generate persona strings.
        ANSWERS_PENDING: Personas have been merged to the task dataframe and answers can be generated.
        ANSWERS_REQUESTED: A request has been sent for question answering.
        FINISHED: Answers have been merged to the task dataframe. The task has finished. 
    """
    PERSONAS_PENDING = "personas_pending"
    PERSONAS_REQUESTED = "personas_requested"
    ANSWERS_PENDING = "answers_pending"
    ANSWERS_REQUESTED = "answers_requested"
    FINISHED = "finished"
