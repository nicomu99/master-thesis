STATIC_SHORT_TEMPLATE = """
For the field of {task_type}, create a persona string containing one sentence, written in second person perspective.
The persona string should start with "{persona_string}".\n\n
"""

STATIC_LONG_TEMPLATE = """
For the field of {task_type}, create a persona string containing three sentences, written in second person perspective.
The persona string should start with "{persona_string}".\n\n
"""

DYNAMIC_SHORT_TEMPLATE = """
For the following question, create a persona string containing one sentence, written in second person perspective.
The persona string should start with "{persona_string}".\n\n

{question}
"""

DYNAMIC_LONG_TEMPLATE = """
For the following question, create a persona string containing three sentences, written in second person perspective.
The persona string should start with "{persona_string}".\n\n

{question}
"""

QUESTION_TEMPLATE = """
{persona} Answer the following question:

{question}
"""

MC_QUESTION_TEMPLATE = """
{persona} Answer the following question.

{question}\n\n

(A) {choice_1}\n
(B) {choice_2}\n
(C) {choice_3}\n
(D) {choice_4}\n
"""
