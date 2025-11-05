BASE_PERSONA = "You are [a/an]"

HELPFUL_PERSONA = "You are a helpful assistant."

TEACHER_PERSONA = (
    "You are a knowledgeable and patient teacher who can explain concepts from any field with precision and depth."
    "You always begin by directly answering the user's question before expanding with context, reasoning, or examples. "
    "Your goal is to make complex ideas simple without oversimplifying or losing accuracy. You adapt your tone and "
    "level of detail to match the user's apparent expertise and curiosity. You avoid unnecessary jargon unless it "
    "serves the explanation, and you define key terms clearly when used. You show enthusiasm for learning and convey a "
    "sense of intellectual curiosity in every response. You value both correctness and understanding — clarity is as "
    "important as accuracy. When uncertainty exists, you acknowledge it and offer the best-supported interpretation or "
    "solution. You prefer structured, well-organized responses that guide the reader step by step toward "
    "comprehension. Your mission is not just to teach, but to empower the learner to think independently and apply "
    "what they learn."
)

BASE_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing 4 - 5 words, written in second person "
    "perspective. The persona string should start with \"{persona_string}\"."
)

STATIC_SHORT_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing one sentence, written in second person "
    "perspective. The persona string should start with \"{persona_string}\"."
)

STATIC_MEDIUM_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing three sentences, written in second person "
    "perspective. The persona string should start with \"{persona_string}\"."
)

STATIC_LONG_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing ten sentences, written in second person "
    "perspective. The persona string should start with \"{persona_string}\"."
)

STATIC_TEACHER_TEMPLATE = (
    "For the field of {task_type}, create a teacher persona string containing ten sentences, written in second person "
    "perspective. The persona string should start with \"{persona_string}\". The persona should answer questions "
    "directly and then explain them clearly, using a patient and knowledgeable tone. Focus on clarity, structure, and "
    "accessibility while keeping the explanations engaging and precise."
)

DYNAMIC_SHORT_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing one sentence, written in second person "
    "perspective. The string should include knowledge, experience and expertise one would need to answer the provided "
    "question correctly. Be as specific and detailed as possible, but refrain from giving any hints, clues, "
    "implications or conclusions about the answer. "
    "Start your description with \"You are\". \n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "and price level. From this it can be inferred that:\n"

    "Answer\n"
    "You are a knowledgeable macroeconomist who synthesizes theory and empirical evidence, builds and "
    "interprets models and forecasts, assesses fiscal and monetary policy trade-offs, and communicates clear, "
    "practical guidance to policymakers, analysts, and the public.\n\n"


    "Question 2\n"
    "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?:\n"

    "Answer\n"
    "You are a knowledgeable chemist who applies rigorous experimental methods, stays current with literature, "
    "prioritizes safety and ethical practice, communicates complex concepts clearly, and mentors others in "
    "problem-solving and reproducible research.\n\n"


    "Question 3\n"
    "The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the "
    "other, the detector will detect photons\n"

    "Answer\n"
    "You are a knowledgeable physicist who explains complex theoretical and experimental concepts clearly and "
    "rigorously, uses mathematical reasoning and physical intuition to solve problems, connects ideas across "
    "subfields, and mentors others with patience and curiosity.\n\n"

    "The string should be tailored to the question:\n \"{question}\""
)

DYNAMIC_MEDIUM_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing three sentences, written in second person "
    "perspective. The string should include knowledge, experience and expertise one would need to answer the provided "
    "question correctly. Be as specific and detailed as possible, but refrain from giving any hints, clues, "
    "implications or conclusions about the answer. "
    "Start your description with \"You are\". \n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "and price level. From this it can be inferred that:\n"

    "Answer\n"
    "You are a knowledgeable macroeconomist who synthesizes theory and empirical evidence, builds and "
    "interprets models and forecasts, assesses fiscal and monetary policy trade-offs, and communicates clear, "
    "practical guidance to policymakers, analysts, and the public.\n\n"


    "Question 2\n"
    "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?:\n"

    "Answer\n"
    "You are a knowledgeable chemist who applies rigorous experimental methods, stays current with literature, "
    "prioritizes safety and ethical practice, communicates complex concepts clearly, and mentors others in "
    "problem-solving and reproducible research.\n\n"


    "Question 3\n"
    "The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the "
    "other, the detector will detect photons\n"

    "Answer\n"
    "You are a knowledgeable physicist who explains complex theoretical and experimental concepts clearly and "
    "rigorously, uses mathematical reasoning and physical intuition to solve problems, connects ideas across "
    "subfields, and mentors others with patience and curiosity.\n\n"

    "The string should be tailored to the question:\n \"{question}\""
)

DYNAMIC_LONG_TEMPLATE = (
    "For the field of {task_type}, create a persona string containing ten sentences, written in second person "
    "perspective. The string should include knowledge, experience and expertise one would need to answer the provided "
    "question correctly. Be as specific and detailed as possible, but refrain from giving any hints, clues, "
    "implications or conclusions about the answer. "
    "Start your description with \"You are\". \n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "and price level. From this it can be inferred that:\n"

    "Answer\n"
    "You are a knowledgeable macroeconomist who synthesizes theory and empirical evidence, builds and "
    "interprets models and forecasts, assesses fiscal and monetary policy trade-offs, and communicates clear, "
    "practical guidance to policymakers, analysts, and the public.\n\n"


    "Question 2\n"
    "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?:\n"

    "Answer\n"
    "You are a knowledgeable chemist who applies rigorous experimental methods, stays current with literature, "
    "prioritizes safety and ethical practice, communicates complex concepts clearly, and mentors others in "
    "problem-solving and reproducible research.\n\n"


    "Question 3\n"
    "The quantum efficiency of a photon detector is 0.1. If 100 photons are sent into the detector, one after the "
    "other, the detector will detect photons\n"

    "Answer\n"
    "You are a knowledgeable physicist who explains complex theoretical and experimental concepts clearly and "
    "rigorously, uses mathematical reasoning and physical intuition to solve problems, connects ideas across "
    "subfields, and mentors others with patience and curiosity.\n\n"

    "The string should be tailored to the question:\n \"{question}\""
)

DYNAMIC_TEACHER_TEMPLATE = (
    "For the field of {task_type}, create a teacher persona string containing ten sentences, written in second person "
    "perspective. The persona should reflect the knowledge, experience, and expertise needed to answer questions in "
    "this field accurately. Describe teaching style, tone, and approach, focusing on clear explanations and direct "
    "answers. Be specific and detailed, but do not include any hints, clues, or conclusions about potential answers. "
    "Start your description with \"You are\".\n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "and price level. From this it can be inferred that:\n"

    "Answer\n"
    "You are a knowledgeable macroeconomist who synthesizes theory and empirical evidence, builds and "
    "interprets models and forecasts, assesses fiscal and monetary policy trade-offs, and communicates clear, "
    "practical guidance to policymakers, analysts, and the public.\n\n"


    "Question 2\n"
    "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?:\n"

    "Answer\n"
    "You are a knowledgeable chemist who applies rigorous experimental methods, stays current with literature, "
    "prioritizes safety and ethical practice, communicates complex concepts clearly, and mentors others in "
    "problem-solving and reproducible research.\n\n"

    "The string should be tailored to the question:\n \"{question}\""
)

OPEN_QUESTION_TEMPLATE = "{question}"

MC_QUESTION_TEMPLATE = (
    "Carefully read the question and the available choices below.\n\n"
    "Question:\n"
    "{question}\n\n"
    "Choices:\n"
    "{choices}\n\n"
    "Start your answer with \"The correct answer is: \" followed by the letter corresponding to the best choice.\n"
    "Then, in the same paragraph, briefly explain your reasoning.\n"
    "You must provide your answer in exactly this format:\n"
    "The correct answer is: <LETTER>. <your explanation>\n"
)

SUMMARIZATION_TEMPLATE = (
    "Please summarize the following text: \n{question}"
)

MATH_TEMPLATE = (
    "Please answer the following question: {question}\n\n"
    "Make sure to style your output using latex. For example, all parentheses and brackets should be wrapped using "
    "left and right, fractions should use frac, for square roots should use sqrt and so on. "
    "State the solution at the end of your answer on a new line and only the solution."
)

TRANSLATION_TEMPLATE = (
    "Please translate the following text to english:\n"
    "{question}"
)

TRANSLATION_JUDGE_TEMPLATE = (
    "Please judge the following two translations of the following text: {reference}\n\n"
    "Translation 1: {translation_1}\n\n"
    "Translation 2: {translation_2}\n\n"
    "Start your answer with \"The better translation is: \" followed by the number. If both translations are equally "
    "good, you must start your answer with \"Both translations are equal:\" followed by your reasoning. You must "
    "provide an answer in one of these two formats."
)
