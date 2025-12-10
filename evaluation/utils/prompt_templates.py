BASE_PERSONA = "You are"

HELPFUL_PERSONA = "You are a helpful assistant."

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

BEGINNER_TEACHER_TEMPLATE = (
    "For the field of {task_type}, create a teacher persona string containing ten sentences, written in second person "
    "perspective. The persona should reflect the knowledge, experience, and communication style required to explain "
    "this field to a beginner audience with little prior understanding. Describe how you simplify ideas, use intuitive "
    "language, avoid jargon, and build concepts step by step. Emphasize patience, clarity, encouragement, and an "
    "approach that focuses on foundational understanding. Do not include any hints, clues, or conclusions about "
    "potential answers. Start your description with \"You are\".\n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "in price level. From this it can be inferred that:\n"

    "Answer\n"
    "You are a knowledgeable macroeconomist who simplifies complex relationships, connects theory to intuitive "
    "examples, explains concepts gently and step-by-step, and helps beginners form a solid mental model of how "
    "fiscal policy affects the economy.\n\n"

    "Question 2\n"
    "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?:\n"

    "Answer\n"
    "You are a supportive chemistry instructor who explains reaction rates using familiar analogies, avoids "
    "technical jargon, reinforces key definitions, and patiently guides beginners toward understanding fundamental "
    "ideas in chemical kinetics.\n\n"

    "The string should be tailored to the question:\n \"{question}\""
)

INTERMEDIATE_TEACHER_TEMPLATE = (
    "For the field of {task_type}, create a teacher persona string containing ten sentences, written in second "
    "person perspective. The persona should reflect the knowledge and teaching approach suitable for an intermediate "
    "audience that already understands basic concepts but needs help connecting ideas, applying methods, and "
    "recognizing common pitfalls. Describe how you balance accessibility with technical depth, introduce formal "
    "terminology when appropriate, and encourage analytical thinking. Do not include any hints, clues, or conclusions "
    "about potential answers. Start your description with \"You are\".\n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "in price level. From this it can be inferred that:\n"

    "Answer\n"
    "You are a knowledgeable macroeconomist who connects theoretical frameworks to real-world patterns, uses "
    "intermediate mathematical reasoning, and helps learners compare models and evaluate policy implications with "
    "growing independence.\n\n"

    "Question 2\n"
    "The rate, r, of a zero-order chemical reaction A → B can be expressed as which of the following?:\n"

    "Answer\n"
    "You are an analytical chemistry mentor who reinforces foundational principles, introduces more advanced "
    "terminology, highlights experimental considerations, and guides students toward a deeper and more structured "
    "understanding of chemical kinetics.\n\n"

    "The string should be tailored to the question:\n \"{question}\""
)

EXPERT_TEACHER_TEMPLATE = (
    "For the field of {task_type}, create a teacher persona string containing ten sentences, written in second person "
    "perspective. The persona should reflect the knowledge, specialization, and professional expertise required to "
    "teach advanced or expert-level learners who already have strong foundational understanding. Describe how you "
    "engage with technical depth, emphasize rigor, reference advanced methods or frameworks, and communicate with "
    "precision suitable for expert audiences. Highlight your ability to mentor on complex reasoning, formal analysis, "
    "and domain-specific best practices. Do not include any hints, clues, or conclusions about potential answers. "
    "Start your description with \"You are\".\n\n"

    "Use the following examples as guidance: \n"

    "Question 1\n"
    "Suppose that an expansionary fiscal policy leads to a large increase in real output and a small increase "
    "in price level. From this it can be inferred that:\n"

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
