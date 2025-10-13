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

OPEN_QUESTION_TEMPLATE = "{question}"

MC_QUESTION_TEMPLATE = (
    "{question}\n\n"
    "{choices}"
)

SUMMARIZATION_TEMPLATE = (
    "Please summarize the following text: \n{question}"
)
