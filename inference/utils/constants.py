from pathlib import Path


TEMP_PATH = Path("temp")  # default temporary directory for intermediate files

DATA_PATH = Path("data/inference")


STATIC_ID_COLUMN = "static_id"   # column name for unique static identifiers
QUESTION_COLUMN = "question"    # column name containing the question text
GROUND_TRUTH_COLUMN = "answers"     # column name containing reference or gold-standard answers
