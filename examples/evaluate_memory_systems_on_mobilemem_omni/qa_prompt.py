from string import Template


def get_mobilemem_omni_qa_prompt() -> Template:
    """Return the shared QA prompt for all four textual memory systems."""
    return Template(
        "Question:\n$question\n"
        "Please answer the question based on the following memories:\n"
        "$context\n"
        "Answer Rules:\n"
        "1. If options are provided in the question, use the original option text as your answer.\n"
        "2. For multiple-choice questions (multiple-select), select all correct options.\n"
        "3. For open-ended questions, respond freely."
    )
