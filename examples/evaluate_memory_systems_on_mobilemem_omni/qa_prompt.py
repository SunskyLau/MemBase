from string import Template


def get_mobilemem_omni_qa_prompt() -> Template:
    """Return the shared QA prompt for all four textual memory systems."""
    return Template(
        "Question:\n$question\n\n"
        "Please answer the question using only the following retrieved memories:\n"
        "$context\n\n"
        "Answer rules:\n"
        "1. If options are present, answer with the original option text.\n"
        "2. For multiple-select questions, include every correct option.\n"
        "3. For open-ended questions, give a direct answer without unsupported details."
    )
