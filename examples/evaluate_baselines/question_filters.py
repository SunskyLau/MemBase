from membase.model_types.dataset import QuestionAnswerPair


def exclude_locomo_adversarial(qa_pair: QuestionAnswerPair) -> bool:
    """Keep LoCoMo categories 1-4 and exclude category-5 adversarial QA."""
    return qa_pair.metadata.get("question_type") != "adversarial"

