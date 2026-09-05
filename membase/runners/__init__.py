from importlib import import_module

_EXPORTS = {
    "ConstructionRunner": "construction",
    "ConstructionRunnerConfig": "construction",
    "SearchRunner": "search",
    "SearchRunnerConfig": "search",
    "EvaluationRunner": "evaluation",
    "EvaluationRunnerConfig": "evaluation",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


__all__ = [
    "ConstructionRunner",
    "ConstructionRunnerConfig",
    "SearchRunner",
    "SearchRunnerConfig",
    "EvaluationRunner",
    "EvaluationRunnerConfig",
]
