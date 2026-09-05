"""Public interfaces, loaded only when requested."""

from importlib import import_module

_EXPORTS = {
    "MEMORY_LAYERS_MAPPING": "layers",
    "CONFIG_MAPPING": "configs",
    "DATASET_MAPPING": "datasets",
    "ONLINE_EVAL_ENV_MAPPING": "datasets",
    "METRIC_MAPPING": "evaluation",
    "ConstructionRunnerConfig": "runners.construction",
    "ConstructionRunner": "runners.construction",
    "SearchRunnerConfig": "runners.search",
    "SearchRunner": "runners.search",
    "EvaluationRunnerConfig": "runners.evaluation",
    "EvaluationRunner": "runners.evaluation",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(f".{_EXPORTS[name]}", __name__), name)
    globals()[name] = value
    return value


# Export the public APIs.
__all__ = [
    "CONFIG_MAPPING",
    "MEMORY_LAYERS_MAPPING",
    "DATASET_MAPPING",
    "ONLINE_EVAL_ENV_MAPPING",
    "METRIC_MAPPING",
    "ConstructionRunnerConfig",
    "ConstructionRunner",
    "SearchRunnerConfig",
    "SearchRunner",
    "EvaluationRunnerConfig",
    "EvaluationRunner",
]
