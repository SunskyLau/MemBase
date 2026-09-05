from importlib import import_module

_EXPORTS = {
    "MonkeyPatcher": "monkey_patch",
    "PatchSpec": "monkey_patch",
    "make_attr_patch": "monkey_patch",
    "token_monitor": "token_monitor",
    "CostStateManager": "token_monitor",
    "CostState": "token_monitor",
    "get_tokenizer_for_model": "token_monitor",
    "import_function_from_path": "files",
    "download_models": "files",
}


def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name = _EXPORTS[name]
    module = import_module(f".{module_name}", __name__)
    # token_monitor 既是模块名又是公共对象名；一同缓存以保留原来的导入语义。
    for export, source in _EXPORTS.items():
        if source == module_name:
            globals()[export] = getattr(module, export)
    return globals()[name]


__all__ = [
    "MonkeyPatcher", 
    "PatchSpec", 
    "make_attr_patch", 
    "token_monitor", 
    "CostStateManager", 
    "CostState", 
    "get_tokenizer_for_model",
    "import_function_from_path",
    "download_models",
]
