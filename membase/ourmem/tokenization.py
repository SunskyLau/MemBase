"""统一分词器；优先使用已经核验的随包资源，离线启动不必重新下载。"""

from functools import lru_cache
from hashlib import sha256
import importlib.util
import os
from pathlib import Path
import shutil

O200K_FILE = "fb374d419588a4632f3f557e76b4b70aebbca790"
O200K_SHA256 = "446a9538cb6c348e3516120d7c08b09f57c36495e2acfffe59a5bf8b0cfb1a2d"


@lru_cache(maxsize=1)
def get_encoder():
    import tiktoken
    cache = Path(os.environ.get("TIKTOKEN_CACHE_DIR", Path(__file__).resolve().parents[2] / "data/cache/tiktoken"))
    target = cache / O200K_FILE
    if not target.exists():
        spec = importlib.util.find_spec("litellm")
        if spec and spec.origin:
            resource = Path(spec.origin).parent / "litellm_core_utils/tokenizers" / O200K_FILE
            if resource.is_file() and sha256(resource.read_bytes()).hexdigest() == O200K_SHA256:
                cache.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(resource, target)
    if target.exists() and sha256(target.read_bytes()).hexdigest() != O200K_SHA256:
        raise ValueError("Cached o200k_base tokenizer does not match its official hash")
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(cache))
    return tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(get_encoder().encode(text, disallowed_special=()))
