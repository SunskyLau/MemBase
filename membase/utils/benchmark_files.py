"""评测数据和官方代码的文件操作；只依赖 Python 标准库。"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from urllib.request import urlopen


REPO_ROOT = Path(__file__).resolve().parents[2]


def read_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value) -> None:
    """先写临时文件，再原子替换；中断不会留下半份完成记录。"""
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == text:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     delete=False) as stream:
        tmp = Path(stream.name)
        stream.write(text)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_hash(path: Path, expected: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"缺少文件：{path}；请先执行 prepare_benchmarks.py")
    if sha256_file(path) != expected:
        raise ValueError(f"文件校验失败：{path}")


def download_verified(url: str, target: Path, expected: str) -> None:
    if target.exists():
        verify_hash(target, expected)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            with urlopen(url, timeout=120) as response:
                shutil.copyfileobj(response, stream)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
    try:
        verify_hash(tmp, expected)
        os.replace(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)


def git_output(path: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def verify_repository(path: Path, revision: str) -> None:
    if not (path / ".git").is_dir():
        raise FileNotFoundError(f"缺少官方仓库：{path}；请先执行 prepare_benchmarks.py")
    if git_output(path, "rev-parse", "HEAD") != revision:
        raise ValueError(f"官方仓库提交与锁定版本不一致：{path}")


def prepare_repository(path: Path, url: str, revision: str, legacy: Path) -> None:
    """复用已有官方副本；迁移保留全部文件和 Git 历史。"""
    if not path.exists() and legacy.exists():
        verify_repository(legacy, revision)
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(legacy), str(path))
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--quiet", "--filter=blob:none", url, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "fetch", "--quiet", "origin", revision], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "--quiet", "--detach", revision], check=True)
    verify_repository(path, revision)


def preserve_incomplete(path: Path) -> None:
    """保留不完整产物后允许官方入口重跑，不把坏文件当作续跑完成标记。"""
    if path.exists():
        from datetime import datetime, timezone
        tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path.rename(path.with_name(f"{path.name}.incomplete.{tag}"))
