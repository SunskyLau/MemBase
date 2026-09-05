"""两个官方评测运行器共用的轻量参数。"""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkRunConfig:
    benchmark: str
    baseline: str
    mode: str
    data_root: Path
    upstream_dir: Path
    run_dir: Path
    answer_model: str = "gpt-4.1-mini"
    internal_model: str = "gpt-4.1-mini"
    judge_model: str = "gpt-4.1-mini"
    base_url: str = "https://api.openai.com/v1"
    top_k: int = 5
    temperature: float = 0.7
    parallel_jobs: int = 1
    workers: int = 4
    judge_workers: int = 4
    check_workers: int = 8
    dry_run: bool = False

    def saved_config(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value
                for key, value in asdict(self).items() if key != "dry_run"}
