"""实验入口的显示层；原运行器、方法指纹和续跑规则保持不变。"""

from __future__ import annotations

import argparse
import codecs
from datetime import datetime, timezone
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from membase.utils.experiment_progress import ExperimentProgress, duration, read_record


class Console:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.key = os.environ.get("OPENAI_API_KEY", "")

    def write(self, text: str) -> None:
        if self.key:
            text = text.replace(self.key, "[REDACTED]")
        print(text, end="", flush=True)
        # 原运行器拒绝非空的新目录，因此只在它创建清单后附加控制台日志。
        if (self.run_dir / "config.json").is_file():
            try:
                with (self.run_dir / "console.log").open("a", encoding="utf-8") as stream:
                    stream.write(text)
            except OSError:
                pass  # 显示层写盘失败不能杀掉正在进行的实验。


def stop_process(process) -> None:
    if process.poll() is not None:
        return
    # 子进程独占进程组。终止的是本次实验及其官方子进程，不波及其他实验。
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise(command: list[str], run_dir: Path, interval: float) -> int:
    console, progress = Console(run_dir), ExperimentProgress(run_dir)
    environment = {**os.environ, "PYTHONUNBUFFERED": "1"}
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=environment, start_new_session=True, bufsize=0)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    decoder, pending = codecs.getincrementaldecoder("utf-8")("replace"), ""
    interrupted = False
    previous_term = signal.getsignal(signal.SIGTERM)

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    console.write(f"[启动] 进度每 {interval:g} 秒刷新；结果目录：{run_dir}\n")
    next_tick = started

    def refresh():
        try:
            lines = progress.poll()
        except Exception as error:
            # 这里只隔离观察器错误；模型和运行器的失败仍由子进程退出码报告。
            console.write(f"[进度] 暂不可读取（{type(error).__name__}），不改变实验状态\n")
            return
        for index, line in enumerate(lines):
            elapsed = f"；本次已运行 {duration(time.monotonic() - started)}" if index == 0 else ""
            console.write(line + elapsed + "\n")

    try:
        while selector.get_map() or process.poll() is None:
            for key, _ in selector.select(timeout=0.25):
                block = os.read(key.fd, 65536)
                if not block:
                    selector.unregister(key.fileobj)
                    pending += decoder.decode(b"", final=True)
                    if pending:
                        console.write(pending + ("" if pending.endswith("\n") else "\n"))
                        pending = ""
                    continue
                pending += decoder.decode(block)
                # 按完整行脱敏，防止密钥刚好跨越两次管道读取而被分段打印。
                lines = pending.splitlines(keepends=True)
                pending = ""
                for line in lines:
                    if line.endswith(("\n", "\r")):
                        console.write(line)
                    else:
                        pending = line
            if time.monotonic() >= next_tick:
                refresh()
                next_tick = time.monotonic() + interval
        code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        console.write("\n[中断] 正在停止本次实验；保留数据库、已完成结果和请求账本。\n")
        stop_process(process)
        code = 130
    finally:
        signal.signal(signal.SIGTERM, previous_term)
        stop_process(process)
        selector.close()
        process.stdout.close()
    if interrupted and (run_dir / "config.json").is_file():
        from membase.utils.benchmark_files import write_json
        if read_record(run_dir / "status.json").get("status") != "complete":
            write_json(run_dir / "status.json", {"status": "incomplete", "error_type": "KeyboardInterrupt"})
    refresh()
    console.write(f"[退出] 返回码 {code}；本次耗时 {duration(time.monotonic() - started)}\n")
    return code if code >= 0 else 128 - code


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--progress-interval", type=float, default=10)
    parser.add_argument("--watch", type=Path)
    parser.add_argument("--once", action="store_true")
    options, remaining = parser.parse_known_args()
    if options.progress_interval <= 0:
        parser.error("progress-interval 必须大于 0")
    if options.watch is not None:
        if remaining or not (options.watch / "config.json").is_file():
            parser.error("--watch 需要已有运行目录，且不能同时传入启动参数")
        progress = ExperimentProgress(options.watch.resolve())
        try:
            while True:
                for line in progress.poll():
                    key = os.environ.get("OPENAI_API_KEY", "")
                    print(line.replace(key, "[REDACTED]") if key else line, flush=True)
                if options.once:
                    return 0
                time.sleep(options.progress_interval)
        except KeyboardInterrupt:
            print("停止查看；原实验未受影响。")
            return 0
    command = [sys.executable, "-u", str(ROOT / "scripts/run_benchmark.py"), *remaining]
    if "--dry-run" in remaining or "--help" in remaining or "-h" in remaining:
        return subprocess.call(command)  # 不启动观察器，也不创建日志或数据库。
    location = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    location.add_argument("--output-dir", type=Path)
    location.add_argument("--run-id", default="")
    args, _ = location.parse_known_args(remaining)
    if args.output_dir is None:
        return subprocess.call(command)  # 由原入口报告缺失参数。
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    if Path(run_id).name != run_id or run_id in {".", ".."}:
        parser.error("run-id 必须是单个目录名")
    if not args.run_id:
        command += ["--run-id", run_id]
    return supervise(command, args.output_dir.resolve() / run_id, options.progress_interval)


if __name__ == "__main__":
    raise SystemExit(main())
