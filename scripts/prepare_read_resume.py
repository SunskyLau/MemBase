"""显式准备只重跑读取阶段；默认只检查，--clear-read-results 才移出旧结果。"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--clear-read-results', action='store_true')
    parser.add_argument('--from-run', type=Path, help='显式复制原构建到新运行；原结果保持不变')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    from membase.utils.read_revision import prepare_revision, fork_read_revision
    if args.from_run:
        if args.clear_read_results:
            parser.error('--from-run 创建新目录，不与 --clear-read-results 混用')
        result = ({**prepare_revision(args.from_run), "new_run_dir": str(args.run_dir.resolve())}
                  if args.dry_run else fork_read_revision(args.from_run, args.run_dir))
    else:
        if args.dry_run and args.clear_read_results:
            parser.error('--dry-run 不能同时清理结果')
        result = prepare_revision(args.run_dir, clear=args.clear_read_results)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
