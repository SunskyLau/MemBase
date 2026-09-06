"""独立短场景验证公共三阶段；只有 --real 才允许发起付费请求。"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--method', choices=['amem', 'ourmem'], default='amem')
    parser.add_argument('--attempt', default='amem_01')
    args = parser.parse_args()
    if Path(args.attempt).name != args.attempt or args.attempt in {'.', '..'}:
        parser.error('attempt must be a single directory name')
    if not args.real:
        print('未指定 --real：不会调用接口。离线见证请运行 tests.benchmarks.test_amem_mab。')
        return 0

    from dotenv import load_dotenv
    import os
    from membase.datasets.official import Episode, Phase, Question, _messages
    from membase.datasets.memoryagentbench import DEFAULT_DATA_ROOT, DEFAULT_UPSTREAM
    from membase.inference_utils.model_client import RequestBudget
    from membase.runners.protocol import OfficialRunConfig, RunContext, run
    from membase.utils.benchmark_files import write_json
    load_dotenv(ROOT / 'envs/.env')
    validation_root = ROOT / 'experiments/mab_comparison_validation'
    ledger = validation_root / 'request_budget.sqlite'
    config = OfficialRunConfig(benchmark='memoryagentbench', baseline=args.method, mode='smoke',
                              data_root=DEFAULT_DATA_ROOT, upstream_dir=DEFAULT_UPSTREAM,
                              run_dir=validation_root / args.attempt, workers=1,
                              base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'),
                              max_llm_requests=20, max_embedding_requests=20, budget_ledger=ledger)
    messages, mapping = _messages([
        {'role': 'user', 'content': '0. Hotel Maple costs 200 dollars per night.'},
        {'role': 'user', 'content': '1. Breakfast at Hotel Maple starts at 8 AM.'},
    ], 'independent-validation', 0, None)
    q = Question('synthetic-price', 'What is the nightly price of Hotel Maple?', None,
                 {'answer': ['200'], 'source': 'factconsolidation_sh_6k'})
    sample = Episode('synthetic', 'independent-validation', (messages,), (Phase('final', 1, (q,)),),
                     {'update_priority': 'newer_source', 'control_roles': ['user'],
                      'description': 'Larger fact serial numbers override conflicting older facts.'}, {}, mapping)
    write_json(validation_root / 'synthetic_input.json', {'kind': 'synthetic_probe_not_MAB_test_results',
               'messages': messages, 'question': q.text, 'expected_answer': '200'})

    def budget_summary():
        budget = RequestBudget(20, 20, ledger)
        try:
            return budget.summary()
        finally:
            budget.close()

    report = {'kind': 'synthetic_transport_and_checkpoint_probe', 'method': args.method,
              'limits': {'llm': 20, 'embedding': 20}, 'before': budget_summary()}
    try:
        # 只替换输入为已明示的独立场景；模型、方法、评分与运行器均是真实组件。
        with patch.object(RunContext, 'samples', lambda self: iter([sample])):
            report['result'] = run(config)
            first = budget_summary()
            if report['result']['score'] != 1 or report['result'].get('technical_failure_questions') or report['result'].get('memory_warning_samples'):
                raise RuntimeError('短场景存在错误或回退，不能宣称验证完整通过')
            run(config)
            report['resume_without_new_requests'] = first == budget_summary()
            if not report['resume_without_new_requests']:
                raise RuntimeError('完整阶段复用产生了额外请求')
        report['status'] = 'passed'
    except Exception as error:
        report.update(status='incomplete', error_type=type(error).__name__)
        print(f'验证未完成：{type(error).__name__}；请查看本次运行日志，预算不扩额。')
    report['after'] = budget_summary()
    write_json(validation_root / f'{args.attempt}_report.json', report)
    print({'status': report['status'], 'llm_requests': report['after']['llm_requests'],
           'embedding_requests': report['after']['embedding_requests']})
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
