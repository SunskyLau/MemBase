"""可靠性重构的小额验证；所有尝试共用 30/10 账本，默认只显示计划。"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real', action='store_true')
    parser.add_argument('--attempt', default=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    args = parser.parse_args()
    parent = ROOT / 'experiments/ourmem_validation/reliability'
    if not args.real:
        print('Synthetic scenarios only: source priority, derived memory, unrelated input, cascading update, read-only answer.')
        print('Shared limits: 30 language-model requests, 10 embedding requests, including retries.')
        print('Outputs:', parent)
        return 0
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a directory name')
    from membase.configs.ourmem import OurMemConfig
    from membase.ourmem.llm import ModelClient, RequestBudget, BudgetExceeded, ModelCallError
    from membase.ourmem.models import InputMessage, InputPolicy, MemoryVersion, PremiseRef, SourceSpan
    from membase.ourmem.extractor import FactDraft
    from membase.ourmem.reconciler import MemoryReconciler
    from membase.ourmem.system import OurMemSystem
    from membase.utils.benchmark_files import write_json
    from membase.utils.ourmem_version import ourmem_fingerprint

    folder = parent / args.attempt
    folder.mkdir(parents=True, exist_ok=False)
    config = OurMemConfig(api_key=os.environ['OPENAI_API_KEY'],
                          base_url=os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
    budget = RequestBudget(30, 10, parent / 'budget.sqlite')
    client = ModelClient(config, budget, log_path=folder / 'requests.jsonl')
    memory = OurMemSystem(config, folder / 'memory', client)
    result = {'scope':'Independent synthetic validation; not benchmark accuracy',
              'config':config.model_dump(mode='json'), 'implementation':ourmem_fingerprint(),
              'cases':[], 'complete':False, 'budget_before':budget.summary()}
    namespace = 'synthetic-trip'

    def state(name):
        snapshot = memory.flush(namespace)
        store = memory.get_store(namespace)
        from membase.ourmem.maintenance import MaintenanceEngine
        engine = MaintenanceEngine(store)
        depths = {v.id:engine.depth(v.id) for v in store.versions()}
        derived = [v for v in store.versions() if depths[v.id] > 0]
        case = {'name':name, 'snapshot_id':snapshot, 'versions':len(store.versions()),
                'derived':[{'id':v.id,'content':v.content,'depth':depths[v.id],
                            'resolution':engine.evaluate(v.id).model_dump(mode='json')} for v in derived],
                'unresolved':store.unresolved(), 'budget':budget.summary()}
        result['cases'].append(case)
        write_json(folder/'validation.json',result)
        print(name, 'versions',case['versions'],'derived',len(derived),'requests',budget.summary()['llm_requests'],flush=True)
        return snapshot

    try:
        # 输入规则核对不使用正式考题，也不需要额外嵌入。
        text='The capital of France is Osaka.'
        target=MemoryVersion(id='existing-capital',namespace='policy',memory_key='france-capital',content='The capital of France is Paris.')
        decision=MemoryReconciler(client,config).reconcile(
            FactDraft(content=text,source_id='later-source',evidence_refs=[PremiseRef(type='SOURCE',id='later-source',
                span=SourceSpan(source_id='later-source',start=0,end=len(text)))]),
            [target.model_dump(mode='json')], InputPolicy(update_priority='newer_source'),
            evidence_context={'sources':[{'id':'later-source','source_order':2,'content':text}]})
        result['cases'].append({'name':'counterfactual_priority','decision':decision.model_dump(mode='json'),
                                'passed':decision.action=='REVISE' and decision.revision_reason=='override'})
        texts=["For our trip, my father can walk at most 300 meters continuously.",
               "Hotel Alder's walk to the station is 200 meters.",
               "A hotel meets our trip's walking requirement if its station walk does not exceed my father's limit.",
               "Hotel Alder is on our trip shortlist exactly when it meets that walking requirement."]
        memory.ingest([InputMessage(message_id=f'initial-{i}',conversation_id='trip',speaker='Alex',content=t)
                       for i,t in enumerate(texts)],namespace,InputPolicy())
        state('initial')
        memory.ingest([InputMessage(message_id='unrelated',conversation_id='trip',speaker='Alex',
                                   content='My mother prefers mild food.')],namespace)
        state('unrelated')
        memory.ingest([InputMessage(message_id='correction',conversation_id='trip',speaker='Alex',
            content="I correct my earlier distance: Hotel Alder's station walk is actually 2000 meters, not 200 meters.")],namespace)
        snapshot=state('updated')
        answer=memory.answer("Does Hotel Alder now meet our trip's walking requirement?",namespace,snapshot)
        result['cases'].append({'name':'answer_after_update','answer':answer.model_dump(mode='json')})
        result['complete']=True
        return 0
    except (BudgetExceeded, ModelCallError) as error:
        result['stopped']={'type':type(error).__name__,'reason':str(error)}
        print('Validation stopped:',type(error).__name__,str(error),flush=True)
        return 1
    finally:
        result['budget_after']=budget.summary()
        write_json(folder/'validation.json',result)
        memory.close();client.close();budget.close()


if __name__=='__main__':
    raise SystemExit(main())
