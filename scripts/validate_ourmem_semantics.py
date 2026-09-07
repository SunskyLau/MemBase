"""小额在线组件验证：语义归并、明确纠错与派生；不调用嵌入，不跑基准数据。"""
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
    if not args.real:
        print('Component checks only; at most 6 language-model requests, including retries; no embeddings.')
        print('The previous 30/10 validation ledger is not modified. Use --real only for the separately authorized follow-up.')
        return 0
    if Path(args.attempt).name != args.attempt:
        parser.error('attempt must be a single directory name')
    from membase.configs.ourmem import OurMemConfig
    from membase.ourmem.llm import ModelClient, RequestBudget, BudgetExceeded, ModelCallError
    from membase.ourmem.models import Source, SourceSpan, PremiseRef, MemoryVersion, DependencyLink, InputPolicy
    from membase.ourmem.extractor import FactDraft
    from membase.ourmem.store import OurMemStore
    from membase.ourmem.maintenance import MaintenanceEngine
    from membase.ourmem.writer import MemoryWriter, StagedWrite
    from membase.utils.benchmark_files import write_json
    from membase.utils.ourmem_version import ourmem_fingerprint

    parent = ROOT / 'experiments/ourmem_validation/semantic_admission'
    folder = parent / args.attempt
    folder.mkdir(parents=True, exist_ok=False)
    cfg = OurMemConfig(api_key=os.environ['OPENAI_API_KEY'], base_url=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'))
    budget = RequestBudget(6, None, parent / 'budget.sqlite')
    client = ModelClient(cfg, budget, log_path=folder / 'requests.jsonl')
    store = OurMemStore(folder / 'memory.sqlite', 'synthetic')
    engine = MaintenanceEngine(store)
    writer = MemoryWriter(store, None, engine, client, cfg)
    report = {'scope':'Separately authorized six-call component validation; not end-to-end or benchmark accuracy',
              'implementation':ourmem_fingerprint(), 'cases':[], 'complete':False, 'budget_before':budget.summary()}

    def source(text, order):
        s=Source(namespace='synthetic',message_id=f'm{order}',conversation_id='trip',speaker='Alex',
                 role='user',source_order=order,content=text)
        store.add_sources([s]);return s

    def ref(s):
        return PremiseRef(type='SOURCE',id=s.id,span=SourceSpan(source_id=s.id,start=0,end=len(s.content)))

    def seed(s):
        v=MemoryVersion(namespace='synthetic',content=s.content)
        store.commit([v],[DependencyLink(namespace='synthetic',target_version_id=v.id,premise_refs=[ref(s)])])
        return v

    def record(case):
        report['cases'].append(case)
        write_json(folder/'validation.json',{**report,'budget_after':budget.summary()})
        print(case['name'], 'passed=',case.get('passed'), 'requests=',budget.summary()['llm_requests'],flush=True)

    try:
        limit=seed(source("Alex's father can walk at most 300 meters continuously.",0))
        distance=seed(source('The walk from Hotel Alder to the station is 200 meters.',1))
        rule=source('Hotel Alder is on our trip shortlist exactly when it meets the walking requirement.',2)
        draft=FactDraft(content=rule.content,source_id=rule.id,evidence_refs=[ref(rule)])
        context=writer._context([rule.id,limit.id,distance.id],2,ref(rule).span)
        decision=writer.reconciler.reconcile(draft,context['versions'],InputPolicy(),evidence_context=context)
        staged=writer._materialize(draft,decision,2,'rule',StagedWrite())
        store.commit(staged.versions,staged.dependencies,staged.operations,source_cutoff=2)
        passed=decision.action=='ADD' and engine.evaluate(distance.id).usable
        record({'name':'rule_coexists_with_distance','decision':decision.model_dump(mode='json'),
                'old_distance_usable':engine.evaluate(distance.id).usable,'passed':passed})
        if not passed:
            return 1

        correction=source("I correct my earlier distance: Hotel Alder's station walk is actually 2000 meters, not 200 meters.",3)
        draft=FactDraft(content=correction.content,source_id=correction.id,evidence_refs=[ref(correction)])
        context=writer._context([correction.id,*[v.id for v in store.versions()]],3,ref(correction).span)
        decision=writer.reconciler.reconcile(draft,context['versions'],InputPolicy(),evidence_context=context)
        staged=writer._materialize(draft,decision,3,'correction',StagedWrite())
        store.commit(staged.versions,staged.dependencies,staged.operations,source_cutoff=3)
        passed=decision.action=='REVISE' and decision.revision_reason=='correction' and not engine.evaluate(distance.id).usable
        record({'name':'explicit_distance_correction','decision':decision.model_dump(mode='json'),'passed':passed})
        if not passed:
            return 1

        context=writer._context([v.id for v in store.versions()],3,ref(correction).span)
        context['input_policy']=InputPolicy().model_dump(mode='json')
        context['trigger_ids']=[staged.target_id]
        graph=writer.inducer.propose(context,[])
        record({'name':'discovery','proposal':graph.model_dump(mode='json'),'passed':bool(graph.claims)})
        if not graph.claims:
            return 1
        claim=graph.claims[0]
        dep=next(d for d in graph.dependencies if d.target_id==claim.temporary_id and d.effect=='SUPPORT')
        if all(r.type=='SOURCE' for r in dep.premise_refs):
            bound=writer._bind_atomic_premises(dep.premise_refs,3)
            if not bound:
                report['uncovered']='Source-only path could not be bound to existing atomic facts.'
                return 1
            dep=dep.model_copy(update={'premise_refs':bound})
        if any(r.type!='SOURCE' and r.id not in {v.id for v in store.versions()} for r in dep.premise_refs):
            report['uncovered']='First path needs additional proposed intermediates; six-call component probe does not expand them.'
            return 1
        writer._check_premises(dep.premise_refs,context,3,ref(correction).span,StagedWrite())
        verified=writer.inducer.verify(dep,{**context,'target':claim.model_dump(mode='json')})
        passed=verified.accepted and verified.sufficient_paths == [list(range(len(dep.premise_refs)))]
        record({'name':'derived_support_verification','result':verified.model_dump(mode='json'),'passed':passed})
        report['complete']=passed
        return 0 if report['complete'] else 1
    except (BudgetExceeded,ModelCallError,ValueError) as error:
        report['stopped']={'type':type(error).__name__,'reason':str(error)}
        return 1
    finally:
        report['budget_after']=budget.summary()
        write_json(folder/'validation.json',report)
        store.close();client.close();budget.close()


if __name__=='__main__':
    raise SystemExit(main())
