"""独立三跳读取验证：程序构造事实，模型只执行规划、绑定与回答，不使用考题。"""
from pathlib import Path
import argparse
import json
import os
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from membase.configs.ourmem import OurMemConfig
from membase.ourmem.models import Source,MemoryVersion,DependencyLink,PremiseRef,SourceSpan
from membase.ourmem.store import OurMemStore
from membase.ourmem.maintenance import MaintenanceEngine
from membase.ourmem.reader import MemoryReader
from membase.ourmem.retriever import RetrievalCandidate
from membase.ourmem.llm import ModelClient,RequestBudget
from membase.utils.benchmark_files import write_json

class LocalCandidates:
    def __init__(self, versions):
        self.versions=versions
        self._query_cache={};self.last_trace={}
    def retrieve(self,queries,*args,**kwargs):
        self.last_trace={'queries':queries,'returned':len(self.versions)}
        return [RetrievalCandidate(id=v.id,kind='memory',text=v.content,record=v) for v in self.versions]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--real',action='store_true')
    args=parser.parse_args()
    if not args.real:
        print('Plan only: seeded three-hop memory, at most 8 language-model requests, no embeddings.');return 0
    args.output_dir.mkdir(parents=True,exist_ok=False)
    cfg=OurMemConfig(model_name='gpt-4o-mini',answer_model='gpt-4o-mini',api_key=os.environ['OPENAI_API_KEY'],
                     base_url=os.environ.get('OPENAI_BASE_URL','https://api.openai.com/v1'))
    budget=RequestBudget(8,None,args.output_dir/'budget.sqlite')
    client=ModelClient(cfg,budget,log_path=args.output_dir/'requests.jsonl')
    store=OurMemStore(args.output_dir/'memory.sqlite','reader-probe')
    report={'scope':'independent synthetic three-hop interface probe; not benchmark accuracy','passed':False}
    try:
        versions=[]
        for i,text in enumerate(['The designer of Project Willow is Nora Ames.',
                'Nora Ames is supervised by Lena Moss.', 'Lena Moss has an office in Quito.',
                'Nora Ames has an office in Oslo.']):
            source=Source(namespace='reader-probe',message_id=str(i),conversation_id='work',speaker='user',role='user',source_order=i,content=text)
            store.add_sources([source]);v=MemoryVersion(namespace='reader-probe',content=text)
            ref=PremiseRef(type='SOURCE',id=source.id,span=SourceSpan(source_id=source.id,start=0,end=len(text)))
            store.commit([v],[DependencyLink(namespace='reader-probe',target_version_id=v.id,premise_refs=[ref])]);versions.append(v)
        reader=MemoryReader(store,LocalCandidates(versions),MaintenanceEngine(store),client,cfg)
        question='Where is the office of the supervisor of the designer of Project Willow?'
        result=reader.prepare(question,store.publish().id)
        response=client.text('Answer the question using only the supplied evidence, in a few words.\n'+question+'\n'+result.context,
                             stage='answer',model=cfg.answer_model,temperature=0,max_tokens=20)
        report.update(result=result.model_dump(mode='json'),answer=str(response),
                      passed='quito' in str(response).lower() and len(result.coverage['bindings'])>=3
                             and result.resolution_status=='resolved')
        return 0 if report['passed'] else 1
    finally:
        report['requests']=budget.summary();write_json(args.output_dir/'validation.json',report)
        print(json.dumps({'passed':report['passed'],'requests':report['requests']},ensure_ascii=False))
        store.close();client.close();budget.close()

if __name__=='__main__':raise SystemExit(main())
