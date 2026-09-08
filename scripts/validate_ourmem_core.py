"""十个核心场景，无网络、无完整测试套件；为日常快速迭代保留单一入口。"""
from pathlib import Path
from tempfile import TemporaryDirectory
import sys
import unittest
import re

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from membase.configs.ourmem import OurMemConfig
from membase.ourmem.models import (Source, MemoryVersion, DependencyLink, PremiseRef, SourceSpan,
                                   Revision, TimePoint, TimeScope, InputPolicy, ControlOperation, PreparedContext)
from membase.ourmem.store import OurMemStore
from membase.ourmem.maintenance import MaintenanceEngine
from membase.ourmem.retriever import MemoryCandidateRetriever
from membase.ourmem.reader import MemoryReader
from membase.ourmem.writer import MemoryWriter
from membase.ourmem.extractor import FactExtractor, FactDraft
from membase.ourmem.inducer import LocalGraphProposal, ClaimProposal, DependencyProposal
from membase.ourmem.reconciler import MemoryReconciler, RelationJudgment


class Model:
    def __init__(self, handler=None):
        self.handler, self.calls = handler, []
        self.vocabulary = {}
    def count_tokens(self, text):
        return max(1, len(text) // 4)
    def embed(self, texts):
        vectors = []
        for text in texts:
            vector = [0.0] * 1024
            for word in re.findall(r"\w+", text.lower()):
                index = self.vocabulary.setdefault(word, len(self.vocabulary))
                vector[index] += 1
            vectors.append(vector)
        return vectors
    def request_json(self, stage, prompt, payload, validator=None, **kwargs):
        self.calls.append((stage, payload))
        if self.handler:
            value = self.handler(stage, payload)
        elif stage in {"reconcile", "reconcile_check"}:
            value = {"relation": "INDEPENDENT", "target_id": None, "reason": "Distinct proposition"}
        elif stage == "verify":
            value = {"accepted": True, "reason": "The supplied joint premises support the consequence"}
        else:
            raise AssertionError(f"Unexpected model stage: {stage}")
        return validator(value) if validator else value


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = OurMemStore(Path(self.temp.name) / "memory.sqlite", "test")
        self.config = OurMemConfig()
        self.model = Model()
        self.engine = MaintenanceEngine(self.store)
        self.retriever = MemoryCandidateRetriever(self.store, self.model.embed, self.config, maintenance=self.engine,
                                                  token_counter=self.model.count_tokens)
        self.writer = MemoryWriter(self.store, self.retriever, self.engine, self.model, self.config)
    def tearDown(self):
        self.store.close()
        self.temp.cleanup()
    def source(self, text, order):
        source = Source(namespace="test", message_id=str(order), conversation_id="chat", speaker="Alex", role="user",
                        source_order=order, content=text)
        self.store.add_sources([source])
        return source
    def ref(self, source):
        return PremiseRef(type="SOURCE", id=source.id,
                          span=SourceSpan(source_id=source.id, start=0, end=len(source.content)))
    def fact(self, text, order, previous=None, reason="update"):
        source = self.source(text, order)
        ref = self.ref(source)
        revision = Revision(previous_version_id=previous.id, reason=reason, effective_time=TimePoint(order=order),
                            evidence_refs=[ref]) if previous else None
        version = MemoryVersion(namespace="test", content=text, valid_time=TimeScope(kind="state"), revision=revision,
                                **({"memory_key": previous.memory_key} if previous else {}))
        self.store.commit([version], [DependencyLink(namespace="test", target_version_id=version.id, premise_refs=[ref])])
        return version, source
    def claim(self, text, parents):
        version = MemoryVersion(namespace="test", content=text)
        refs = [PremiseRef(type="CURRENT", id=v.id) for v in parents]
        self.store.commit([version], [DependencyLink(namespace="test", target_version_id=version.id, premise_refs=refs)])
        return version

    def test_01_grouped_hybrid_and_no_query_model(self):
        old,_ = self.fact("The office is in Delta.", 0)
        new,_ = self.fact("The office is in Harbor.", 1, old)
        self.fact("The lunch is soup.", 2)
        self.source("An unextracted private remark.", 3)
        snap = self.store.publish()
        reader = MemoryReader(self.store, self.retriever, self.engine, self.model, self.config)
        result = reader.prepare("Where is the office?", snap.id, top_k=1)
        self.assertEqual(result.coverage["selected_memory_keys"], [new.memory_key])
        self.assertIn("Harbor", result.context)
        self.assertIn("HISTORICAL", result.context)
        self.assertNotIn("unextracted private", result.context)
        self.assertEqual(self.model.calls, [])

    def test_02_correction_and_delete_cannot_return_as_history(self):
        old,_ = self.fact("The vault code is 111.", 0)
        new,_ = self.fact("The vault code is 222.", 1, old, reason="correction")
        self.store.commit(operations=[ControlOperation(namespace="test", kind="delete", scope="version", target_id=new.id,
                                                     source_cutoff=1, topic="vault code")])
        groups = self.retriever.search_memories("vault code", self.store.publish())
        self.assertEqual(groups, [])

    def test_03_shared_proof_dedup_and_budget(self):
        a,_ = self.fact("The project lead is Mira.", 0)
        b,_ = self.fact("Mira works in Harbor Lab.", 1)
        c = self.claim("The project lead works in Harbor Lab.", [a,b])
        d = self.claim("The project has a lead associated with Harbor Lab.", [c,b])
        proof = self.engine.evidence(d.id)
        from membase.ourmem.evidence_format import render_evidence
        text = render_evidence([proof, proof], self.store.view(), self.engine._evaluation())
        self.assertEqual(text.count("[source order=0;"), 1)
        self.assertEqual(text.count("[source order=1;"), 1)
        self.config.max_evidence_tokens = 20
        result = MemoryReader(self.store,self.retriever,self.engine,self.model,self.config).prepare("project lead",self.store.publish().id)
        self.assertEqual(result.coverage["selected_memory_keys"], [])

    def test_04_independent_support_survives(self):
        a,_ = self.fact("Permission was granted by the manager.",0)
        b,_ = self.fact("Permission was granted by the owner.",1)
        c = self.claim("Permission is granted.",[a])
        self.store.commit(dependencies=[DependencyLink(namespace="test",target_version_id=c.id,
            premise_refs=[PremiseRef(type="CURRENT",id=b.id)])])
        self.store.commit(operations=[ControlOperation(namespace="test",kind="delete",scope="version",target_id=a.id,source_cutoff=1)])
        self.assertTrue(self.engine.evaluate(c.id).usable)
        self.assertNotIn("manager",self.engine.evidence(c.id).text)

    def test_05_inconsistent_identity_cannot_silently_become_add(self):
        old,_ = self.fact("Team North plays volleyball.",0)
        source = self.source("Team South plays volleyball.",1)
        # 真实失败形状：主判定说同一属性冲突，解释字段却把“值不同”当成“属性不同”。
        with self.assertRaises(ValueError):
            RelationJudgment.model_validate({"relation":"CONFLICTING","reason":"Different values of the same property",
                "same_subject":True,"same_attribute":False,"same_scope":True,"values_compatible":False})
        RelationJudgment.model_validate({"relation":"CONFLICTING","reason":"Same property, different values"})
        model = Model(lambda stage,payload: {"relation":"INDEPENDENT","reason":"Different teams",
            "same_subject":False,"same_attribute":True,"same_scope":True,"values_compatible":True})
        draft = FactDraft(content=source.content,source_id=source.id,evidence_refs=[self.ref(source)])
        decision = MemoryReconciler(model,self.config).reconcile(draft,[old.model_dump(mode="json")],InputPolicy(update_priority="newer_source"))
        self.assertEqual(decision.action,"ADD")
        self.assertTrue(self.engine.evaluate(old.id).usable)
        def mistaken_match(stage, payload):
            if stage == "reconcile_check":
                return {"relation":"INDEPENDENT", "reason":"The actual candidate has a different subject",
                        "same_subject":False,"same_attribute":True,"same_scope":True,"values_compatible":True}
            return {"relation":"EQUIVALENT","target_id":"c0","reason":"Mistook the new text for the candidate",
                    "same_subject":True,"same_attribute":True,"same_scope":True,"values_compatible":True}
        decision = MemoryReconciler(Model(mistaken_match),self.config).reconcile(
            draft,[old.model_dump(mode="json")],InputPolicy(),identity_recheck=True)
        self.assertEqual(decision.action,"ADD")
        def exhausted(stage, payload):
            if stage == "reconcile_check":
                from membase.ourmem.llm import StructuredOutputError
                raise StructuredOutputError("Inconsistent relation after bounded correction")
            return {"relation":"CONFLICTING","target_id":"c0","reason":"A provisional same-property conflict"}
        decision = MemoryReconciler(Model(exhausted),self.config).reconcile(
            draft,[old.model_dump(mode="json")],InputPolicy())
        self.assertEqual(decision.action,"DEFER")

    def test_06_new_identity_checked_even_same_candidates(self):
        old,_ = self.fact("The lab director is Mira.",0)
        source = self.source("The lab director is Lena.",1)
        def handler(stage,payload):
            if stage == "reconcile_check":
                return {"relation":"CONFLICTING","reason":"same director property, new value",
                        "same_subject":True,"same_attribute":True,"same_scope":True,"values_compatible":False}
            if not payload.get("identity_recheck"):
                return {"relation":"INDEPENDENT","reason":"first-pass mistake"}
            candidate = next(c for c in payload["candidates"] if c["id"]==old.id)
            return {"relation":"CONFLICTING","target_id":candidate["candidate_ref"],"reason":"new value",
                    "same_subject":True,"same_attribute":True,"same_scope":True,"values_compatible":False}
        self.model.handler = handler
        draft = FactDraft(content=source.content,source_id=source.id,evidence_refs=[self.ref(source)],valid_time=TimeScope(kind="state"))
        stage, decision = self.writer._coordinate(draft,InputPolicy(update_priority="newer_source"),1,self.ref(source).span,"test")
        self.store.commit(stage.versions,stage.dependencies,stage.operations)
        self.assertEqual(decision.action,"REVISE")
        self.assertFalse(self.engine.evaluate(old.id).usable)
        self.assertEqual(len(self.model.calls),3)
        recheck = next(payload for stage,payload in self.model.calls if payload.get("identity_recheck"))
        self.assertEqual(recheck["evidence_context"], {})
        self.assertNotIn("evidence_refs", recheck["proposed_content"])

    def test_07_local_quote_repair_preserves_other_fact(self):
        first = self.source("Mira lives in Delta.",0)
        second = self.source("The univeristy is Harbor College.",1)
        def handler(stage,payload):
            if stage=="extract_quote":
                return {"repairs":[{"index":1,"quote":second.content}]}
            return {"facts":[{"source_id":first.id,"quote":first.content,"content":first.content},
                             {"source_id":second.id,"quote":"The university is Harbor College.","content":"The university is Harbor College."}]}
        result = FactExtractor(Model(handler),self.config).extract([first,second],[],InputPolicy())
        self.assertEqual(len(result.drafts),2)
        self.assertFalse(result.unresolved)
        self.assertEqual(result.drafts[1].evidence_refs[0].span.end,len(second.content))

    def test_08_duplicate_alias_and_joint_derivation(self):
        a,_ = self.fact("The project lead is Mira.",0)
        b,source = self.fact("Mira works in Harbor Lab.",1)
        graph = LocalGraphProposal(claims=[ClaimProposal(temporary_id="copy",content=a.content),
            ClaimProposal(temporary_id="new",content="The project lead works in Harbor Lab.")],dependencies=[
                DependencyProposal(temporary_id="d0",target_id="copy",effect="SUPPORT",premise_refs=[PremiseRef(type="CURRENT",id=a.id)]),
                DependencyProposal(temporary_id="d1",target_id="new",effect="SUPPORT",premise_refs=[PremiseRef(type="CURRENT",id="copy"),PremiseRef(type="CURRENT",id=b.id)])])
        context = self.writer._context([a.id,b.id],1,self.ref(source).span)
        changed,failed = self.writer._commit_graph(graph,context,InputPolicy(),1,self.ref(source).span,"graph-test")
        self.assertFalse(failed)
        self.assertEqual(len(self.store.versions()),3)
        count = len(self.model.calls)
        self.writer._commit_graph(graph,context,InputPolicy(),1,self.ref(source).span,"graph-test")
        self.assertEqual(len(self.model.calls),count)
        verify = next(payload for stage,payload in self.model.calls if stage=="verify")
        self.assertEqual(len(verify["premise_evidence"]),2)

    def test_09_five_layers_cascade_and_sixth_rejected(self):
        root,_ = self.fact("The switch is on.",0)
        last = root
        for i in range(5):
            last = self.claim(f"Consequence {i} holds.",[last])
        with self.assertRaises(ValueError):
            self.claim("Beyond five layers.",[last])
        self.fact("The switch is off.",1,root)
        self.assertFalse(self.engine.evaluate(last.id).usable)
        self.assertIn(last.id,self.engine.recompute([root.id]).pending_ids)

    def test_10_source_mapping_and_public_k(self):
        fact,source = self.fact("The office is in Harbor.",0)
        context = self.writer._context([source.id],0,self.ref(source).span)
        self.assertIn(fact.id,context["version_ids"])
        from membase.layers.ourmem import OurMemLayer
        from types import SimpleNamespace
        captured = {}
        def prepare(*args,**kwargs):
            captured.update(kwargs)
            return PreparedContext(context="evidence",resolution_status="not_assessed")
        layer = OurMemLayer.__new__(OurMemLayer)
        layer.config = self.config
        layer.system = SimpleNamespace(prepare_evidence=prepare)
        layer.retrieve("office",k=7,snapshot_id="snapshot")
        self.assertEqual(captured["top_k"],7)


if __name__=="__main__":
    unittest.main(verbosity=1)
