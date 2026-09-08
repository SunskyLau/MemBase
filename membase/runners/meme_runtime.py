"""在固定官方 MEME 流程外补齐检查点、逐题终态和执行审计。"""
from __future__ import annotations

import ast
from copy import deepcopy
from functools import wraps
import importlib
import inspect
import json
import os
from pathlib import Path
import time

from ..inference_utils.model_client import RecoverableModelError, StructuredOutputError, failure_details
from ..utils.benchmark_files import read_json, write_json
from .meme_calls import EpisodeAudit, active, scope, digest, audited_client_factory

PROTOCOL = "meme-native-audit-v1"


def settings():
    return read_json(Path(os.environ["MEMBASE_MEME_SETTINGS"]))


def record_path(rt, phase, question):
    return rt.directory / phase / (digest(question) + ".json")


def state(agent):
    if hasattr(agent, "fs"):
        return {"type": "md_file", "files": deepcopy(agent.fs.files)}
    return {"type": "dense", "chunks": agent._chunks,
            "embeddings": None if agent._embeddings is None else agent._embeddings.tolist()}


def restore(agent, saved):
    if saved["type"] == "md_file":
        agent.fs.files = deepcopy(saved["files"])
    else:
        import numpy as np
        agent._chunks = saved["chunks"]
        agent._embeddings = None if saved["embeddings"] is None else np.asarray(saved["embeddings"], dtype=np.float32)


def feed_sessions(original, agent, sessions, begin, end):
    rt = active()
    phase = "before" if begin == 0 else "after"
    rt.phase = phase
    path = rt.directory / f"{phase}_state.json"
    saved = read_json(path) if path.exists() else None
    if saved:
        if saved["range"] != [begin, end]:
            raise ValueError("MEME observation boundary changed")
        restore(agent, saved["state"])
        start, logs = saved["next_index"], saved["logs"]
    else:
        start, logs = begin, []
    for index in range(start, end):
        # 单个会话提交检查点；before和after各自保存，不能加载未来状态回答过去问题。
        request_ids = []
        try:
            with scope(stage="ingest", phase=phase, session_index=index, request_ids=request_ids):
                entry = original(agent, sessions, index, index+1)[0]
        except RecoverableModelError as error:
            if not error.request_ids:
                error.request_ids = request_ids
            raise
        logs.append(entry)
        write_json(path, {"range": [begin,end], "next_index": index+1, "state": state(agent), "logs": logs})
        print(f"[{rt.binding['episode']}/{phase}] sessions {index+1}/{end}", flush=True)
    return logs


def answer_questions(agent, questions, client=None, model=None, *, transcript=None, ask=None, phase=None):
    rt = active()
    phase = phase or rt.phase
    snapshot = digest(transcript if transcript is not None else agent.get_memory_snapshot())
    results = []
    for q in questions:
        signature = {"question": q["question"], "snapshot": snapshot, "phase": phase}
        path = record_path(rt, phase, q)
        if path.exists():
            saved = read_json(path)
            if saved["binding"] != signature:
                raise ValueError("MEME question checkpoint does not match its observation state")
            results.append(saved["result"])
            continue
        started = time.monotonic()
        row = {**q, "agent_answer": None, "retrieved_context": "", "stage_times": {"retrieve_seconds":0.0}}
        rt.question_times = row["stage_times"]
        rt.retrieve_trajectory = []
        request_ids = []
        try:
            with scope(stage="answer", phase=phase, question_id=digest(q), request_ids=request_ids):
                if transcript is not None:
                    row["retrieved_context"] = transcript
                    row["agent_answer"] = ask(client, model, transcript, q["question"])
                else:
                    row["agent_answer"] = agent.answer_question(q["question"], client=client, model=model)
                    row["retrieved_context"] = agent.get_retrieved_context()
        except RecoverableModelError as error:
            if not error.request_ids:
                error.request_ids = request_ids
            if agent is not None:
                row["retrieved_context"] = agent.get_retrieved_context()
            row["technical_failure"] = {"stage": getattr(error, "failed_stage", "answer"),
                                         **failure_details(error), "assigned_score": 0}
        row["answer_time_sec"] = time.monotonic()-started
        row["stage_times"]["answer_seconds"] = row["answer_time_sec"] - row["stage_times"].get("retrieve_seconds", 0)
        row["retrieval_trajectory"] = rt.retrieve_trajectory
        row["context_sha256"] = digest(row["retrieved_context"])
        from ..utils.tokenization import count_tokens
        row["context_tokens"] = count_tokens(row["retrieved_context"])
        write_json(path, {"binding": signature, "result": row})
        results.append(row)
        print(f"[{rt.binding['episode']}/{phase}] questions {len(results)}/{len(questions)}; "
              f"{'technical failure' if row.get('technical_failure') else 'complete'}", flush=True)
    return results


def instrument_retrieve(original):
    @wraps(original)
    def retrieve(self, *args, **kwargs):
        rt = active()
        start = time.monotonic()
        self._last_retrieved_context = ""
        try:
            with scope(stage="retrieve"):
                return original(self, *args, **kwargs)
        except RecoverableModelError as error:
            error.failed_stage = "retrieval"
            raise
        finally:
            rt.question_times["retrieve_seconds"] = time.monotonic()-start
    return retrieve


def tool_loop(self, messages, tools=None):
    """保留原提示、工具和五轮预算；合法不更新不是故障。"""
    from agents.md_file import TOOLS, MAX_TOOL_ROUNDS
    tools = TOOLS if tools is None else tools
    allowed = {t["function"]["name"] for t in tools}
    trajectory, usage = [], {"input_tokens": 0, "output_tokens": 0}
    active().retrieve_trajectory = trajectory
    unresolved_error = False
    for index in range(MAX_TOOL_ROUNDS):
        response = self._internal_client.chat.completions.create(
            model=self._internal_model, messages=messages, tools=tools, temperature=0, max_tokens=2000)
        if response.usage:
            usage["input_tokens"] += getattr(response.usage, "prompt_tokens", 0) or 0
            usage["output_tokens"] += getattr(response.usage, "completion_tokens", 0) or 0
        msg = response.choices[0].message
        if not msg.tool_calls:
            if unresolved_error:
                raise StructuredOutputError("MD-flat ended with an unresolved tool error")
            if allowed == {"read_memory"} and self.fs.files and not any(
                    x["tool"] == "read_memory" and x["success"] for x in trajectory):
                raise StructuredOutputError("MD-flat returned retrieved facts without reading its memory")
            result = {"response": msg.content or "", "trajectory": trajectory,
                      "token_usage": usage, "termination": "completed" if trajectory else "no_op"}
            self._tool_termination = result["termination"]
            return result
        messages.append(msg)
        round_error, effective_action = False, False
        for call in msg.tool_calls:
            name, args = call.function.name, None
            try:
                args = json.loads(call.function.arguments)
                if not isinstance(args, dict) or name not in allowed:
                    raise ValueError("Tool or arguments not allowed")
                if name in {"write_memory", "append_memory"} and not isinstance(args.get("content"), str):
                    raise ValueError("Memory write needs string content")
                result = self.fs.execute_tool(name, args)
                # 参数已校验；空记忆的首次读取合法，原文以“Error:”开头也不是工具故障。
                success = True
            except (json.JSONDecodeError, ValueError) as error:
                result, success = f"Error: {error}", False
            round_error |= not success
            effective_action |= success and (allowed == {"read_memory"} or name in {"write_memory", "append_memory"})
            trajectory.append({"tool": name, "args": args, "result": result, "success": success})
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
        # 一次成功读取不能掩盖失败的写入；后续有效操作才算完成修正。
        unresolved_error = round_error or (unresolved_error and not effective_action)
    active().retrieve_trajectory = trajectory
    if allowed == {"read_memory"} or unresolved_error:
        raise StructuredOutputError("MD-flat exhausted tool rounds without a complete retrieval/result")
    # 摄入的有效工具操作已完成，保留官方有界执行结果；记录轮数耗尽而不扩充算法轮数。
    self._tool_termination = "round_limit"
    return {"response": "", "trajectory": trajectory, "token_usage": usage, "termination": "round_limit"}


def parse_judge(text, expected=None):
    text = text.strip()
    if text.startswith("```"):
        text = text.removeprefix("```").removesuffix("```").removeprefix("json").strip()
    try:
        result = json.loads(text)
    except ValueError as error:
        raise StructuredOutputError("Judge did not return valid JSON") from error
    if not isinstance(result, dict):
        raise StructuredOutputError("Judge result must be an object")
    if expected is None:
        if type(result.get("correct")) is not bool or not isinstance(result.get("reason"), str) or not result["reason"].strip():
            raise StructuredOutputError("Judge requires boolean correct and a nonempty reason")
    else:
        items = result.get("results")
        if not isinstance(items, list) or len(items) != expected or any(
                not isinstance(x, dict) or type(x.get("present")) is not bool for x in items):
            raise StructuredOutputError("Aggregation judge must return one boolean present per target")
    return result


def judged_question(original, multi=False):
    @wraps(original)
    def check(self, *args, **kwargs):
        values = inspect.signature(original).bind(self, *args, **kwargs)
        values.apply_defaults(); params = values.arguments
        rt = self._meme_audit; phase = params["phase"]
        q = rt.judge_questions[(phase, params["question"])]
        if q.get("technical_failure"):
            return {"u_pass": False, "u_reason": "technical_failure", "technical_failure": q["technical_failure"]}
        path = record_path(rt, "judge_"+phase, q)
        if path.exists():
            return read_json(path)
        start = time.monotonic()
        count = len(params["entity_values"]) if multi else None
        try:
            with scope(stage="judge", phase=phase, question_id=digest(q), validator=lambda t:parse_judge(t,count)):
                result = original(self, *args, **kwargs)
        except RecoverableModelError as error:
            result = {"u_pass": False, "u_reason": "technical_failure", "technical_failure": {
                "stage": "judge", **failure_details(error), "assigned_score": 0}}
        result["judge_time_sec"] = time.monotonic()-start
        write_json(path, result)
        return result
    return check


def make_audit(episode, output_dir, stage, config):
    from ..datasets.meme import episode_key
    opts = settings()
    key = episode_key(episode)
    directory = Path(output_dir).parent / "audit" / key / stage
    return EpisodeAudit(directory, {"episode": key, "input_sha256": digest(episode),
                                   "stage": stage, "config": config, "settings": opts}, opts)


def install(module_name):
    import openai
    # SDK默认重试和上游预算猴子补丁不叠加；本地持久账本是唯一成本来源。
    import eval.budget_tracker as tracker
    tracker.install_patches = lambda: None
    openai.OpenAI = audited_client_factory(openai.OpenAI)
    module = importlib.import_module(module_name)
    from agents.base import UNIFIED_ANSWER_PROMPT

    if module_name == "eval.run_agent":
        from agents.md_file import MDFlatMemory
        from agents.dense_memory import DenseMemory
        MDFlatMemory._run_tool_loop = tool_loop
        for cls in [MDFlatMemory, DenseMemory]:
            cls.retrieve = instrument_retrieve(cls.retrieve)
        original_ingest = MDFlatMemory.ingest_session
        def ingest(self, session):
            result = original_ingest(self, session)
            result["termination"] = self._tool_termination
            return result
        MDFlatMemory.ingest_session = ingest
        if os.environ.get("MEMBASE_EMBEDDING_MODEL"):
            from .native_transport import configure_meme_dense
            configure_meme_dense()
        original_feed = module.feed_sessions
        module.feed_sessions = lambda a,s,b,e: feed_sessions(original_feed,a,s,b,e)
        module.ask_questions = answer_questions
        original_process = module.process_one_episode
        @wraps(original_process)
        def process(ep_path, agent_type, model, api_key, output_dir, internal_model=None, neo4j_port=None, top_k=None):
            episode = read_json(Path(ep_path))
            with make_audit(episode, output_dir, "answers", {"agent":agent_type,"model":model,"internal":internal_model,"k":top_k}).activate() as rt:
                try:
                    result = original_process(ep_path,agent_type,model,api_key,output_dir,internal_model,neo4j_port,top_k)
                except RecoverableModelError as error:
                    # 基础构建不能计作成功；保存原因，让官方循环继续其他独立样本。
                    write_json(rt.directory / "construction_failure.json", failure_details(error))
                    print(f"[{rt.binding['episode']}] construction incomplete: {error}", flush=True)
                    return {"ep_id":episode["episode_id"], "elapsed":0, "incomplete":True}
                path = Path(result["out_path"]); output = read_json(path)
                output["instrumentation"] = {"protocol":PROTOCOL,"binding":rt.binding,"calls":rt.budget.summary()}
                # 上游内存计数只覆盖本次进程的成功请求，不再作为完整费用使用。
                output.pop("budget", None)
                output.pop("token_usage", None)
                write_json(path, output)
                return result
        module.process_one_episode = process
    elif module_name == "eval.in_context_baseline":
        def ask(client, model, transcript, question):
            response = client.chat.completions.create(model=model, messages=[{"role":"user", "content":
                UNIFIED_ANSWER_PROMPT.format(context=transcript,question=question)}],temperature=0,max_tokens=500)
            return response.choices[0].message.content.strip()
        original_process = module.process_one_episode
        @wraps(original_process)
        def process(ep_path, model, api_key, output_dir):
            episode = read_json(Path(ep_path))
            with make_audit(episode, output_dir, "answers", {"agent":"in_context","model":model}).activate() as rt:
                client = module._make_client(model,api_key)
                output = {"episode_id":episode["episode_id"],"domain":episode["domain"],"config":{
                    "agent_type":"in_context","agent_model":model,"internal_model":None}}
                for phase in ["before","after"]:
                    spec = episode[phase+"_questions"]
                    transcript = module._flatten_sessions(episode["sessions"],spec["position_after_session"]+1)
                    output[phase+"_answers"] = answer_questions(None,spec["questions"],client,model,
                        transcript=transcript,ask=ask,phase=phase)
                output["instrumentation"] = {"protocol":PROTOCOL,"binding":rt.binding,"calls":rt.budget.summary()}
                from ..evaluation.meme import output_name
                path = Path(output_dir)/output_name(episode,"in_context",model)
                write_json(path, output); client.close()
                return str(path)
        module.process_one_episode = process
    elif module_name == "eval.judge":
        original_init = module.LLMJudge.__init__
        def init(self, client, model="gpt-4o", max_retries=3):
            original_init(self,client,model,max_retries=1)
            self._meme_audit = active()
        module.LLMJudge.__init__ = init
        module.LLMJudge.u_check = judged_question(module.LLMJudge.u_check)
        module.LLMJudge.u_check_multi = judged_question(module.LLMJudge.u_check_multi,True)
        original_episode = module.judge_episode
        def judge_episode(output, judge, max_workers=8):
            rt = active()
            rt.judge_questions = {(phase,q["question"]):q for phase in ["before","after"] for q in output[phase+"_answers"]}
            result = original_episode(output,judge,max_workers)
            for row in result["before_answers"]:
                if row["task_type"].split(" (")[0] == "ER":
                    entity,value = next(iter(row["entity_values"].items()))
                    row.update(judge.u_check(row["question"],entity,value,row["agent_answer"],row["task_type"],phase="before"))
            result["totals"]["before_pass"] = sum(q["u_pass"] for q in result["before_answers"])
            return result
        module.judge_episode = judge_episode
        original_process = module.process_one_judge
        @wraps(original_process)
        def process(fp, api_key, judge_model, output_dir, check_workers):
            answer = read_json(Path(fp))
            with make_audit(answer,output_dir,"judge",{"model":judge_model}).activate() as rt:
                result = original_process(fp,api_key,judge_model,output_dir,check_workers)
                result["domain"] = answer["domain"]
                result["instrumentation"] = {"protocol":PROTOCOL,"binding":rt.binding,"calls":rt.budget.summary()}
                result.pop("judge_usage", None)
                from ..evaluation.meme import judge_name
                config = answer["config"]
                path = Path(output_dir)/judge_name(answer,config["agent_type"],config["agent_model"],judge_model)
                write_json(path,result)
                return result
        module.process_one_judge = process
    return module


def execute(module_name):
    module = install(module_name)
    if module_name == "eval.in_context_baseline":
        module.main()
    else:
        # 执行官方CLI主体，但保留上面已包装的函数，不另建一套样本调度器。
        tree = ast.parse(inspect.getsource(module))
        main = next(n for n in tree.body if isinstance(n,ast.If) and ast.unparse(n.test)=="__name__ == '__main__'")
        exec(compile(ast.Module(body=main.body,type_ignores=[]),module.__file__,"exec"),module.__dict__)
