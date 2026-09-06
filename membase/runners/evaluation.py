import json
import os
import inspect
from string import Template
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
)
from smartcomment import (
    comment_graph,
    comment_session,
    comment_op,
    comment_variable,
    comment_op_scope, 
    enable_tracing,
    disable_tracing, 
    is_tracing_enabled, 
)
from smartcomment.runtime import ExecNetwork
from ..datasets import DATASET_MAPPING
from ..inference_utils.operators import QuestionAnsweringOperator
from ..model_types.dataset import QuestionAnswerPair, MemoryDataset
from ..model_types.memory import MemoryEntry
from typing import Any, Callable


def answer_question(config, sample, question, retrieval, client):
    """在线观察点与普通评测共用同一个问答算子，官方消息不加包装。"""
    import time
    from ..evaluation.official import answer_prompt, answer_system, validate_answers
    from ..inference_utils.backends import BoundedModelInterface
    from .protocol import digest
    prompt, limit = answer_prompt(config.benchmark, config.upstream_dir, sample, question, retrieval["context"])
    system = answer_system(config.benchmark, config.upstream_dir)
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    operator = QuestionAnsweringOperator("default-question-answering", model_name=config.answer_model,
                                        interface=BoundedModelInterface(client, model=config.answer_model),
                                        message_builder=lambda query, context: messages)
    started = time.monotonic()
    response = operator([question.text], [retrieval["context"]], aggregate=False, batch_size=1,
                        temperature=config.temperature if config.benchmark == "memoryagentbench" else 0,
                        max_tokens=limit, allow_truncated=True)[0]
    record = {**retrieval, "retrieval_sha256": digest(retrieval), "answer_text": response["processed_content"],
              "answer_finish_reason": response["finish_reason"], "answer_request_id": response["request_id"],
              "answer_seconds": time.monotonic() - started}
    if response["finish_reason"] == "length":
        record.update(resolution_status="incomplete", reason="answer_output_truncated")
    validate_answers([record], (question,))
    return record


def grade_question(runtime, sample, phase, question, snapshot_id, client, scorer):
    """一次只隔离当前题的模型故障；损坏产物、预算和程序错误仍向上传播。"""
    from .protocol import checked, question_file, digest
    from .search import validate_retrieval
    from .question_outcomes import read_failure, save_failure, failure_result, stored_failure_result
    from ..evaluation.official import validate_answers, validate_scores
    from ..inference_utils.model_client import RecoverableModelError
    from ..utils.benchmark_files import write_json
    cfg, folder = runtime.config, runtime.directory(sample) / phase.name
    failure = read_failure(runtime, sample, phase, question, snapshot_id)
    if failure is not None:
        return stored_failure_result(runtime, sample, phase, question, failure)
    identity = {"question_id": question.id, "question": question.text, "snapshot_id": snapshot_id}
    retrieval = checked(question_file(folder / "retrievals", question.id), identity, "检索结果")
    validate_retrieval(retrieval)
    path = question_file(folder / "answers", question.id)
    if path.exists():
        record = checked(path, {**identity, "retrieval_sha256": digest(retrieval)}, "回答")
        validate_answers([record], (question,))
    else:
        try:
            record = answer_question(cfg, sample, question, retrieval, client)
        except RecoverableModelError as error:
            return failure_result(save_failure(runtime, sample, phase, question, snapshot_id, "answer", error, payload=retrieval))
        write_json(path, record)
    path = question_file(folder / "scores", question.id)
    identity = {"answer_sha256": digest(record), "judge_model": cfg.judge_model}
    if path.exists():
        score = checked(path, identity, "评分")
    else:
        try:
            scores = runtime.dataset_cls.evaluate([question.as_pair()], [record["answer_text"]],
                                                 protocol="official", official_scorer=scorer)[0]
        except RecoverableModelError as error:
            return failure_result(save_failure(runtime, sample, phase, question, snapshot_id, "judge", error, payload=record), record)
        score = {**identity, "scores": scores}
    validate_scores(cfg.benchmark, score["scores"], locomo_judge=cfg.locomo_judge)
    write_json(path, score)
    return {**record, "scores": score["scores"], "score_origin": "official"}


def evaluate_memory(
    retrievals: list[dict[str, Any]],
    qa_model: str,
    judge_model: str,
    dataset_cls: type[MemoryDataset],
    qa_batch_size: int = 4,
    judge_batch_size: int = 4,
    add_question_timestamp: bool = False,
    prompt_template: Callable[[], Template] | None = None,
    context_builder: Callable[[list[MemoryEntry]], str] | None = None,
    interface_kwargs: dict[str, Any] | None = None,
    user_id: str | None = None,
    metrics: list[str] | None = None,
    metric_configs: dict[str, dict[str, Any]] | None = None,
    traced_data_save_dir: str | None = None,
) -> list[dict[str, Any]]:
    """Given a list of retrieval results, evaluate the memory layer.

    If you want to trace the evaluation process, you need to provide 
    both the user identifier and the traced data save directory.

    Args:
        retrievals (`list[dict[str, Any]]`):
            The retrieval results produced by the search runner.
        qa_model (`str`):
            Model name for question answering.
        judge_model (`str`):
            Model name for judgment.
        dataset_cls (`type`):
            Dataset class that provides the logic to evaluate the memory layer.
        qa_batch_size (`int`, defaults to `4`):
            Batch size for question-answering.
        judge_batch_size (`int`, defaults to `4`):
            Batch size for judgment.
        add_question_timestamp (`bool`, defaults to `False`):
            Whether to append the question timestamp to the prompt.
        prompt_template (`Callable[[], Template] | None`, optional):
            A factory that returns a ``string.Template`` with
            `$question` and `$context` placeholders.
        context_builder (`Callable[[list[MemoryEntry]], str] | None`, optional):
            A callable that converts memory entries into a context string.
        interface_kwargs (`dict[str, Any] | None`, optional):
            Extra keyword arguments for the LLM operator.
        user_id (`str | None`, optional):
            Unique identifier of the user. If not provided, the graph importing and 
            exporting are skipped.
        metrics (`list[str] | None`, optional):
            Metric names to compute. If not provided, the default metrics will be used.
        metric_configs (`dict[str, dict[str, Any]] | None`, optional):
            Per-metric configuration overrides keyed by metric name.
        traced_data_save_dir (`str | None`, optional):
            Directory where execution graph artefacts are saved. If not provided,
            the graph importing and exporting are skipped.

    Returns:
        `list[dict[str, Any]]`:
            Per-query evaluation results.
    """
    interface_kwargs = interface_kwargs or {}

    if context_builder is None:
        context_builder = lambda memories: "\n\n".join(
            f"### Memory {i + 1}:\n{mem.formatted_content or mem.content}"
            for i, mem in enumerate(memories)
        )

    imported_graph = None
    if traced_data_save_dir is not None and user_id is not None:
        traced_data_path = os.path.join(
            traced_data_save_dir,
            user_id,
            "graph_search.json",
        )
        if os.path.exists(traced_data_path):
            with open(traced_data_path, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
            imported_graph = ExecNetwork.import_graph(graph_data)
        else:
            print(
                f"The execution graph for user '{user_id}' is not found "
                f"in the path '{traced_data_path}'."
            )


    final_results = []
    with comment_graph(graph=imported_graph) as graph:
        with comment_session(
            category="memory_evaluation",
            comment=(
                "Evaluate the memory layer by checking whether "
                "the question-answering model can generate the correct answer "
                "based on the retrieved memories." 
            ),
            metadata={
                "qa_model": qa_model,
                "judge_model": judge_model,
                "qa_batch_size": qa_batch_size,
                "judge_batch_size": judge_batch_size,
                "add_question_timestamp": add_question_timestamp,
            },
        ):
            questions = []
            contexts = []

            for item in retrievals:
                qa_pair = item["qa_pair"]
                question = qa_pair.question
                if "name" in qa_pair.metadata:
                    question = f"{qa_pair.metadata['name']}: {question}"
                if add_question_timestamp:
                    question = f"{question}\nQuestion Timestamp: {qa_pair.timestamp}"
                questions.append(question)

                context = context_builder(item["retrieved_memories"])
                contexts.append(context)


            qa_operator = QuestionAnsweringOperator(
                prompt_name="default-question-answering",
                model_name=qa_model,
                timeout=120.0,
                **interface_kwargs,
            )
            if prompt_template is not None:
                qa_operator.set_prompt(prompt_template())

            runtime_qa_template = comment_variable(
                qa_operator.prompt.template,
                to_runtime=True,
                id_strategy=lambda v: "question-answering-prompt",
                comment=(
                    "The prompt template for the question-answering model. "
                    "It is a `string.Template` object with `$question` and `$context` placeholders. "
                    "It tells the question-answering model to generate an answer based on " 
                    "the question and the context retrieved from the memory system."
                ),
                category="prompt",
                metadata={
                    "op_type": "question-answering",  
                }
            )

            qa_responses = qa_operator(
                questions,
                contexts,
                batch_size=qa_batch_size,
                aggregate=False,
                temperature=0.0,
            )

            predictions = []
            for idx, resp in enumerate(qa_responses):
                pred = resp.get("processed_content")
                if pred is None:
                    raise ValueError(
                        "The question-answering model returns an empty prediction."
                    )
                predictions.append(pred)


                # Construct the graph based on the related and stored variables. 
                item = retrievals[idx]
                question = (
                    questions[idx], 
                    {
                        "class_name": "query", 
                        "id_strategy": lambda _: item["qa_pair"].id,
                    }
                )
                context = (
                    contexts[idx], 
                    {
                        "class_name": "context", 
                        "category": "memory_context",
                        "comment": "The formatted memory context.",
                    }
                )
                pred = (
                    pred, 
                    {
                        "class_name": "prediction", 
                        "category": "llm_response",
                        "comment": "The model's response to the question.",
                    }
                )

                with comment_op_scope(
                    op_name="question-answering",
                    category="evaluation",
                    comment=(
                        "The question-answering model generates an answer based on "
                        "the question, the context retrieved from the memory system, "
                        "and a question-answering prompt."
                    ),
                ):
                    input_memories = [] 
                    for memory in item["retrieved_memories"]:
                        if is_tracing_enabled():
                            assert "trace_id" in memory.metadata, (
                                "The memory metadata must contain an 'trace_id' field. "
                                "Please check the memory construction process."
                            )
                        input_memories.append(
                            (
                                # A very lightweight representation of the memory.
                                {"id": memory.metadata.get("trace_id")}, 
                                {
                                    "id_strategy": lambda v: v["id"], 
                                    "identity_only": True,  # We don't need snapshot consistency check here.
                                },
                            )
                        )
                    
                    comment_op(
                        inputs=input_memories,
                        outputs=[context],
                        metadata={
                            "source_code": inspect.getsource(context_builder),
                        }, 
                        comment=(
                            "The formatted context is constructed from the retrieved memories. "
                            "Depending on the memory system implementation, the resulting context "
                            "may include only selected portions of those memories rather than the "
                            "full content of every retrieved memory."
                        ),
                        reuse_op=True,
                    )
                    comment_op(
                        inputs=[question, context, runtime_qa_template],
                        outputs=[pred],
                        comment=(
                            "The question-answering model generates an answer based on "
                            "the question, the context retrieved from the memory system, "
                            "and a question-answering prompt."
                        ),
                        reuse_op=True,
                    )
                
            qa_pairs = [item["qa_pair"] for item in retrievals]
            judge_results = dataset_cls.evaluate(
                qa_pairs=qa_pairs,
                predictions=predictions,
                metrics=metrics,
                metric_configs=metric_configs,
                judge_model=judge_model,
                judge_batch_size=judge_batch_size,
                **interface_kwargs,
            )

            # Assemble final outputs.
            for i, item in enumerate(retrievals):
                qa_pair = item["qa_pair"]
                final_results.append(
                    {
                        "qa_pair": qa_pair.model_dump(mode="python"),
                        "prediction": predictions[i],
                        "metrics": judge_results[i],
                        "retrieved_memories": [
                            mem.model_dump(mode="python")
                            for mem in item["retrieved_memories"]
                        ],
                        "user_id": item["user_id"],
                    }
                )


    if graph is not None and traced_data_save_dir is not None:
        graph_data = graph.export_graph()
        traced_data_path = os.path.join(
            traced_data_save_dir,
            user_id,
            "graph_evaluation.json",
        )
        os.makedirs(
            os.path.dirname(traced_data_path), 
            exist_ok=True
        )
        with open(
            traced_data_path, 
            "w", 
            encoding="utf-8",
        ) as f:
            json.dump(
                graph_data, 
                f, 
                indent=4, 
                ensure_ascii=False,
            )

    return final_results


class EvaluationRunnerConfig(BaseModel):
    """Configuration for the evaluation runner."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    search_results_path: str = Field(
        ...,
        description="Path to the search results.",
    )
    dataset_type: str = Field(
        ...,
        description="The type of the dataset used to evaluate the memory layer.",
    )
    qa_model: str = Field(
        default="gpt-4.1-mini",
        description="Model name or path for question answering.",
    )
    judge_model: str = Field(
        default="gpt-4.1-mini",
        description="Model name or path for judgment.",
    )
    qa_batch_size: int = Field(
        default=4,
        description="Batch size for question-answering.",
    )
    judge_batch_size: int = Field(
        default=4,
        description="Batch size for judgment.",
    )
    api_config_path: str | None = Field(
        default=None,
        description="Path to the API config file.",
    )
    api_keys: list[str] | None = Field(
        default=None,
        description=(
            "API keys for the LLM operator. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    base_urls: list[str] | None = Field(
        default=None,
        description=(
            "Base URLs for the LLM operator. "
            "If provided, they take precedence over ``api_config_path``."
        ),
    )
    context_builder: Callable[[list[MemoryEntry]], str] | None = Field(
        default=None,
        description=(
            "A callable that converts a list of memory entries into a context string."
        ),
    )
    prompt_template: Callable[[], Template] | None = Field(
        default=None,
        description=(
            "A factory that returns a ``string.Template`` with "
            "``$question`` and ``$context`` placeholders."
        ),
    )
    add_question_timestamp: bool = Field(
        default=False,
        description="Append the question timestamp to the prompt.",
    )
    metrics: list[str] | None = Field(
        default=None,
        description="Metric names to compute.",
    )
    metric_configs: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-metric configuration overrides keyed by metric name.",
    )
    traced_data_save_dir: str = Field(
        default="traced_data",
        description="Directory where execution graph artefacts are saved.",
    )
    tracing: bool = Field(
        default=False,
        description=(
            "Whether to enable execution graph tracing. "
            "Note that this only applies to memory systems that currently support tracing."
        ),
    )


class EvaluationRunner:
    """The runner that orchestrates the question-answering and evaluation stage.

    It loads retrieval results, generates answers via an LLM, and then
    delegates judgment to the dataset-specific evaluation logic.
    """

    def __init__(self, config: EvaluationRunnerConfig | None, *, runtime=None) -> None:
        """Initialize the evaluation runner.

        Args:
            config (`EvaluationRunnerConfig`):
                The runner configuration.
        """
        self.config = config
        self.runtime = runtime

    def _evaluate_sample(self, sample):
        from .protocol import question_file, digest
        from .search import require_observation_answers
        from ..datasets.online_base import OnlineMemBaseDataset
        from ..utils.benchmark_files import read_json, write_json, sha256_file
        from ..evaluation.official import validate_answers
        runtime, cfg = self.runtime, self.runtime.config
        directory = runtime.prepare_sample(sample)
        runtime.require_stage(sample, "search")
        all_answers, phase_rows = [], {}
        # 评测本身不必加载记忆；只使用检索阶段已经保存的完整证据。
        from ..inference_utils.model_client import ModelClient
        client = ModelClient(runtime.call_config, budget=runtime.client.budget, log_path=directory / "requests.jsonl") if runtime.owns_client else runtime.client
        scorer = runtime.scorer(sample, client)
        try:
            for phase in sample.phases:
                folder = directory / phase.name
                snapshot_id = runtime.require_snapshot(sample, phase)
                if issubclass(runtime.dataset_cls, OnlineMemBaseDataset):
                    rows = require_observation_answers(runtime, sample, phase, snapshot_id)
                else:
                    rows = [grade_question(runtime, sample, phase, question, snapshot_id, client, scorer)
                            for question in phase.questions]
                validate_answers(rows, phase.questions, allow_failures=True)
                write_json(folder / "answers.json", rows)
                phase_rows[phase.name] = rows
                if cfg.benchmark == "meme":
                    continue
                all_answers.extend(rows)
            output = {"sample": sample.key, "manifest": read_json(directory / "manifest.json"),
                      "answers": all_answers, "locomo_judge_enabled": cfg.locomo_judge}
            if cfg.benchmark == "meme":
                raw = sample.reference
                official = {"episode_id": raw["episode_id"], "domain": raw["domain"], "root": raw.get("root", ""),
                            "config": {"agent_type": "ourmem", "agent_model": cfg.answer_model, "internal_model": cfg.internal_model},
                            "memory_snapshots": {f"{p.name}_questions": read_json(directory / p.name / "memory_snapshot.json")["text"] for p in sample.phases}}
                for phase in sample.phases:
                    official[f"{phase.name}_answers"] = [
                        {**question.reference, "agent_answer": row["answer_text"], "retrieved_context": row.get("context"),
                         "read_audit": {k: row.get(k) for k in ("resolution_status", "reason", "coverage", "read_trace")},
                         **({"technical_failure": row} if row.get("status") == "technical_failure" else {})}
                        for question, row in zip(phase.questions, phase_rows[phase.name])]
                write_json(directory / "official_answers.json", official)
                path, receipt = directory / "official_judge.json", directory / "judge_receipt.json"
                identity = {"answer_sha256": digest(official), "judge_model": cfg.judge_model}
                if path.exists() and receipt.exists() and read_json(receipt) == {**identity, "judge_sha256": sha256_file(path)}:
                    judged = read_json(path)
                else:
                    judged = runtime.dataset_cls.evaluate([], [], protocol="official", official_scorer=scorer,
                                                          episode=official, check_workers=cfg.check_workers)
                    write_json(path, judged)
                    write_json(receipt, {**identity, "judge_sha256": sha256_file(path)})
                from ..evaluation.meme import validate_judge
                validate_judge(path, official, cfg.judge_model, allow_failures=True)
                from .question_outcomes import save_failure
                from ..inference_utils.model_client import RecoverableModelError
                for phase in sample.phases:
                    for question, row, judgment in zip(phase.questions, phase_rows[phase.name], judged[f"{phase.name}_answers"]):
                        failure = judgment.get("technical_failure")
                        if failure and row.get("status") != "technical_failure":
                            payload = read_json(question_file(directory / phase.name / "answers", question.id))
                            save_failure(runtime, sample, phase, question, runtime.require_snapshot(sample, phase), "judge",
                                         RecoverableModelError(failure["reason"], request_ids=failure["request_ids"]), payload=payload)
                output["judge"] = judged
            output["memory_warnings"] = [phase.name for phase in sample.phases
                                         if read_json(directory / phase.name / "memory_snapshot.json").get("maintenance_incomplete")]
            write_json(directory / "result.json", output)
            runtime.finish_stage(sample, "evaluation")
            warnings = output["memory_warnings"] or any(row.get("status") == "technical_failure" for row in all_answers)
            if cfg.benchmark == "meme":
                warnings = warnings or any(row.get("technical_failure") for phase in sample.phases for row in judged[f"{phase.name}_answers"])
            write_json(directory / "status.json", {"status": "complete_with_warnings" if warnings else "complete"})
            return output
        finally:
            if runtime.owns_client:
                client.close()

    def _resolve_interface_kwargs(self) -> dict[str, Any]:
        """Build the interface keyword arguments for the LLM operator."""
        cfg = self.config
        interface_kwargs = {}

        if cfg.api_keys is not None and cfg.base_urls is not None:
            interface_kwargs["api_keys"] = cfg.api_keys
            interface_kwargs["base_urls"] = cfg.base_urls
        elif cfg.api_config_path is not None:
            with open(cfg.api_config_path, "r") as f:
                api_config = json.load(f)
            interface_kwargs["api_keys"] = api_config["api_keys"]
            interface_kwargs["base_urls"] = api_config["base_urls"]
        elif os.environ.get("OPENAI_API_KEY") is not None:
            interface_kwargs["api_keys"] = [os.environ["OPENAI_API_KEY"]]
            interface_kwargs["base_urls"] = [os.environ.get("OPENAI_API_BASE")]

        return interface_kwargs

    def run(self) -> list[dict[str, Any]]:
        """Execute the question-answering and evaluation pipeline.

        Returns:
            `list[dict[str, Any]]`:
                A list of evaluation results. Each element is a dictionary
                containing the question-answer pair, the prediction, the metrics,
                the retrieved memories, and the user id.
        """
        if self.runtime is not None:
            return self.runtime.execute("evaluation", self._evaluate_sample)
        cfg = self.config
        interface_kwargs = self._resolve_interface_kwargs()
        dataset_cls = DATASET_MAPPING[cfg.dataset_type]

        # Load and deserialize retrieval results.
        with open(cfg.search_results_path, "r") as f:
            retrievals = json.load(f)
        for item in retrievals:
            item["qa_pair"] = QuestionAnswerPair(**item["qa_pair"])
            item["retrieved_memories"] = [
                MemoryEntry(**mem) for mem in item["retrieved_memories"]
            ]
        print(
            f"✅ {len(retrievals)} retrieval results are loaded "
            f"from {cfg.search_results_path}."
        )

        if cfg.tracing:
            enable_tracing()
        else:
            disable_tracing()

        # Group retrieval results by user so that the tracing results can be saved per-user.
        user_groups = {}
        for item in retrievals:
            uid = item["user_id"]
            user_groups.setdefault(uid, []).append(item)

        print(f"🧠 Running evaluation for {len(user_groups)} users...")
        all_results = []
        for user_id, user_items in user_groups.items():
            print(
                f"  ⚙️ Evaluating user '{user_id}' ({len(user_items)} queries in total)..."
            )
    
            user_results = evaluate_memory(
                retrievals=user_items,
                qa_model=cfg.qa_model,
                judge_model=cfg.judge_model,
                dataset_cls=dataset_cls,
                qa_batch_size=cfg.qa_batch_size,
                judge_batch_size=cfg.judge_batch_size,
                add_question_timestamp=cfg.add_question_timestamp,
                prompt_template=cfg.prompt_template,
                context_builder=cfg.context_builder,
                interface_kwargs=interface_kwargs,
                user_id=user_id,
                metrics=cfg.metrics,
                metric_configs=cfg.metric_configs,
                traced_data_save_dir=cfg.traced_data_save_dir,
            )
            all_results.extend(user_results)

        # Persist results.
        output_path = (
            cfg.search_results_path.rsplit(".", 1)[0] + "_evaluation.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                all_results,
                f,
                ensure_ascii=False,
                indent=4,
            )
        print(f"✅ {len(all_results)} evaluation results are saved to {output_path}.")

        return all_results
