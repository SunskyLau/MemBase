"""官方基线的接口连接适配；不改写检索、提示词、评分或官方仓库文件。"""
from __future__ import annotations

import argparse
import ast
import inspect
import os
from pathlib import Path
import runpy
import sys
import textwrap


def openai_method(method):
    """MAB 的三个方法以模型名判断协议；两套实验都显式走兼容接口。"""
    tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
    class ProtocolBranch(ast.NodeTransformer):
        def visit_Compare(self, node):
            if (isinstance(node.left, ast.Constant) and node.left.value in {"gpt", "gpt-4"}
                    and len(node.ops) == 1 and isinstance(node.ops[0], ast.In)
                    and ast.unparse(node.comparators[0]) == "self.model"):
                return ast.copy_location(ast.Constant(True), node)
            return self.generic_visit(node)
    tree = ast.fix_missing_locations(ProtocolBranch().visit(tree))
    namespace = {}
    exec(compile(tree, inspect.getsourcefile(method), "exec"), method.__globals__, namespace)
    return namespace[method.__name__]


def configure_mab():
    from agent import AgentWrapper
    from openai import OpenAI
    AgentWrapper._create_oai_client = lambda self: OpenAI(
        api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_BASE_URL"])
    for name in ("_initialize_long_context_agent", "_query_long_context_agent", "_handle_bm25_rag"):
        setattr(AgentWrapper, name, openai_method(getattr(AgentWrapper, name)))


def configure_meme_dense():
    from agents.dense_memory import DenseMemory
    from openai import OpenAI
    original = DenseMemory.__init__
    def initialize(self, *args, **kwargs):
        kwargs.setdefault("embedding_model", os.environ["MEMBASE_EMBEDDING_MODEL"])
        original(self, *args, **kwargs)
    def embedding_client(self):
        if self._client is None:
            self._client = OpenAI(api_key=os.environ["MEMBASE_EMBEDDING_API_KEY"],
                                  base_url=os.environ["MEMBASE_EMBEDDING_BASE_URL"])
        return self._client
    DenseMemory.__init__, DenseMemory._openai = initialize, embedding_client


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", choices=["memoryagentbench", "meme"], required=True)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--module", default="eval.run_agent")
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    arguments = args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
    sys.argv = [args.module, *arguments]
    if args.benchmark == "memoryagentbench":
        sys.path.insert(0, str(args.upstream))
        configure_mab()
        runpy.run_path(str(args.upstream / "main.py"), run_name="__main__")
    else:
        sys.path.insert(0, str(args.upstream / "code"))
        configure_meme_dense()
        runpy.run_module(args.module, run_name="__main__")
