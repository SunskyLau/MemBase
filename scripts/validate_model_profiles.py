"""服务配置的少量离线检查；不发起接口请求。"""
import argparse
import ast
from pathlib import Path
from types import SimpleNamespace as NS
import sys
import tempfile
import unittest
import contextlib
import io
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from membase.configs.model_profiles import add_profile_arguments, apply_profile, load_environment
from membase.inference_utils.model_client import ModelClient, ModelClientConfig
from membase.runners.native_transport import openai_method


class Profiles(unittest.TestCase):
    def test_selection_and_fixed_judge(self):
        parser = argparse.ArgumentParser()
        add_profile_arguments(parser)
        with patch.dict('os.environ', {'OPENAI_BASE_URL':'https://gpt.test/v1', 'DASHSCOPE_BASE_URL':'https://qwen.test/v1'}):
            for profile, expected in [('gpt','gpt-custom'),('qwen','qwen-custom')]:
                args = parser.parse_args(['--model-profile',profile,'--gpt-model','gpt-custom','--qwen-model','qwen-custom'])
                args.internal_model = args.answer_model = args.judge_model = args.base_url = None
                route = apply_profile(args)
                self.assertEqual(args.internal_model, expected)
                self.assertEqual(args.answer_model, expected)
                self.assertEqual(args.judge_model, 'gpt-4o-2024-11-20')
                self.assertEqual(route['judge_api_key_env'], 'OPENAI_API_KEY')
                self.assertEqual(route['embedding_api_key_env'], 'OPENAI_API_KEY')

    def test_env_is_literal_and_export_wins(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict('os.environ', {'TEST_PROFILE':'exported'}):
            path = Path(directory)/'.env'
            path.write_text('TEST_PROFILE=file\nTEST_LITERAL="$(not-executed)"\n')
            load_environment(path)
            import os
            self.assertEqual(os.environ['TEST_PROFILE'], 'exported')
            self.assertEqual(os.environ['TEST_LITERAL'], '$(not-executed)')

    def test_actual_client_routing_and_redaction(self):
        calls, created = [], []
        class Backend:
            def __init__(self, **kwargs):
                self.options = kwargs
                self.closed = 0
                created.append(self)
                self.chat = NS(completions=NS(create=self.chat_call))
                self.embeddings = NS(create=self.embed_call)
            def chat_call(self, **kwargs):
                calls.append(('chat', self.options['base_url'], kwargs['model']))
                return NS(choices=[NS(message=NS(content='ok'),finish_reason='stop')],usage=NS(total_tokens=2))
            def embed_call(self, **kwargs):
                calls.append(('embedding', self.options['base_url'], kwargs['model']))
                return NS(data=[NS(index=0,embedding=[1.,0.])],usage=NS(total_tokens=1))
            def close(self): self.closed += 1
        config = ModelClientConfig(model_name='qwen-custom',answer_model='qwen-custom',api_key='private-qwen',base_url='https://qwen.test/v1',
            judge_model='gpt-4o-2024-11-20',judge_api_key='private-gpt',judge_base_url='https://gpt.test/v1',
            embedding_api_key='private-gpt',embedding_base_url='https://gpt.test/v1')
        with patch('openai.OpenAI', Backend):
            client = ModelClient(config)
            client.count_tokens = lambda text: 1
            client.text('hello',stage='answer')
            client.text('judge',stage='judge',model=config.judge_model)
            client.embed(['text'])
            self.assertEqual(calls,[('chat','https://qwen.test/v1','qwen-custom'),
                ('chat','https://gpt.test/v1','gpt-4o-2024-11-20'),('embedding','https://gpt.test/v1','text-embedding-3-small')])
            self.assertNotIn('private-',config.model_dump_json())
            self.assertEqual(client._safe('private-qwen private-gpt'),'[REDACTED] [REDACTED]')
            self.assertEqual(len(created),2)
            client.close()
            self.assertTrue(all(b.closed == 1 for b in created))

    def test_mab_transport_only(self):
        tree = ast.parse((ROOT/'external/MemoryAgentBench/agent.py').read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name=='AgentWrapper')
        for name in ['_initialize_long_context_agent','_query_long_context_agent','_handle_bm25_rag']:
            node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
            # 使用真实官方方法的语法树；不导入其模型依赖或调用网络。
            with tempfile.TemporaryDirectory() as directory:
                path=Path(directory)/'method.py'; path.write_text(ast.unparse(node)+'\n')
                namespace={};exec(compile(path.read_text(),str(path),'exec'),namespace)
                import linecache
                linecache.cache[str(path)]=(path.stat().st_size,None,path.read_text().splitlines(True),str(path))
                result = openai_method(namespace[name])
                self.assertEqual(result.__name__,name)
                if name=='_initialize_long_context_agent':
                    obj=NS(model='qwen-custom',_create_oai_client=lambda:'compatible-client')
                    result(obj);self.assertEqual(obj.client,'compatible-client')
                else:
                    captured = []
                    response = NS(choices=[NS(message=NS(content='answer'))],usage=NS(prompt_tokens=1,completion_tokens=1))
                    backend = NS(chat=NS(completions=NS(create=lambda **kwargs: (captured.append(kwargs),response)[1])))
                    namespace.update(time=NS(time=lambda:1), tiktoken=NS(encoding_for_model=lambda model:None),
                        get_template=lambda *args:'official-system', format_chat=lambda **kwargs:kwargs)
                    obj = NS(model='qwen-custom',context='history',input_length_limit=1000000,context_max_length=0,
                        sub_dataset='factconsolidation',agent_name='test',temperature=0.7,max_tokens=10,
                        client=backend,_format_openai_response=lambda value,start:value,context_id=1,retrieve_num=10,
                        _extract_retrieval_query=lambda text:text,_create_oai_client=lambda:backend,
                        bm25_retriever=NS(get_relevant_documents=lambda query:[NS(page_content='official chunk')]))
                    with contextlib.redirect_stdout(io.StringIO()):
                        result(obj,'question') if name=='_query_long_context_agent' else result(obj,'question',1,None)
                    self.assertEqual(captured[0]['model'],'qwen-custom')
                    self.assertEqual(captured[0]['max_tokens'],10)
                    self.assertEqual(captured[0]['messages']['system_message'],'official-system')
        # 条件替换不删除原提示词、分块或评分函数。

    def test_meme_dense_keeps_embedding_service(self):
        import importlib
        from membase.runners.native_transport import configure_meme_dense
        with patch.dict('os.environ', {'OPENAI_API_KEY':'qwen-key','OPENAI_BASE_URL':'https://qwen.test/v1',
                'MEMBASE_EMBEDDING_API_KEY':'embedding-key','MEMBASE_EMBEDDING_BASE_URL':'https://gpt.test/v1',
                'MEMBASE_EMBEDDING_MODEL':'embedding-model'}), patch.object(sys,'path',[str(ROOT/'external/MEME-public/code'),*sys.path]):
            with patch('openai.OpenAI',side_effect=lambda **kwargs:NS(**kwargs)), patch('tiktoken.get_encoding',return_value=NS()):
                configure_meme_dense()
                dense = importlib.import_module('agents.dense_memory').DenseMemory(model='qwen-custom')
                self.assertEqual(dense._embedding_model,'embedding-model')
                self.assertEqual(dense._openai().api_key,'embedding-key')
                self.assertEqual(dense._openai().base_url,'https://gpt.test/v1')


if __name__ == '__main__':
    unittest.main()
