"""按实际发送格式计入完整请求及纠错空间；重试仍由共享调用器执行。"""

import json

from .llm import ContextLimitError, OutputLimitError, StructuredOutputError

FEEDBACK_TOKENS = 512
CORRECTION_PREFIX = "Return corrected JSON only. Previous output failed validation: "


def request_tokens(llm, prompt, payload):
    system = prompt if "json" in prompt.casefold() else prompt + "\nReturn a JSON object only."
    body = json.dumps(payload, ensure_ascii=False, default=lambda item: item.model_dump(mode="json"))
    return llm.count_tokens(system) + llm.count_tokens(body) + 32


def fits_request(llm, config, prompt, payload):
    # 最坏的一次纠错会带回前次输出；额外空间留给错误说明与消息包装。
    reserve = (config.max_model_output_tokens + FEEDBACK_TOKENS + llm.count_tokens(CORRECTION_PREFIX) + 32
               if config.max_llm_retries else 0)
    return request_tokens(llm, prompt, payload) + reserve <= config.max_context_tokens


def request_json(llm, config, stage, prompt, payload, *, validator):
    if not fits_request(llm, config, prompt, payload):
        raise ContextLimitError(f"{stage} complete request plus retry reserve exceeds {config.max_context_tokens}")

    def bounded_validator(raw):
        try:
            return validator(raw)
        except ValueError as error:
            message = str(error)
            if llm.count_tokens(message) <= FEEDBACK_TOKENS:
                raise
            # 只压缩诊断文本，不截断事实或成立依据；避免长校验报告挤爆重试。
            low, high = 0, len(message)
            suffix = "\nFurther validation errors omitted; correct the listed errors first."
            while low < high:
                middle = (low + high + 1) // 2
                if llm.count_tokens(message[:middle] + suffix) <= FEEDBACK_TOKENS:
                    low = middle
                else:
                    high = middle - 1
            raise ValueError(message[:low] + suffix) from error

    return llm.request_json(stage, prompt, payload, validator=bounded_validator)
