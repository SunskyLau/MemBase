"""请求内的短引用；仅转换标识，不改变事实文本，回复再还原为内部标识。"""

class ReferenceCodec:
    def __init__(self, payload):
        self.forward = {}
        def visit(value):
            if isinstance(value, dict):
                for key, item in value.items():
                    if key in {"id", "source_id", "target_id", "memory_key", "operation_id", "version_id",
                               "previous_version_id", "replacement_id"} and isinstance(item, str):
                        self.add(item)
                    elif key.endswith("_ids") and isinstance(item, list):
                        for entry in item:
                            if isinstance(entry, str):
                                self.add(entry)
                    visit(item)
            elif isinstance(value, list):
                for item in value:
                    visit(item)
        visit(payload)
        self.reverse = {v: k for k, v in self.forward.items()}

    def add(self, value):
        if len(value) > 16 and value not in self.forward:
            self.forward[value] = f"r{len(self.forward)}"

    @staticmethod
    def transform(value, mapping, key=None):
        if isinstance(value, dict):
            return {k: ReferenceCodec.transform(v, mapping, k) for k, v in value.items()}
        if isinstance(value, list):
            return [ReferenceCodec.transform(v, mapping, key) for v in value]
        if isinstance(value, str) and key not in {"content", "text", "quote", "reason", "description", "identity_description"}:
            return mapping.get(value, value)
        return value

    def encode(self, value):
        return self.transform(value, self.forward)

    def decode(self, value):
        return self.transform(value, self.reverse)

    def encode_text(self, value):
        for original, alias in self.forward.items():
            value = value.replace(original, alias)
        return value
