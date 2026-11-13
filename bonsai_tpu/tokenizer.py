"""Tokenizer and chat template built from the GGUF's own metadata (no external tokenizer files)."""
import json
from tokenizers import Tokenizer, Regex, AddedToken, decoders, models, normalizers, pre_tokenizers

# llama.cpp LLAMA_VOCAB_PRE_TYPE_QWEN35 (original regex from tokenizer.json)
QWEN35_SPLIT = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*"
                r"|\s*[\r\n]+|\s+(?!\S)|\s+")


class BonsaiTokenizer:
    def __init__(self, metadata):
        if metadata['tokenizer.ggml.model'] != 'gpt2' or metadata['tokenizer.ggml.pre'] != 'qwen35':
            raise ValueError('Unsupported tokenizer contract')
        tokens = metadata['tokenizer.ggml.tokens']
        types = metadata['tokenizer.ggml.token_type']
        vocab = {t: i for i, t in enumerate(tokens)}
        merges = [tuple(m.split(' ', 1)) for m in metadata['tokenizer.ggml.merges']]
        tk = Tokenizer(models.BPE(vocab=vocab, merges=merges, fuse_unk=False, byte_fallback=False))
        tk.normalizer = normalizers.NFC()
        tk.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN35_SPLIT), behavior='isolated'),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
        tk.decoder = decoders.ByteLevel()
        # type 3 = control, 4 = user-defined: matched verbatim in text (parse_special=true semantics)
        special = [AddedToken(tokens[i], special=(types[i] == 3), normalized=False)
                   for i in range(len(tokens)) if types[i] in (3, 4)]
        tk.add_tokens(special)
        self.tk = tk
        self.tokens = tokens
        self.control = {i for i, t in enumerate(types) if t == 3}
        self.eos_id = metadata['tokenizer.ggml.eos_token_id']
        self.bos_id = metadata.get('tokenizer.ggml.bos_token_id')
        self.eot_ids = {self.eos_id, vocab['<|endoftext|>'], vocab['<|im_end|>']}
        self.template_source = metadata.get('tokenizer.chat_template')
        self._template = None

    def encode(self, text):
        return self.tk.encode(text, add_special_tokens=False).ids

    def decode(self, ids, skip_special=True):
        return self.tk.decode(list(ids), skip_special_tokens=skip_special)

    def apply_chat_template(self, messages, add_generation_prompt=True, **kwargs):
        if self._template is None:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            def raise_exception(msg):
                raise jinja2.TemplateError(msg)
            env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
            env.globals['raise_exception'] = raise_exception
            env.filters['tojson'] = lambda v, **kw: json.dumps(v, ensure_ascii=False, **kw)
            self._template = env.from_string(self.template_source)
        return self._template.render(messages=messages, add_generation_prompt=add_generation_prompt, **kwargs)


class StreamDecoder:
    """Incremental detokenizer that never emits partial UTF-8 sequences."""

    def __init__(self, tokenizer, skip_special=True):
        self.t = tokenizer
        self.skip = skip_special
        self.ids = []
        self.emitted = 0
        self.prefix = 0

    def push(self, token_id):
        self.ids.append(token_id)
        text = self.t.decode(self.ids[self.prefix:], self.skip)
        if text.endswith('�'):
            return ''
        out = text[self.emitted:]
        self.emitted = len(text)
        if len(self.ids) - self.prefix > 64 and '\n' in out:
            self.prefix = len(self.ids)
            self.emitted = 0
        return out
