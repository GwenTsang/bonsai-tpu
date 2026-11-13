"""Representative evaluation prompts (rendered with the model's own chat template) for reference comparison."""
import json, sys
sys.path.insert(0, '.')
from bonsai_tpu.gguf import GGUF
from bonsai_tpu.tokenizer import BonsaiTokenizer
tk = BonsaiTokenizer(GGUF('weights/Ternary-Bonsai-2-27B-PTQ1_0.gguf').metadata)
tools = [{'type': 'function', 'function': {'name': 'get_weather', 'description': 'Get current weather for a city.',
          'parameters': {'type': 'object', 'properties': {'city': {'type': 'string'}, 'unit': {'type': 'string', 'enum': ['c', 'f']}}, 'required': ['city']}}}]
long_doc = ' '.join('Record %d: the %s warehouse holds %d crates of %s.' % (i, n, 17 * i + 3, g) for i, (n, g) in enumerate(
    [('northern', 'copper wire'), ('harbor', 'dried figs'), ('upland', 'glass lenses'), ('river', 'salt'), ('eastern', 'wool'),
     ('quarry', 'granite tiles'), ('orchard', 'cider'), ('coastal', 'rope'), ('desert', 'dates'), ('mountain', 'tin ore')] * 4))
cases = {
 'knowledge': dict(messages=[{'role': 'user', 'content': 'Explain briefly why the sky is blue.'}]),
 'math': dict(messages=[{'role': 'user', 'content': 'A train travels 180 km in 2.5 hours, then 120 km in 1.5 hours. What is its average speed for the whole trip? Show the calculation.'}]),
 'coding': dict(messages=[{'role': 'user', 'content': 'Write a Python function that merges two sorted lists into one sorted list without using sort().'}]),
 'instruction': dict(messages=[{'role': 'user', 'content': 'List exactly three fruits, one per line, in uppercase, with no other text.'}]),
 'tool_use': dict(messages=[{'role': 'user', 'content': 'What is the weather in Lyon in celsius?'}], tools=tools),
 'multi_turn': dict(messages=[{'role': 'system', 'content': 'You are a terse assistant.'}, {'role': 'user', 'content': 'My name is Mirela and I keep bees.'},
                              {'role': 'assistant', 'content': 'Noted, Mirela.'}, {'role': 'user', 'content': 'What do I keep, and what is my name?'}]),
 'retrieval': dict(messages=[{'role': 'user', 'content': long_doc + '\n\nHow many crates does record 23 mention, and of what?'}]),
 'thinking': dict(messages=[{'role': 'user', 'content': 'Is 221 a prime number?'}], thinking=True),
 'french': dict(messages=[{'role': 'user', 'content': 'Explique en deux phrases ce qu\'est la photosynthèse.'}]),
 'raw_completion': dict(raw='def quicksort(arr):\n    """Sort a list using quicksort."""\n'),
}
out = {}
for name, c in cases.items():
    if 'raw' in c:
        text = c['raw']
    else:
        kw = {'tools': c['tools']} if 'tools' in c else {}
        text = tk.apply_chat_template(c['messages'], add_generation_prompt=True, enable_thinking=c.get('thinking', False), **kw)
    out[name] = dict(text=text, ids=tk.encode(text))
    print(name, len(out[name]['ids']), 'tokens')
json.dump(out, open('eval/prompts.json', 'w'), indent=1)
