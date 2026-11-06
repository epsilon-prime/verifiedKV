"""Answer a question over a compressed KV cache with VerifiedKV.

    python examples/quickstart.py
"""

from transformers import pipeline

from verifiedkv import make_press

MODEL = "Qwen/Qwen3-4B-Instruct-2507"

facts = [f"Report {i} was filed in room {100 + i}." for i in range(2000)]
facts.insert(1000, "The access code for the archive is 7392.")
context = " ".join(facts)
question = "What is the access code for the archive?"

pipe = pipeline("kv-press-text-generation", model=MODEL, device_map="auto", dtype="auto",
                model_kwargs={"attn_implementation": "sdpa"})
press = make_press(compression_ratio=0.9375, scorer="kvzip", sketch="2-bit", eps0=0.05, delta=0.1)

print(pipe(context, question=question, press=press, max_new_tokens=32)["answer"])
