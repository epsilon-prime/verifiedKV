"""LOFT answers all of a task's questions on ONE compressed context. Check that a question's answer does not depend
on the questions asked before it (the per-question rewind / state restore works with the vkv_final arms):
answers from pipe(context, questions=[q0, q1, q2]) must equal pipe(context, question=qi) one by one.

    source env.sh && python -u check_multiq.py --model Qwen/Qwen3.5-9B [--arm vkv_final_kvzip] [--task nq_32k]

Uses LOFT dev questions (never test). Prints IDENTICAL or the differing answers; exit code 1 on a difference.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "kvpress"))
sys.path.insert(0, str(HERE.parents[1] / "kvpress" / "evaluation"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import evaluate  # noqa: E402
import arms  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--arm", default="vkv_final_kvzip")
ap.add_argument("--task", default="nq_32k")
ap.add_argument("--ratio", type=float, default=0.9375)
ap.add_argument("--n", type=int, default=3)
args = ap.parse_args()

presses = arms.register()
cfg = evaluate.EvaluationConfig(dataset="loft", data_dir=args.task, model=args.model, press_name=args.arm,
                                compression_ratio=args.ratio, model_kwargs={"attn_implementation": "sdpa"})
runner = evaluate.EvaluationRunner(cfg)
runner._setup_press()
runner._setup_model_pipeline()
pipe, press = runner.pipeline, runner.press
df = evaluate.load_loft_dataset(args.task)
df = df[df["split"] == "dev"].head(args.n)
ctx, qs = df["context"].iloc[0], df["question"].tolist()
kw = dict(answer_prefix=df["answer_prefix"].iloc[0], press=press, max_new_tokens=int(df["max_new_tokens"].iloc[0]),
          max_context_length=200000)
with torch.inference_mode():
    multi = pipe(ctx, questions=qs, **kw)["answers"]
    single = [pipe(ctx, question=q, **kw)["answer"] for q in qs]
bad = [(i, a, b) for i, (a, b) in enumerate(zip(multi, single)) if a != b]
print(f"{args.model} {args.arm} {args.task}: {len(qs)} questions, GPU {torch.cuda.get_device_name(0)}")
for i, (a, b) in enumerate(zip(multi, single)):
    print(f"  q{i}: {'same' if a == b else 'DIFFERENT'} {a[:100]!r}" + ("" if a == b else f" vs alone {b[:100]!r}"))
print("IDENTICAL" if not bad else f"{len(bad)} DIFFERENT")
sys.exit(1 if bad else 0)
