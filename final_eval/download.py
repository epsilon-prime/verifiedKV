"""Download everything the final grid reads into HF_HOME, and check it against the revisions the grid used.

    HF_HOME=<cache> python final_eval/download.py                  # models, gates, RULER, LOFT (needs internet)
    HF_HOME=<cache> python final_eval/download.py --gate-training  # also FineWeb-Edu, for fastkvzip_gates/
    HF_HOME=<cache> python final_eval/download.py --check          # no downloads: report what is missing or differs
    ... --only loft-rag-nq                                          # restrict to names containing a substring

The jobs run offline (HF_HUB_OFFLINE=1, HF_DATASETS_OFFLINE=1), so everything must be in HF_HOME first. Datasets
are fetched through the same ``load_dataset`` calls kvpress makes at run time, because a dataset that was only
``snapshot_download``-ed cannot be loaded offline by config. Run it on a machine with internet that shares
HF_HOME with the GPU nodes (e.g. a cluster login node).

Each repository is fetched at its current main branch, then its commit is compared with the one the final grid
used (REVISIONS). A mismatch means the repository changed upstream since: results may then differ from ours.
"""

import argparse
import os
import sys

# repo id -> commit the final grid used (HF_HOME/hub/<repo>/refs/main on the machine that ran it)
MODELS = {
    "Qwen/Qwen3-4B-Instruct-2507": "cdbee75f17c01a7cc42f958dc650907174af0554",
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": "f6fae9795746f63c9be8344932f01275f3c63734",
    "Qwen/Qwen3.5-9B": "c202236235762e1c871ad0ccb60c8ee5ba337b9a",
    # the Fast KVzip authors' released gates (Qwen3-4B-Instruct-2507's are used; ours cover the other models)
    "Jang-Hyun/Fast-KVzip": "67ce1265f510a946bf22486a30269f99066b3cbb",
}
DOWNLOAD_KW = {
    # the consolidated (Mistral-format) checkpoint duplicates the safetensors shards; transformers does not read it
    "mistralai/Ministral-3-8B-Instruct-2512-BF16": {"ignore_patterns": ["consolidated.safetensors"]},
    "Jang-Hyun/Fast-KVzip": {"allow_patterns": ["qwen3-4b-instruct-2507/*", "README.md"]},
}
RULER_TASKS = ["niah_single_1", "niah_single_2", "niah_single_3", "niah_multikey_1", "niah_multikey_2",
               "niah_multikey_3", "niah_multiquery", "niah_multivalue", "vt", "cwe", "fwe", "qa_1", "qa_2"]
RULER_32K = ("xAlg-AI/att-hub-ruler-32k", "525f667545a6c96767aa7047f205de07ed7b9455")
RULER_128K = ("SaylorTwift/RULER-131072-llama-3.2-tokenizer", "758174ef5e9a6a2da6c4c3a25c18b9aca57b2881")
LOFT = {
    "f20180301/loft-rag-nq-32k": "d2edac5ebed53d2a882bf44916d942fabb0db80b",
    "f20180301/loft-rag-hotpotqa-32k": "68245514f0c09de5905e1bf7d1a5f188ad918ab1",
    "f20180301/loft-rag-musique-32k": "f769b496c75b81e808bf3a865abc1feda78ae42a",
    "f20180301/loft-rag-qampari-32k": "aada108438f93ffaf62618f37d8ee33919a3993d",
    "f20180301/loft-rag-quest-32k": "19306ce4cab43621b10e6f02cd41c558a680be2c",
    "f20180301/loft-rag-nq-128k": "71964da6826d669fdceb1058095f56b7148cd0cc",
    "f20180301/loft-rag-hotpotqa-128k": "9758c8c9b153a5ed65ea301a0a7902b313458f54",
    "f20180301/loft-rag-musique-128k": "c2c321aeed28e6f8276f2b9d1fb18e17f3e08a48",
    "f20180301/loft-rag-qampari-128k": "db38e707a240aee398a0f961b45a2d0d4620a944",
    "f20180301/loft-rag-quest-128k": "9ba2b24fe14c4415853d572ff3d513aea1ff36b4",
}
FINEWEB = ("HuggingFaceFW/fineweb-edu", "87f09149ef4734204d70ed1d046ddc9ca3f2b8f9", "sample/10BT/000_00000.parquet")


def cached_revision(repo: str, kind: str) -> str | None:
    """The commit HF_HOME's refs/main points at, which is what an offline load reads."""
    from huggingface_hub.constants import HF_HUB_CACHE

    ref = os.path.join(HF_HUB_CACHE, f"{kind}s--{repo.replace('/', '--')}", "refs", "main")
    return open(ref).read().strip() if os.path.isfile(ref) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="download nothing; report missing or changed repositories")
    ap.add_argument("--gate-training", action="store_true", help="also FineWeb-Edu (only for training gates)")
    ap.add_argument("--only", help="only repositories whose id contains this substring")
    args = ap.parse_args()
    want = lambda repo: args.only is None or args.only in repo  # noqa: E731

    from huggingface_hub import hf_hub_download, snapshot_download
    from huggingface_hub.constants import HF_HUB_CACHE

    print(f"HF hub cache: {HF_HUB_CACHE}")
    todo = []  # (repo, kind, expected commit, fetch)
    for repo, sha in MODELS.items():
        todo.append((repo, "model", sha, lambda r=repo: snapshot_download(r, **DOWNLOAD_KW.get(r, {}))))

    def ruler_fetch(repo, by_config):
        from datasets import load_dataset

        for task in RULER_TASKS:  # exactly kvpress's dataset_loaders.load_ruler_length_dataset calls
            load_dataset(repo, task, split=task) if by_config else load_dataset(repo, split=task)

    todo.append((RULER_32K[0], "dataset", RULER_32K[1], lambda: ruler_fetch(RULER_32K[0], True)))
    todo.append((RULER_128K[0], "dataset", RULER_128K[1], lambda: ruler_fetch(RULER_128K[0], False)))
    for repo, sha in LOFT.items():
        def loft_fetch(r=repo):
            from datasets import load_dataset

            load_dataset(r)  # as dataset_loaders.load_loft_dataset

        todo.append((repo, "dataset", sha, loft_fetch))
    if args.gate_training:
        repo, sha, fname = FINEWEB
        todo.append((repo, "dataset", sha, lambda: hf_hub_download(repo, fname, repo_type="dataset")))

    problems = 0
    for repo, kind, sha, fetch in todo:
        if not want(repo):
            continue
        if not args.check:
            print(f"fetching {repo} ...", flush=True)
            fetch()
        got = cached_revision(repo, kind)
        if got is None:
            status, problems = "MISSING", problems + 1
        elif got != sha:
            status, problems = f"DIFFERS: cached {got[:12]}, final grid used {sha[:12]}", problems + 1
        else:
            status = "ok"
        print(f"  {repo}: {status}", flush=True)
    print("all present at the final grid's revisions" if not problems else f"{problems} problem(s), see above")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
