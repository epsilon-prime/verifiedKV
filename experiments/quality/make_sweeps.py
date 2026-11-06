"""Write the sweep files: sweeps/ (VerifiedKV), sweeps_baselines/ (Dense, KVzip, Fast KVzip, PoissonComp) and
sweeps_smoke/. The output is deterministic, and the files are committed.

    python make_sweeps.py

Grid: three models x {RULER-32K, RULER-128K: 4 tasks, 50 test rows; LOFT-32K, LOFT-128K: 5 tasks, all test questions}
x {8x, 16x, 32x}. One file per (model, length, ratio, task).
"""

from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
MODELS = {"qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507", "qwen3.5-9b": "Qwen/Qwen3.5-9B",
          "ministral3-8b": "mistralai/Ministral-3-8B-Instruct-2512-BF16"}
RATIOS = {8: 0.875, 16: 0.9375, 32: 0.96875}
RULER_TASKS = ["niah_multikey_3", "qa_1", "niah_multivalue", "cwe"]
LOFT_TASKS = ["nq", "hotpotqa", "musique", "qampari", "quest"]
ARMS = ["vkv_final_kvzip", "vkv_final_fastkvzip", "vkv_final_perfect"]
BASELINES = ["no_press", "kvzip_c", "fastkvzip", "bernoulli_kvzip_c"]  # Dense, KVzip, Fast KVzip, PoissonComp
N_RULER = 50
HEADER = "# Test rows, written by make_sweeps.py. Do not edit by hand.\n"


def write(folder: Path, name: str, sweep: dict, header: str):
    folder.mkdir(exist_ok=True)
    (folder / f"{name}.yaml").write_text(header + yaml.safe_dump(sweep, sort_keys=False))


def main():
    for d in ("sweeps", "sweeps_baselines", "sweeps_smoke"):
        for f in (HERE / d).glob("*.yaml"):
            f.unlink()
    n = 0
    for short, model in MODELS.items():
        for L in ("32k", "128k"):
            for rate, ratio in RATIOS.items():
                bl = {"group": f"final_baselines_{short}", "arms": BASELINES}
                base = {"group": f"final_vkv_{short}", "model": model, "ratio": ratio, "max_context_length": 200000,
                        "arms": ARMS, "require_clean": True}
                for t in RULER_TASKS:
                    write(HERE / "sweeps", f"{short}_{L}_r{rate}_ruler_{t}",
                          {**base, "n_samples": N_RULER, "cells": [{"dataset": f"ruler{L}", "task": t}]}, HEADER)
                    write(HERE / "sweeps_baselines", f"{short}_{L}_r{rate}_ruler_{t}",
                          {**base, **bl, "n_samples": N_RULER, "cells": [{"dataset": f"ruler{L}", "task": t}]}, HEADER)
                    n += 1
                for t in LOFT_TASKS:
                    write(HERE / "sweeps", f"{short}_{L}_r{rate}_loft_{t}",
                          {**base, "loft_split": "test", "cells": [{"dataset": "loft", "task": f"{t}_{L}"}]}, HEADER)
                    write(HERE / "sweeps_baselines", f"{short}_{L}_r{rate}_loft_{t}",
                          {**base, **bl, "loft_split": "test", "cells": [{"dataset": "loft", "task": f"{t}_{L}"}]},
                          HEADER)
                    n += 1
        # smoke tests: development rows and questions only, 16x, every arm and task
        smoke = {"group": f"smoke_{short}", "model": model, "ratio": RATIOS[16], "max_context_length": 200000,
                 "arms": ARMS}
        hdr = "# Smoke test (development rows and questions), written by make_sweeps.py\n"
        write(HERE / "sweeps_smoke", f"{short}_32k_ruler", {**smoke, "ruler_rows": "150:151",
              "cells": [{"dataset": "ruler32k", "task": t} for t in RULER_TASKS]}, hdr)
        write(HERE / "sweeps_smoke", f"{short}_32k_loft", {**smoke, "loft_split": "dev",
              "cells": [{"dataset": "loft", "task": "nq_32k"}]}, hdr)
        write(HERE / "sweeps_smoke", f"{short}_128k_ruler", {**smoke, "ruler_rows": "400:401",
              "cells": [{"dataset": "ruler128k", "task": t} for t in ("niah_multikey_3", "cwe")]}, hdr)
        write(HERE / "sweeps_smoke", f"{short}_128k_loft", {**smoke, "loft_split": "dev",
              "cells": [{"dataset": "loft", "task": "nq_128k"}]}, hdr)
    print(f"wrote {n} sweeps to sweeps/ and sweeps_baselines/, and the smoke sweeps to sweeps_smoke/")


if __name__ == "__main__":
    main()
