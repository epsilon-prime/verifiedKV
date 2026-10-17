Copied unmodified from https://github.com/Janghyun1230/FastKVzip at commit e04afaa (2026-04-04),
the official Fast KVzip code (Kim, Han, Yun, arXiv:2601.17668). Used only to train gates for models
without released ones, with the authors' own training procedure. The repo has no LICENSE file; do not
redistribute. Read before use: pure PyTorch (BCE on KVzip scores, SGD, 5000 steps, step LR schedule).
Known quirk kept as is: `Weight.forward` adds the bias `b` twice; kvpress's `FastKVzipGate` adds it once.
