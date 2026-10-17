#!/bin/bash
# Fetch the Fast KVzip authors' training code into vendor/ (needed by train_gates.py and eval_gate.py).
# Their repository has no license, so it is not redistributed here; this pins the exact commit the gates were
# trained with and checks each file against its git blob hash.
set -euo pipefail
cd "$(dirname "$0")"
REPO=Janghyun1230/FastKVzip
COMMIT=e04afaa6007c58d18500d0698d2d8ae592ff22c2
mkdir -p vendor
while read -r f sha; do
  curl -fsSL "https://raw.githubusercontent.com/$REPO/$COMMIT/$f" -o "vendor/$f"
  got=$(git hash-object "vendor/$f")
  [ "$got" = "$sha" ] || { echo "vendor/$f: blob $got, expected $sha" >&2; exit 1; }
  echo "vendor/$f ok"
done <<'LIST'
optim.py 5a8b0324c994d7b7830e91d33c9a605f3c5e510c
load_data.py c59c1322ca4f93f6ca8b36a319e1a7650f97fa4d
utils.py 2c5785fbeb2dcf15835bcfd8f75faa54f961194b
LIST
