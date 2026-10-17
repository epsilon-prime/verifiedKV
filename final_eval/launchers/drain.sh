#!/bin/bash
# Submission queue: submits the lines of pending.txt as SLURM caps allow (QOS MaxSubmitPU: dgx 5, a40 6, l40 5;
# account MaxSubmit 20, kept at 19 for headroom). Line format:
#     [need:<path>] [part:<dgx|a40|l40>] <shell command that prints the job id, e.g. sbatch --parsable ...>
# `need:` holds a line back until <path> exists; `part:` names the partition it counts against (default dgx).
# A submitted line moves to done.txt with its job id; a refused one stays and is retried (errors.txt).
# Exits after 6 hours with nothing left to submit.
cd "$(dirname "$0")"
declare -A CAP=([dgx]=5 [a40]=6 [l40]=5)
TOTAL_CAP=19
idle=0
while true; do
  declare -A N=()
  for p in dgx a40 l40; do N[$p]=$(squeue -h -A 23b1835 -p $p 2>/dev/null | wc -l); done
  total=$(squeue -h -A 23b1835 2>/dev/null | wc -l)
  picked=""; rest=""
  if [ -s pending.txt ]; then
    while IFS= read -r l; do
      [ -z "$l" ] && continue
      if [ -z "$picked" ]; then
        x="$l"; need=""; part=dgx
        if [[ "$x" == need:* ]]; then need="${x%% *}"; need="${need#need:}"; x="${x#* }"; fi
        if [[ "$x" == part:* ]]; then part="${x%% *}"; part="${part#part:}"; x="${x#* }"; fi
        if { [ -z "$need" ] || [ -e "$need" ]; } && [ "${N[$part]:-0}" -lt "${CAP[$part]:-5}" ] && [ "$total" -lt $TOTAL_CAP ]; then
          picked="$x"; continue
        fi
      fi
      rest+="$l"$'\n'
    done < pending.txt
  fi
  if [ -n "$picked" ]; then
    id=$(bash -c "$picked" 2>&1 | tail -1)
    if [[ "$id" =~ ^[0-9]+$ ]]; then
      echo "$(date '+%m-%d %H:%M') $id :: $picked" >> done.txt
      printf "%s" "$rest" > pending.txt
    else
      echo "$(date '+%m-%d %H:%M') RETRY ($id) :: $picked" >> errors.txt
      sleep 60
    fi
    idle=0
    sleep 5
    continue
  fi
  idle=$((idle + 1)); [ ! -s pending.txt ] && [ $idle -gt 180 ] && exit 0
  sleep 120
done
