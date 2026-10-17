#!/bin/bash
# Submission queue for clusters that cap how many jobs a user may have submitted at once: submits the lines of
# pending.txt as the caps allow. Run it in the background from the folder holding pending.txt:
#     cd <queue dir> && nohup bash <repo>/final_eval/launchers/drain.sh > drain.log 2>&1 &
# Line format:
#     [need:<path>] [part:<partition>] <shell command that prints the job id>
# e.g.
#     part:dgx cd <repo>/final_eval && SWEEPS="sweeps/qwen3-4b_128k_r16_ruler.yaml" sbatch --parsable -p dgx --time=3-00:00:00 launchers/job_grid.sbatch
# `need:` holds a line back until <path> exists; `part:` names the partition it counts against (default: the
# first in PARTITIONS). A submitted line moves to done.txt with its job id; a refused one stays and is retried
# (errors.txt). Exits after 6 hours with nothing left to submit.
# Settings (environment): ACCOUNT (default $USER), PARTITIONS (default "dgx a40 l40"), CAPS (per-partition
# submitted-job caps in the same order, default "5 6 5"), TOTAL_CAP (account-wide, default 19).
# The defaults are the caps of the cluster the final grid ran on (QOS MaxSubmitPU dgx 5, a40 6, l40 5; account
# MaxSubmit 20, kept at 19 for headroom).
ACCOUNT="${ACCOUNT:-$USER}"
read -r -a PARTS <<< "${PARTITIONS:-dgx a40 l40}"
read -r -a CAPS_ <<< "${CAPS:-5 6 5}"
declare -A CAP=()
for i in "${!PARTS[@]}"; do CAP[${PARTS[$i]}]=${CAPS_[$i]}; done
TOTAL_CAP="${TOTAL_CAP:-19}"
idle=0
while true; do
  declare -A N=()
  for p in "${PARTS[@]}"; do N[$p]=$(squeue -h -A "$ACCOUNT" -p "$p" 2>/dev/null | wc -l); done
  total=$(squeue -h -A "$ACCOUNT" 2>/dev/null | wc -l)
  picked=""; rest=""
  if [ -s pending.txt ]; then
    while IFS= read -r l; do
      [ -z "$l" ] && continue
      if [ -z "$picked" ]; then
        x="$l"; need=""; part="${PARTS[0]}"
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
