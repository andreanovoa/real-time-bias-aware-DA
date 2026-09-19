#!/usr/bin/env bash
# Run ablation studies of one case in parallel (resumable — completed runs are skipped).
# Usage: [CASE=lorenz63|lorenz96|annular|cylinder] ./run_parallel.sh [study ...]
# Default: CASE=lorenz63 and its full study matrix. Logs: ../../results/ablations/<case>_<study>.log
cd "$(dirname "$0")"
CASE=${CASE:-lorenz63}
[ $# -eq 0 ] && case $CASE in
    lorenz63) set -- ensemble_size assimilation_frequency inflation observation_noise regularization ;;
    lorenz96) set -- ensemble_size observation_sparsity inflation ;;
    annular)  set -- ensemble_size assimilation_frequency regularization equivalence_ratio ;;
    cylinder) set -- ensemble_size assimilation_frequency sensors wout_estimation inflation ;;
    *) echo "Unknown CASE=$CASE" >&2; exit 1 ;;
esac

logdir=../../results/ablations
mkdir -p "$logdir"
for study in "$@"; do
    MPLBACKEND=Agg python "run_$CASE.py" --study "$study" > "$logdir/${CASE}_$study.log" 2>&1 &
done
wait
