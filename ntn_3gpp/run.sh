#!/bin/sh
# Campaign runs.  ./run.sh GPU SCENARIO BAND [ELEVATION...]
#   SCENARIO = urban or rural, BAND = S, Ka or Ku, elevations default to 10..90.
#   RUNS=<dir> overrides the run folder (default: runs/ next to this script).
# Rerun the same command to resume: finished elevations are skipped, partial ones resume.
B=$(cd "$(dirname "$0")" && pwd)
R=${RUNS:-$B/runs}
export PYTHONPATH=$B:$B/site${PYTHONPATH:+:$PYTHONPATH}
export OPENBLAS_NUM_THREADS=1      # numpy dot on 1e5-path arrays: 64 BLAS threads ran ~30x slower than 1
mkdir -p $R/logs
sed -i "s#@ROOT@#$R#g" $R/groups/*.json

gpu=$1; s=$2; band=$3; shift 3
[ $# -gt 0 ] || set -- 10 20 30 40 50 60 70 80 90
g=$R/groups/${s}_$band.json
dirs=$(grep '"dir"' $g | cut -d'"' -f4)
for e in "$@"; do
    el=$(printf %03d $e)
    done_all=1
    for d in $dirs; do [ -f $d/results_elev${el}deg.pkl ] || done_all=0; done
    [ $done_all = 1 ] && continue
    CUDA_VISIBLE_DEVICES=$gpu ${PY:-python3} -W ignore $B/sim/run_stochastic_sim.py \
        --elevation $e --gpu_id $gpu --group $g --resume > $R/logs/${s}_${band}_e$e.log 2>&1 \
        || echo "FAILED ${s}_$band e$e" >> $R/logs/failed.txt
done
echo "${s}_$band finished $(date)" >> $R/logs/failed.txt
