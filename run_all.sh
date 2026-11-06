#!/usr/bin/env bash
# Run every check script single-threaded; print a summary and exit non-zero
# if any check fails.  Expected total runtime: 8-13 minutes (fixed_dim_mini alone ~4.5).
set -u
cd "$(dirname "$0")" || exit 1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
PY=${PYTHON:-python3}
status=0
summary=""
start_all=$(date +%s)
for s in chord_selberg gap_void finite_n_reduction asymptotic_c1 asymptotic_c2c3 exact_reductions \
         quadrilateral_trichotomy simple_pentagon sorted_sampler figure_data \
         printed_formulas alphabet_check fixed_dimension fixed_dim_mini; do
    echo "=== $s.py ==="
    t0=$(date +%s)
    if "$PY" "$s.py"; then r=OK; else r=FAIL; status=1; fi
    t1=$(date +%s)
    summary="$summary$(printf '%-26s %-4s %5ss' "$s" "$r" "$((t1 - t0))")"$'\n'
    echo
done
echo "=== summary ==="
printf '%s' "$summary"
echo "total: $(( $(date +%s) - start_all )) s"
if [ $status -eq 0 ]; then echo "ALL CHECKS PASS"; else echo "SOME CHECKS FAILED (see above)"; fi
exit $status
