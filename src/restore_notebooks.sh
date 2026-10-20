#!/usr/bin/env bash
# Restore the verified notebooks after an editor buffer overwrites them.
# Run from /path/to/rician_urban_sim.  Close the notebook tab FIRST.
set -e
cd "$(dirname "$0")"
cp .stochastic_simulation.GOOD.ipynb stochastic_simulation.ipynb
cp .site_simulation.GOOD.ipynb       site_simulation.ipynb
echo "restored both notebooks from the verified copies"
md5sum stochastic_simulation.ipynb site_simulation.ipynb
