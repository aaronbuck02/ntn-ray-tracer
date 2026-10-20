# Fixed-UE Canary Wharf pass sweep

This campaign uses the existing site-specific `run_elevation_sim.py` runner
with a separate output directory for every time sample. One outdoor UE stays
at exactly the same position for every sample and every pass. The default
scene is `RT Scenarios/wharf_scene_v3/du/canarywharf3_du.xml`, using the
Canary Wharf 2 GHz configuration (RHCP antennas, configured scene
materials and propagation flags).

The campaign selects **real satellite TLEs** with predicted passes nearest
30°, 60°, and 90° peak elevation. These are selection targets, not imposed
orbital geometry. Inclination, RAAN, eccentricity and mean anomaly remain
exactly those in each satellite's TLE. SGP4 supplies time-varying position,
azimuth and range. Continuous predicted peak and sampled peak are recorded
separately, along with satellite name, NORAD ID, original TLE lines, TLE epoch
and UTC rise/peak/set times. Predicted passes are not measured satellite tracks.

By default, preparation reads `tle_catalog/starlink.tle` and starts the search
at the median TLE epoch. This is a reproducible **historical replay**; the
repository's existing catalog is from August 2026, not current October data.
The default window is 24 hours, inspecting each satellite's first visible pass.
Different real satellites may be selected for the three peak targets. The
nearest selected peak must be within 5° (`--peak-tolerance-deg`). Exactly 90°
is not assumed: the highest qualifying real pass near zenith is recorded with
its actual peak. No synthetic fallback or orbit-element fitting is allowed.

For another catalog or search time, use `--tle-catalog FILE --epoch ISO_UTC`
and optionally `--search-hours HOURS`. TLEs must be within 3 days of both ends
of the search window (`--max-tle-age-days`). For a present-day campaign,
download fresh elements with the repository's fetcher first:

```bash
python src/utils/fetch_tle.py --group starlink --cache_dir tle_catalog --force
python src/simulations/run_pass_snr_sweep.py prepare \
  --output Results/wharf_real_pass_snr_fresh
python src/simulations/run_pass_snr_sweep.py run \
  --output Results/wharf_real_pass_snr_fresh
```

Preparation selects the eligible outdoor occupancy-grid cell nearest the
scene centre, records its XYZ coordinates and approximate geographic location,
and writes a single-receiver input file. `--ue-xy X Y` selects a different
nearby eligible grid cell; the actual snapped position is recorded. UE height
comes from the source configuration (1.5 m). Geographic conversion assumes the
configured SIM_LAT/SIM_LON is at SCENE_CENTER and uses the configured scene
heading. The orbit's topocentric origin is the ground point below the UE;
the scene solver includes the UE's 1.5 m antenna height.

Run these commands from the repository with a Python environment containing
the dependencies in README.md. Preparation requires Mitsuba but no GPU; the
simulation requires the repository's patched Sionna and working CUDA driver.

For a notebook run on both GPUs, open
`src/custom_drivers/site_simulation_wharf_pass_snr.ipynb` with your Sionna kernel
and run its cells in order. It reads the already prepared campaign; there are
no atmosphere/ionosphere settings or orbital preparation cells. Set `GPU_IDS`
(default `[0, 1]`) and `PYTHON_EXECUTABLE` in the run settings. Its shared queue
keeps at most one `run_elevation_sim.py` subprocess on each GPU, assigning a
new sample whenever a GPU becomes free. Each worker sets `CUDA_VISIBLE_DEVICES`
before GPU imports, matching the other site notebooks. Interrupting the launch
cell or a worker failure stops both workers; completed samples are retained
for the next run. Full-campaign analysis runs after all samples finish.

The current prepared replay contains 147, 138 and 137 RT samples for the
30.00°, 60.10° and 89.93° peak passes: **422 solves** in total. The respective
time intervals are 4.987, 4.971 and 4.983 seconds, covering both rise/set legs.
Each solve uses the saved configuration's 10 million rays.

The same scheduler is available from the CLI:

```bash
python src/simulations/run_pass_snr_sweep.py run --gpu-ids 0 1
```

`--gpu-id 0` remains available for the existing single-GPU CLI workflow.

The notebook's `CAMPAIGN` setting selects `"original3"` or `"additional10"`.
The additional campaign lives in `Results/wharf_real_pass_snr_additional10`.
It contains ten different NORAD IDs, excludes all three original satellites,
and copies the original UE, simulation configuration and catalog snapshot
exactly. It uses the same August 2026 historical replay, rather than current
satellite predictions. Each new pass is selected jointly by peak elevation
and compass azimuth at that peak (clockwise from north), with tolerances of
5° elevation and 30° azimuth. Incomplete passes at search-window boundaries
are excluded. `pass_summary.csv` records actual rise/peak/set azimuths, peak
elevation, sample count and UTC peak for inspection.

To reproduce preparation in a new output directory:

```bash
python src/simulations/run_pass_snr_sweep.py prepare-additional \
  --base-campaign Results/wharf_real_pass_snr \
  --output Results/wharf_real_pass_snr_additional10 \
  --max-elevations 15 25 35 45 55 65 75 85 30 60 \
  --peak-azimuths 30 330 60 300 90 270 150 210 240 0
```

Preparation does not run RT. Select `CAMPAIGN = "additional10"` and execute
the notebook normally to run the new suite on both GPUs with 5-second-or-less
midpoint sampling, retaining the original beamforming and link-budget settings.

```bash
python src/simulations/run_pass_snr_sweep.py prepare
python src/simulations/run_pass_snr_sweep.py run
```

Default sampling covers the complete horizon-to-horizon pass (mask 0°), both
rising and setting legs. Equal-duration intervals of at most 5 s cover the
whole pass, with RT evaluated at each interval midpoint. Thus the fraction of
samples approximates the fraction of **time** in each elevation bin. The number
of RT solves depends on the selected real passes; every solve retains the source config's
10 million rays. For a link mask of 10° or finer time resolution:

```bash
python src/simulations/run_pass_snr_sweep.py prepare \
  --output Results/wharf_real_pass_snr_10deg --min-elevation 10 --step-s 1
python src/simulations/run_pass_snr_sweep.py run \
  --output Results/wharf_real_pass_snr_10deg
```

Position perturbation and Doppler time averaging are disabled. Each sample
uses the waypoint's actual azimuth and slant range. The ray cone includes
the full scene, as in the site pipeline; no additional satellite antenna gain
is introduced. Random scattering rays follow the existing runner's seeds.
This measures temporal link availability for one UE, not a receiver ensemble.

SNR is the band-mean received power of a flat-spectrum signal over the
bandwidth, i.e. the CIR convolved with that signal:

`H(f) = sum_i a_i exp(-j 2π f τ_i)`, f = baseband offset on a 15 kHz grid

`RX power [W] = TX power [W] * mean_f |H(f)|²`

`Noise [W] = k * 290 K * bandwidth [Hz] * 10^(noise figure [dB]/10)`

The CIR coefficients already contain the carrier phase, so only the baseband
offset is applied. `snr_samples.csv` also carries `snr_effective_db` (SNR with
the mean per-subcarrier capacity) and `snr_narrowband_db` (the carrier-only
`|sum a_i|²`, the earlier definition). Bandwidth defaults to **10 MHz**, receiver noise figure to **7 dB**;
both are explicit experiment assumptions, adjustable during preparation with
`--bandwidth-hz` and `--noise-figure-db`. SNR depends on those choices. No paths
means zero received power and SNR = -infinity, counted as an outage.

For each pass, bins are [0°,10°), [10°,20°), …, [80°,90°]. The last includes
90°. Both legs are pooled and the requested statistic is:

`100 * count(SNR > threshold in bin) / count(all samples in bin)`

The threshold defaults to **0 dB**, with strict `>`; equality does not pass.
Empty bins have count zero and NaN percentage, rather than 0% availability.
Incomplete passes fail analysis instead of reporting biased percentages.
You can change the threshold without repeating RT:

```bash
python src/simulations/run_pass_snr_sweep.py analyze --threshold-db 3
```

The stored `Results/wharf_real_pass_snr` campaign was ray traced at 43 dBm.
CIR amplitudes are pure channel gain, so `--tx-power-dbm 74` re-analyzes it at
the TR 38.821 S-band satellite EIRP (34 dBW/MHz over 10 MHz) without new RT:

```bash
python src/simulations/run_pass_snr_sweep.py analyze \
  --output Results/wharf_real_pass_snr --tx-power-dbm 74
```

Outputs in `Results/wharf_real_pass_snr/`:

- `campaign.json`: fixed UE, link-budget assumptions and orbit metadata for every pass.
- `catalog_snapshot.tle`: exact source catalog; its SHA-256 hash is recorded in the manifest.
- `simulation_config.json`, `ue.pkl`: exact inputs used by the site runner.
- `geometry.csv`, each pass's `waypoints.pkl`: full time/elevation/azimuth/range sequence.
- Each sample's results/checkpoint and `run.log`, retaining Sionna patch provenance.
- `snr_samples.csv`: per-sample SNR, geometry, satellite/NORAD ID, inclination and fixed UE XYZ.
- `elevation_bins.csv`: sample counts, success counts and percentage for each pass/bin.
- `pass_snr.png`: SNR over time and percentage above threshold versus elevation bin.

`run` skips completed sample directories and then analyzes the complete campaign.
Preparation refuses to overwrite a prepared campaign. Use another output
directory for changes to geometry, UE, bandwidth or noise figure. Analysis
checks saved result configuration and UE position against campaign inputs.
The earlier synthetic preparation in `Results/wharf_pass_snr/` is separate;
the revised runner rejects that campaign, so it cannot accidentally simulate
the synthetic inputs when a real orbit is requested.
