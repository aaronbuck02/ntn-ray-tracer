"""Exercise GPU scheduling and cleanup with real, CPU-only child processes."""

import json
import os
import pickle
import sys

import pytest

from Ray_Tracing.src.simulations import run_pass_snr_sweep as sweep


def setup_campaign(tmp_path, monkeypatch, fail=False):
    output = tmp_path / "campaign"
    folder = output / "pass_00"
    folder.mkdir(parents=True)
    (output / "campaign.json").write_text(json.dumps({"passes": [{
        "directory": folder.name,
        "orbit_source": "real catalog TLE propagated with SGP4",
        "tle_line1": "saved line 1", "tle_line2": "saved line 2",
    }]}))
    (output / "simulation_config.json").write_text("{}")
    (output / "ue.pkl").write_bytes(pickle.dumps({}))
    (folder / "waypoints.pkl").write_bytes(pickle.dumps({"waypoints": [{}, {}, {}, {}]}))
    repo = tmp_path / "repo"
    scripts = repo / "src/simulations"
    scripts.mkdir(parents=True)
    (scripts / "run_elevation_sim.py").write_text('''
import argparse, json, os, time
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--output')
p.add_argument('--gpu_id')
args, _ = p.parse_known_args()
assert args.gpu_id == '0'
gpu = os.environ['CUDA_VISIBLE_DEVICES']
dest = Path(args.output)
events = dest.parents[1] / 'events.jsonl'
def record(kind):
    with events.open('a') as f:
        f.write(json.dumps(dict(kind=kind, gpu=gpu, pid=os.getpid(), sample=dest.name)) + '\\n')
record('start')
if FAIL:
    if gpu == '0':
        time.sleep(0.15)
        raise SystemExit(7)
    time.sleep(30)
else:
    time.sleep(0.15)
(dest / 'results_elev001deg.pkl').write_bytes(b'result')
record('finish')
'''.replace('FAIL', repr(fail)))
    monkeypatch.setattr(sweep, "REPO", repo)
    collected = []
    monkeypatch.setattr(sweep, "collect", lambda *args: collected.append(args))
    return output, folder, collected


def test_two_workers_skip_completed_and_collect(tmp_path, monkeypatch):
    output, folder, collected = setup_campaign(tmp_path, monkeypatch)
    completed = folder / "sample_00000"
    completed.mkdir()
    (completed / "results_elev001deg.pkl").write_bytes(b"retained")
    sweep.run_campaign(output, [0, 1], sys.executable, 3, poll_interval=0.01)
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    active, maximum = set(), 0
    for event in events:
        if event["kind"] == "start":
            assert event["gpu"] not in active
            active.add(event["gpu"])
        else:
            active.remove(event["gpu"])
        maximum = max(maximum, len(active))
    assert maximum == 2 and not active
    assert len([e for e in events if e["kind"] == "start"]) == 3
    assert all(e["sample"] != "sample_00000" for e in events)
    assert (completed / "results_elev001deg.pkl").read_bytes() == b"retained"
    assert collected == [(output, 3)]


def test_failure_stops_other_worker_and_skips_analysis(tmp_path, monkeypatch):
    output, _, collected = setup_campaign(tmp_path, monkeypatch, fail=True)
    with pytest.raises(RuntimeError, match="exited 7 on GPU 0"):
        sweep.run_campaign(output, [0, 1], sys.executable, poll_interval=0.01)
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert len(events) == 2
    for event in events:
        with pytest.raises(ProcessLookupError):
            os.kill(event["pid"], 0)
    assert not collected


@pytest.mark.parametrize("gpu_ids", [[], [0, 0], [-1]])
def test_invalid_gpu_assignment_rejected(tmp_path, gpu_ids):
    with pytest.raises(ValueError, match="GPU IDs"):
        sweep.run_campaign(tmp_path, gpu_ids)
