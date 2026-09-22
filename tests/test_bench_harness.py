"""The benchmark harness selects experimental winners, so its gates and ratio
definitions are tested with deliberately corrupted records. No GPU needed."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "benchmarks"))
bench = pytest.importorskip("bench_single_gather")

GOOD = {"rel_l2": 0.001, "max_norm": 0.001, "finite": True}
BAD = {"rel_l2": 0.5, "max_norm": 0.5, "finite": True}


def record(caps=("fwd", "bwd"), lse_kind="natural", **over):
    r = {"provider": "p", "caps": sorted(caps), "lse_kind": lse_kind, "status": "pass",
         "fwd_ok": True, "lse_ok": True if lse_kind == "natural" else None, "bwd_ok": True,
         "repeat_ok": True, "full_shape_ok": True, "Y": GOOD, **{g: GOOD for g in bench.GRADS}}
    r.update(over)
    return r


def test_clean_record_passes_every_gate():
    r = record()
    assert all(bench.gate(r, op) for op in ("fwd", "bwd", "fwd_bwd", "api_fwd", "api_fwd_bwd"))


def test_corrupted_Y_blocks_forward_and_backward():
    r = record(fwd_ok=False, Y=BAD)
    assert not bench.gate(r, "fwd")
    assert not bench.gate(r, "bwd"), "a backward needs its own forward state validated"
    assert not bench.gate(r, "fwd_bwd")


def test_corrupted_lse_blocks_only_natural_claims():
    assert not bench.gate(record(lse_ok=False), "fwd")
    other = record(caps=("fwd",), lse_kind="other", lse_ok=None, lse_max_err=2.56)
    assert bench.gate(other, "fwd"), "a provider that declares another statistic is reported, not gated, on it"
    assert not bench.gate(other, "bwd"), "forward-only providers publish no backward"


def test_fbgemm_training_refuses_uncorrected_forward_state():
    # Forward Y can be correct while the stock mixed-base normalizer makes
    # every backward gradient wrong. Refuse that composition before launch.
    stock = bench.FbgemmTritonFwd.__new__(bench.FbgemmTritonFwd)
    with pytest.raises(ValueError, match="natural-LSE"):
        bench.FbgemmTraining(stock)


def test_one_bad_gradient_blocks_backward_not_forward():
    r = record(bwd_ok=False, dVs=BAD)
    assert bench.gate(r, "fwd")
    assert not bench.gate(r, "bwd")
    assert not bench.gate(r, "fwd_bwd")
    assert not bench.gate(r, "api_fwd_bwd")


def test_full_shape_and_repeat_checks_update_the_gate():
    assert not bench.gate(record(full_shape_ok=False), "fwd")
    assert bench.gate(record(repeat_ok=False), "fwd")
    assert not bench.gate(record(repeat_ok=False), "bwd")


def test_failed_status_and_unsupported_carry_no_ratio():
    for status in ("error", "oom", "unsupported", "unavailable", "compile_error"):
        assert not bench.gate(record(status=status), "fwd")
    assert not bench.gate(record(), "legacy_state_fwd"), "component metrics carry no ratio"


def _run_dir(tmp_path, name, session, records, timings, rounds=3, iters=4):
    """Synthetic session dir: `records` = {provider: correctness record}, `timings` =
    {provider: {op: (event_ms, wall_ms)}}."""
    import json
    d = tmp_path / f"{name}_s{session}"
    d.mkdir()
    cfg = bench.make_cfg(1, 1, 1, 128, 64)
    json.dump({"run_id": d.name, "session": session, "matrix": [cfg],
               "providers": {p: {"ops": list(t)} for p, t in timings.items()}}, open(d / "manifest.json", "w"))
    json.dump({cfg["name"]: {"providers": records}}, open(d / "correctness.json", "w"))
    with open(d / "samples.jsonl", "w") as f:
        for rnd in range(rounds):
            for p, ops in timings.items():
                for op, (ev, wall) in ops.items():
                    for i in range(iters):
                        f.write(json.dumps({"config": cfg["name"], "provider": p, "op": op, "session": session,
                                            "round": rnd, "iter": i, "event_ms": ev, "wall_ms": wall}) + "\n")
    return str(d)


def test_reference_own_gate_suppresses_ratio(tmp_path):
    """The reference fails its backward gate, so bwd speedups vanish while fwd survive;
    event and wall speedups come from their own columns (event != wall in the samples)
    and the API op is judged on wall time."""
    d = _run_dir(tmp_path, "run", 0, {"ref": record(bwd_ok=False, dQ=BAD), "p": record()},
                 {"ref": {"fwd": (1.0, 1.5), "bwd": (1.0, 1.5), "api_fwd": (1.0, 1.5)},
                  "p": {"fwd": (2.0, 6.0), "bwd": (2.0, 6.0), "api_fwd": (2.0, 6.0)}})
    rows, paired = bench.analyze([d], str(tmp_path / "out"), "ref")
    p_fwd = next(r for r in rows if r["provider"] == "p" and r["op"] == "fwd")
    p_bwd = next(r for r in rows if r["provider"] == "p" and r["op"] == "bwd")
    assert p_fwd["event_speedup_ref_vs_provider"] == pytest.approx(2.0)
    assert p_fwd["wall_speedup_ref_vs_provider"] == pytest.approx(4.0), "wall speedup must come from wall times"
    assert p_fwd["event_speedup_ci95_lo"] == pytest.approx(2.0) and p_fwd["event_speedup_ci95_hi"] == pytest.approx(2.0)
    assert p_bwd["event_speedup_ref_vs_provider"] is None, "reference failed its own backward gate"
    assert {x["op"] for x in paired} == {"fwd", "api_fwd"}
    api = next(x for x in paired if x["op"] == "api_fwd")
    assert api["primary_metric"] == "wall" and api["provider_median_ms"] == pytest.approx(6.0) and api["ref_faster"]


def test_known_win_and_loss_are_labelled_from_the_reference(tmp_path):
    """A provider twice as slow is a 2x speedup of the reference (ref_faster); a provider
    twice as fast is 0.5x (provider_faster); values, intervals and win counts agree."""
    d = _run_dir(tmp_path, "run", 0, {"ref": record(), "slow": record(), "fast": record()},
                 {"ref": {"fwd": (1.0, 1.0)}, "slow": {"fwd": (2.0, 2.0)}, "fast": {"fwd": (0.5, 0.5)}})
    rows, paired = bench.analyze([d], str(tmp_path / "out"), "ref")
    slow = next(x for x in paired if x["provider"] == "slow")
    fast = next(x for x in paired if x["provider"] == "fast")
    assert slow["event_speedup_ref_vs_provider"] == pytest.approx(2.0) and slow["ref_faster"] and not slow["provider_faster"]
    assert fast["event_speedup_ref_vs_provider"] == pytest.approx(0.5) and fast["provider_faster"] and not fast["ref_faster"]
    fam = bench.family_summary(paired, {bench.make_cfg(1, 1, 1, 128, 64)["name"]: bench.make_cfg(1, 1, 1, 128, 64)})
    by = {f["provider"]: f for f in fam}
    assert by["slow"]["ref_faster"] == 1 and by["slow"]["provider_faster"] == 0
    assert by["fast"]["provider_faster"] == 1 and by["fast"]["geomean_speedup_ref_vs_provider"] == pytest.approx(0.5)


def test_per_head_only_failure_blocks_the_gate():
    """Global metrics inside the threshold but one (b, h) slice outside: not within."""
    m = dict(GOOD, bh_worst_rel_l2=0.5, bh_worst_max_norm=0.001)
    assert not bench.within("Y", m)
    assert bench.within("Y", dict(GOOD, bh_worst_rel_l2=0.001, bh_worst_max_norm=0.001))


def test_listing3_composite_publishes_complete_operation():
    """Listing 3's forward is the ordinary Listing 1, so its fwd_bwd samples are eligible;
    the legacy three-gather state stays ineligible for a single-gather forward claim."""
    l3 = record(caps=bench.PaperTritonSmallW2.caps)
    assert bench.gate(l3, "fwd_bwd") and bench.gate(l3, "bwd")
    legacy = record(caps=bench.CudaLegacyUnspecialized.caps)
    assert bench.gate(legacy, "bwd") and not bench.gate(legacy, "fwd") and not bench.gate(legacy, "fwd_bwd")


def test_pass_in_one_session_does_not_hide_failure_in_another(tmp_path):
    d0 = _run_dir(tmp_path, "run", 0, {"ref": record(), "p": record()}, {"ref": {"fwd": (1.0, 1.0)}, "p": {"fwd": (2.0, 2.0)}})
    d1 = _run_dir(tmp_path, "run", 1, {"ref": record(), "p": record(fwd_ok=False, Y=BAD)}, {"ref": {"fwd": (1.0, 1.0)}, "p": {"fwd": (2.0, 2.0)}})
    rows, paired = bench.analyze([d0, d1], str(tmp_path / "out"), "ref")
    p = next(r for r in rows if r["provider"] == "p")
    assert p["gate_pass"] is False and p["event_speedup_ref_vs_provider"] is None
    assert not paired


def test_rerender_adapter_recomputes_gates_from_metrics():
    man = {"providers": {"cuda_specialized": {"ops": ["fwd", "bwd", "fwd_bwd"]},
                         "fbgemm_triton_fwd": {"ops": ["fwd"]},
                         "paper_triton_listing3_w32": {"ops": ["bwd", "fwd_bwd"]}}}
    old = {"providers": {
        "cuda_specialized": {"pass": True, "Y": GOOD, "lse_ok": True, "lse_max_err": 0.001, **{g: GOOD for g in bench.GRADS}},
        "fbgemm_triton_fwd": {"pass": True, "Y": GOOD, "lse_ok": False, "lse_max_err": 2.56},
        "paper_triton_listing3_w32": {"pass": False, **{g: GOOD for g in bench.GRADS[:-1]}, "dVs": BAD}}}
    new = bench.adapt_correctness(old, man)["providers"]
    assert bench.gate(new["cuda_specialized"], "fwd_bwd")
    assert bench.gate(new["fbgemm_triton_fwd"], "fwd"), "FBGEMM's statistic is not a natural-log LSE claim"
    assert not bench.gate(new["paper_triton_listing3_w32"], "bwd")


def test_calibration_bounds():
    assert bench.calibrate_repeats([0.01], 150) == 200
    assert bench.calibrate_repeats([500.0], 150) == 5
    assert bench.calibrate_repeats([1.0, 4.0], 150) == 75      # geometric mean 2 ms


def test_matrix_dedup_keeps_all_families():
    m = bench.build_matrix(["core", "anchors"])
    names = [c["name"] for c in m]
    assert len(names) == len(set(names))
    anchor = next(c for c in m if c["name"] == "B1_H4_D128_N256_wfull")
    assert set(anchor["families"]) == {"core", "anchors"}
    assert bench.cfg_name(bench.make_cfg(1, 64, 1, 256, 128, 32)) == "B1_Hq64_Hkv1_D128_N256_w32"
    assert bench.cfg_name(bench.make_cfg(1, 4, 4, 256, 128, mask="none")) == "B1_H4_D128_N256_unmasked"


def test_useful_flops_counts_visibility_only():
    full = bench.make_cfg(1, 1, 1, 128, 64)
    win = bench.make_cfg(1, 1, 1, 128, 64, 32)
    none = bench.make_cfg(1, 1, 1, 128, 64, mask="none")
    assert bench.useful_flops(win) < bench.useful_flops(full) < bench.useful_flops(none)
    assert bench.useful_flops(none) == 4 * 64 * 128 ** 3
