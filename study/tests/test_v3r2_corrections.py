"""v3r2: receipted v3r1 corrections, controller revision, gate amendment, reuse."""
from __future__ import annotations

import dataclasses
import gzip
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from semuav import v3r1_import
from semuav.control import (
    CandidateProfile, ControllerObservation, FullPolicy, LinkEstimate, QueueSnapshot,
)
from semuav.protocol import PayloadMode, SenderReceiverBelief, StateKind
from semuav.experiments import (
    _drop_verified_episode_keys, build_causal_candidate_profiles,
    experiment3_static_transport, registered_wire_format,
)
from semuav.registry import RunContext, StudyGateError, frame_stage, hash_object
from semuav.visdrone_runner import (
    CONTROLLER_REVISION_V3R2, _controller_policies, _selected_calibration_trusted,
    _validation_policy_spec, calibration_gate_row,
)


def _observation(*, uncertainty: float = 0.0) -> ControllerObservation:
    return ControllerObservation(
        now_us=1_000_000, capture_us=1_000_000, action_deadline_us=5_000_000,
        link=LinkEstimate(1_000_000, 10, 100_000_000.0, 1_000.0, 0.0, 10_000.0),
        queue=QueueSnapshot(1_000_000, (0,) * 11, 0, 0, (None,) * 11, (0,) * 11),
        receiver_belief=SenderReceiverBelief(status="known"),
        mission_priority=0.0, risk=0.0, uncertainty=uncertainty,
        age_of_information_us=None, battery_fraction=0.8, thermal_c=55.0,
    )


def _candidate(mode: PayloadMode, *, success: float, evidence: float = 0.55,
               high_confidence_error: float = 0.0) -> CandidateProfile:
    return CandidateProfile(
        mode=mode, payload_bytes=20, generation_us=0, energy_j=0.0,
        predicted_task_success=success, predicted_high_confidence_error=high_confidence_error,
        evidence_strength=evidence, state_kind=StateKind.NONE, available_us=0,
        variant=mode.name.lower(),
    )


class WireFormatAndEpisodeKeyTests(unittest.TestCase):
    def test_registered_level_variants_are_cbor(self) -> None:
        for variant in ("l0_cbor", "l1_cbor", "l2_cbor", "l1_cbor__detector_only", "cbor:full", "cbor+roi"):
            self.assertEqual(registered_wire_format(variant), "cbor", variant)
        self.assertEqual(registered_wire_format("json-debug:x"), "json-debug")
        for variant in ("jpeg_q50_s1.0", "registered_feature_int8", "png_lossless", "l3_roi"):
            self.assertIsNone(registered_wire_format(variant), variant)

    @unittest.skipUnless(importlib.util.find_spec("cbor2"), "cbor2 is installed in the lab candidate")
    def test_level_cbor_messages_decode_in_static_transport(self) -> None:
        from semuav.protocol import encode_mapping

        manifest = pd.DataFrame([{"sample_id": "s1", "capture_time_us": 1_000_000,
                                  "trace_time_us": 1_000_000, "deadline_us": 5_000_000}])
        heartbeat = {"schema_version": "fixture", "observation_id": "s1", "mission": "visdrone_urban",
                     "capture_time_us": 1_000_000, "telemetry": {}}
        payload, wire_format = encode_mapping(heartbeat, scientific=True)
        self.assertEqual(wire_format, "cbor")
        candidates = pd.DataFrame([{
            "sample_id": "s1", "payload_blob": payload.hex(), "generation_ms": 1.0, "energy_j": 0.0,
            "receiver_latency_ms": 1.0, "receiver_energy_j": 0.0, "native_receiver_correct": True,
            "mode": "L0_HEARTBEAT", "variant": "l0_cbor", "source_model": "detector_only",
        }])
        result = experiment3_static_transport(
            candidates, manifest, seed=3, rates_mbps=(20.0,), rtts_ms=(20.0,), losses=(0.0,),
            loss_models=("iid",), jitters_ms=(0.0,), queue_kb=(512,),
        )
        self.assertFalse(result["apply_status"].astype(str).eq("decode_error").any())

    def test_episode_keys_are_verified_then_dropped(self) -> None:
        manifest = pd.DataFrame({"sample_id": ["a", "b"], "visual_group_id": ["g1", "g2"],
                                 "mission_profile": ["m", "m"]})
        candidates = pd.DataFrame({"sample_id": ["a", "a", "b"], "visual_group_id": ["g1", "g1", "g2"],
                                   "mission_profile": ["m", "m", "m"], "wire_bytes": [1, 2, 3]})
        cleaned = _drop_verified_episode_keys(candidates, manifest)
        self.assertEqual(list(cleaned.columns), ["sample_id", "wire_bytes"])
        conflicting = candidates.assign(visual_group_id=["g1", "g9", "g2"])
        with self.assertRaises(StudyGateError):
            _drop_verified_episode_keys(conflicting, manifest)
        untouched = candidates.drop(columns=["visual_group_id", "mission_profile"])
        self.assertIs(_drop_verified_episode_keys(untouched, manifest), untouched)


def _profile_rows() -> pd.DataFrame:
    rows = []
    # Variant A predicts RESPOND for everything: raw accuracy 0.75, balanced 0.5.
    # Variant B is right on every NO_ACTION and half of RESPOND: raw 0.625, balanced 0.75.
    gold = {"s1": "RESPOND", "s2": "RESPOND", "s3": "RESPOND", "s4": "RESPOND",
            "s5": "RESPOND", "s6": "RESPOND", "n1": "NO_ACTION", "n2": "NO_ACTION"}
    for sample, action in gold.items():
        rows.append({"sample_id": sample, "mode": "L1_ALERT", "variant": "a",
                     "native_receiver_correct": action == "RESPOND"})
        rows.append({"sample_id": sample, "mode": "L2_SEMANTICS", "variant": "b",
                     "native_receiver_correct": action == "NO_ACTION" or sample in {"s1", "s2", "s3"}})
    frame = pd.DataFrame(rows)
    for column, value in (("wire_bytes", 1000), ("generation_ms", 1.0), ("energy_j", 0.1),
                          ("receiver_latency_ms", 1.0), ("receiver_energy_j", 0.1)):
        frame[column] = value
    return frame, gold


class ProfileRevisionTests(unittest.TestCase):
    def test_balanced_success_and_untrusted_placeholder(self) -> None:
        frame, gold = _profile_rows()
        raw = build_causal_candidate_profiles(frame, list(gold)).set_index("variant")
        self.assertAlmostEqual(raw.loc["a", "predicted_task_success"], 0.75)
        self.assertEqual(raw.loc["a", "predicted_high_confidence_error"], 1.0)
        revised = build_causal_candidate_profiles(
            frame, list(gold), gold_actions=gold, calibration_trusted=False,
        ).set_index("variant")
        self.assertAlmostEqual(revised.loc["a", "predicted_task_success"], 0.5)
        self.assertAlmostEqual(revised.loc["b", "predicted_task_success"], 0.75)
        self.assertEqual(revised.loc["a", "predicted_high_confidence_error"], 0.0)
        supported = frame.assign(calibrated_correctness_probability=0.95)
        untrusted_supported = build_causal_candidate_profiles(
            supported, list(gold), gold_actions=gold, calibration_trusted=False,
        ).set_index("variant")
        self.assertTrue(untrusted_supported["predicted_high_confidence_error"].eq(0.0).all())
        trusted_supported = build_causal_candidate_profiles(
            supported, list(gold), gold_actions=gold, calibration_trusted=True,
        ).set_index("variant")
        self.assertAlmostEqual(trusted_supported.loc["a", "predicted_high_confidence_error"], 0.25)
        trusted = build_causal_candidate_profiles(
            frame, list(gold), gold_actions=gold, calibration_trusted=True,
        ).set_index("variant")
        self.assertEqual(trusted.loc["a", "predicted_high_confidence_error"], 1.0)
        with self.assertRaises(StudyGateError):
            build_causal_candidate_profiles(frame, list(gold), gold_actions={"s1": "RESPOND"})


class ControllerRevisionTests(unittest.TestCase):
    def test_untrusted_calibration_disables_evidence_escalation(self) -> None:
        candidates = (
            _candidate(PayloadMode.L1_ALERT, success=0.60),
            _candidate(PayloadMode.L3_SEMANTICS_ROI, success=0.50, evidence=0.9),
        )
        uncertain = _observation(uncertainty=0.9)
        self.assertEqual(FullPolicy().select(uncertain, candidates).mode, PayloadMode.L3_SEMANTICS_ROI)
        self.assertEqual(
            FullPolicy(calibration_trusted=False).select(uncertain, candidates).mode,
            PayloadMode.L1_ALERT,
        )
        placeholder = (
            _candidate(PayloadMode.L1_ALERT, success=0.60, high_confidence_error=1.0),
            _candidate(PayloadMode.L3_SEMANTICS_ROI, success=0.50, evidence=0.9),
        )
        calm = _observation(uncertainty=0.0)
        self.assertEqual(FullPolicy().select(calm, placeholder).mode, PayloadMode.L3_SEMANTICS_ROI)

    def test_policy_spec_flag_reaches_every_full_policy_variant(self) -> None:
        cfg = mock.Mock()
        cfg.controller = {
            "minimum_dwell_us": 2_000_000, "hysteresis_margin": 0.04,
            "risk_escalation": 0.7, "uncertainty_escalation": 0.35, "byte_weight": 0.02,
            "energy_weight": 0.08, "evidence_weight": 0.3, "deadline_weight": 1.0,
            "vlm_trigger_risk": 0.45, "vlm_trigger_uncertainty": 0.35,
        }
        cfg.hardware = {"thermal_limit_c": 82}
        spec = {
            "fixed": [{"name": "fixed_l1_alert", "mode": "L1_ALERT", "variant": "l1_cbor"}],
            "best_fixed": {"mode": "L1_ALERT", "variant": "l1_cbor"},
            "feature_baseline_available": False, "calibration_trusted": False,
        }
        frontier = pd.DataFrame({"mode": ["L2_SEMANTICS"], "variant": ["l2_cbor__detector_only"]})
        with mock.patch("semuav.visdrone_runner._section", side_effect=lambda c, name: getattr(c, name)):
            policies = _controller_policies(cfg, spec, frontier)
        full = [policy for policy in policies if isinstance(policy, FullPolicy)]
        self.assertGreaterEqual(len(full), 6)
        self.assertTrue(all(policy.calibration_trusted is False for policy in full))


class PolicySpecAndGateTests(unittest.TestCase):
    def _spec(self, under_l4: bool):
        cfg = mock.Mock()
        cfg.evaluation = {"primary_wire_budget_bytes_per_s": 10_000.0, "primary_energy_budget_j_per_s": 100.0}
        cfg.controller = {"risk_escalation": 0.7}
        manifest = pd.DataFrame({"sample_id": ["s1", "s2"], "trace_time_us": [0, 200_000],
                                 "sequence_id": ["q", "q"], "mission_profile": ["m", "m"]})
        profiles = pd.DataFrame([
            {"mode": "L1_ALERT", "variant": "l1_cbor", "observations": 2, "payload_bytes": 1000,
             "total_compute_energy_j": 0.1, "total_processing_latency_ms": 1.0, "predicted_task_success": 0.52},
            {"mode": "L4_FEATURES", "variant": "int8", "observations": 2,
             "payload_bytes": 1000 if under_l4 else 47_000,
             "total_compute_energy_j": 0.1, "total_processing_latency_ms": 1.0, "predicted_task_success": 0.70},
        ])
        with mock.patch("semuav.visdrone_runner._section", side_effect=lambda c, name: getattr(c, name)):
            return _validation_policy_spec(cfg, profiles, manifest, calibration_trusted=False, balanced_success=True)

    def test_best_fixed_is_rate_screened(self) -> None:
        spec = self._spec(under_l4=False)
        self.assertEqual(spec["best_fixed"]["mode"], "L1_ALERT")
        self.assertIs(spec["calibration_trusted"], False)
        self.assertEqual(spec["controller_revision"], CONTROLLER_REVISION_V3R2)
        self.assertEqual(spec["predicted_task_success_estimand"], "class_balanced_receiver_recall_v3r2")
        self.assertEqual(self._spec(under_l4=True)["best_fixed"]["mode"], "L4_FEATURES")

    def test_calibration_gate_amendment_routes_only_untrusted_revised_controller(self) -> None:
        revised = {"calibration_trusted": False, "controller_revision": CONTROLLER_REVISION_V3R2}
        routed = calibration_gate_row(calibration_pass=False, selected_model="detector_only", policy_spec=revised)
        self.assertTrue(routed["passed"])
        self.assertFalse(routed["calibration_measurement_passed"])
        self.assertTrue(routed["calibration_amendment_routed"])
        self.assertIn("calibration=FAIL", routed["detail"])
        for spec in ({"calibration_trusted": True, "controller_revision": CONTROLLER_REVISION_V3R2},
                     {"calibration_trusted": False}, {}):
            row = calibration_gate_row(calibration_pass=False, selected_model="m", policy_spec=spec)
            self.assertFalse(row["passed"], spec)
        passed = calibration_gate_row(calibration_pass=True, selected_model="m", policy_spec={})
        self.assertTrue(passed["passed"])
        self.assertFalse(passed["calibration_amendment_routed"])

    def test_selected_calibration_trust_reads_the_selected_model_row(self) -> None:
        selection = {"selected_model": "detector_only", "calibration_gate": [
            {"model_id": "detector_only", "passed": False}, {"model_id": "qwen3_vl_2b", "passed": True}]}
        self.assertFalse(_selected_calibration_trusted(selection))
        selection["selected_model"] = "qwen3_vl_2b"
        self.assertTrue(_selected_calibration_trusted(selection))
        selection["selected_model"] = "absent"
        with self.assertRaises(StudyGateError):
            _selected_calibration_trusted(selection)


def _gzip_csv(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as handle:
        handle.write(frame.to_csv(index=False).encode())
    return buffer.getvalue()


class V3R1ImportTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        project = root / "project"
        project.mkdir()
        self.successor_config = {"paths": {"project_root": str(project)}, "grid": [1, 2], "study_id": "v3r2",
                                 "runtime": {"pre_test_calibration_gate_amendment_record": "x.md"}}
        self.ctx = RunContext.create(project, root / "cache", "full", "semuav_visdrone_v3r2",
                                     self.successor_config)
        predecessor_config = {"paths": {"project_root": "/old/project"}, "grid": [1, 2], "study_id": "v3r1",
                              "runtime": {}}
        config_hash = hash_object(predecessor_config)
        code = v3r1_import.PREDECESSOR_CODE_SHA256
        self.run_id = f"semuav_visdrone_v3r1-full-{config_hash[:10]}-{code[:10]}"
        for name, value in (("PREDECESSOR_CONFIG_PREFIX", config_hash[:10]), ("PREDECESSOR_RUN_ID", self.run_id)):
            patcher = mock.patch.object(v3r1_import, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        state = mock.patch.dict(v3r1_import._STATE, {"receipt": None, "imported": [], "computed": [], "results": []})
        state.start()
        self.addCleanup(state.stop)
        self.run_root = root / "old_cache" / "runs" / self.run_id
        self.run_root.mkdir(parents=True)
        manifest = (json.dumps({"run_name": "semuav_visdrone_v3r1", "profile": "full", "code_hash": code,
                                "config_hash": config_hash, "config": predecessor_config}) + "\n").encode()
        (self.run_root / "run_manifest.json").write_bytes(manifest)
        self.key_material = {"candidate_hash": "abc", "seed": 1}
        self.frame = pd.DataFrame({"x": [1, 2, 3]})
        stages = []
        for name in ("03_static_transport_seed_1_shard_00000", "04_policy_partition_00000"):
            key = hash_object({"name": name, "key": self.key_material, "config": config_hash, "code": code})
            payload = _gzip_csv(self.frame)
            path = self.run_root / "stages" / name / f"{key}.csv.gz"
            path.parent.mkdir(parents=True)
            path.write_bytes(payload)
            if name.startswith("03"):
                stages.append({"name": name, "cache_key": key, "relative_path": f"stages/{name}/{key}.csv.gz",
                               "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(), "rows": 3})
        native = _gzip_csv(pd.DataFrame({"sample_id": ["a"]}))
        (self.run_root / "results").mkdir()
        (self.run_root / "results" / "validation_detector_only_native_receiver_cache.csv.gz").write_bytes(native)
        results = [{"relative_path": "results/validation_detector_only_native_receiver_cache.csv.gz",
                    "bytes": len(native), "sha256": hashlib.sha256(native).hexdigest()}]
        receipt = {
            "schema_version": v3r1_import.IMPORT_SCHEMA, "status": v3r1_import.IMPORT_STATUS,
            "predecessor_run_name": "semuav_visdrone_v3r1", "predecessor_run_root": str(self.run_root),
            "predecessor_code_sha256": code, "predecessor_config_sha256": config_hash,
            "run_manifest": {"bytes": len(manifest), "sha256": hashlib.sha256(manifest).hexdigest()},
            "untouched_test_accessed": False, "experiment4_partitions_imported": False,
            "stages": stages, "results": results,
            "inventory_sha256": hash_object({"stages": stages, "results": results}),
        }
        receipt_path = root / "receipt.json"
        receipt_path.write_bytes(json.dumps(receipt).encode())
        environment = mock.patch.dict(os.environ, {
            v3r1_import.ENABLE_VARIABLE: "1",
            v3r1_import.RECEIPT_VARIABLE: str(receipt_path),
            v3r1_import.RECEIPT_SHA256_VARIABLE: hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
        })
        environment.start()
        self.addCleanup(environment.stop)

    def _fail(self):
        raise AssertionError("an exact v3r1 stage must not be recomputed")

    def test_exact_stage_is_reused_and_changed_inputs_are_computed(self) -> None:
        reused = frame_stage(self.ctx, "03_static_transport_seed_1_shard_00000", self.key_material, self._fail)
        pd.testing.assert_frame_equal(reused, self.frame)
        record = self.ctx.stage_record("03_static_transport_seed_1_shard_00000")
        self.assertTrue(record["message"].startswith("IMPORTED_V3R1"))
        computed = frame_stage(self.ctx, "03_static_transport_seed_1_shard_00000",
                               {"candidate_hash": "changed", "seed": 1}, lambda: pd.DataFrame({"x": [9]}))
        self.assertEqual(computed["x"].tolist(), [9])
        report = v3r1_import.import_report()
        self.assertEqual((report["stages_reused"], report["stages_computed"]), (1, 1))

    def test_exact_shard_is_reused_after_its_index_shifts(self) -> None:
        shifted = frame_stage(self.ctx, "03_static_transport_seed_1_shard_00004", self.key_material, self._fail)
        pd.testing.assert_frame_equal(shifted, self.frame)
        record = self.ctx.stage_record("03_static_transport_seed_1_shard_00004")
        self.assertIn("from=03_static_transport_seed_1_shard_00000", record["message"])
        other_seed = frame_stage(self.ctx, "03_static_transport_seed_2_shard_00000", self.key_material,
                                 lambda: pd.DataFrame({"x": [4]}))
        self.assertEqual(other_seed["x"].tolist(), [4])

    def test_reusable_scope_matches_real_stage_names(self) -> None:
        pattern = v3r1_import.REUSABLE_STAGE_PATTERN
        for name in ("01_vlm_qwen3_vl_2b_shard_00003", "validation_detector",
                     "validation_02_representations_shard_00012",
                     "validation_02_detector_representations_shard_00000",
                     "validation_temporal_02_representations_shard_00041",
                     "validation_temporal_02_detector_representations_shard_00007",
                     "03_static_transport_seed_20260823_shard_00065"):
            self.assertIsNotNone(pattern.fullmatch(name), name)
        for name in ("04_policy_partition_00000", "04_policy_partition_aggregate", "test_detector",
                     "test_02_representations_shard_00000", "test_temporal_02_representations_shard_00000"):
            self.assertIsNone(pattern.fullmatch(name), name)

    def test_stale_pre_fix_static_rows_are_recomputed(self) -> None:
        name = "03_static_transport_seed_1_shard_00000"
        stale = pd.DataFrame({"variant": ["l1_cbor", "jpeg"], "wire_format": [None, None],
                              "apply_status": ["decode_error", "applied"]})
        record = next(iter(v3r1_import.load_receipt(self.ctx)["stages"].values()))
        payload = _gzip_csv(stale)
        path = self.run_root.joinpath(*Path(record["relative_path"]).parts)
        path.write_bytes(payload)
        record["bytes"], record["sha256"] = len(payload), hashlib.sha256(payload).hexdigest()
        record["rows"] = 2
        result = frame_stage(self.ctx, name, self.key_material, lambda: pd.DataFrame({"x": [8]}))
        self.assertEqual(result["x"].tolist(), [8])

    def test_controller_partitions_are_never_reused(self) -> None:
        result = frame_stage(self.ctx, "04_policy_partition_00000", self.key_material,
                             lambda: pd.DataFrame({"x": [7]}))
        self.assertEqual(result["x"].tolist(), [7])

    def test_result_caches_are_copied_exactly_and_tampering_fails(self) -> None:
        self.assertEqual(v3r1_import.import_result_caches(self.ctx),
                         ["validation_detector_only_native_receiver_cache.csv.gz"])
        self.assertEqual(v3r1_import.import_result_caches(self.ctx), [])
        stage = self.run_root / "stages" / "03_static_transport_seed_1_shard_00000"
        next(stage.iterdir()).write_bytes(b"tampered")
        with self.assertRaises(StudyGateError):
            frame_stage(self.ctx, "03_static_transport_seed_1_shard_00000", self.key_material, self._fail)

    def test_configuration_drift_and_test_access_are_refused(self) -> None:
        self.successor_config["grid"] = [1, 3]
        ctx = dataclasses.replace(self.ctx, config=self.successor_config)
        with self.assertRaisesRegex(StudyGateError, "differs from v3r1"):
            v3r1_import.load_receipt(ctx)
        self.successor_config["grid"] = [1, 2]
        (self.run_root / "test_access_ledger.json").write_text("{}")
        with self.assertRaises(StudyGateError):
            v3r1_import.load_receipt(self.ctx)

    def test_disabled_import_is_inert(self) -> None:
        with mock.patch.dict(os.environ, {v3r1_import.ENABLE_VARIABLE: "0"}):
            result = frame_stage(self.ctx, "03_static_transport_seed_1_shard_00000", self.key_material,
                                 lambda: pd.DataFrame({"x": [5]}))
        self.assertEqual(result["x"].tolist(), [5])


if __name__ == "__main__":
    unittest.main()
