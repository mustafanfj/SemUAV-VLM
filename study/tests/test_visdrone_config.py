from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from semuav.config import (
    ConfigError,
    StudyScope,
    TruthMode,
    load_study_config,
)
from semuav.perception import (
    VISDRONE_GENERATION_POLICY,
    VISDRONE_GENERATION_POLICY_SHA256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class VisDronePaperConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        # These assertions describe checked-in configuration templates, not
        # environment-resolved live-run configuration.
        clean_environment = {
            name: value
            for name, value in os.environ.items()
            if not name.startswith("SEMUAV_")
        }
        environment_patch = patch.dict(
            os.environ, clean_environment, clear=True,
        )
        environment_patch.start()
        self.addCleanup(environment_patch.stop)

    def test_registered_visdrone_config_is_scientific_and_narrow(self) -> None:
        cfg = load_study_config(PROJECT_ROOT / "configs" / "visdrone_paper.yaml")
        self.assertEqual(cfg.study_id, "semuav_vlm_visdrone_paper_v3r2")
        self.assertTrue(
            str(cfg.paths.cache_root).endswith("/.cache/visdrone_paper_v3r2")
        )
        self.assertTrue(cfg.is_scientific)
        self.assertEqual(cfg.study_scope, StudyScope.VISDRONE_PRIMARY)
        self.assertEqual(cfg.truth_mode, TruthMode.PUBLISHER_SOURCE_RULE)
        self.assertEqual([item.adapter for item in cfg.enabled_datasets], ["visdrone"])
        self.assertEqual(cfg.dataset("visdrone").mission, "visdrone_urban")
        self.assertEqual(cfg.evaluation["hypotheses"], ["H1", "H2", "H3", "H4"])
        self.assertEqual(
            cfg.evaluation["energy_claim_scope"],
            "combined_sender_and_ground_gpu_board_nvml_compute_only_excludes_radio_system",
        )
        self.assertFalse(cfg.evaluation["radio_energy_included"])
        self.assertEqual(cfg.evaluation["not_applicable_hypotheses"], ["H5"])
        self.assertEqual(cfg.benchmark["occupancy_rule"], "center")
        self.assertEqual(cfg.benchmark["min_roi_visible_fraction"], 0.25)
        self.assertEqual(cfg.benchmark["min_source_score"], 0.50)
        self.assertEqual(cfg.models["input_size"], [768, 768])
        self.assertEqual(cfg.models["detector"]["image_size"], 768)
        self.assertTrue(
            cfg.models["detector"]["training_receipt"].endswith(
                "/semuav_visdrone_v1/frozen_inputs/DETECTOR_TRAINING_RECEIPT.json"
            )
        )
        self.assertEqual(
            cfg.models["semantic_architecture"],
            "detector_plus_bounded_vlm_verifier_plus_deterministic_host_composer_v2",
        )
        self.assertEqual(
            cfg.models["semantic_contract"],
            "semuav.visdrone-proposal-decision.v2",
        )
        self.assertEqual(
            cfg.runtime["inference_environment_id"], "release_20260912_v2r3",
        )
        self.assertEqual(
            cfg.runtime["pre_validation_runtime_correction_record"],
            "PRE_VALIDATION_NATIVE_RECEIVER_RECOVERY_20260921_V2R10.md",
        )
        self.assertEqual(
            cfg.runtime[
                "pre_validation_acceptance_recovery_isolation_record"
            ],
            "PRE_VALIDATION_ACCEPTANCE_RECOVERY_ISOLATION_20260921_V2R11.md",
        )
        self.assertEqual(
            cfg.runtime[
                "pre_smoke_predecessor_boundary_routing_correction_record"
            ],
            "PRE_SMOKE_PREDECESSOR_BOUNDARY_ROUTING_CORRECTION_20260921_V2R12.md",
        )
        self.assertEqual(
            cfg.runtime["pre_validation_low_evidence_packet_correction_record"],
            "PRE_VALIDATION_LOW_EVIDENCE_PACKET_CORRECTION_20260921_V2R13.md",
        )
        self.assertEqual(
            cfg.runtime["experiment1_recovery"],
            {
                "schema": "semuav.visdrone-v2r9-exp1-recovery.v1",
                "predecessor_release": "release_20260920_visdrone_paper_v2r9",
                "predecessor_run_name": "semuav_visdrone_v2r9",
                "imported_scope": "raw_experiment1_detector_and_vlm_shards_only",
                "derived_results_recomputed": True,
                "downstream_results_recomputed": True,
                "untouched_test_accessed": False,
            },
        )
        self.assertTrue(
            cfg.runtime["required_inference_python"].endswith(
                "/venvs/release_20260912_visdrone_inference_candidate_v2r3/bin/python"
            )
        )
        self.assertEqual(
            cfg.runtime["candidate_ready_receipt"],
            "/data/Lane Detection Datasets/musta/semuav_vlm/venvs/"
            "release_20260912_visdrone_inference_candidate_v2r3/"
            "SEMUAV_CANDIDATE_READY.json",
        )
        self.assertEqual(
            cfg.runtime["candidate_ready_receipt_schema"],
            "semuav.visdrone-inference-candidate-environment.v1",
        )
        self.assertEqual(
            cfg.runtime["candidate_ready_status"],
            "READY_FOR_DEVELOPMENT_SMOKE_ONLY",
        )
        self.assertEqual(
            cfg.runtime["inference_environment_seal_schema"],
            "semuav.visdrone-inference-environment-seal.v1",
        )
        self.assertEqual(
            cfg.runtime["inference_environment_seal_status"],
            "SEALED_FOR_SCIENTIFIC_VALIDATION_AND_TEST",
        )
        self.assertNotIn("moondream_tokenizer_cache", cfg.runtime)
        self.assertNotIn("moondream_tokenizer_lock", cfg.runtime)
        self.assertEqual(
            cfg.runtime["required_environment_guards"],
            {
                "CUDA_VISIBLE_DEVICES": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "YOLO_AUTOINSTALL": "false",
                "PIP_NO_INDEX": "1",
                "PIP_NO_INPUT": "1",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "SEMUAV_MODELS_LOCAL_ONLY": "1",
                "HF_HUB_CACHE": (
                    "/data/Lane Detection Datasets/musta/semuav_vlm/"
                    ".cache/huggingface"
                ),
                "HUGGINGFACE_HUB_CACHE": (
                    "/data/Lane Detection Datasets/musta/semuav_vlm/"
                    ".cache/huggingface"
                ),
                "TRANSFORMERS_CACHE": (
                    "/data/Lane Detection Datasets/musta/semuav_vlm/"
                    ".cache/huggingface"
                ),
                "HF_MODULES_CACHE": (
                    "/data/Lane Detection Datasets/musta/semuav_vlm/"
                    ".cache/huggingface/modules-v3r1"
                ),
            },
        )
        active_model_ids = [
            entry["id"] for entry in cfg.models["vlms"]
        ] + [cfg.models["upper_bound"]["id"]]
        self.assertEqual(
            active_model_ids,
            [
                "qwen3_vl_2b",
                "smolvlm2_2b",
                "qwen3_vl_8b_upper_bound",
            ],
        )
        self.assertNotIn("moondream2", active_model_ids)
        self.assertNotIn(
            "moondream2",
            cfg.models["generation_policy"]["strategy_by_model"],
        )
        self.assertNotIn("native_query", cfg.models["generation_policy"])
        generation_policy = dict(cfg.models["generation_policy"])
        self.assertEqual(
            generation_policy.pop("sha256"),
            VISDRONE_GENERATION_POLICY_SHA256,
        )
        self.assertEqual(generation_policy, VISDRONE_GENERATION_POLICY)
        reuse = cfg.models["reused_v1_artifacts"]
        self.assertEqual(
            reuse["receipt_schema"], "semuav.reused-v1-frozen-artifacts.v1",
        )
        self.assertEqual(reuse["source_run_name"], "semuav_visdrone_v1")
        self.assertEqual(len(reuse["expected_sha256"]), 8)
        self.assertTrue(all(
            re.fullmatch(r"[0-9a-f]{64}", str(value))
            for value in reuse["expected_sha256"].values()
        ))
        self.assertEqual(
            cfg.evaluation["model_selection"],
            {
                "estimand": "group_equal_critical_action_plus_object_exact_v2",
                "weights": {
                    "critical_action_accuracy": 0.5,
                    "objects_exact_accuracy": 0.5,
                },
                "tie_break": [
                    "selection_score_desc",
                    "critical_action_accuracy_desc",
                    "objects_exact_accuracy_desc",
                    "full_path_latency_p95_ms_asc",
                    "model_id_asc",
                ],
            },
        )
        self.assertEqual(
            cfg.benchmark["temporal_sampling"]["method"],
            "outcome_blind_mot_sequence_hash_contiguous_v1",
        )
        temporal = cfg.benchmark["temporal_sampling"]
        amendment = cfg.benchmark["temporal_feasibility_amendment"]
        self.assertEqual(temporal["validation_sequences"], 7)
        self.assertEqual(temporal["sequences_per_split"]["validation"], 7)
        self.assertEqual(temporal["test_sequences"], 15)
        self.assertEqual(temporal["sequences_per_split"]["test"], 15)
        self.assertEqual(cfg.network["validation_trace_realizations"], 12)
        self.assertEqual(
            cfg.network["validation_fully_paired_scene_trace_cells"], 84,
        )
        self.assertEqual(
            cfg.network["validation_fully_paired_scene_trace_cells"],
            temporal["sequences_per_split"]["validation"]
            * cfg.network["validation_trace_realizations"],
        )
        self.assertEqual(amendment, {
            "schema": "semuav.visdrone-temporal-feasibility-amendment.v1",
            "record": (
                "PRE_EXPERIMENT_TEMPORAL_FEASIBILITY_AMENDMENT_20260914_V2R7.md"
            ),
            "predecessor_release": "release_20260914_visdrone_paper_v2r6",
            "predecessor_run_name": "semuav_visdrone_v2r6",
            "predecessor_registered_validation_sequences": 15,
            "predecessor_observed_eligible_validation_sequences": 7,
            "amended_validation_sequences": 7,
            "retained_test_sequences": 15,
            "observations_per_sequence": 64,
            "validation_trace_realizations": 12,
            "amended_validation_fully_paired_scene_trace_cells": 84,
            "validation_development_bytes_read": True,
            "validation_outcomes_evaluated": False,
            "temporal_artifact_written": False,
            "cache_build_started": False,
            "experiments_1_to_7_started": False,
            "protocol_frozen": False,
            "untouched_test_accessed": False,
            "scientific_contract_changed": True,
        })
        self.assertEqual(
            temporal["sequences_per_split"]["test"]
            * cfg.network["held_out_trace_realizations"],
            450,
        )
        self.assertEqual(cfg.network["validation_static_min_independent_groups"], 15)
        self.assertEqual(temporal["minimum_temporal_groups"], 30)
        self.assertEqual(temporal["minimum_group_trace_cells"], 500)
        self.assertEqual(cfg.evaluation["mission_priority_positive_units"], 140)
        self.assertEqual(cfg.evaluation["mission_priority_negative_units"], 73)
        self.assertEqual(
            cfg.evaluation["minimum_calibration_independent_groups"], 30,
        )
        self.assertEqual(
            cfg.evaluation["calibration_fitting_estimand"],
            "visual_group_equal_temperature_nll_v1",
        )
        self.assertEqual(
            cfg.evaluation["calibration_bootstrap_upper_quantile"], 0.95,
        )

    def test_runtime_smoke_canary_is_exact_development_train_proposal_input(self) -> None:
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "visdrone_paper.yaml").read_text(
                encoding="utf-8"
            )
        )
        canary_path = PROJECT_ROOT / config["models"]["runtime_smoke_canary"]
        canary_bytes = canary_path.read_bytes()
        self.assertEqual(
            hashlib.sha256(canary_bytes).hexdigest(),
            config["models"]["runtime_smoke_canary_sha256"],
        )
        canary = json.loads(canary_bytes)
        self.assertEqual(
            canary["schema_version"],
            "semuav.visdrone-runtime-smoke-canary.v1",
        )
        self.assertEqual(canary["scope"], "development_registered_train_canary")
        self.assertFalse(canary["scientific_results_eligible"])
        self.assertEqual(canary["split"], "train")
        self.assertEqual(canary["component"], "det_train")
        self.assertEqual(canary["publisher_partition"], "train")
        self.assertEqual(canary["study_partition"], "development_train")
        self.assertFalse(Path(canary["image_relative_path"]).is_absolute())
        self.assertNotIn("..", Path(canary["image_relative_path"]).parts)
        proposals = canary["detector_proposals"]
        self.assertEqual(len(proposals), 32)
        self.assertEqual(canary["detector_proposal_count"], 32)
        source_bytes = (
            json.dumps(proposals, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
        ).encode("utf-8")
        self.assertEqual(
            hashlib.sha256(source_bytes).hexdigest(),
            canary["source_detector_proposals_sha256"],
        )
        self.assertEqual(
            canary["source_detector_proposals_sha256"],
            "5ece693644bf870124563475fe4186271f44fa4b5fc1397599af3060335093b7",
        )

    def test_visdrone_scope_rejects_total_system_energy_claim(self) -> None:
        source = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "visdrone_paper.yaml").read_text()
        )
        source["evaluation"]["energy_claim_scope"] = "total_system_measured"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "combined sender-and-ground"):
                load_study_config(path)

    def test_full_proposal_default_still_requires_all_four_datasets(self) -> None:
        source = yaml.safe_load((PROJECT_ROOT / "configs" / "full.yaml").read_text())
        source["datasets"] = {"visdrone": source["datasets"]["visdrone"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            with self.assertRaisesRegex(ConfigError, "all proposal datasets"):
                load_study_config(path)

    def test_source_rule_cannot_be_used_outside_visdrone_scope(self) -> None:
        source = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "visdrone_paper.yaml").read_text()
        )
        source["study_scope"] = "full_proposal"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsafe.yaml"
            path.write_text(yaml.safe_dump(source, sort_keys=False), encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_study_config(path)


if __name__ == "__main__":
    unittest.main()
