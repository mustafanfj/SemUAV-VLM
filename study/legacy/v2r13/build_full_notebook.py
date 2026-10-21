from __future__ import annotations

import json
import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "SemUAV_VLM_Full_Study.ipynb"


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(source).strip() + "\n"}


def code(source: str, *tags: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {"tags": list(tags)} if tags else {},
        "outputs": [],
        "source": textwrap.dedent(source).strip() + "\n",
    }


cells = [
    md("""
    # SemUAV-VLM: full proposal study

    **Adaptive semantic communication for bandwidth-constrained UAV visual intelligence**

    This notebook is the experiment orchestrator for the complete proposal. The
    substantive implementation lives in the adjacent `semuav/` package so GPU
    stages can be tested, cached, resumed, and run outside notebook cell state.

    Scientific guardrail: generated fixtures and mock models are software tests,
    never findings. A publishable result requires the `full` profile, real licensed
    data, two blinded annotators plus adjudication, frozen group-safe splits,
    pinned model commits, registered native receivers, held-out traces, one locked
    test execution, and measured target-hardware costs.
    """),
    md("""
    ## Study endpoint and confirmatory rule

    The primary endpoint is the first **correct committed receiver action by the
    absolute mission deadline measured from capture**. Missing, malformed, stale,
    base-mismatched, or late outputs are failures; a later correction does not
    rescue an earlier wrong action.

    The main claim is eligible only when the full policy satisfies the frozen wire
    and energy caps and its Holm-controlled, two-way clustered paired-bootstrap
    lower confidence bound exceeds every registered deployable baseline by more
    than 3 percentage points. A privileged one-step heuristic lookahead is
    reported separately; it is neither an oracle nor an upper bound because it
    does not counterfactually replay queues, retries, or receiver state.
    """),
    md("""
    ## Lab installation (run once in the remote terminal)

    ```bash
    STUDY_ROOT="/data/Lane Detection Datasets/musta/semuav_vlm"
    REPO_ROOT="$STUDY_ROOT/releases/release_20260830/SemUAV-VLM-Full-Study"
    VENV_ROOT="$STUDY_ROOT/venvs/release_20260830"
    cd "$REPO_ROOT"
    python3 -m venv "$VENV_ROOT"
    source "$VENV_ROOT/bin/activate"
    python -m pip install --no-cache-dir --upgrade pip
    python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu128 torch==2.7.0 torchvision==0.22.0
    python -m pip install --no-cache-dir -r requirements.txt
    python -c "from semuav.runtime import resolve_ffmpeg_executable, probe_ffmpeg_encoder; executable = resolve_ffmpeg_executable(); available, detail = probe_ffmpeg_encoder(executable, 'libx264') if executable else (False, 'FFmpeg unavailable'); print(detail); assert available"
    python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); assert torch.cuda.is_available()"
    python -m ipykernel install --user --name semuav-vlm --display-name "Python (SemUAV-VLM)"
    ```

    Keep Hugging Face, Torch, pip, output, and checkpoint storage below `/data`,
    not `/home`. The complete Windows VPN/SSH/tunnel procedure is in `README.md`.
    """),
    code("""
    # EDIT ONLY THIS CELL for ordinary operation.
    import os
    from pathlib import Path

    RUN_PROFILE = os.environ.get("SEMUAV_PROFILE", "dry_run")  # dry_run | smoke_real | full
    FULL_PHASE = os.environ.get("SEMUAV_FULL_PHASE", "prepare")  # prepare | validation | test | embedded
    PROJECT_ROOT = Path(os.environ.get("SEMUAV_PROJECT_ROOT", Path.cwd())).resolve()
    CACHE_ROOT = Path(os.environ.get("SEMUAV_CACHE_ROOT", PROJECT_ROOT / ".cache" / RUN_PROFILE)).resolve()
    os.environ.setdefault("MPLCONFIGDIR", str(CACHE_ROOT / "matplotlib"))
    RUN_NAME = os.environ.get("SEMUAV_RUN_NAME", "semuav_v1")
    DEVICE = os.environ.get("SEMUAV_DEVICE", "cuda:0" if RUN_PROFILE != "dry_run" else "cpu")
    RESUME = True

    # Publisher acquisition is status-only unless both download controls are explicit.
    DATASET_ACTION = os.environ.get("SEMUAV_DATASET_ACTION", "status").strip().lower()
    ACQUISITION_ROOT_TEXT = os.environ.get("SEMUAV_ACQUISITION_ROOT", "").strip()
    DATASETS = tuple(
        item.strip().lower()
        for item in os.environ.get(
            "SEMUAV_DATASETS", "visdrone,uavid,uavreason,flame3"
        ).split(",")
        if item.strip()
    )
    ACCEPT_DATASET_TERMS = os.environ.get(
        "SEMUAV_ACCEPT_DATASET_TERMS", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    if DATASET_ACTION not in {"status", "download", "prepare"}:
        raise ValueError(
            "SEMUAV_DATASET_ACTION must be 'status', 'download', or 'prepare'"
        )
    if not DATASETS:
        raise ValueError("SEMUAV_DATASETS must name at least one dataset")

    if RUN_PROFILE == "full" and FULL_PHASE in {"test", "embedded"}:
        # Locked phases may consume only snapshots prefetched during validation.
        os.environ["SEMUAV_MODELS_LOCAL_ONLY"] = "1"
        print("Model snapshot policy: SEMUAV_MODELS_LOCAL_ONLY=1 (network fetch disabled)")

    # Expensive or irreversible scientific transitions require explicit opt-in.
    EXECUTE_FULL_STAGES = False
    FREEZE_SPLITS = False
    FREEZE_PROTOCOL = False
    RUN_UNTOUCHED_TEST_ONCE = False

    # Frozen receiver artifacts required by Experiment 2 in full mode.
    NATIVE_RECEIVER_CACHE = Path(os.environ.get("SEMUAV_NATIVE_RECEIVER_CACHE", "REQUIRED_NATIVE_RECEIVER_CACHE.csv.gz"))
    REGISTERED_FEATURE_CACHE = Path(os.environ.get("SEMUAV_FEATURE_CACHE", "REQUIRED_REGISTERED_FEATURE_CACHE.csv.gz"))
    EMBEDDED_RETEST_SUMMARY = Path(os.environ.get("SEMUAV_EMBEDDED_RETEST_SUMMARY", "REQUIRED_EMBEDDED_RETEST_SUMMARY.csv.gz"))
    print({
        "profile": RUN_PROFILE, "phase": FULL_PHASE,
        "project": str(PROJECT_ROOT), "device": DEVICE,
        "dataset_action": DATASET_ACTION, "datasets": DATASETS,
        "acquisition_root_override": ACQUISITION_ROOT_TEXT or None,
        "dataset_terms_accepted": ACCEPT_DATASET_TERMS,
    })
    """, "parameters"),
    code("""
    import gc, json, subprocess, sys, warnings
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    try:
        from IPython.display import display
    except ImportError:
        display = print

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from semuav import __version__
    from semuav.acquisition import (
        SUPPORTED_DATASETS, acquire_datasets, dataset_source_table,
        dataset_status, disk_preflight, prepare_dataset_manifests,
    )
    from semuav.config import Profile, load_config
    from semuav.figures import (
        build_hypothesis_effects_frame, generate_publication_figures,
    )
    from semuav.registry import (
        EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS,
        RunContext, StudyGateError, atomic_write_frame, atomic_write_json, frame_stage,
        authorize_test_once, confirmatory_gate_pass, finalize_test_once, freeze_protocol,
        frozen_file_manifest, hash_file, load_finalized_test_handoff,
        matched_coverage_semantic_errors, notebook_source_hash,
        resolve_frozen_artifact, selected_vlm_claim_gate, test_access_guard,
    )
    from semuav.hardware import (
        assert_environment_ready, embedded_gate, environment_fingerprint, environment_report,
        evaluate_embedded_retest, evaluate_h5_status,
        import_embedded_retest_summary, netem_command_plan, render_command,
    )
    from semuav.data import (
        apply_frozen_splits, assert_study_data_ready, assign_group_safe_splits,
        audit_annotations, audit_split_leakage, build_canonical_manifest,
        export_preparation_bundle, freeze_split_manifest, load_annotations,
        prepare_analysis_manifest,
    )
    from semuav.perception import (
        DetectorOnlyBackend, HFMultimodalBackend, ModelSpec, MoondreamBackend,
        TemperatureCalibrator, UltralyticsDetectorTracker,
        calibrated_semantic_packet_cache, calibration_metrics,
        detector_execution_revision,
        run_detector_cache,
    )
    from semuav.experiments import (
        EXPERIMENT6_CAMPAIGN_VERSION, EXPERIMENT6_REGISTERED_CASES,
        EXPERIMENT6_REGISTERED_STATE_STRATEGIES,
        EXPERIMENT6_STATE_STRATEGY_VERSION,
        add_experiment2_total_costs, build_causal_candidate_profiles,
        assert_experiment3_exact_replicate_coverage,
        assert_experiment3_partition_index_coverage,
        dry_run_all, experiment1_perception,
        experiment2_representation_frontier, experiment2_representation_frontier_sharded,
        experiment3_static_transport, experiment3_static_transport_partitioned,
        experiment4_controller_behavior, experiment5_risk_uncertainty,
        experiment6_receiver_robustness,
        experiment6_registered_campaign,
        experiment7_readiness, final_completion_status, run_paired_policy_trials,
        run_paired_policy_trials_sharded, select_experiment3_candidate_variants,
        select_experiment3_manifest, summarize_experiment3_static_transport,
    )
    from semuav.exp6_strategies import experiment6_state_strategy_comparison
    from semuav.control import (
        AdaptiveFeaturePolicy, AdaptiveImagePolicy, AdaptiveSemanticPolicy, AlwaysVLMPolicy, ChannelMissionPolicy,
        ChannelOnlyPolicy, DeadlineGreedyPolicy, FixedModePolicy, FixedVariantPolicy, FullPolicy,
        MissionOnlyPolicy, NoEvidenceEscalationPolicy, NoHysteresisPolicy,
        LookaheadReferencePolicy, NoMissionRiskPolicy, NoReceiverStatePolicy,
        RiskAwareNoUncertaintyPolicy,
        causal_schema_audit,
    )
    from semuav.protocol import PayloadMode
    from semuav.network import (
        EndBehavior, load_trace_design_registry, load_trace_manifest,
        scientific_episode_coverage_audit, scientific_trace_design_audit,
    )
    from semuav.evaluation import (
        assert_paired_design, gate_summary, holm_adjust, hypothesis_report,
        budget_compliance, minimum_sample_gate, pareto_front, resource_summary,
        two_way_clustered_paired_bootstrap, validation_power_simulation,
    )
    print(f"SemUAV package {__version__}; Python {sys.version.split()[0]}")
    """),
    code("""
    config_path = PROJECT_ROOT / "configs" / f"{RUN_PROFILE}.yaml"
    cfg = load_config(config_path)
    ctx = RunContext.create(
        project_root=PROJECT_ROOT,
        cache_root=CACHE_ROOT,
        profile=cfg.profile.value,
        run_name=RUN_NAME,
        config=cfg.as_dict(),
        resume=RESUME,
    )
    print("Run root:", ctx.run_root)
    print("Config hash:", ctx.config_hash)
    print("Code hash:", ctx.code_hash)
    print("Scientific status:", "SCIENTIFIC PROFILE; ALL GATES PENDING" if ctx.is_scientific else "NON_SCIENTIFIC")
    run_id = ctx.run_root.name
    PAPER_FIGURE_DIR = (
        cfg.paths.output_root / RUN_NAME / run_id / "paper_figures"
    ).resolve()
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    print("Publication figure output directory:", PAPER_FIGURE_DIR)
    ACQUISITION_ROOT_EXPLICIT = bool(ACQUISITION_ROOT_TEXT)
    ACQUISITION_ROOT = (
        Path(ACQUISITION_ROOT_TEXT).expanduser().resolve()
        if ACQUISITION_ROOT_EXPLICIT
        else cfg.paths.data_root.resolve()
    )
    print(
        "Dataset acquisition root:", ACQUISITION_ROOT,
        "(explicit override)" if ACQUISITION_ROOT_EXPLICIT else "(profile default)",
    )
    """),
    md("""
    ## Publisher-grounded dataset acquisition

    Source URLs, licence/terms notes, disk estimates, and local status are
    always visible. The default action is `status`; `prepare` rebuilds manifests
    from already extracted UAVReason/FLAME trees. Manual UAVid/FLAME ZIPs use
    the `download` action, which requires both
    `SEMUAV_DATASET_ACTION=download` and
    `SEMUAV_ACCEPT_DATASET_TERMS=1`. Select a comma-separated subset with
    `SEMUAV_DATASETS`. Dry-run mutation is blocked unless
    `SEMUAV_ACQUISITION_ROOT` names an explicit non-`.fixtures` destination.
    """),
    code("""
    unknown_datasets = sorted(set(DATASETS) - set(SUPPORTED_DATASETS))
    if unknown_datasets:
        raise ValueError(
            f"SEMUAV_DATASETS contains unsupported names: {unknown_datasets}; "
            f"choose from {list(SUPPORTED_DATASETS)}"
        )

    print("Publisher sources (review every landing page and its current terms):")
    display(dataset_source_table())

    acquisition_disk = disk_preflight(ACQUISITION_ROOT, DATASETS)
    acquisition_disk_row = {
        "data_root": str(ACQUISITION_ROOT),
        "selected_datasets": ",".join(DATASETS),
    }
    acquisition_disk_row.update(acquisition_disk)
    display(pd.DataFrame([acquisition_disk_row]))

    print("Dataset status before any requested action:")
    display(dataset_status(ACQUISITION_ROOT))

    acquisition_result = None
    acquisition_mutates = DATASET_ACTION in {"download", "prepare"}
    acquisition_targets_fixtures = ".fixtures" in {
        part.lower() for part in ACQUISITION_ROOT.parts
    }
    if acquisition_mutates and (
        acquisition_targets_fixtures
        or (RUN_PROFILE == "dry_run" and not ACQUISITION_ROOT_EXPLICIT)
    ):
        raise StudyGateError(
            "Dataset download/prepare cannot target the dry-run .fixtures tree. "
            "Set SEMUAV_ACQUISITION_ROOT to an explicit non-.fixtures data directory."
        )
    if DATASET_ACTION == "download":
        print("Opt-in download requested; publisher terms acknowledgement will be enforced.")
        acquisition_result = acquire_datasets(
            ACQUISITION_ROOT,
            DATASETS,
            accept_terms=ACCEPT_DATASET_TERMS,
        )
        display(acquisition_result)
        print("Dataset status after download:")
        display(dataset_status(ACQUISITION_ROOT))
    elif DATASET_ACTION == "prepare":
        print("Rebuilding manifests from already extracted publisher trees; no download will be attempted.")
        acquisition_result = prepare_dataset_manifests(ACQUISITION_ROOT, DATASETS)
        display(acquisition_result)
        if "status" not in acquisition_result.columns:
            raise StudyGateError("Dataset manifest preparation returned no status column")
        blocked_preparation = acquisition_result[
            acquisition_result["status"].astype(str).str.strip().str.lower().eq("blocked")
        ]
        if len(blocked_preparation):
            raise StudyGateError(
                "Dataset manifest preparation blocked: "
                + json.dumps(
                    blocked_preparation.to_dict(orient="records"),
                    sort_keys=True,
                    default=str,
                )
            )
        print("Dataset status after manifest preparation:")
        display(dataset_status(ACQUISITION_ROOT))
    else:
        print("Status-only acquisition mode; no network download was attempted.")
    """),
    code("""
    preflight = environment_report(PROJECT_ROOT, CACHE_ROOT, DEVICE, RUN_PROFILE)
    display(preflight)
    environment_fingerprint_payload = environment_fingerprint(preflight)
    environment_fingerprint_artifact = (
        ctx.run_root / "results" / f"{FULL_PHASE}_environment_fingerprint.json"
    )
    atomic_write_json(environment_fingerprint_artifact, environment_fingerprint_payload)
    prerequisite_stage = (
        "embedded" if FULL_PHASE == "embedded" else
        "test" if FULL_PHASE == "test" else
        "validation" if FULL_PHASE == "validation" else "ingest"
    )
    if RUN_PROFILE == "full" and FULL_PHASE == "prepare":
        # Preparation needs publisher media only. Annotation, telemetry, and
        # trace contracts are exported below and become strict at validation.
        prerequisites = [
            f"dataset {spec.name!r} root is missing: {spec.root}"
            for spec in cfg.enabled_datasets
            if not spec.root.is_dir()
        ]
    else:
        prerequisites = cfg.prerequisite_errors(prerequisite_stage)
    if prerequisites:
        display(pd.DataFrame({"BLOCKING_PREREQUISITE": prerequisites}))
    else:
        print("Configured prerequisites for this phase are present.")
    if RUN_PROFILE == "full" and EXECUTE_FULL_STAGES:
        if FULL_PHASE == "prepare":
            # Dataset indexing/split preparation is CPU-safe and does not load
            # detector/VLM packages, so CUDA and model-package checks do not apply.
            environment_scopes = {"all", "full"}
        else:
            if (
                FULL_PHASE in {"validation", "test", "embedded"}
                and not DEVICE.strip().lower().startswith("cuda")
            ):
                raise StudyGateError(
                    f"Full {FULL_PHASE} execution requires SEMUAV_DEVICE to select CUDA; got {DEVICE!r}"
                )
            environment_scopes = {"all", "full", "real_profiles"}
            if FULL_PHASE in {"validation", "test"}:
                environment_scopes.add("experiments_2_7")
        assert_environment_ready(preflight, sorted(environment_scopes))
        print("Scientific environment gate: PASS", environment_fingerprint_artifact)
    if RUN_PROFILE == "full" and not EXECUTE_FULL_STAGES:
        print("FULL MODE IS ARMED BUT PAUSED: review the tables, then set EXECUTE_FULL_STAGES=True in the parameters cell.")
    """),
    md("""
    ## Software acceptance tests

    These tests cover group-safe splitting, full-mode rejection of missing inputs,
    MTU boundaries, reassembly, INT8/INT4 tensors, exact cross-segment/outage
    serialization, byte conservation, state keyframes/deltas/repair, paired
    statistics, and controller-schema causality.
    """),
    code("""
    test_process = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-s", str(PROJECT_ROOT / "tests"), "-v"],
        cwd=PROJECT_ROOT, text=True, capture_output=True,
        env={**os.environ, "SEMUAV_LIVE_NOTEBOOK_SELFTEST": "1"},
    )
    print(test_process.stdout)
    print(test_process.stderr)
    if test_process.returncode:
        raise RuntimeError("Acceptance tests failed; do not run study stages")
    print("Acceptance tests: PASS")
    """),
    md("""
    # Integration execution (dry run)

    This path exercises software paths for Experiments 1–5, the Experiment 6
    receiver-state unit contract, and the Experiment 7 readiness scaffold with
    generated fixtures. It deliberately fails every empirical/scientific claim
    and writes only below `dry_run_NON_SCIENTIFIC`.
    """),
    code("""
    dry_outputs = None
    if RUN_PROFILE == "dry_run":
        dry_outputs = dry_run_all(ctx, device=DEVICE)
        display(dry_outputs["status"])
        assert not dry_outputs["status"]["scientific"].any()
    else:
        print("Skipped: selected profile uses real inputs.")
    """),
    code("""
    if dry_outputs is not None:
        dry_summary = resource_summary(dry_outputs["exp4_policy"])
        display(dry_summary.sort_values("correct_timely_rate", ascending=False))
        display(dry_outputs["exp4_controller_summary"])
        print(
            "Dry-run visual diagnostics are rendered by the publication suite "
            "below and watermarked NON-SCIENTIFIC."
        )
    """),
    md("""
    # Full study: data, annotations, and frozen split

    Whole flights/videos/locations/events remain in one connected split. New
    mission labels require two blinded annotators, documented adjudication, ≥0.80
    overall chance-corrected agreement, and visible evidence. Mission profile and
    deadline assignments come from the telemetry manifest and may not depend on
    the hidden event label.

    Use `FULL_PHASE="prepare"` to propose the split. Inspect it before setting
    `FREEZE_SPLITS=True`; the notebook never silently overwrites a frozen split.
    """),
    code("""
    analysis_manifest = annotations = raw_manifest = split_manifest = trace_table = frozen_test_protocol = None
    test_handoff = embedded_deadline_ms = None
    sample_gate = annotation_report = data_snapshot_artifact = None
    preparation_output_dir = preparation_artifacts = None
    test_access_opened = False
    test_results_verified = False
    if RUN_PROFILE == "full" and EXECUTE_FULL_STAGES and FULL_PHASE == "embedded":
        cfg.assert_ready("embedded")
        target_log_path = Path(str(cfg.hardware.get("target_log", ""))).expanduser()
        if not target_log_path.is_file():
            raise StudyGateError(f"Embedded target-hardware log is missing: {target_log_path}")
        test_handoff = load_finalized_test_handoff(
            ctx, ctx.run_root / "frozen_protocol.json",
            required_artifacts=EMBEDDED_HANDOFF_REQUIRED_ARTIFACTS,
        )
        frozen_test_protocol = test_handoff.protocol
        test_trials = pd.read_csv(test_handoff.artifact("test_policy_trials.csv.gz"), compression="gzip")
        timing_columns = {"capture_us", "deadline_us"}
        if missing_timing := timing_columns.difference(test_trials.columns):
            raise StudyGateError(f"Finalized test trials lack timing columns: {sorted(missing_timing)}")
        deadline_budgets_ms = (
            pd.to_numeric(test_trials["deadline_us"], errors="coerce")
            - pd.to_numeric(test_trials["capture_us"], errors="coerce")
        ) / 1000.0
        if deadline_budgets_ms.empty or not np.isfinite(deadline_budgets_ms).all() or (deadline_budgets_ms <= 0).any():
            raise StudyGateError("Finalized test trials contain invalid absolute mission deadlines")
        embedded_deadline_ms = float(deadline_budgets_ms.min())
        test_results_verified = True
        print(
            "Embedded phase verified the frozen protocol and finalized test-result ledger; "
            f"test data will not be reopened (strictest frozen deadline={embedded_deadline_ms:.3f} ms)."
        )
    elif RUN_PROFILE == "full" and EXECUTE_FULL_STAGES:
        if FULL_PHASE not in {"prepare", "validation", "test"}:
            raise StudyGateError(
                "SEMUAV_FULL_PHASE must be prepare, validation, test, or embedded"
            )
        if FULL_PHASE in {"validation", "test"}:
            # These phases retain the complete annotation, telemetry, frozen
            # split, trace, model, and hardware prerequisite gates.
            cfg.assert_ready(FULL_PHASE)
        if FULL_PHASE == "test":
            if not RUN_UNTOUCHED_TEST_ONCE:
                raise StudyGateError("Untouched test is locked. Set RUN_UNTOUCHED_TEST_ONCE=True only for the preregistered run.")
            with test_access_guard(ctx, ctx.run_root / "frozen_protocol.json") as protocol_preview:
                frozen_environment = protocol_preview["decisions"]["artifact_manifest"].get(
                    "environment_fingerprint"
                )
                if not frozen_environment:
                    raise StudyGateError("Frozen protocol has no validation environment fingerprint")
                expected_environment = json.loads(
                    Path(frozen_environment["path"]).read_text(encoding="utf-8")
                )
                if expected_environment != environment_fingerprint_payload:
                    raise StudyGateError(
                        "Runtime or installed-package environment changed after validation freeze"
                    )
            frozen_test_protocol = authorize_test_once(ctx, ctx.run_root / "frozen_protocol.json")
            test_access_opened = True
            print("ONE-TIME test unblinding authorized by frozen protocol", frozen_test_protocol["protocol_hash"])
        raw_manifest = build_canonical_manifest(
            cfg,
            enforce_prerequisites=FULL_PHASE != "prepare",
        )
        if cfg.paths.split_manifest and cfg.paths.split_manifest.is_file():
            split_manifest = apply_frozen_splits(raw_manifest, cfg.paths.split_manifest)
        else:
            split_manifest = assign_group_safe_splits(raw_manifest, cfg.split)
            display(split_manifest.groupby(["split", "dataset"])["split_unit"].nunique().unstack(fill_value=0))
            if FREEZE_SPLITS:
                if cfg.paths.split_manifest is None:
                    raise StudyGateError("Configure paths.split_manifest before freezing the split")
                freeze_split_manifest(split_manifest, cfg.paths.split_manifest)
                print("Frozen split written:", cfg.paths.split_manifest)
            elif FULL_PHASE == "prepare":
                print(
                    "Split proposal is not frozen. Inspect the balance and templates, "
                    "then rerun prepare with FREEZE_SPLITS=True exactly once."
                )
            else:
                raise StudyGateError(
                    "Validation/test requires the inspected frozen split manifest."
                )
        leakage = audit_split_leakage(split_manifest, cfg.split)
        leakage.raise_for_errors()
        display(pd.DataFrame([leakage.as_dict()]))

        if FULL_PHASE == "prepare":
            preparation_output_dir = (
                cfg.paths.output_root / RUN_NAME / run_id / "preparation_templates"
            ).resolve()
            preparation_artifacts = dict(export_preparation_bundle(
                split_manifest,
                preparation_output_dir,
                cfg.annotations,
                split_config=cfg.split,
                overwrite=True,
            ))
            print(
                "Atomic preparation bundle exported without invented labels or telemetry:",
                preparation_artifacts["bundle_manifest"],
            )

            annotation_tasks = (
                "gold_action", "critical", "cohort", "sufficient_evidence",
                "event", "objects", "relations", "evidence_roi",
            )

            def build_annotation_template(manifest_rows):
                sample_ids = sorted(manifest_rows["sample_id"].astype(str).unique())
                template = pd.MultiIndex.from_product(
                    [
                        sample_ids,
                        annotation_tasks,
                        range(1, int(cfg.annotations.required_annotators) + 1),
                    ],
                    names=["sample_id", "task", "annotator_number"],
                ).to_frame(index=False)
                template["label"] = ""
                template["annotator_id"] = template.pop("annotator_number").map(
                    lambda number: f"ANNOTATOR_{number}_REPLACE"
                )
                template["blinded"] = True
                template["adjudicated_label"] = ""
                template["adjudicator_id"] = ""
                template["is_critical"] = ""
                template["evidence_geometry_json"] = ""
                template["notes"] = ""
                return template[[
                    "sample_id", "task", "label", "annotator_id", "blinded",
                    "adjudicated_label", "adjudicator_id", "is_critical",
                    "evidence_geometry_json", "notes",
                ]]

            annotation_scopes = {
                "development_annotation_template": split_manifest[
                    ~split_manifest["split"].eq("test")
                ],
                "untouched_test_annotation_template": split_manifest[
                    split_manifest["split"].eq("test")
                ],
            }
            for artifact_name, manifest_rows in annotation_scopes.items():
                preparation_artifacts[artifact_name] = atomic_write_frame(
                    preparation_output_dir / f"{artifact_name}.csv.gz",
                    build_annotation_template(manifest_rows),
                )

            telemetry_columns = (
                "sample_id", "capture_time_us", "gps_lat", "gps_lon",
                "altitude_m", "yaw_deg", "mission_profile", "deadline_us",
                "trace_time_us", "battery_fraction", "thermal_c",
            )
            telemetry_template = split_manifest[["sample_id"]].copy()
            telemetry_template["sample_id"] = telemetry_template["sample_id"].astype(str)
            telemetry_template = telemetry_template.drop_duplicates().sort_values(
                "sample_id", kind="stable"
            )
            for column in telemetry_columns[1:]:
                telemetry_template[column] = ""
            preparation_artifacts["telemetry_template"] = atomic_write_frame(
                preparation_output_dir / "telemetry_template.csv.gz",
                telemetry_template[list(telemetry_columns)],
            )

            for template_name in (
                "trace_manifest_template",
                "native_receiver_cache_template",
                "registered_feature_cache_template",
            ):
                bundled_template = PROJECT_ROOT / "annotations" / f"{template_name}.csv"
                template_frame = pd.read_csv(bundled_template, nrows=0)
                preparation_artifacts[template_name] = atomic_write_frame(
                    preparation_output_dir / f"{template_name}.csv.gz",
                    template_frame,
                )

            print("Preparation template directory:", preparation_output_dir)
            display(pd.DataFrame([
                {"artifact": name, "path": str(path)}
                for name, path in sorted(preparation_artifacts.items())
            ]))
        else:
            # Validation and test remain fail-closed on adjudicated annotations,
            # exogenous telemetry, held-out traces, and their scientific gates.
            annotation_path = cfg.paths.test_annotations if FULL_PHASE == "test" else cfg.paths.annotations
            all_annotations = load_annotations(annotation_path)
            if FULL_PHASE == "test":
                visible_manifest = split_manifest[split_manifest["split"].eq("test")].copy()
            else:
                visible_manifest = split_manifest[~split_manifest["split"].eq("test")].copy()
            visible_ids = set(visible_manifest["sample_id"].astype(str))
            annotations = all_annotations[all_annotations[cfg.annotations.sample_column].astype(str).isin(visible_ids)].copy()
            annotation_report = audit_annotations(annotations, visible_manifest, cfg.annotations)
            display(pd.DataFrame([annotation_report.as_dict()]))
            annotation_report.raise_for_errors()
            analysis_manifest = prepare_analysis_manifest(visible_manifest, annotations, cfg.annotations, cfg.paths.telemetry_manifest)
            # Before the selected validation/test episode is opened below, load
            # provenance only. Held-out rates/loss/delay values do not enter the
            # notebook namespace during development sample/design checks.
            trace_table = load_trace_design_registry(str(cfg.paths.channel_traces))
            trace_design_gate = scientific_trace_design_audit(
                trace_table, cfg.network["held_out_trace_families"],
            )
            display(trace_design_gate)
            atomic_write_frame(
                ctx.run_root / "results" / f"{FULL_PHASE}_trace_design_gate.csv.gz",
                trace_design_gate,
            )
            if not trace_design_gate["passed"].astype(bool).all():
                raise StudyGateError("Scientific trace design/provenance gate failed")
            sample_gate = minimum_sample_gate(analysis_manifest, trace_table)
            display(sample_gate)
            atomic_write_frame(ctx.run_root / "results" / f"{FULL_PHASE}_sample_gate.csv.gz", sample_gate)
            atomic_write_json(
                ctx.run_root / "results" / f"{FULL_PHASE}_annotation_audit.json",
                annotation_report.as_dict(),
            )
            if FULL_PHASE == "validation":
                snapshot_rows = []
                seen_paths = set()
                for manifest_row in raw_manifest.itertuples(index=False):
                    candidate_paths = [Path(manifest_row.media_path)]
                    clip_value = getattr(manifest_row, "clip_paths_json", "[]")
                    clip_paths = json.loads(clip_value) if isinstance(clip_value, str) and clip_value else []
                    for clip_path in clip_paths:
                        candidate_paths.append(Path(clip_path))
                    for candidate_path in candidate_paths:
                        resolved = candidate_path.resolve()
                        if resolved in seen_paths:
                            continue
                        if not resolved.is_file():
                            raise StudyGateError(f"Frozen dataset input is missing: {resolved}")
                        seen_paths.add(resolved)
                        snapshot_rows.append({
                            "path": str(resolved), "bytes": resolved.stat().st_size,
                            "sha256": hash_file(resolved),
                        })
                data_snapshot_artifact = ctx.run_root / "results" / "frozen_dataset_content_snapshot.json"
                atomic_write_json(data_snapshot_artifact, {"files": sorted(snapshot_rows, key=lambda row: row["path"])})
            if FULL_PHASE == "test" and not sample_gate["passed"].all():
                raise StudyGateError("Confirmatory sample/trace minimum gate failed before model evaluation")
    else:
        print("Full data stage is paused.")
    """, "full-only"),
    md("""
    # Experiment 1 — perception, calibration, and packet validity

    All models receive the same predeclared image size, detector proposals,
    prompt, schema, output-token cap, quantization declaration, and hardware;
    native visual-token counts are recorded because tokenizer designs differ.
    The larger VLM is an
    upper bound only. Confidence uses observable logits/token likelihood—not a
    model's self-reported confidence. Fit calibration on `calibration`, select
    thresholds/model on `validation`, and freeze before `test`.
    """),
    code("""
    exp1_detail = exp1_summary = detector_cache = test_calibration_report = None
    if RUN_PROFILE == "full" and EXECUTE_FULL_STAGES and FULL_PHASE in {"validation", "test"} and analysis_manifest is not None:
        detector_cfg = cfg.models["detector"]
        detector_checkpoint_sha256 = hash_file(detector_cfg["checkpoint"])
        detector_tracker_id = detector_cfg.get("tracker", "botsort.yaml")
        detector_revision = detector_execution_revision(
            detector_checkpoint_sha256, detector_tracker_id,
        )
        detector = UltralyticsDetectorTracker(detector_cfg["checkpoint"], detector_cfg.get("tracker", "botsort.yaml"), DEVICE)
        detector_input_hashes = [
            {"sample_id": str(row.sample_id), "image_sha256": hash_file(row.image_path)}
            for row in analysis_manifest[["sample_id", "image_path"]].itertuples(index=False)
        ]
        detector_cache = frame_stage(
            ctx, "01_detector_tracker",
            {
                "checkpoint": detector_cfg["checkpoint"],
                "checkpoint_sha256": detector_checkpoint_sha256,
                "tracker": detector_cfg.get("tracker", "botsort.yaml"),
                "inputs": detector_input_hashes,
            },
            lambda: run_detector_cache(
                detector, analysis_manifest, measure_energy=True,
                device_index=int(DEVICE.split(":")[-1]) if DEVICE.startswith("cuda") else 0,
            ),
        )
        detector_residency = detector.ensure_resident()
        detector_residency_artifact = (
            ctx.run_root / "results" / f"{FULL_PHASE}_detector_residency.json"
        )
        atomic_write_json(detector_residency_artifact, detector_residency)
        inference_cache_context = {
            "detector_revision": detector_revision,
            "checkpoint_sha256": detector_checkpoint_sha256,
            "tracker": detector_tracker_id,
            "residency": detector_residency,
        }
        print("Detector residency verified:", detector_residency)
        study_prompt = (PROJECT_ROOT / "prompts" / "mission_packet_v1.txt").read_text(encoding="utf-8")
        model_input_size = tuple(int(value) for value in cfg.models.get("input_size", [768, 768]))
        allowed_splits = ["train", "calibration", "validation"] if FULL_PHASE == "validation" else ["test"]
        stage_manifest = analysis_manifest[analysis_manifest["split"].isin(allowed_splits)]
        perception_detail_parts = []
        perception_summary_parts = []
        # The always-on detector remains resident while one context model is
        # loaded/measured at a time. Thus peak CUDA allocation represents the
        # registered deployed residency strategy (detector + triggered VLM),
        # without unrealistically holding every VLM simultaneously.
        model_entries = [None, *cfg.models["vlms"], cfg.models["upper_bound"]]
        for item in model_entries:
            if item is None:
                backend = DetectorOnlyBackend(detector_revision)
            else:
                spec = ModelSpec(
                    item["id"], item["repo"], item["revision"],
                    "upper_bound" if item["id"] == cfg.models["upper_bound"]["id"] else "compact",
                    item.get("quantization", "none"), item["id"] == "moondream2",
                    model_input_size,
                )
                backend = (
                    MoondreamBackend(spec, DEVICE) if item["id"] == "moondream2"
                    else HFMultimodalBackend(
                        spec, DEVICE, cfg.runtime.get("dtype", "bfloat16")
                    )
                )
            model_detail, model_summary = experiment1_perception(
                stage_manifest, [backend], scientific=True, detector_cache=detector_cache,
                ctx=ctx, shard_size=int(cfg.raw.get("cache", {}).get("shard_size", 32)),
                max_new_tokens=int(cfg.models.get("max_new_tokens", 384)), prompt=study_prompt,
                inference_cache_context=inference_cache_context,
            )
            perception_detail_parts.append(model_detail)
            perception_summary_parts.append(model_summary)
            del backend
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
        del detector
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        exp1_detail = pd.concat(perception_detail_parts, ignore_index=True)
        exp1_summary = pd.concat(perception_summary_parts, ignore_index=True)
        atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp1_perception_detail.csv.gz",
            exp1_detail,
        )
        atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp1_perception_summary.csv.gz",
            exp1_summary,
        )
        display(exp1_summary)
        if FULL_PHASE == "validation":
            calibration_rows = exp1_detail[exp1_detail["split"].eq("calibration") & exp1_detail["confidence"].notna()]
            calibrators = {}
            for model_id, group in calibration_rows.groupby("model_id"):
                calibrators[model_id] = TemperatureCalibrator().fit(group.confidence, group.action_correct.astype(int))
            print("Fitted calibration models:", sorted(calibrators))
            validation_rows = exp1_detail[exp1_detail["split"].eq("validation")]
            calibration_payload = {
                model_id: {"temperature": calibrator.temperature, "confidence_threshold": 0.80}
                for model_id, calibrator in calibrators.items()
            }
            calibration_gate_rows = []
            for model_id, calibrator in calibrators.items():
                validation_conf = validation_rows[(validation_rows.model_id == model_id) & validation_rows.confidence.notna()]
                validation_all = validation_rows[validation_rows.model_id == model_id]
                if validation_conf.empty:
                    continue
                before = calibration_metrics(validation_conf.confidence, validation_conf.action_correct.astype(int))
                calibrated_probability = calibrator.transform(validation_conf.confidence)
                after = calibration_metrics(calibrated_probability, validation_conf.action_correct.astype(int))
                critical_conf = validation_conf[validation_conf.critical.astype(bool)]
                calibration_fit = calibration_rows[calibration_rows.model_id == model_id]
                critical_ece = (
                    calibration_metrics(calibrator.transform(critical_conf.confidence), critical_conf.action_correct.astype(int))["ece_15_equal_mass"]
                    if len(critical_conf) else np.nan
                )
                calibration_gate_rows.append({
                    "model_id": model_id, "ece": after["ece_15_equal_mass"],
                    "critical_ece": critical_ece,
                    "brier_before": before["brier"], "brier_after": after["brier"],
                    "calibration_units": int(calibration_fit.visual_group_id.nunique()),
                    "calibration_correct_units": int(calibration_fit.loc[calibration_fit.action_correct, "visual_group_id"].nunique()),
                    "calibration_incorrect_units": int(calibration_fit.loc[~calibration_fit.action_correct, "visual_group_id"].nunique()),
                    "calibration_critical_units": int(calibration_fit.loc[calibration_fit.critical.astype(bool), "visual_group_id"].nunique()),
                    "validation_confidence_coverage": float(len(validation_conf) / len(validation_all)) if len(validation_all) else 0.0,
                    "passed": (
                        after["ece_15_equal_mass"] <= 0.05
                        and np.isfinite(critical_ece) and critical_ece <= 0.10
                        and after["brier"] <= before["brier"]
                        and calibration_fit.visual_group_id.nunique() >= 200
                        and calibration_fit.loc[calibration_fit.action_correct, "visual_group_id"].nunique() >= 100
                        and calibration_fit.loc[~calibration_fit.action_correct, "visual_group_id"].nunique() >= int(cfg.evaluation.get("minimum_calibration_incorrect_units", 50))
                        and calibration_fit.loc[calibration_fit.critical.astype(bool), "visual_group_id"].nunique() >= 100
                        and len(validation_conf) == len(validation_all)
                    ),
                })
            calibration_gate_frame = pd.DataFrame(calibration_gate_rows)
            display(calibration_gate_frame)
            atomic_write_frame(
                ctx.run_root / "results" / "validation_calibration_gate.csv.gz",
                calibration_gate_frame,
            )
            score_fields = ["action_correct", "event_correct", "objects_exact", "relations_exact", "evidence_state_correct"]
            validation_group_scores = validation_rows.groupby(
                ["model_id", "visual_group_id"], as_index=False,
            )[score_fields].mean()
            validation_model_scores = validation_group_scores.groupby("model_id")[
                score_fields
            ].mean().mean(axis=1)
            contextual_scores = validation_group_scores.groupby("model_id")[[
                "event_correct", "relations_exact",
            ]].mean().mean(axis=1)
            constraint_rows = []
            for model_id, model_rows in validation_rows.groupby("model_id"):
                critical_rows = model_rows[model_rows["critical"].astype(bool)]
                constraint_rows.append({
                    "model_id": model_id,
                    "latency_p95_ms": float(model_rows["full_path_latency_ms"].quantile(0.95)),
                    "peak_memory_mb": float(pd.to_numeric(model_rows["peak_memory_mb"], errors="coerce").max()),
                    "schema_valid_rate": float(
                        model_rows.groupby("visual_group_id")["schema_valid"].mean().mean()
                    ),
                    "semantic_valid_rate": float(
                        model_rows.groupby("visual_group_id")["semantic_valid"].mean().mean()
                    ),
                    "authoritative_consistency_rate": float(
                        model_rows.groupby("visual_group_id")["authoritative_consistent"].mean().mean()
                    ),
                    "critical_action_recall": (
                        float(critical_rows.groupby("visual_group_id")["action_correct"].mean().mean())
                        if len(critical_rows) else np.nan
                    ),
                })
            validation_constraint_summary = pd.DataFrame(constraint_rows)
            summary_lookup = validation_constraint_summary.set_index("model_id").to_dict("index")
            limits = {
                "max_p95_latency_ms": float(cfg.evaluation.get("max_vlm_p95_latency_ms", 1000.0)),
                "max_peak_memory_mb": float(cfg.evaluation.get("max_vlm_peak_memory_mb", 12000.0)),
                "min_schema_valid_rate": float(cfg.evaluation.get("min_schema_valid_rate", 0.98)),
                "min_semantic_valid_rate": float(cfg.evaluation.get("min_semantic_valid_rate", 0.98)),
                "min_critical_recall": float(cfg.evaluation.get("min_critical_recall", 0.85)),
                "minimum_context_gain": float(cfg.evaluation.get("minimum_vlm_context_gain", 0.03)),
            }
            compact_ids = [item["id"] for item in cfg.models["vlms"]]
            compact_scored = [model_id for model_id in compact_ids if model_id in validation_model_scores]
            if not compact_scored:
                raise StudyGateError("No compact VLM produced validation records")
            calibration_pass = {
                row["model_id"]: bool(row["passed"]) for row in calibration_gate_rows
            }
            def model_constraint_pass(model_id):
                row = summary_lookup[model_id]
                return bool(
                    float(row["latency_p95_ms"]) <= limits["max_p95_latency_ms"]
                    and float(row["peak_memory_mb"]) <= limits["max_peak_memory_mb"]
                    and float(row["schema_valid_rate"]) >= limits["min_schema_valid_rate"]
                    and float(row["semantic_valid_rate"]) >= limits["min_semantic_valid_rate"]
                    and float(row["authoritative_consistency_rate"]) == 1.0
                    and float(row["critical_action_recall"]) >= limits["min_critical_recall"]
                )
            eligible_compact = [
                model_id for model_id in compact_scored
                if calibration_pass.get(model_id, False) and model_constraint_pass(model_id)
            ]
            best_vlm = max(
                eligible_compact or compact_scored,
                key=lambda model_id: (contextual_scores.get(model_id, -np.inf), validation_model_scores[model_id]),
            )
            detector_score = float(contextual_scores.get("detector_only", np.nan))
            vlm_gain = float(contextual_scores[best_vlm] - detector_score)
            vlm_claim_enabled = bool(
                best_vlm in eligible_compact and vlm_gain >= limits["minimum_context_gain"]
            )
            selected_model = best_vlm if vlm_claim_enabled else "detector_only"
            calibration_claim_enabled = calibration_pass.get(selected_model, False)
            if not vlm_claim_enabled:
                print(
                    f"VLM GO/NO-GO FAILED (gain={vlm_gain:.4f}, "
                    f"calibration={calibration_pass.get(best_vlm, False)}); contextual VLM claim is disabled."
                )
            selection_artifact = ctx.run_root / "results" / "selected_deployed_model.json"
            atomic_write_json(selection_artifact, {
                "selected_model": selected_model, "best_compact_vlm": best_vlm,
                "vlm_gain_over_detector": vlm_gain, "calibration": calibration_payload,
                "calibration_gate": calibration_gate_rows,
                "selection_limits": limits,
                "eligible_compact_models": eligible_compact,
                "vlm_claim_enabled": vlm_claim_enabled,
                "calibration_claim_enabled": calibration_claim_enabled,
            })
            print("Frozen-selection candidate (validation only):", selection_artifact)
        elif FULL_PHASE == "test":
            selection_artifact = ctx.run_root / "results" / "selected_deployed_model.json"
            if not selection_artifact.is_file():
                raise StudyGateError("Frozen validation calibration artifact is missing")
            selection = json.loads(selection_artifact.read_text())
            calibration_rows = []
            for model_id, specification in selection.get("calibration", {}).items():
                model_rows = exp1_detail[exp1_detail["model_id"].eq(model_id)]
                observed = model_rows[model_rows["confidence"].notna()]
                if observed.empty:
                    continue
                calibrator = TemperatureCalibrator(float(specification["temperature"]), True)
                calibrated_probability = calibrator.transform(observed["confidence"])
                overall = calibration_metrics(calibrated_probability, observed["action_correct"].astype(int))
                critical = observed[observed["critical"].astype(bool)]
                critical_ece = (
                    calibration_metrics(calibrator.transform(critical["confidence"]), critical["action_correct"].astype(int))["ece_15_equal_mass"]
                    if len(critical) else np.nan
                )
                calibration_rows.append({
                    "model_id": model_id, "ece": overall["ece_15_equal_mass"],
                    "critical_ece": critical_ece, "brier": overall["brier"],
                    "confidence_coverage": float(len(observed) / len(model_rows)) if len(model_rows) else 0.0,
                    "passed": bool(
                        overall["ece_15_equal_mass"] <= 0.05
                        and np.isfinite(critical_ece) and critical_ece <= 0.10
                        and len(observed) == len(model_rows)
                    ),
                })
            test_calibration_report = pd.DataFrame(calibration_rows)
            atomic_write_frame(
                ctx.run_root / "results" / "test_calibration_report.csv.gz",
                test_calibration_report,
            )
            display(test_calibration_report)
    else:
        print("Experiment 1 full execution is paused.")
    """, "full-only"),
    md("""
    # Experiment 2 — static representation frontier

    Candidate construction uses actual serialization bytes and native frozen
    receivers. Gold labels never generate packets or receiver outputs. L4 is
    excluded unless a registered encoder/split/tensor/quantizer/head cache exists.
    """),
    code("""
    exp2_frontier = causal_profiles = policy_manifest = None
    if RUN_PROFILE == "full" and EXECUTE_FULL_STAGES and FULL_PHASE in {"validation", "test"} and exp1_detail is not None:
        selection_artifact = ctx.run_root / "results" / "selected_deployed_model.json"
        if not selection_artifact.is_file():
            raise StudyGateError("Validation-selected deployed-model artifact is missing")
        selection = json.loads(selection_artifact.read_text())
        selected_model = selection["selected_model"]
        requested_model = os.environ.get("SEMUAV_SELECTED_MODEL", selected_model)
        if requested_model != selected_model:
            raise StudyGateError("SEMUAV_SELECTED_MODEL differs from the validation-frozen selection")
        if frozen_test_protocol is not None and frozen_test_protocol["decisions"].get("selected_model") != selected_model:
            raise StudyGateError("Selected model differs from the frozen protocol")
        calibration_spec = selection.get("calibration", {}).get(selected_model)
        if not selection.get("calibration_claim_enabled") or calibration_spec is None:
            raise StudyGateError("Selected model failed the frozen calibration/sample-size gate")
        semantic_rows = exp1_detail[exp1_detail.model_id == selected_model].copy()
        semantic_cache = calibrated_semantic_packet_cache(
            semantic_rows, calibration_spec["temperature"], source_model=selected_model,
            authoritative_manifest=analysis_manifest,
        )
        detector_rows = exp1_detail[exp1_detail.model_id == "detector_only"]
        detector_calibration = selection.get("calibration", {}).get("detector_only")
        detector_gate = {
            row["model_id"]: bool(row["passed"])
            for row in selection.get("calibration_gate", [])
        }
        if detector_calibration is None or not detector_gate.get("detector_only", False):
            raise StudyGateError("Detector-only policy lacks a passing frozen calibration gate")
        detector_semantic_cache = (
            semantic_cache.copy() if selected_model == "detector_only" else
            calibrated_semantic_packet_cache(
                detector_rows, detector_calibration["temperature"], source_model="detector_only",
                authoritative_manifest=analysis_manifest,
            )
        )
        perception_context = semantic_cache.copy()
        policy_manifest = analysis_manifest.drop(columns=[column for column in ("risk", "uncertainty") if column in analysis_manifest]).merge(
            perception_context[["sample_id", "risk", "uncertainty"]], on="sample_id", validate="one_to_one",
        )
        policy_manifest = policy_manifest.merge(
            detector_semantic_cache[["sample_id", "risk", "uncertainty"]].rename(columns={
                "risk": "detector_risk", "uncertainty": "detector_uncertainty",
            }),
            on="sample_id", validate="one_to_one",
        )
        trigger_risk = float(cfg.controller.get("vlm_trigger_risk", 0.45))
        trigger_uncertainty = float(cfg.controller.get("vlm_trigger_uncertainty", 0.35))
        policy_manifest["context_triggered"] = bool(selected_model != "detector_only") & (
            policy_manifest["detector_risk"].ge(trigger_risk)
            | policy_manifest["detector_uncertainty"].ge(trigger_uncertainty)
            | policy_manifest["mission_profile"].eq("high_priority")
        )
        policy_manifest = policy_manifest.merge(
            detector_cache[["sample_id", "latency_ms", "energy_j"]].rename(columns={
                "latency_ms": "detector_latency_ms", "energy_j": "detector_energy_j",
            }),
            on="sample_id", validate="one_to_one",
        )
        selected_cost = semantic_rows[["observation_id", "latency_ms", "energy_j"]].rename(columns={
            "observation_id": "sample_id", "latency_ms": "context_model_latency_ms",
            "energy_j": "context_model_energy_j",
        })
        if selected_model == "detector_only":
            selected_cost[["context_model_latency_ms", "context_model_energy_j"]] = 0.0
        policy_manifest = policy_manifest.merge(selected_cost, on="sample_id", validate="one_to_one")
        measured_costs = ["detector_latency_ms", "detector_energy_j", "context_model_latency_ms", "context_model_energy_j"]
        if policy_manifest[measured_costs].isna().any().any():
            raise StudyGateError("Detector/VLM latency or energy measurement is incomplete")
        if FULL_PHASE == "test":
            if frozen_test_protocol is None:
                raise StudyGateError("Test receiver caches require the verified frozen protocol")
            native_cache_path = resolve_frozen_artifact(
                frozen_test_protocol, "native_receiver_cache", NATIVE_RECEIVER_CACHE,
            )
            feature_cache_path = resolve_frozen_artifact(
                frozen_test_protocol, "registered_feature_cache", REGISTERED_FEATURE_CACHE,
            )
        else:
            native_cache_path = NATIVE_RECEIVER_CACHE.expanduser().resolve()
            feature_cache_path = REGISTERED_FEATURE_CACHE.expanduser().resolve()
        if not native_cache_path.is_file():
            raise StudyGateError(f"Missing registered native receiver cache: {native_cache_path}")
        if not feature_cache_path.is_file():
            raise StudyGateError(f"Missing registered feature cache: {feature_cache_path}")
        native_cache = pd.read_csv(native_cache_path)
        feature_cache = pd.read_csv(feature_cache_path)
        exp2_frontier = experiment2_representation_frontier_sharded(
            analysis_manifest, ctx, scientific=True, mtu_bytes=int(cfg.payloads["mtu_bytes"]),
            semantic_packet_cache=semantic_cache, native_receiver_cache=native_cache,
            registered_feature_cache=feature_cache, semantic_source_model=selected_model,
            image_codecs=cfg.payloads["image_codecs"],
            image_qualities=cfg.payloads["image_qualities"],
            image_scales=cfg.payloads["image_scales"],
            shard_size=16, stage_prefix="02_deployed_representations",
        )
        semantic_modes = {"L1_ALERT", "L2_SEMANTICS", "L3_SEMANTICS_ROI"}
        selected_suffix = ":detector_only" if selected_model == "detector_only" else ":deployed"
        exp2_frontier.loc[exp2_frontier["mode"].isin(semantic_modes), "variant"] += selected_suffix
        if selected_model != "detector_only":
            detector_frontier = experiment2_representation_frontier_sharded(
                analysis_manifest, ctx, scientific=True, mtu_bytes=int(cfg.payloads["mtu_bytes"]),
                semantic_packet_cache=detector_semantic_cache, native_receiver_cache=native_cache,
                registered_feature_cache=None, semantic_source_model="detector_only",
                image_codecs=cfg.payloads["image_codecs"],
                image_qualities=cfg.payloads["image_qualities"],
                image_scales=cfg.payloads["image_scales"],
                shard_size=16, stage_prefix="02_detector_only_representations",
            )
            detector_frontier = detector_frontier[detector_frontier["mode"].isin(semantic_modes)].copy()
            detector_frontier["variant"] += ":detector_only"
            exp2_frontier = pd.concat([exp2_frontier, detector_frontier], ignore_index=True)
        if not exp2_frontier["variant"].str.startswith("h264_").any():
            raise StudyGateError("No real consecutive-frame H.264 candidate was generated; the practical video baseline is incomplete")
        exp2_frontier = add_experiment2_total_costs(
            exp2_frontier, policy_manifest, require_measured_costs=True,
        )
        atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp2_representation_frontier.csv.gz",
            exp2_frontier,
        )
        group_equal_frontier = exp2_frontier.groupby(
            ["mode", "variant", "visual_group_id"], as_index=False,
        ).agg(
            native_receiver_correct=("native_receiver_correct", "mean"),
            wire_bytes=("wire_bytes", "mean"),
            total_processing_latency_ms=("total_processing_latency_ms", "mean"),
            total_compute_energy_j=("total_compute_energy_j", "mean"),
            lossless_wire_compression_ratio=("lossless_wire_compression_ratio", "mean"),
            lossless_wire_reduction_fraction=("lossless_wire_reduction_fraction", "mean"),
        ).groupby(["mode", "variant"], as_index=False).mean(numeric_only=True)
        display(pareto_front(
            group_equal_frontier, "native_receiver_correct",
            ["wire_bytes", "total_processing_latency_ms", "total_compute_energy_j"],
        ))
        profile_artifact = ctx.run_root / "results" / "validation_candidate_profiles.csv.gz"
        if FULL_PHASE in {"prepare", "validation"}:
            validation_ids = analysis_manifest.loc[analysis_manifest.split.eq("validation"), "sample_id"].tolist()
            causal_profiles = build_causal_candidate_profiles(exp2_frontier, validation_ids)
            atomic_write_frame(profile_artifact, causal_profiles)
            print("Validation-only causal profiles:", profile_artifact, hash_file(profile_artifact))
        else:
            if not profile_artifact.is_file():
                raise StudyGateError("Frozen validation candidate profiles are missing")
            causal_profiles = pd.read_csv(profile_artifact, compression="gzip")
    else:
        print("Experiment 2 full execution is paused.")
    """, "full-only"),
    md("""
    # Experiments 3–4 — static transport and held-out causal adaptation

    Experiment 3 covers 0.1–20 Mbps with RTT/deadline combinations and a
    registered independent-IID versus Gilbert-Elliott burst-loss dimension.
    Experiment 4 uses held-out sudden drops/recoveries, ramps, Markov/fading,
    outages, and combined impairments. Only delayed ACK-derived estimates enter
    ordinary policies. Random streams exclude policy identity.
    """),
    code("""
    exp3_transport = exp3_partition_index = exp3_summary = exp4_policy = None
    exp4_controller_detail = exp4_controller_summary = None
    if RUN_PROFILE == "full" and EXECUTE_FULL_STAGES and FULL_PHASE in {"validation", "test"} and exp2_frontier is not None:
        static_split = "test" if FULL_PHASE == "test" else "validation"
        static_manifest = policy_manifest[policy_manifest["split"].eq(static_split)].copy()
        static_manifest = select_experiment3_manifest(
            static_manifest,
            max_groups_per_dataset=int(cfg.network["static_max_groups_per_dataset"]),
            seed=int(cfg.seed),
            max_observations_per_group=int(
                cfg.network["static_max_observations_per_group"]
            ),
        )
        static_ids = set(static_manifest["sample_id"].astype(str))
        static_candidates = exp2_frontier[
            exp2_frontier["sample_id"].astype(str).isin(static_ids)
        ].copy()
        available_modes = [mode for mode in PayloadMode if mode.name in set(exp2_frontier["mode"])]
        policy_spec_artifact = ctx.run_root / "results" / "validation_policy_spec.json"
        if FULL_PHASE == "validation":
            validation_manifest_for_rate = policy_manifest[policy_manifest["split"].eq("validation")].sort_values("trace_time_us")
            capture_times = np.sort(validation_manifest_for_rate["trace_time_us"].unique())
            if len(capture_times) >= 2:
                capture_interval_us = max(1, int(np.median(np.diff(capture_times))))
                exposure_s = (capture_times[-1] - capture_times[0] + capture_interval_us) / 1e6
            else:
                exposure_s = 1.0
            observation_rate_hz = len(validation_manifest_for_rate) / max(exposure_s, 1e-9)
            detector_energy_per_observation = float(validation_manifest_for_rate["detector_energy_j"].mean())
            context_energy_per_observation = float(validation_manifest_for_rate["context_model_energy_j"].mean())
            causal_profiles = causal_profiles.copy()
            causal_profiles["estimated_wire_rate_bytes_per_s"] = causal_profiles["payload_bytes"] * observation_rate_hz
            causal_profiles["estimated_energy_rate_j_per_s"] = (
                causal_profiles["total_compute_energy_j"]
                + causal_profiles["payload_bytes"] * float(cfg.hardware["radio_tx_j_per_byte"])
            ) * observation_rate_hz
            fixed_specs = []
            for mode in available_modes:
                mode_candidates = causal_profiles[
                    causal_profiles["mode"].eq(mode.name)
                    & causal_profiles["observations"].eq(len(set(validation_ids)))
                ]
                if mode_candidates.empty:
                    raise StudyGateError(
                        f"No full-coverage validation candidate is available for fixed mode {mode.name}"
                    )
                within_caps = mode_candidates[
                    mode_candidates["estimated_wire_rate_bytes_per_s"].le(1.05 * float(cfg.evaluation["primary_wire_budget_bytes_per_s"]))
                    & mode_candidates["estimated_energy_rate_j_per_s"].le(1.05 * float(cfg.evaluation["primary_energy_budget_j_per_s"]))
                ]
                candidates = (within_caps if len(within_caps) else mode_candidates).sort_values(
                    ["predicted_task_success", "payload_bytes", "total_compute_energy_j", "total_processing_latency_ms"],
                    ascending=[False, True, True, True],
                )
                if mode is PayloadMode.L5_IMAGE_OR_CLIP:
                    practical = candidates[~candidates["variant"].str.contains("lossless", case=False, na=False)]
                    if len(practical):
                        candidates = practical
                if len(candidates):
                    chosen = candidates.iloc[0]
                    fixed_specs.append({
                        "name": f"fixed_{mode.name.lower()}", "mode": mode.name,
                        "variant": str(chosen.variant),
                        "selection": "validation_accuracy_then_resources_with_rate_screen",
                        "rate_screen_passed": bool(len(within_caps)),
                    })
            if not fixed_specs:
                raise StudyGateError("No validation-selected fixed representation is available")
            best_fixed = max(
                fixed_specs,
                key=lambda spec: float(causal_profiles.loc[
                    causal_profiles["mode"].eq(spec["mode"]) & causal_profiles["variant"].eq(spec["variant"]),
                    "predicted_task_success",
                ].iloc[0]),
            )
            policy_spec = {
                "fixed": fixed_specs,
                "best_fixed": {**best_fixed, "name": "best_fixed_validation"},
                "feature_baseline_available": "L4_FEATURES" in {item["mode"] for item in fixed_specs},
                "controller": cfg.controller,
            }
            policy_spec["required_h2_baselines"] = [
                *[item["name"] for item in fixed_specs], "best_fixed_validation",
                "adaptive_image_codec", "adaptive_semantic_rate",
                *(["adaptive_feature_rate"] if policy_spec["feature_baseline_available"] else []),
                "detector_only_semantics", "channel_only_hysteresis", "mission_only",
                "queue_deadline_greedy", "channel_plus_mission",
                "risk_aware_no_uncertainty", "ablate_mission_risk",
                "ablate_evidence_escalation", "ablate_receiver_state",
                "ablate_hysteresis", "always_vlm_full",
            ]
            atomic_write_json(policy_spec_artifact, policy_spec)
        elif not policy_spec_artifact.is_file():
            raise StudyGateError("Frozen validation policy specification is missing")
        policy_spec = json.loads(policy_spec_artifact.read_text())
        if frozen_test_protocol is not None and frozen_test_protocol["decisions"].get("policy_spec_hash") != hash_file(policy_spec_artifact):
            raise StudyGateError("Policy specification differs from the frozen protocol")

        exp3_variant_spec_artifact = (
            ctx.run_root / "results" / "validation_exp3_variant_spec.csv.gz"
        )
        max_static_variants = int(
            cfg.network["static_max_variants_per_mode_source"]
        )
        if FULL_PHASE == "validation":
            static_candidates, exp3_variant_spec = select_experiment3_candidate_variants(
                static_candidates,
                max_variants_per_mode_source=max_static_variants,
            )
            # The validation-selected fixed comparator for every mode is always
            # present even when it is not one of the wire-size span points.
            selected_keys = set(exp3_variant_spec[[
                "mode", "source_model", "variant"
            ]].itertuples(index=False, name=None))
            extra_spec_rows = []
            for fixed_item in policy_spec["fixed"]:
                matching = exp2_frontier[
                    exp2_frontier["mode"].eq(fixed_item["mode"])
                    & exp2_frontier["variant"].eq(fixed_item["variant"])
                    & exp2_frontier["sample_id"].astype(str).isin(static_ids)
                ]
                if matching.empty:
                    raise StudyGateError(
                        "Validation-selected fixed variant is absent from the "
                        f"Experiment 3 subset: {fixed_item['mode']}/{fixed_item['variant']}"
                    )
                for source_model, source_rows in matching.groupby("source_model", sort=True):
                    key = (
                        str(fixed_item["mode"]), str(source_model),
                        str(fixed_item["variant"]),
                    )
                    if key in selected_keys:
                        continue
                    if source_rows["sample_id"].astype(str).nunique() != len(static_ids):
                        raise StudyGateError(
                            f"Fixed Experiment 3 variant lacks full sample coverage: {key}"
                        )
                    existing_group = exp3_variant_spec[
                        exp3_variant_spec["mode"].eq(key[0])
                        & exp3_variant_spec["source_model"].eq(key[1])
                    ]
                    next_rank = (
                        int(existing_group["selection_rank"].max()) + 1
                        if len(existing_group) else 0
                    )
                    extra_spec_rows.append({
                        "mode": key[0], "source_model": key[1], "variant": key[2],
                        "validation_median_wire_bytes": float(
                            pd.to_numeric(source_rows["wire_bytes"], errors="raise").median()
                        ),
                        "selection_rank": next_rank,
                        "selection_version": "semuav.exp3.variant-wire-span.v1",
                    })
                    selected_keys.add(key)
            if extra_spec_rows:
                exp3_variant_spec = pd.concat(
                    [exp3_variant_spec, pd.DataFrame(extra_spec_rows)],
                    ignore_index=True,
                ).sort_values(
                    ["mode", "source_model", "selection_rank", "variant"],
                    kind="mergesort",
                ).reset_index(drop=True)
                static_candidates = exp2_frontier[
                    exp2_frontier["sample_id"].astype(str).isin(static_ids)
                ].copy()
                static_candidates = static_candidates[
                    static_candidates.apply(
                        lambda row: (
                            str(row["mode"]), str(row["source_model"]),
                            str(row["variant"]),
                        ) in selected_keys,
                        axis=1,
                    )
                ].reset_index(drop=True)
            atomic_write_frame(exp3_variant_spec_artifact, exp3_variant_spec)
        else:
            if frozen_test_protocol is None:
                raise StudyGateError("Test Experiment 3 requires the frozen validation protocol")
            frozen_variant_spec = resolve_frozen_artifact(
                frozen_test_protocol, "exp3_variant_spec",
            )
            exp3_variant_spec = pd.read_csv(frozen_variant_spec, compression="gzip")
            static_candidates, exp3_variant_spec = select_experiment3_candidate_variants(
                static_candidates,
                max_variants_per_mode_source=max_static_variants,
                frozen_spec=exp3_variant_spec,
            )

        exp3_summary, exp3_partition_index = experiment3_static_transport_partitioned(
            static_candidates, static_manifest, ctx,
            registered_seeds=cfg.seeds, shard_size=64,
            rates_mbps=cfg.network["static_rate_mbps"],
            rtts_ms=cfg.network["rtt_ms"], losses=cfg.network["loss"],
            loss_models=cfg.network["static_loss_models"],
            burst_recovery_probability=float(
                cfg.network["static_burst_recovery_probability"]
            ),
            burst_slot_ms=float(cfg.network["static_burst_slot_ms"]),
            jitters_ms=cfg.network.get("static_jitter_ms", [0.0, 10.0]),
            queue_kb=cfg.network["queue_kb"],
            radio_tx_j_per_byte=float(cfg.hardware["radio_tx_j_per_byte"]),
            radio_rx_j_per_byte=float(cfg.hardware["radio_rx_j_per_byte"]),
            require_perception_costs=True,
            require_registered_loss_models=True,
        )
        assert_experiment3_partition_index_coverage(
            exp3_partition_index, cfg.seeds, verify_artifacts=True,
        )
        exp3_partition_artifact = atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp3_static_partition_index.csv.gz",
            exp3_partition_index,
        )
        exp3_summary_artifact = atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp3_static_summary.csv.gz",
            exp3_summary,
        )
        # Existing completion code treats this as the Experiment 3 evidence
        # object; raw rows remain in checksummed resumable partitions.
        exp3_transport = exp3_partition_index
        print("Experiment 3 partition index:", exp3_partition_artifact)
        print("Experiment 3 aggregate summary:", exp3_summary_artifact)
        display(exp3_variant_spec)
        display(exp3_summary.head(100))

        fixed_policies = [
            FixedVariantPolicy(PayloadMode[item["mode"]], item["variant"], item["name"])
            for item in policy_spec["fixed"]
        ]
        fixed_policies.append(FixedVariantPolicy(
            PayloadMode[policy_spec["best_fixed"]["mode"]],
            policy_spec["best_fixed"]["variant"], "best_fixed_validation",
        ))
        controller_kwargs = {
            "minimum_dwell_us": int(cfg.controller.get("minimum_dwell_us", 2_000_000)),
            "hysteresis_margin": float(cfg.controller.get("hysteresis_margin", 0.04)),
        }
        full_kwargs = {
            **controller_kwargs,
            "thermal_limit_c": float(cfg.hardware.get("thermal_limit_c", 82.0)),
            "risk_escalation": float(cfg.controller.get("risk_escalation", 0.7)),
            "uncertainty_escalation": float(cfg.controller.get("uncertainty_escalation", 0.35)),
            "byte_weight": float(cfg.controller.get("byte_weight", 0.02)),
            "energy_weight": float(cfg.controller.get("energy_weight", 0.08)),
            "evidence_weight": float(cfg.controller.get("evidence_weight", 0.30)),
            "deadline_weight": float(cfg.controller.get("deadline_weight", 1.0)),
            "vlm_trigger_risk": float(cfg.controller.get("vlm_trigger_risk", 0.45)),
            "vlm_trigger_uncertainty": float(cfg.controller.get("vlm_trigger_uncertainty", 0.35)),
        }
        policies = [
            *fixed_policies, AdaptiveImagePolicy(), AdaptiveSemanticPolicy(),
            ChannelOnlyPolicy(**controller_kwargs), MissionOnlyPolicy(), DeadlineGreedyPolicy(),
            ChannelMissionPolicy(), RiskAwareNoUncertaintyPolicy(**full_kwargs),
            NoMissionRiskPolicy(**full_kwargs), NoEvidenceEscalationPolicy(**full_kwargs),
            NoReceiverStatePolicy(**full_kwargs), NoHysteresisPolicy(
                thermal_limit_c=full_kwargs["thermal_limit_c"],
                risk_escalation=full_kwargs["risk_escalation"],
                uncertainty_escalation=full_kwargs["uncertainty_escalation"],
                byte_weight=full_kwargs["byte_weight"],
                energy_weight=full_kwargs["energy_weight"],
                evidence_weight=full_kwargs["evidence_weight"],
                deadline_weight=full_kwargs["deadline_weight"],
                vlm_trigger_risk=full_kwargs["vlm_trigger_risk"],
                vlm_trigger_uncertainty=full_kwargs["vlm_trigger_uncertainty"],
            ), AlwaysVLMPolicy(**full_kwargs), FullPolicy(**full_kwargs),
            LookaheadReferencePolicy(),
        ]
        if policy_spec["feature_baseline_available"]:
            policies.append(AdaptiveFeaturePolicy())
        detector_variants = exp2_frontier.loc[
            exp2_frontier["mode"].eq("L2_SEMANTICS") & exp2_frontier["variant"].str.contains("detector_only"), "variant"
        ].unique().tolist()
        if not detector_variants:
            raise StudyGateError("Required detector-only semantic baseline is unavailable")
        policies.append(FixedVariantPolicy(PayloadMode.L2_SEMANTICS, detector_variants[0], "detector_only_semantics"))
        if not causal_schema_audit()["passed"]:
            raise StudyGateError("Causal controller schema audit failed")
        selected_split = "test" if FULL_PHASE == "test" else "validation"
        trial_manifest = policy_manifest[policy_manifest.split.eq(selected_split)]
        trial_ids = set(trial_manifest["sample_id"].astype(str))
        for fixed_item in [*policy_spec["fixed"], policy_spec["best_fixed"]]:
            available_ids = set(exp2_frontier.loc[
                exp2_frontier["mode"].eq(fixed_item["mode"])
                & exp2_frontier["variant"].eq(fixed_item["variant"]),
                "sample_id",
            ].astype(str))
            if not trial_ids.issubset(available_ids):
                raise StudyGateError(
                    f"Frozen fixed candidate {fixed_item['mode']}/{fixed_item['variant']} "
                    f"does not cover every {selected_split} observation"
                )
        selected_traces, selected_trace_rows = load_trace_manifest(
            str(cfg.paths.channel_traces), required_split=selected_split,
            end_behavior=EndBehavior.TERMINATE,
        )
        episode_coverage_gate = scientific_episode_coverage_audit(
            selected_traces, trial_manifest,
        )
        display(episode_coverage_gate)
        atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_episode_trace_coverage_gate.csv.gz",
            episode_coverage_gate,
        )
        if not episode_coverage_gate["passed"].astype(bool).all():
            raise StudyGateError("Selected episode exceeds measured trace coverage")
        if selected_split == "test":
            if not RUN_UNTOUCHED_TEST_ONCE:
                raise StudyGateError("Untouched test is locked; set RUN_UNTOUCHED_TEST_ONCE=True only after the protocol is frozen")
            if not test_access_opened:
                raise StudyGateError("One-time test ledger is not open")
            exp4_policy = run_paired_policy_trials_sharded(
                trial_manifest, exp2_frontier, selected_traces, policies, cfg.seeds,
                causal_profiles, ctx, scientific=True,
                radio_tx_j_per_byte=float(cfg.hardware["radio_tx_j_per_byte"]),
                radio_rx_j_per_byte=float(cfg.hardware["radio_rx_j_per_byte"]),
            )
        else:
            exp4_policy = run_paired_policy_trials_sharded(
                trial_manifest, exp2_frontier, selected_traces, policies, cfg.seeds,
                causal_profiles, ctx, scientific=True,
                radio_tx_j_per_byte=float(cfg.hardware["radio_tx_j_per_byte"]),
                radio_rx_j_per_byte=float(cfg.hardware["radio_rx_j_per_byte"]),
            )
        assert_paired_design(exp4_policy)
        exp4_controller_detail, exp4_controller_summary = experiment4_controller_behavior(exp4_policy)
        exp4_policy = exp4_controller_detail
        controller_artifact_prefix = "test" if selected_split == "test" else "validation"
        controller_detail_artifact = (
            ctx.run_root / "results" / f"{controller_artifact_prefix}_controller_behavior_detail.csv.gz"
        )
        controller_summary_artifact = (
            ctx.run_root / "results" / f"{controller_artifact_prefix}_controller_behavior_summary.csv.gz"
        )
        atomic_write_frame(controller_detail_artifact, exp4_controller_detail)
        atomic_write_frame(controller_summary_artifact, exp4_controller_summary)
        display(resource_summary(exp4_policy))
        display(exp4_controller_summary)
    else:
        print("Experiments 3–4 full execution is paused.")
    """, "full-only"),
    md("""
    # Experiment 5 — risk, uncertainty, OOD, and evidence escalation

    Report natural-prevalence, enriched critical/hard-negative, and OOD cohorts
    separately at the same frozen average wire and energy budgets. The 5% false-
    alert limit is fixed before testing.
    """),
    code("""
    exp5_detail = exp5_summary = None
    if exp4_policy is not None:
        exp5_detail, exp5_summary = experiment5_risk_uncertainty(exp4_policy)
        atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp5_risk_uncertainty_detail.csv.gz",
            exp5_detail,
        )
        atomic_write_frame(
            ctx.run_root / "results" / f"{FULL_PHASE}_exp5_risk_uncertainty_summary.csv.gz",
            exp5_summary,
        )
        display(exp5_summary)
    else:
        print("Experiment 5 awaits Experiment 4 outputs.")
    """, "full-only"),
    md("""
    # Experiment 6 — receiver-state robustness

    Every profile runs the small receiver-state unit contract. The one-time
    untouched-test phase additionally runs the protocol-frozen deterministic
    fault campaign: identity-targeted data/fragment-ACK/application-ACK loss,
    data and feedback reordering, CRC corruption, missing bases, stale/expired
    state, schema/model revision mismatch, cumulative keyframe repair, and
    urgent bypass of queued bulk fragments. It also runs the registered paired
    comparison of application-ACKed deltas, periodic semantic keyframes, and
    explicit resynchronization after an application-ACK loss. A campaign case
    passes only when its explicit receiver/sender-state and event-log criterion
    passes under every frozen seed; strategy outcomes are reported, not forced.
    """),
    code("""
    exp6_campaign_detail = exp6_campaign_summary = None
    exp6_strategy_detail = exp6_strategy_summary = None
    exp6_registered_pass = False
    exp6_robustness = experiment6_receiver_robustness()
    display(exp6_robustness)
    if not exp6_robustness.passed.all():
        raise RuntimeError("Receiver-state robustness contract failed")
    if RUN_PROFILE == "full" and FULL_PHASE == "test" and EXECUTE_FULL_STAGES:
        campaign_spec = frozen_test_protocol["decisions"].get("experiment6_campaign")
        expected_campaign_spec = {
            "version": EXPERIMENT6_CAMPAIGN_VERSION,
            "cases": list(EXPERIMENT6_REGISTERED_CASES),
            "seeds": [int(seed) for seed in cfg.seeds],
        }
        if campaign_spec != expected_campaign_spec:
            raise StudyGateError(
                "Frozen Experiment 6 campaign is missing or differs from the executable contract; "
                "return to validation and freeze a new protocol before opening test data"
            )
        strategy_spec = frozen_test_protocol["decisions"].get(
            "experiment6_state_strategy_comparison"
        )
        expected_strategy_spec = {
            "version": EXPERIMENT6_STATE_STRATEGY_VERSION,
            "strategies": list(EXPERIMENT6_REGISTERED_STATE_STRATEGIES),
            "seeds": [int(seed) for seed in cfg.seeds],
            "periodic_keyframe_interval": 3,
            "fault": "drop_update_2_application_ack_once",
        }
        if strategy_spec != expected_strategy_spec:
            raise StudyGateError(
                "Frozen Experiment 6 state-strategy comparison is missing or differs "
                "from the executable contract; freeze a new protocol before test"
            )
        exp6_campaign_detail, exp6_campaign_summary = experiment6_registered_campaign(
            seeds=tuple(campaign_spec["seeds"]), scientific=True,
            registered_cases=tuple(campaign_spec["cases"]),
        )
        atomic_write_frame(
            ctx.run_root / "results" / "test_receiver_robustness_detail.csv.gz",
            exp6_campaign_detail,
        )
        atomic_write_frame(
            ctx.run_root / "results" / "test_receiver_robustness_summary.csv.gz",
            exp6_campaign_summary,
        )
        exp6_strategy_detail, exp6_strategy_summary = experiment6_state_strategy_comparison(
            seeds=tuple(strategy_spec["seeds"]), scientific=True,
            registered_strategies=tuple(strategy_spec["strategies"]),
            periodic_keyframe_interval=int(strategy_spec["periodic_keyframe_interval"]),
        )
        atomic_write_frame(
            ctx.run_root / "results" / "test_receiver_state_strategy_detail.csv.gz",
            exp6_strategy_detail,
        )
        atomic_write_frame(
            ctx.run_root / "results" / "test_receiver_state_strategy_summary.csv.gz",
            exp6_strategy_summary,
        )
        campaign_pass = bool(
            len(exp6_campaign_summary) == len(EXPERIMENT6_REGISTERED_CASES)
            and exp6_campaign_summary["fault_supported"].all()
            and exp6_campaign_summary["all_passed"].all()
        )
        strategy_execution_valid = bool(
            len(exp6_strategy_summary) == len(EXPERIMENT6_REGISTERED_STATE_STRATEGIES)
            and set(exp6_strategy_summary["strategy"])
                == set(EXPERIMENT6_REGISTERED_STATE_STRATEGIES)
            and exp6_strategy_summary["coverage_complete"].all()
            and exp6_strategy_summary["ack_loss_fault_exercised"].all()
        )
        exp6_registered_pass = bool(campaign_pass and strategy_execution_valid)
        display(exp6_campaign_summary)
        display(exp6_strategy_summary)
        print(
            "Experiment 6 registered campaign and strategy comparison:",
            "PASSED" if exp6_registered_pass else "FAILED_OR_INCOMPLETE",
        )
    else:
        print("Experiment 6 registered campaign runs only in the authorized full test phase.")
    """),
    md("""
    # Experiment 7 — embedded end-to-end

    This cell validates environment/target-log readiness and prints manually
    reviewed `tc/netem` commands; it does not apply privileged rules or invent a
    two-host result. A registered Phase-B sender/receiver run and its target log
    are external prerequisites. Phase C uses ns-3/5G-LENA only if the paper makes
    PHY/cellular claims. RTX 4090 timing cannot support a Jetson deployment claim.
    """),
    code("""
    exp7_readiness, embedded_records = experiment7_readiness(
        PROJECT_ROOT, CACHE_ROOT, DEVICE, RUN_PROFILE,
        cfg.hardware.get("target_log") if RUN_PROFILE == "full" and FULL_PHASE == "embedded" and EXECUTE_FULL_STAGES else None,
        expected_hardware_id=(
            str(cfg.hardware.get("hardware_id"))
            if RUN_PROFILE == "full" and FULL_PHASE == "embedded" and EXECUTE_FULL_STAGES else None
        ),
        expected_platform_class=(
            str(cfg.hardware.get("target"))
            if RUN_PROFILE == "full" and FULL_PHASE == "embedded" and EXECUTE_FULL_STAGES else None
        ),
    )
    display(exp7_readiness)
    embedded_gates = embedded_retest_effects = embedded_retest_status = h5_embedded_status = None
    if embedded_records is not None:
        if RUN_PROFILE == "full" and FULL_PHASE == "embedded":
            if embedded_deadline_ms is None:
                raise StudyGateError("Embedded deadline is unavailable from the finalized test handoff")
            deadline_ms = float(embedded_deadline_ms)
        else:
            deadline_ms = float(
                ((analysis_manifest["deadline_us"] - analysis_manifest["capture_time_us"]) / 1000).min()
            ) if analysis_manifest is not None else float(cfg.evaluation.get("max_vlm_p95_latency_ms", 1000.0))
        embedded_gates = embedded_gate(
            embedded_records, deadline_ms=deadline_ms,
            memory_limit_mb=float(cfg.evaluation.get("target_memory_limit_mb", 8000.0)),
            thermal_limit_c=float(cfg.hardware.get("thermal_limit_c", 82.0)),
        )
        display(embedded_gates)
        atomic_write_frame(ctx.run_root / "results" / "embedded_gate.csv.gz", embedded_gates)
        atomic_write_frame(ctx.run_root / "results" / "embedded_records.csv.gz", embedded_records)
        if RUN_PROFILE == "full" and FULL_PHASE == "embedded":
            if EMBEDDED_RETEST_SUMMARY.is_file():
                retest_input = import_embedded_retest_summary(EMBEDDED_RETEST_SUMMARY)
                decisions = frozen_test_protocol["decisions"]
                embedded_retest_status = evaluate_embedded_retest(
                    retest_input,
                    required_h2_baselines=decisions["required_h2_baselines"],
                    primary_margin=float(decisions["primary_margin"]),
                    false_alert_limit=float(decisions["false_alert_limit"]),
                    protocol_hash=str(frozen_test_protocol["protocol_hash"]),
                    target_log_sha256=hash_file(cfg.hardware["target_log"]),
                    expected_hardware_id=str(embedded_records["hardware_id"].iloc[0]),
                    analysis_plan_version=str(decisions["analysis_plan_version"]),
                    wire_budget_bytes_per_s=float(decisions["wire_budget"]),
                    energy_budget_j_per_s=float(decisions["energy_budget"]),
                    sample_minima=decisions["sample_minima"],
                    registered_calibration_gate_pass=bool(decisions["calibration_gate_pass"]),
                    bootstrap_repetitions=int(cfg.evaluation["bootstrap_repetitions"]),
                    bootstrap_seed=cfg.seed,
                    h4_matched_coverage=float(decisions["h4_matched_coverage"]),
                    h4_coverage_selection_rule=str(decisions["h4_coverage_selection_rule"]),
                )
                # The evaluator has now recomputed every comparison from the
                # hashed raw target-stage trials and reconciled these reported
                # estimates/intervals exactly. Only after that verification may
                # they become H5 replication-component figure inputs.
                embedded_retest_effects = retest_input.copy()
                original_hypotheses = pd.read_csv(
                    test_handoff.artifact("hypothesis_status.csv.gz"), compression="gzip"
                )
                h5_embedded_status = evaluate_h5_status(
                    original_hypotheses, embedded_retest_status, embedded_gates
                )
                atomic_write_frame(
                    ctx.run_root / "results" / "embedded_retest_status.csv.gz",
                    embedded_retest_status,
                )
                display(embedded_retest_status)
                display(h5_embedded_status)
            else:
                print(
                    "No registered embedded H2-H4 retest summary was supplied; "
                    "target metrics can be checked, but H5 remains incomplete."
                )
    example_netem = netem_command_plan("veth-uav", 0.5, 40, 5, 5, 128, cfg.seed)
    print("Manual-review netem example (NOT executed):")
    for command in example_netem:
        print("  ", render_command(command))
    """),
    md("""
    # Freeze, statistics, and completion decision

    Use validation-only power analysis before test access. Required minima are
    500 independent natural-prevalence visual groups, 30 traces from ≥6 held-out
    families, a separate safety cohort with ≥200 critical positives and ≥500
    noncritical/hard negatives, and ≥150 OOD units. Use 10,000 paired bootstrap
    replicates resampling visual groups and trace realizations independently.
    """),
    code("""
    if RUN_PROFILE == "full" and EXECUTE_FULL_STAGES and FULL_PHASE == "validation" and exp4_policy is not None:
        policy_spec_artifact = ctx.run_root / "results" / "validation_policy_spec.json"
        policy_spec = json.loads(policy_spec_artifact.read_text())
        required_h2_baselines = policy_spec["required_h2_baselines"]
        observed_policies = set(exp4_policy["policy"])
        missing_required = sorted(set(required_h2_baselines + ["proposed_full"]) - observed_policies)
        if missing_required:
            raise StudyGateError(f"Validation is missing required policies: {missing_required}")
        if not policy_spec.get("feature_baseline_available", False):
            raise StudyGateError("Registered L4 feature baseline is absent; broad H2 protocol cannot be frozen")
        validation_budget = budget_compliance(
            exp4_policy,
            float(cfg.evaluation["primary_wire_budget_bytes_per_s"]),
            float(cfg.evaluation["primary_energy_budget_j_per_s"]),
            tolerance=0.05,
        )
        atomic_write_frame(ctx.run_root / "results" / "validation_budget_compliance.csv.gz", validation_budget)
        display(validation_budget)
        required_budget = validation_budget[
            validation_budget["policy"].isin(required_h2_baselines + ["proposed_full"])
        ]
        if len(required_budget) != len(set(required_h2_baselines + ["proposed_full"])) or not required_budget["eligible"].all():
            raise StudyGateError("Matched wire/energy budget gate failed on validation; retune without opening test")
        h4_matched_coverage = 0.10
        _, validation_h4_coverage_gate = matched_coverage_semantic_errors(
            exp4_policy,
            ("proposed_full", "ablate_evidence_escalation"),
            h4_matched_coverage,
        )
        validation_h4_coverage_artifact = (
            ctx.run_root / "results" / "validation_h4_matched_coverage_gate.csv.gz"
        )
        atomic_write_frame(validation_h4_coverage_artifact, validation_h4_coverage_gate)
        display(validation_h4_coverage_gate)
        if not validation_h4_coverage_gate["passed"].astype(bool).all():
            raise StudyGateError(
                "H4 cannot be registered: validation lacks the predeclared "
                "group-matched semantic coverage"
            )
        validation_rates = resource_summary(exp4_policy)
        strongest_baseline = validation_rates[
            validation_rates["policy"].isin(required_h2_baselines)
        ].sort_values("correct_timely_rate", ascending=False).iloc[0].policy
        power_rows = []
        for baseline in required_h2_baselines:
            power_pair = exp4_policy[
                exp4_policy["policy"].isin(["proposed_full", baseline])
            ]
            power_pivot = power_pair.pivot(
                index=["sample_id", "visual_group_id", "trace_id", "replicate_seed"],
                columns="policy", values="correct_timely_action",
            ).reset_index()
            power_pivot["discordant"] = power_pivot["proposed_full"].ne(
                power_pivot[baseline]
            )
            power_pivot["delta"] = (
                power_pivot["proposed_full"].astype(float)
                - power_pivot[baseline].astype(float)
            )
            group_trace_power = power_pivot.groupby(
                ["visual_group_id", "trace_id"], as_index=False,
            ).agg(discordant=("discordant", "mean"), delta=("delta", "mean"))
            discordance = float(group_trace_power["discordant"].mean())
            validation_effect = float(group_trace_power["delta"].mean())
            comparison_alpha = 0.05 / max(1, len(required_h2_baselines))
            estimated_power = validation_power_simulation(
                max(discordance, 1e-3), exp4_policy.visual_group_id.nunique(),
                exp4_policy.trace_id.nunique(), effect=validation_effect,
                null_margin=float(cfg.evaluation["primary_margin"]),
                alpha=comparison_alpha,
            )
            power_rows.append({
                "baseline": baseline,
                "discordance": discordance,
                "validation_effect": validation_effect,
                "null_margin": float(cfg.evaluation["primary_margin"]),
                "familywise_alpha_per_comparison": comparison_alpha,
                "estimated_power": estimated_power,
            })
        validation_power_by_baseline = pd.DataFrame(power_rows)
        validation_power_artifact = (
            ctx.run_root / "results" / "validation_power_by_baseline.csv.gz"
        )
        atomic_write_frame(validation_power_artifact, validation_power_by_baseline)
        display(validation_power_by_baseline)
        validation_power = float(validation_power_by_baseline["estimated_power"].min())
        print("Worst-case validation-only power across every frozen H2 comparator:", validation_power)
        if validation_power < float(cfg.evaluation["minimum_power"]):
            raise StudyGateError("Power gate failed; enlarge the independent test design before unblinding")
        selection_artifact = ctx.run_root / "results" / "selected_deployed_model.json"
        profile_artifact = ctx.run_root / "results" / "validation_candidate_profiles.csv.gz"
        calibration_gate_artifact = ctx.run_root / "results" / "validation_calibration_gate.csv.gz"
        data_snapshot_artifact = ctx.run_root / "results" / "frozen_dataset_content_snapshot.json"
        exp3_variant_spec_artifact = (
            ctx.run_root / "results" / "validation_exp3_variant_spec.csv.gz"
        )
        required_validation_artifacts = [
            selection_artifact, profile_artifact, policy_spec_artifact,
            calibration_gate_artifact, data_snapshot_artifact,
            validation_power_artifact, environment_fingerprint_artifact,
            validation_h4_coverage_artifact, detector_residency_artifact,
            exp3_variant_spec_artifact,
        ]
        if not all(path.is_file() for path in required_validation_artifacts):
            raise StudyGateError("One or more validation/frozen-input artifacts are missing")
        selection = json.loads(selection_artifact.read_text())
        artifact_paths = {
            **cfg.dataset_provenance_paths(),
            "split_manifest": cfg.paths.split_manifest,
            "development_annotations": cfg.paths.annotations,
            "untouched_test_annotations": cfg.paths.test_annotations,
            "telemetry_manifest": cfg.paths.telemetry_manifest,
            "channel_trace_manifest": cfg.paths.channel_traces,
            "detector_checkpoint": cfg.models["detector"]["checkpoint"],
            "native_receiver_cache": NATIVE_RECEIVER_CACHE,
            "registered_feature_cache": REGISTERED_FEATURE_CACHE,
            "prompt": PROJECT_ROOT / "prompts" / "mission_packet_v1.txt",
            "mission_ontology": PROJECT_ROOT / "annotations" / "mission_ontology.yaml",
            "dataset_content_snapshot": data_snapshot_artifact,
            "model_selection": selection_artifact,
            "calibration_gate": calibration_gate_artifact,
            "candidate_profiles": profile_artifact,
            "policy_spec": policy_spec_artifact,
            "exp3_variant_spec": exp3_variant_spec_artifact,
            "validation_power_by_baseline": validation_power_artifact,
            "validation_h4_matched_coverage_gate": validation_h4_coverage_artifact,
            "environment_fingerprint": environment_fingerprint_artifact,
            "detector_residency": detector_residency_artifact,
        }
        decisions = {
            "selected_model": selection["selected_model"],
            "model_selection_hash": hash_file(selection_artifact),
            "candidate_profile_hash": hash_file(profile_artifact),
            "policy_spec_hash": hash_file(policy_spec_artifact),
            "exp3_variant_spec_hash": hash_file(exp3_variant_spec_artifact),
            "artifact_manifest": frozen_file_manifest(artifact_paths),
            "required_dataset_provenance_artifacts": sorted(
                cfg.dataset_provenance_paths()
            ),
            "required_h2_baselines": required_h2_baselines,
            "strongest_validation_baseline": strongest_baseline,
            "feature_baseline_required": True,
            "wire_budget": cfg.evaluation["primary_wire_budget_bytes_per_s"],
            "energy_budget": cfg.evaluation["primary_energy_budget_j_per_s"],
            "false_alert_limit": cfg.evaluation["false_alert_limit"],
            "primary_margin": cfg.evaluation["primary_margin"],
            "h4_matched_coverage": h4_matched_coverage,
            "h4_coverage_selection_rule": "top_calibrated_semantic_score_within_visual_group",
            "policy_parameters": cfg.controller,
            "model_revisions": cfg.models,
            "sample_minima": {
                "natural_test_units": cfg.evaluation["natural_test_units"],
                "safety_critical_positive_units": cfg.evaluation["safety_critical_positive_units"],
                "safety_noncritical_units": cfg.evaluation["safety_noncritical_units"],
                "ood_units": cfg.evaluation["ood_units"],
                "held_out_trace_realizations": cfg.network["held_out_trace_realizations"],
                "held_out_trace_families": len(cfg.network["held_out_trace_families"]),
            },
            "calibration_gate_pass": bool(all(
                any(
                    row["model_id"] == required_model and bool(row["passed"])
                    for row in selection["calibration_gate"]
                )
                for required_model in {selection["selected_model"], "detector_only"}
            )),
            "experiment6_campaign": {
                "version": EXPERIMENT6_CAMPAIGN_VERSION,
                "cases": list(EXPERIMENT6_REGISTERED_CASES),
                "seeds": [int(seed) for seed in cfg.seeds],
            },
            "experiment6_state_strategy_comparison": {
                "version": EXPERIMENT6_STATE_STRATEGY_VERSION,
                "strategies": list(EXPERIMENT6_REGISTERED_STATE_STRATEGIES),
                "seeds": [int(seed) for seed in cfg.seeds],
                "periodic_keyframe_interval": 3,
                "fault": "drop_update_2_application_ack_once",
            },
            "analysis_plan_version": "semuav.analysis.v1",
            "notebook_path": str(PROJECT_ROOT / "SemUAV_VLM_Full_Study.ipynb"),
            "notebook_source_hash": notebook_source_hash(PROJECT_ROOT / "SemUAV_VLM_Full_Study.ipynb"),
        }
        if FREEZE_PROTOCOL:
            frozen_path = freeze_protocol(ctx, decisions, {
                "validation_power": validation_power,
                "strongest_baseline": strongest_baseline,
                "budget_compliance_hash": hash_file(ctx.run_root / "results" / "validation_budget_compliance.csv.gz"),
            })
            print("Protocol frozen:", frozen_path)
        else:
            print("Protocol not frozen. Inspect all validation outputs, then set FREEZE_PROTOCOL=True exactly once.")
    """, "full-only"),
    code("""
    confirmatory_report = h1_report = h3_report = h4_report = hypothesis_status = None
    final_gate_table = test_budget = None
    if RUN_PROFILE == "full" and FULL_PHASE == "test" and exp4_policy is not None:
        decisions = frozen_test_protocol["decisions"]
        required_baselines = list(decisions["required_h2_baselines"])
        required_policies = required_baselines + ["proposed_full"]
        observed_policies = set(exp4_policy["policy"])
        policy_gate_scope = {
            "proposed_full": "H2-H4",
            "channel_only_hysteresis": "H2-H3",
            "ablate_evidence_escalation": "H2-H4",
        }
        coverage_rows = [{
            "gate": f"policy_present:{policy}", "scope": "H2",
            "passed": policy in observed_policies,
            "required_for": policy_gate_scope.get(policy, "H2"),
            "policy": policy,
            "evidence_path": "test_policy_trials.csv.gz",
            "detail": "present" if policy in observed_policies else "missing",
        } for policy in required_policies]
        test_trials_artifact = ctx.run_root / "results" / "test_policy_trials.csv.gz"
        atomic_write_frame(test_trials_artifact, exp4_policy)
        test_budget = budget_compliance(
            exp4_policy, float(decisions["wire_budget"]), float(decisions["energy_budget"]),
            tolerance=0.05,
        )
        test_budget_artifact = ctx.run_root / "results" / "test_budget_compliance.csv.gz"
        atomic_write_frame(test_budget_artifact, test_budget)
        display(test_budget)
        budget_rows = []
        budget_lookup = test_budget.set_index("policy").to_dict("index")
        for policy in required_policies:
            row = budget_lookup.get(policy)
            budget_rows.append({
                "gate": f"matched_budget:{policy}", "scope": "H2",
                "passed": bool(row and row["eligible"]),
                "required_for": policy_gate_scope.get(policy, "H2"),
                "policy": policy,
                "evidence_path": str(test_budget_artifact),
                "detail": (
                    f"wire={row['wire_bytes_per_s']:.6g}, energy={row['energy_j_per_s']:.6g}"
                    if row else "policy missing"
                ),
            })
        h4_matched_trials, h4_coverage_gate = matched_coverage_semantic_errors(
            exp4_policy,
            ("proposed_full", "ablate_evidence_escalation"),
            float(decisions["h4_matched_coverage"]),
        )
        h4_matched_trials_artifact = (
            ctx.run_root / "results" / "test_h4_matched_coverage_trials.csv.gz"
        )
        h4_coverage_gate_artifact = (
            ctx.run_root / "results" / "test_h4_matched_coverage_gate.csv.gz"
        )
        atomic_write_frame(h4_matched_trials_artifact, h4_matched_trials)
        atomic_write_frame(h4_coverage_gate_artifact, h4_coverage_gate)
        display(h4_coverage_gate)
        h4_coverage_rows = [{
            "gate": f"h4_matched_coverage:{row.policy}",
            "scope": "H4",
            "passed": bool(row.passed),
            "required_for": "H4",
            "policy": str(row.policy),
            "evidence_path": str(h4_coverage_gate_artifact),
            "detail": (
                f"target={row.target_coverage}, "
                f"realized={row.realized_group_equal_coverage}, "
                f"insufficient_groups={row.insufficient_visual_groups}"
            ),
        } for row in h4_coverage_gate.itertuples(index=False)]
        sample_rows = [{
            "gate": str(row.gate), "scope": "all_confirmatory",
            "passed": bool(row.passed), "required_for": "H1-H5",
            "policy": "",
            "evidence_path": str(ctx.run_root / "results" / f"{FULL_PHASE}_sample_gate.csv.gz"),
            "detail": f"observed={row.observed}, required={row.required}",
        } for row in sample_gate.itertuples(index=False)]
        selection = json.loads((ctx.run_root / "results" / "selected_deployed_model.json").read_text())
        selected_calibration = next(
            (row for row in selection["calibration_gate"] if row["model_id"] == selection["selected_model"]),
            None,
        )
        detector_calibration = next(
            (row for row in selection["calibration_gate"] if row["model_id"] == "detector_only"),
            None,
        )
        test_calibration_lookup = (
            test_calibration_report.set_index("model_id").to_dict("index")
            if test_calibration_report is not None and len(test_calibration_report) else {}
        )
        rare_agreements = [
            float(value["chance_corrected_agreement"])
            for task, value in annotation_report.by_task.items()
            if str(task).startswith("__rare__:")
        ]
        minimum_rare_agreement = min(rare_agreements) if rare_agreements else float("nan")
        scientific_rows = [
            {"gate": "selected_context_is_vlm", "scope": "H1-H5", "passed": selected_vlm_claim_gate(selection), "required_for": "H1-H5", "policy": "", "evidence_path": str(selection_artifact), "detail": selection["selected_model"]},
            {"gate": "annotation_overall", "scope": "all_confirmatory", "passed": annotation_report.chance_corrected_agreement >= 0.80, "required_for": "H1-H5", "policy": "", "evidence_path": str(ctx.run_root / "results" / f"{FULL_PHASE}_annotation_audit.json"), "detail": f"kappa={annotation_report.chance_corrected_agreement}"},
            {"gate": "annotation_rare", "scope": "all_confirmatory", "passed": np.isfinite(minimum_rare_agreement) and minimum_rare_agreement >= 0.70, "required_for": "H1-H5", "policy": "", "evidence_path": str(ctx.run_root / "results" / f"{FULL_PHASE}_annotation_audit.json"), "detail": f"minimum rare kappa={minimum_rare_agreement}"},
            {"gate": "selected_model_calibration", "scope": "H2-H4", "passed": bool(selected_calibration and selected_calibration["passed"]), "required_for": "H2-H4", "policy": "", "evidence_path": str(ctx.run_root / "results" / "validation_calibration_gate.csv.gz"), "detail": selection["selected_model"]},
            {"gate": "detector_calibration", "scope": "H2-H4", "passed": bool(detector_calibration and detector_calibration["passed"]), "required_for": "H2-H4", "policy": "", "evidence_path": str(ctx.run_root / "results" / "validation_calibration_gate.csv.gz"), "detail": "detector_only"},
            {"gate": "selected_model_test_calibration", "scope": "H2-H4", "passed": bool(test_calibration_lookup.get(selection["selected_model"], {}).get("passed", False)), "required_for": "H2-H4", "policy": "", "evidence_path": str(ctx.run_root / "results" / "test_calibration_report.csv.gz"), "detail": selection["selected_model"]},
            {"gate": "detector_test_calibration", "scope": "H2-H4", "passed": bool(test_calibration_lookup.get("detector_only", {}).get("passed", False)), "required_for": "H2-H4", "policy": "", "evidence_path": str(ctx.run_root / "results" / "test_calibration_report.csv.gz"), "detail": "detector_only"},
            {"gate": "registered_feature_baseline", "scope": "H1-H2", "passed": bool(decisions.get("feature_baseline_required")), "required_for": "H1-H2", "policy": "", "evidence_path": decisions["artifact_manifest"]["registered_feature_cache"]["path"], "detail": "frozen L4 cache"},
        ]
        final_gate_table = pd.DataFrame([
            *sample_rows, *coverage_rows, *budget_rows,
            *h4_coverage_rows, *scientific_rows,
        ])
        final_gate_artifact = ctx.run_root / "results" / "confirmatory_gate_table.csv.gz"
        atomic_write_frame(final_gate_artifact, final_gate_table)
        display(final_gate_table)

        h1_preconditions = confirmatory_gate_pass(final_gate_table, "H1")
        h2_preconditions = confirmatory_gate_pass(
            final_gate_table, "H2", required_policies=required_policies,
        )
        h3_preconditions = confirmatory_gate_pass(
            final_gate_table, "H3",
            required_policies=("proposed_full", "channel_only_hysteresis"),
        )
        h4_preconditions = confirmatory_gate_pass(
            final_gate_table, "H4",
            required_policies=("proposed_full", "ablate_evidence_escalation"),
        )
        effects = []
        if set(required_policies).issubset(observed_policies):
            family_confidence = 1.0 - 0.05 / max(1, len(required_baselines))
            effects = [
                two_way_clustered_paired_bootstrap(
                    exp4_policy, "proposed_full", baseline,
                    repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed,
                    confidence=family_confidence,
                ) for baseline in required_baselines
            ]
        adjusted = holm_adjust({
            f"{effect.method}_vs_{effect.reference}": effect.p_two_sided for effect in effects
        }) if effects else {}
        confirmatory_report = hypothesis_report(
            effects, adjusted, margin=float(decisions["primary_margin"])
        )
        if len(confirmatory_report):
            confirmatory_report["familywise_method"] = "Bonferroni simultaneous percentile intervals"
            confirmatory_report["matched_budget_gate"] = h2_preconditions
        confirmatory_artifact = ctx.run_root / "results" / "h2_confirmatory_report.csv.gz"
        atomic_write_frame(confirmatory_artifact, confirmatory_report)
        display(confirmatory_report)
        h2_supported = bool(h2_preconditions and len(confirmatory_report) == len(required_baselines) and confirmatory_report["exceeds_margin"].all())

        # H1: validation-frozen L2/L3 variants versus the validation-frozen
        # practical L5 image baseline on identical static scene/link tuples.
        policy_spec = json.loads((ctx.run_root / "results" / "validation_policy_spec.json").read_text())
        fixed_lookup = {item["mode"]: item for item in policy_spec["fixed"]}
        h1_rows = []
        image_spec = fixed_lookup.get("L5_IMAGE_OR_CLIP")
        if image_spec and exp3_partition_index is not None:
            h1_mode_variants = {
                (mode, fixed_lookup[mode]["variant"])
                for mode in ("L2_SEMANTICS", "L3_SEMANTICS_ROI")
                if mode in fixed_lookup
            }
            h1_mode_variants.add(("L5_IMAGE_OR_CLIP", image_spec["variant"]))
            h1_read_columns = {
                "observation_id", "mode", "variant", "source_model",
                "replicate_seed", "rate_mbps", "rtt_ms", "loss", "loss_model",
                "burst_recovery_probability", "burst_slot_ms", "jitter_ms", "queue_kb",
                "latency_ms", "deadline_us", "capture_us", "total_wire_bytes",
                "energy_j", "correct_timely_action",
            }
            h1_parts = []
            for partition_path in exp3_partition_index["artifact_path"].astype(str):
                partition = pd.read_csv(
                    partition_path,
                    compression="infer",
                    usecols=lambda column: column in h1_read_columns,
                )
                relevant = partition.apply(
                    lambda row: (str(row["mode"]), str(row["variant"]))
                    in h1_mode_variants,
                    axis=1,
                )
                if relevant.any():
                    h1_parts.append(partition.loc[relevant].copy())
                del partition
            if not h1_parts:
                raise StudyGateError(
                    "Experiment 3 partitions contain none of the frozen H1 variants"
                )
            static = pd.concat(h1_parts, ignore_index=True).merge(
                analysis_manifest[["sample_id", "visual_group_id", "mission_profile"]].rename(columns={"sample_id": "observation_id"}),
                on="observation_id", validate="many_to_one",
            )
            static["sample_id"] = static["observation_id"].astype(str)
            static["trace_id"] = static.apply(
                lambda row: (
                    f"rate={row.rate_mbps}|rtt={row.rtt_ms}|loss={row.loss}"
                    f"|loss_model={row.loss_model}"
                    f"|burst_q={row.burst_recovery_probability}"
                    f"|burst_slot_ms={row.burst_slot_ms}"
                    f"|jitter={row.jitter_ms}|queue={row.queue_kb}"
                ),
                axis=1,
            )
            assert_experiment3_exact_replicate_coverage(static, cfg.seeds)
            static["deadline_censored_latency_ms"] = np.where(
                np.isfinite(static["latency_ms"]),
                np.minimum(static["latency_ms"], (static["deadline_us"] - static["capture_us"]) / 1000),
                (static["deadline_us"] - static["capture_us"]) / 1000,
            )
            comparison_confidence = 1.0 - 0.05 / 8
            for mode in ("L2_SEMANTICS", "L3_SEMANTICS_ROI"):
                semantic_spec = fixed_lookup.get(mode)
                if not semantic_spec:
                    continue
                subset = static[
                    ((static["mode"].eq(mode)) & static["variant"].eq(semantic_spec["variant"]))
                    | ((static["mode"].eq("L5_IMAGE_OR_CLIP")) & static["variant"].eq(image_spec["variant"]))
                ].copy()
                subset["policy"] = np.where(subset["mode"].eq(mode), "semantic", "image")
                if set(subset["policy"]) != {"semantic", "image"}:
                    continue
                utility = two_way_clustered_paired_bootstrap(subset, "semantic", "image", repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed, confidence=comparison_confidence)
                bytes_saved = two_way_clustered_paired_bootstrap(subset, "image", "semantic", endpoint="total_wire_bytes", repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed, confidence=comparison_confidence)
                latency_saved = two_way_clustered_paired_bootstrap(subset, "image", "semantic", endpoint="deadline_censored_latency_ms", repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed, confidence=comparison_confidence)
                energy_saved = two_way_clustered_paired_bootstrap(subset, "image", "semantic", endpoint="energy_j", repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed, confidence=comparison_confidence)
                h1_rows.append({
                    "semantic_mode": mode, "image_variant": image_spec["variant"],
                    "utility_difference": utility.estimate,
                    "utility_ci_low": utility.ci_low, "utility_ci_high": utility.ci_high,
                    "wire_bytes_saved": bytes_saved.estimate,
                    "wire_bytes_saved_ci_low": bytes_saved.ci_low,
                    "wire_bytes_saved_ci_high": bytes_saved.ci_high,
                    "latency_ms_saved": latency_saved.estimate,
                    "latency_ms_saved_ci_low": latency_saved.ci_low,
                    "latency_ms_saved_ci_high": latency_saved.ci_high,
                    "energy_j_saved": energy_saved.estimate,
                    "energy_j_saved_ci_low": energy_saved.ci_low,
                    "energy_j_saved_ci_high": energy_saved.ci_high,
                    "noninferiority_margin": -float(decisions["primary_margin"]),
                    "supported": bool(
                        utility.ci_low > -float(decisions["primary_margin"])
                        and bytes_saved.ci_low > 0
                        and latency_saved.ci_low > 0
                        and energy_saved.ci_low > 0
                    ),
                })
        h1_report = pd.DataFrame(h1_rows)
        h1_artifact = ctx.run_root / "results" / "h1_semantic_noninferiority.csv.gz"
        atomic_write_frame(h1_artifact, h1_report)

        critical_trials = exp4_policy[exp4_policy["critical"].astype(bool)]
        h3_effect = two_way_clustered_paired_bootstrap(
            critical_trials, "channel_only_hysteresis", "proposed_full", endpoint="critical_miss",
            repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed,
        )
        full_noncritical = exp4_policy[(exp4_policy.policy == "proposed_full") & ~exp4_policy.critical.astype(bool)]
        channel_noncritical = exp4_policy[(exp4_policy.policy == "channel_only_hysteresis") & ~exp4_policy.critical.astype(bool)]
        full_far = (
            float(full_noncritical.groupby("visual_group_id")["false_alert"].mean().mean())
            if len(full_noncritical) else np.nan
        )
        channel_far = (
            float(channel_noncritical.groupby("visual_group_id")["false_alert"].mean().mean())
            if len(channel_noncritical) else np.nan
        )

        h4_coverage_pass = bool(h4_coverage_gate["passed"].astype(bool).all())
        h4_effect = (
            two_way_clustered_paired_bootstrap(
                h4_matched_trials,
                "ablate_evidence_escalation", "proposed_full",
                endpoint="matched_coverage_semantic_error",
                repetitions=int(cfg.evaluation["bootstrap_repetitions"]), seed=cfg.seed,
            )
            if h4_coverage_pass else None
        )
        secondary_adjusted = holm_adjust({
            "H3_critical_miss": h3_effect.p_two_sided,
            # A failed matched-coverage gate makes H4 untestable; p=1 keeps the
            # registered two-hypothesis Holm family conservative.
            "H4_high_confidence_error": h4_effect.p_two_sided if h4_effect is not None else 1.0,
        })
        false_alert_limit = float(decisions["false_alert_limit"])
        h3_report = pd.DataFrame([{
            "critical_miss_reduction": h3_effect.estimate, "ci_low": h3_effect.ci_low,
            "ci_high": h3_effect.ci_high, "full_false_alert_rate": full_far,
            "channel_only_false_alert_rate": channel_far,
            "holm_adjusted_p": secondary_adjusted["H3_critical_miss"],
            "false_alert_limit": false_alert_limit,
            "all_applicable_gates_passed": h3_preconditions,
            "supported": bool(
                h3_preconditions
                and h3_effect.ci_low > 0
                and secondary_adjusted["H3_critical_miss"] <= 0.05
                and np.isfinite(full_far) and full_far <= false_alert_limit
                and np.isfinite(channel_far) and channel_far <= false_alert_limit
            ),
        }])
        h3_artifact = ctx.run_root / "results" / "h3_mission_awareness.csv.gz"
        atomic_write_frame(h3_artifact, h3_report)

        h4_coverage_lookup = h4_coverage_gate.set_index("policy")
        proposed_selective_coverage = float(h4_coverage_lookup.loc[
            "proposed_full", "realized_group_equal_coverage"
        ])
        proposed_selective_risk = float(h4_coverage_lookup.loc[
            "proposed_full", "group_equal_selective_risk"
        ])
        ablation_selective_coverage = float(h4_coverage_lookup.loc[
            "ablate_evidence_escalation", "realized_group_equal_coverage"
        ])
        ablation_selective_risk = float(h4_coverage_lookup.loc[
            "ablate_evidence_escalation", "group_equal_selective_risk"
        ])
        h4_coverage_exactly_matched = bool(
            h4_coverage_pass
            and np.isclose(
                proposed_selective_coverage, ablation_selective_coverage,
                rtol=0.0, atol=1e-12,
            )
        )
        h4_report = pd.DataFrame([{
            "high_confidence_error_reduction": h4_effect.estimate if h4_effect is not None else np.nan,
            "ci_low": h4_effect.ci_low if h4_effect is not None else np.nan,
            "ci_high": h4_effect.ci_high if h4_effect is not None else np.nan,
            "holm_adjusted_p": secondary_adjusted["H4_high_confidence_error"],
            "proposed_semantic_selective_coverage": proposed_selective_coverage,
            "proposed_semantic_selective_risk": proposed_selective_risk,
            "ablation_semantic_selective_coverage": ablation_selective_coverage,
            "ablation_semantic_selective_risk": ablation_selective_risk,
            "predeclared_target_coverage": float(decisions["h4_matched_coverage"]),
            "coverage_selection_rule": decisions["h4_coverage_selection_rule"],
            "coverage_exactly_matched_within_visual_group": h4_coverage_exactly_matched,
            "all_applicable_gates_passed": h4_preconditions,
            "supported": bool(
                h4_preconditions
                and h4_effect is not None
                and h4_effect.ci_low > 0
                and secondary_adjusted["H4_high_confidence_error"] <= 0.05
                and h4_coverage_exactly_matched
                and np.isfinite(proposed_selective_coverage)
                and np.isfinite(ablation_selective_coverage)
                and proposed_selective_coverage > 0
                and ablation_selective_coverage > 0
                and np.isfinite(proposed_selective_risk)
                and np.isfinite(ablation_selective_risk)
                and proposed_selective_risk < ablation_selective_risk
            ),
        }])
        h4_artifact = ctx.run_root / "results" / "h4_uncertainty_escalation.csv.gz"
        atomic_write_frame(h4_artifact, h4_report)

        h1_complete = bool(h1_preconditions and len(h1_report) == 2)
        h2_complete = bool(h2_preconditions and len(confirmatory_report) == len(required_baselines))
        h3_complete = bool(h3_preconditions and h3_effect is not None)
        h4_complete = bool(h4_preconditions and h4_effect is not None and h4_coverage_exactly_matched)
        h1_supported = bool(h1_complete and h1_report["supported"].all())
        h3_supported = bool(h3_report.iloc[0]["supported"])
        h4_supported = bool(h4_report.iloc[0]["supported"])
        embedded_pass = bool(embedded_gates is not None and embedded_gates["passed"].all())
        hypothesis_status = pd.DataFrame([
            {"hypothesis": "H1", "execution_status": "EMPIRICAL_COMPLETE" if h1_complete else "BLOCKED_GATES_OR_INCOMPLETE_ANALYSIS", "claim_supported": h1_supported},
            {"hypothesis": "H2", "execution_status": "EMPIRICAL_COMPLETE" if h2_complete else "BLOCKED_GATES_OR_INCOMPLETE_ANALYSIS", "claim_supported": h2_supported},
            {"hypothesis": "H3", "execution_status": "EMPIRICAL_COMPLETE" if h3_complete else "BLOCKED_GATES_OR_INCOMPLETE_ANALYSIS", "claim_supported": h3_supported},
            {"hypothesis": "H4", "execution_status": "EMPIRICAL_COMPLETE" if h4_complete else "BLOCKED_GATES_OR_INCOMPLETE_ANALYSIS", "claim_supported": h4_supported},
            {"hypothesis": "H5", "execution_status": "TARGET_METRICS_COMPLETE_RETEST_REQUIRED" if embedded_pass else "BLOCKED_EXTERNAL_VALIDATION", "claim_supported": False},
        ])
        hypothesis_artifact = ctx.run_root / "results" / "hypothesis_status.csv.gz"
        atomic_write_frame(hypothesis_artifact, hypothesis_status)
        display(hypothesis_status)
    """, "full-only"),
    code("""
    print("Run artifacts:", ctx.run_root)
    display(ctx.status_frame())
    if RUN_PROFILE == "dry_run":
        print("FINAL STATUS: SOFTWARE CHECKS PASSED; NO EMPIRICAL STUDY EXECUTED")
    elif RUN_PROFILE == "smoke_real":
        print("FINAL STATUS: PRECHECK_ONLY; NO EMPIRICAL STUDY EXECUTED")
    elif RUN_PROFILE == "full" and FULL_PHASE == "embedded":
        if not EXECUTE_FULL_STAGES:
            print("FINAL STATUS: EMBEDDED EVIDENCE PHASE PAUSED")
        else:
            if not test_results_verified:
                raise StudyGateError("Embedded completion requires verified finalized test results")
            test_completion_artifact = test_handoff.artifact("final_completion_status.csv.gz")
            test_hypothesis_artifact = test_handoff.artifact("hypothesis_status.csv.gz")
            completion_status = pd.read_csv(test_completion_artifact, compression="gzip")
            hypothesis_status = pd.read_csv(test_hypothesis_artifact, compression="gzip")
            embedded_pass = bool(embedded_gates is not None and embedded_gates["passed"].all())
            h5_mask = hypothesis_status["hypothesis"].eq("H5")
            if int(h5_mask.sum()) != 1:
                raise StudyGateError("Finalized test hypothesis table does not contain exactly one H5 row")
            if h5_embedded_status is not None:
                hypothesis_status = pd.concat(
                    [hypothesis_status.loc[~h5_mask].copy(), h5_embedded_status.copy()],
                    ignore_index=True,
                )
                h5_execution = str(h5_embedded_status.iloc[0]["execution_status"])
                h5_complete = h5_execution == "EMPIRICAL_COMPLETE"
                h5_supported = bool(h5_embedded_status.iloc[0]["claim_supported"])
            else:
                hypothesis_status.loc[h5_mask, "execution_status"] = (
                    "TARGET_METRICS_COMPLETE_RETEST_REQUIRED" if embedded_pass else "BLOCKED_EXTERNAL_VALIDATION"
                )
                hypothesis_status.loc[h5_mask, "claim_supported"] = False
                h5_complete = h5_supported = False
                h5_execution = (
                    "TARGET_METRICS_COMPLETE_RETEST_REQUIRED" if embedded_pass else "BLOCKED_EXTERNAL_VALIDATION"
                )
            experiment7_mask = completion_status["experiment"].eq(7)
            if int(experiment7_mask.sum()) != 1:
                raise StudyGateError("Finalized test completion table does not contain exactly one Experiment 7 row")
            completion_status.loc[experiment7_mask, "execution_status"] = h5_execution
            completion_status.loc[experiment7_mask, "evidence_class"] = (
                "target_hardware_retest" if h5_embedded_status is not None
                else "target_hardware_log" if embedded_gates is not None else "readiness_only"
            )
            completion_status.loc[experiment7_mask, "empirical_complete"] = h5_complete
            completion_status.loc[experiment7_mask, "scientific_claim_eligible"] = h5_complete
            completion_status.loc[experiment7_mask, "claim_status"] = (
                "SUPPORTED" if h5_supported else "NOT_SUPPORTED" if h5_complete else "NOT_SUPPORTED_OR_INELIGIBLE"
            )
            atomic_write_frame(ctx.run_root / "results" / "embedded_completion_status.csv.gz", completion_status)
            atomic_write_frame(ctx.run_root / "results" / "embedded_hypothesis_status.csv.gz", hypothesis_status)
            atomic_write_json(ctx.run_root / "results" / "EMBEDDED_STATUS.json", {
                "overall_status": h5_execution,
                "test_results_verified": True,
                "hypothesis_supported": h5_supported,
            })
            display(completion_status)
            display(hypothesis_status[hypothesis_status["hypothesis"].eq("H5")])
            h5_outcome = "SUPPORTED" if h5_supported else "NOT SUPPORTED" if h5_complete else "INELIGIBLE"
            print("FINAL STATUS:", h5_execution, h5_outcome)
    else:
        completion_rows = []
        test_phase = FULL_PHASE == "test"
        all_confirmatory_gates_passed = bool(
            test_phase
            and final_gate_table is not None
            and len(final_gate_table)
            and final_gate_table["passed"].astype(bool).all()
        )
        all_test_hypothesis_analyses_complete = bool(
            hypothesis_status is not None
            and set(hypothesis_status["hypothesis"]) >= {"H1", "H2", "H3", "H4"}
            and hypothesis_status[
                hypothesis_status["hypothesis"].isin(["H1", "H2", "H3", "H4"])
            ]["execution_status"].eq("EMPIRICAL_COMPLETE").all()
        )
        empirical_evidence_gates_passed = bool(
            test_access_opened
            and all_confirmatory_gates_passed
            and all_test_hypothesis_analyses_complete
        )
        empirical_objects = {
            1: exp1_detail, 2: exp2_frontier, 3: exp3_transport,
            4: exp4_policy, 5: exp5_detail,
        }
        for number, name in {
            1: "Perception and packet validity", 2: "Static representation frontier",
            3: "Static transport grid", 4: "Adaptive controller on held-out traces",
            5: "Mission risk and uncertainty", 6: "Receiver-state robustness",
            7: "Embedded end-to-end",
        }.items():
            if number <= 5:
                data_executed = bool(
                    test_phase and empirical_objects[number] is not None
                    and len(empirical_objects[number])
                )
                complete = bool(data_executed and empirical_evidence_gates_passed)
                execution = (
                    "EMPIRICAL_COMPLETE_ALL_GATES_PASSED" if complete
                    else "EXECUTED_BUT_EMPIRICALLY_INELIGIBLE" if data_executed
                    else "NOT_RUN_OR_VALIDATION_ONLY"
                )
                evidence_class = "held_out_real_test" if data_executed else "development_or_none"
            elif number == 6:
                campaign_ran = bool(
                    test_phase and exp6_campaign_detail is not None
                    and exp6_campaign_summary is not None
                    and exp6_strategy_detail is not None
                    and exp6_strategy_summary is not None
                    and len(exp6_campaign_detail) and len(exp6_campaign_summary)
                    and len(exp6_strategy_detail) and len(exp6_strategy_summary)
                )
                complete = bool(
                    campaign_ran and exp6_registered_pass
                    and empirical_evidence_gates_passed
                )
                if complete:
                    execution = "REGISTERED_TEST_CAMPAIGN_COMPLETE_ALL_GATES_PASSED"
                    evidence_class = "registered_fault_campaign_and_state_strategy_comparison"
                elif campaign_ran:
                    execution = "REGISTERED_CAMPAIGN_FAILED_OR_EMPIRICALLY_INELIGIBLE"
                    evidence_class = "registered_fault_campaign_and_state_strategy_comparison"
                else:
                    execution = "PARTIAL_UNIT_CONTRACT_ONLY"
                    evidence_class = "software_unit_contract"
            else:
                complete = bool(embedded_gates is not None and embedded_gates["passed"].all())
                execution = "EMPIRICAL_COMPLETE" if complete else "BLOCKED_EXTERNAL_VALIDATION"
                evidence_class = "target_hardware_log" if embedded_gates is not None else "readiness_only"
            claim_status = "NOT_EVALUATED"
            claim_eligible = False
            if hypothesis_status is not None:
                linked = {2: ["H1"], 3: ["H1"], 4: ["H2"], 5: ["H3", "H4"], 7: ["H5"]}.get(number, [])
                if linked:
                    linked_rows = hypothesis_status[hypothesis_status["hypothesis"].isin(linked)]
                    claim_eligible = bool(len(linked_rows) == len(linked) and linked_rows["execution_status"].eq("EMPIRICAL_COMPLETE").all())
                    claim_status = "SUPPORTED" if len(linked_rows) == len(linked) and linked_rows["claim_supported"].all() else "NOT_SUPPORTED_OR_INELIGIBLE"
            completion_rows.append({
                "experiment": number, "name": name, "execution_status": execution,
                "evidence_class": evidence_class, "empirical_complete": complete,
                "scientific_claim_eligible": bool(complete and claim_eligible),
                "claim_status": claim_status,
            })
        completion_status = pd.DataFrame(completion_rows)
        completion_artifact = ctx.run_root / "results" / "final_completion_status.csv.gz"
        atomic_write_frame(completion_artifact, completion_status)
        overall_complete = bool(completion_status["empirical_complete"].all())
        confirmatory_complete = bool(
            len(completion_status[completion_status["experiment"].le(6)]) == 6
            and completion_status.loc[
                completion_status["experiment"].le(6), "empirical_complete"
            ].astype(bool).all()
        )
        atomic_write_json(ctx.run_root / "results" / "STUDY_STATUS.json", {
            "overall_status": "FULL_EMPIRICAL_STUDY_COMPLETE" if overall_complete else "PARTIAL_OR_BLOCKED",
            "confirmatory_test_status": (
                "COMPLETE_ALL_GATES_PASSED" if confirmatory_complete
                else "BLOCKED_OR_EMPIRICALLY_INELIGIBLE"
            ),
            "all_confirmatory_gates_passed": all_confirmatory_gates_passed,
            "all_hypotheses_supported": bool(hypothesis_status is not None and hypothesis_status["claim_supported"].all()),
            "note": "Empirical completion and hypothesis support are separate; a completed null result remains complete.",
        })
        display(completion_status)
        if test_access_opened and hypothesis_status is not None:
            result_artifacts = [
                ctx.run_root / "results" / "test_detector_residency.json",
                ctx.run_root / "results" / "test_exp1_perception_detail.csv.gz",
                ctx.run_root / "results" / "test_exp1_perception_summary.csv.gz",
                ctx.run_root / "results" / "test_exp2_representation_frontier.csv.gz",
                ctx.run_root / "results" / "test_exp3_static_partition_index.csv.gz",
                ctx.run_root / "results" / "test_exp3_static_summary.csv.gz",
                ctx.run_root / "results" / "test_policy_trials.csv.gz",
                ctx.run_root / "results" / "test_controller_behavior_detail.csv.gz",
                ctx.run_root / "results" / "test_controller_behavior_summary.csv.gz",
                ctx.run_root / "results" / "test_exp5_risk_uncertainty_detail.csv.gz",
                ctx.run_root / "results" / "test_exp5_risk_uncertainty_summary.csv.gz",
                ctx.run_root / "results" / "test_budget_compliance.csv.gz",
                ctx.run_root / "results" / "confirmatory_gate_table.csv.gz",
                ctx.run_root / "results" / "test_calibration_report.csv.gz",
                ctx.run_root / "results" / "h1_semantic_noninferiority.csv.gz",
                ctx.run_root / "results" / "h2_confirmatory_report.csv.gz",
                ctx.run_root / "results" / "h3_mission_awareness.csv.gz",
                ctx.run_root / "results" / "h4_uncertainty_escalation.csv.gz",
                ctx.run_root / "results" / "test_h4_matched_coverage_trials.csv.gz",
                ctx.run_root / "results" / "test_h4_matched_coverage_gate.csv.gz",
                ctx.run_root / "results" / "test_receiver_robustness_detail.csv.gz",
                ctx.run_root / "results" / "test_receiver_robustness_summary.csv.gz",
                ctx.run_root / "results" / "test_receiver_state_strategy_detail.csv.gz",
                ctx.run_root / "results" / "test_receiver_state_strategy_summary.csv.gz",
                ctx.run_root / "results" / "hypothesis_status.csv.gz",
                completion_artifact,
            ]
            ledger_status = "COMPLETE" if confirmatory_complete else "BLOCKED"
            finalize_test_once(ctx, status=ledger_status, result_artifacts=result_artifacts)
            print(
                "One-time untouched-test ledger finalized:", ledger_status,
                "(claim support reported separately).",
            )
        print("FINAL STATUS:", "FULL EMPIRICAL STUDY COMPLETE" if overall_complete else "PARTIAL/BLOCKED; SEE FINAL COMPLETION TABLE")
    """),
    md("""
    # Publication figure suite

    Available measured frames are rendered through the shared fail-closed
    publication suite. Missing inputs produce explicit manifest entries rather
    than placeholder plots. Dry-run and incomplete-study figures are visibly
    watermarked as non-scientific.
    """),
    code("""
    # Dataset balance must describe the complete frozen allocation. The phase
    # analysis manifest is deliberately excluded because validation/test filter
    # it by visibility and cohort, which would make the paper split counts
    # phase-dependent. Embedded mode reads the same checksum-verified frozen
    # split artifact without reopening image or annotation data.
    publication_split_manifest = split_manifest
    if publication_split_manifest is None and frozen_test_protocol is not None:
        frozen_split_path = resolve_frozen_artifact(
            frozen_test_protocol, "split_manifest",
        )
        publication_split_manifest = pd.read_csv(
            frozen_split_path, compression="infer",
        )

    publication_h1_report = h1_report
    publication_h2_report = confirmatory_report
    publication_h3_report = h3_report
    publication_h4_report = h4_report
    if test_handoff is not None:
        # Embedded publication consumes only the already verified finalized
        # test handoff; it never recomputes or reopens the untouched test.
        publication_h1_report = pd.read_csv(
            test_handoff.artifact("h1_semantic_noninferiority.csv.gz"),
            compression="gzip",
        )
        publication_h2_report = pd.read_csv(
            test_handoff.artifact("h2_confirmatory_report.csv.gz"),
            compression="gzip",
        )
        publication_h3_report = pd.read_csv(
            test_handoff.artifact("h3_mission_awareness.csv.gz"),
            compression="gzip",
        )
        publication_h4_report = pd.read_csv(
            test_handoff.artifact("h4_uncertainty_escalation.csv.gz"),
            compression="gzip",
        )
    hypothesis_effects = build_hypothesis_effects_frame(
        h1_report=publication_h1_report,
        h2_report=publication_h2_report,
        h3_report=publication_h3_report,
        h4_report=publication_h4_report,
        hypothesis_status=hypothesis_status,
        embedded_retest_effects=embedded_retest_effects,
        embedded_retest_status=embedded_retest_status,
    )

    publication_frames = {
        "dataset_manifest": publication_split_manifest,
        "exp1_summary": exp1_summary,
        "exp2_frontier": exp2_frontier,
        "exp3_summary": exp3_summary,
        "exp4_policy": exp4_policy,
        "exp5_summary": exp5_summary,
        "exp6_campaign_summary": (
            exp6_campaign_summary
            if exp6_campaign_summary is not None
            else exp6_robustness
        ),
        "exp7_readiness": exp7_readiness,
        "exp7_hardware": embedded_records,
        "hypothesis_effects": hypothesis_effects,
    }
    if dry_outputs is not None:
        publication_frames.update({
            key: value
            for key, value in dry_outputs.items()
            if isinstance(value, pd.DataFrame)
        })

    publication_scientific = bool(
        RUN_PROFILE == "full"
        and (
            (
                FULL_PHASE == "test"
                and globals().get("empirical_evidence_gates_passed", False)
            )
            or (
                FULL_PHASE == "embedded"
                and h5_embedded_status is not None
                and len(h5_embedded_status)
                and h5_embedded_status["execution_status"].eq("EMPIRICAL_COMPLETE").all()
            )
        )
    )
    publication_figures = generate_publication_figures(
        frames=publication_frames,
        output_dir=PAPER_FIGURE_DIR,
        scientific=publication_scientific,
    )
    for figure_id, figure in publication_figures.items():
        print("Publication figure:", figure_id)
        display(figure)
        plt.close(figure)
    print("Publication figure files and manifests:", PAPER_FIGURE_DIR)
    """),
]

# Notebook format 4.5 requires stable cell IDs.  Emitting deterministic IDs
# prevents Jupyter from injecting random values on first save.
for cell_index, cell in enumerate(cells):
    cell["id"] = f"semuav-{cell_index:02d}"


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python (SemUAV-VLM)", "language": "python", "name": "semuav-vlm"},
        "language_info": {"name": "python", "version": "3.10"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

OUTPUT.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT} with {len(cells)} cells")
