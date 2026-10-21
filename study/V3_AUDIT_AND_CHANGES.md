# SemUAV-VLM v3 family: audit and implementation record

This release continues the supplied 2026-09-21 v2r13 source. It does not use the
separate V6 notebook design. The v2r13 execution evidence includes the
Experiment 2 native-cache exception from FFmpeg on a 1360 × 765 image.
The subsequent v3 deployment stopped at CPU acceptance: 568 tests reported
zero assertion failures and three test-helper import errors. V3R1 fixes those
imports and adds a regression with an unrelated installed `tests` package.
The full gate remains mandatory. See `REPAIR_NOTES.md`; the v3r1 source, run,
cache, kernel and receipt namespaces preserve the failed v3 deployment.
The earlier handoff's statement that v2r13 validation had not been reported is
therefore superseded by that traceback. No completed untouched-test campaign
or empirical paper results were supplied.

## Corrections

1. **Odd video dimensions.** The H264/YUV420 encoder previously rejected an
   odd width or height. V3 pads the right/bottom edge to an even coded extent,
   transmits original dimensions in the video payload, and restores that
   extent before receiver inference. This avoids resizing the source or
   shifting normalized ROI coordinates. Added metadata counts toward wire
   size. Mixed-size clips and invalid geometry are rejected explicitly.
2. **Results appeared too late.** Experiment 1 results previously were saved
   only after the Experiment 2 cache/frontier returned. V3 commits completed
   experiment tables before the next experiment, then calls a synchronous
   result-and-figure hook. The notebook displays each experiment's tables and
   publication figures before computation continues to the next experiment.
3. **Receiver progress was lost on interruption.** V3 adds verified recovery
   of completed native-receiver measurement rows within the same run and
   measurement context. Raw Experiment 1 recovery remains a separate,
   hash-verified v2r9 import. A partial v2r13 native cache is not trusted as v3
   measurement evidence.
4. **Cache identity could disagree with consumed data.** Existing canonical
   cache files must agree with the frame supplied to downstream experiments.
   A different in-memory cache cannot silently be paired with an older file
   path in the frozen protocol.
5. **The Pareto plot could compare unmatched samples.** Video variants exist
   only for clip-bearing observations. V3 aligns variant support for frontier
   comparisons and reports the paired support. Equal-byte inferior points
   cannot appear on the Pareto frontier.
6. **Figure failures were too easy to miss.** Each completed experiment has a
   required figure set, a separate phase/experiment destination and a source
   hash manifest. Missing required inputs or rendering errors are surfaced.
   Validation figures are labeled development evidence; interim test figures
   do not establish a completed confirmatory campaign.
7. **An optional readiness check could block the active study.** The
   `tc_netem` check belongs to deferred Experiment 7b. Its absence is reported
   without treating it as an active VisDrone software prerequisite. Required
   GPU, packages, codecs, and all other active checks retain their gates.
8. **Successor provenance.** V3 has a new release, run, cache, kernel and
   deployment receipt. It verifies its supplied v2r13 predecessor and the
   existing candidate environment, with a fresh three-model development smoke
   and scientific-use seal required. Historical receipts are not rewritten.
9. **Saved notebook acceptance.** Software tests now accept documented
   parameter edits and saved execution outputs. They cannot reject a valid
   prepare/validation notebook solely because it has already run.
10. **Final-report artifact discovery.** Interim previews live in the sibling
    `experiment_previews` tree. Final report hashing includes files in
    `paper_figures`, avoiding a directory-as-file error when previews exist.
    Presentation failures are contained so they cannot consume or re-finalize
    the one-time scientific test ledger.

## Scientific design retained

The three VLMs remain Qwen3-VL-2B, SmolVLM2-2.2B and the Qwen3-VL-8B upper-bound
reference, with detector-only as the fourth general perception pipeline.
Only the two compact VLMs are eligible for selection. Physical GPU 1 remains
mapped to logical `cuda:0`. No training, dataset acquisition, model download,
package installation or unregistered prediction repair is added.

The seven registered experiments, source-rule truth, mission profiles,
thresholds, sample sizes, seeds, native receivers, measured repetition counts,
static grid, temporal design and claim gates are retained. Validation uses
seven temporal sequences and twelve traces; test retains fifteen sequences
and thirty traces. Test data remain behind protocol freeze and the one-time
authorization ledger. H2/H3 design limitations and H5's absent target-device
evidence remain visible; v3 does not convert them to positive claims.

## Recovery and measurement compatibility

The preserved recovery source is the authenticated 132 raw v2r9 inference
artifacts containing 4,152 rows. The producer, model implementation, prompts,
measurement code, canary and requirements are checked byte-for-byte against
the supplied predecessor inventory. Runtime request, model, image and shard
checks still apply. V3 recomputes derived metrics, selection, semantic packets,
receiver measurements and downstream experiments in the new namespace.

An audit also found that inherited `profile_callable` latency includes NVML
sampler setup and teardown. V3 retains that measurement scope consistently,
including recovered raw timings; it does not retroactively describe them as
GPU-kernel-only or operation-only latency. Sampler overhead can matter for
small payloads. GPU-board energy excludes CPU codecs, radio and whole-system
energy. Separating profiler overhead would require a separately identified
measurement change and new comparable raw timing evidence.

## Evidence limits

Local checks exercise software contracts, real CPU video codecs, cache
recovery and plotting with clearly non-scientific fixtures. They cannot
establish model accuracy, throughput on the lab workstation, scientific
hypothesis support or publication acceptance. The lab dataset, model weights,
private receipts, historical raw shards and CUDA/NVML hardware are not present
in this workspace. See `VALIDATION_REPORT.md` for the actual checks completed.

The source package contains no empirical results. Run it on the existing lab
environment using `VISDRONE_PAPER_RUN_GUIDE.md`.
